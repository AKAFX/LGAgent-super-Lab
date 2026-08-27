"""Unified production runner for corrected LGAgent and LGAgent++."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from .cape_v import (
    CapeVConfig,
    CapeVResult,
    CapeVRunner,
    ReasoningCandidate,
    calculate_cape_v_metrics,
)
from .config import LGAgentConfig, ModelConfig
from .counterfactual import (
    CounterfactualObservation,
    TransformationExpectation,
    calculate_counterfactual_metrics,
    harmless_text_normalization,
    option_reorder_transformations,
)
from .corpus import CorpusValidationError, load_jsonl_corpus
from .domain import SingleQuestionResult, SingleQuestionRunner
from .evidence_audit import (
    AuditedEvidenceMatrix,
    EvidenceAuditConfig,
    EvidenceAuditPipeline,
    EvidenceAuditor,
    RetrievalFailureMode,
    TemporalStatus,
)
from .model import BudgetedSeededChatModel, ChatModel, ExecutionBudget
from .oath_rag import OathRagConfig, OathRagRetriever
from .protocol import LawyerAOutput, OptionClaim
from .risk import (
    AdaptiveRiskOrchestrator,
    BudgetCost,
    CandidateSignals,
    RiskRoute,
    RiskSnapshot,
)
from .serialization import to_jsonable
from .trace import RunTrace
from .verification import VerificationContext, VerificationReport, VerificationRunner


class LGAgentPlusRunner:
    """Execute the baseline and enabled LGAgent++ stages in dependency order."""

    def __init__(
        self,
        config: LGAgentConfig,
        *,
        reasoning_model: ChatModel,
        evaluation_model: ChatModel,
        evaluation_config: ModelConfig | Mapping[str, Any] | None = None,
        verifier_model: ChatModel | None = None,
        evidence_pipeline: EvidenceAuditPipeline | None = None,
        project_root: str | Path | None = None,
        max_dialogue_rounds: int = 2,
        evidence_lanes: Sequence[str] = ("support", "refute", "exception"),
        fixed_compute: bool = False,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.config = config
        self.reasoning_model = reasoning_model
        self.evaluation_model = evaluation_model
        self.evaluation_config = (
            evaluation_config or config.generation
        )
        self.verifier_model = verifier_model or evaluation_model
        self.project_root = Path(project_root or Path.cwd())
        self.max_dialogue_rounds = max_dialogue_rounds
        self.evidence_lanes = tuple(evidence_lanes)
        self.fixed_compute = fixed_compute
        self._evidence_pipeline = evidence_pipeline
        self.clock = clock

    def _build_evidence_pipeline(
        self,
        reasoning_model: ChatModel,
    ) -> EvidenceAuditPipeline:
        if self._evidence_pipeline is not None:
            auditor = copy(self._evidence_pipeline.auditor)
            auditor.model = reasoning_model
            return EvidenceAuditPipeline(
                self._evidence_pipeline.retriever,
                auditor,
            )
        settings = self.config.lgagent_plus.oath_rag
        corpus_path = Path(settings.corpus_path)
        if not corpus_path.is_absolute():
            corpus_path = self.project_root / corpus_path
        corpus = load_jsonl_corpus(corpus_path, allow_legacy=False)
        retriever = OathRagRetriever(
            corpus,
            config=OathRagConfig(
                lexical_top_k=settings.lexical_top_k,
                dense_top_k=settings.dense_top_k,
                final_top_k_per_lane=settings.final_top_k_per_lane,
                min_authority_level=(
                    1 if settings.require_authoritative_source else 0
                ),
                graph_hops=settings.graph_hops,
                enabled_lanes=self.evidence_lanes,
                require_temporal_match=settings.require_temporal_match,
            ),
        )
        failure_mode = (
            RetrievalFailureMode.FAIL_CLOSED
            if settings.corpus_failure_mode == "fail_closed"
            else RetrievalFailureMode.EMPTY_EVIDENCE
        )
        auditor = EvidenceAuditor(
            reasoning_model,
            self.config.generation,
            config=EvidenceAuditConfig(retrieval_failure_mode=failure_mode),
        )
        return EvidenceAuditPipeline(retriever, auditor)

    def _evidence_provider(
        self,
        analysis: LawyerAOutput,
        trace: RunTrace,
        reasoning_model: ChatModel,
    ) -> AuditedEvidenceMatrix:
        settings = self.config.lgagent_plus.oath_rag
        trace.add_route(
            "oath-rag",
            "retrieve and audit before Lawyer B1 and CAPE-V",
        )
        try:
            pipeline = self._build_evidence_pipeline(reasoning_model)
        except CorpusValidationError as exc:
            if settings.corpus_failure_mode == "fail_closed":
                raise
            auditor = EvidenceAuditor(
                reasoning_model,
                self.config.generation,
                config=EvidenceAuditConfig(
                    retrieval_failure_mode=RetrievalFailureMode.EMPTY_EVIDENCE
                ),
            )
            matrix = auditor.empty_matrix(
                analysis,
                retrieval_error=f"{type(exc).__name__}: {exc}",
            )
            trace.add_route(
                "oath-rag-fallback",
                "strict corpus failed; configured empty evidence fallback",
                {"error_type": type(exc).__name__},
            )
        else:
            matrix = pipeline.run(
                analysis,
                jurisdiction=analysis.jurisdiction or None,
                case_date=analysis.case_date,
                trace=trace,
            )
        trace.add_route(
            "evidence-audited",
            "audited evidence matrix is ready for downstream candidates",
            {"retrieval_status": matrix.retrieval_status.value},
        )
        return matrix

    @staticmethod
    def _candidate_signals(
        candidate: ReasoningCandidate,
        report: VerificationReport,
        matrix: AuditedEvidenceMatrix | None,
        permutation_consistency: float | None,
    ) -> CandidateSignals:
        if matrix is None:
            coverage = authority = temporal = 1.0
        else:
            option = matrix.options[candidate.answer]
            coverage = option.coverage
            authority = option.authority
            temporal = {
                TemporalStatus.VALID: 1.0,
                TemporalStatus.MIXED: 0.5,
                TemporalStatus.INVALID: 0.0,
                TemporalStatus.NO_EVIDENCE: 0.0,
            }[option.temporal]
        return CandidateSignals(
            candidate=candidate,
            evidence_coverage=coverage,
            authority_score=authority,
            temporal_validity=temporal,
            verification_report=report,
            permutation_consistency=(
                1.0
                if permutation_consistency is None
                else permutation_consistency
            ),
        )

    def _verify(
        self,
        question: str,
        analysis: LawyerAOutput,
        candidates: Sequence[ReasoningCandidate],
        matrix: AuditedEvidenceMatrix | None,
        permutation_consistency: float | None,
        trace: RunTrace,
        verifier_model: ChatModel,
    ) -> tuple[CandidateSignals, ...]:
        cape = self.config.lgagent_plus.cape_v
        verifier_config = replace(cape.verifier, enabled=cape.verifiers)
        verifier = VerificationRunner(
            verifier_model,
            cape.verifier_model or self.config.generation,
            verifier_config,
        )
        evidence = () if matrix is None else (matrix.as_dict(),)
        exception_hints = tuple(
            hint
            for claim in analysis.option_claims.values()
            for hint in claim.possible_exceptions
        )
        signals = []
        for candidate in candidates:
            report = verifier.run(
                VerificationContext(
                    question=question,
                    candidate=candidate,
                    facts=analysis.facts,
                    evidence=evidence,
                    exception_hints=exception_hints,
                ),
                trace=trace,
            )
            signals.append(
                self._candidate_signals(
                    candidate,
                    report,
                    matrix,
                    permutation_consistency,
                )
            )
        return tuple(signals)

    def _cape_run(
        self,
        question: str,
        *,
        candidate_count: int,
        permutation_count: int,
        trusted_context: Mapping[str, Any],
        trace: RunTrace,
        evaluation_model: ChatModel,
        candidate_prefix: str = "",
    ) -> CapeVResult:
        settings = self.config.lgagent_plus.cape_v
        return CapeVRunner(
            evaluation_model,
            self.evaluation_config,
            CapeVConfig(
                candidate_count=candidate_count,
                permutation_count=permutation_count,
                seed=self.config.lgagent_plus.seed,
                max_attempts=settings.verifier.max_attempts,
                max_tokens=settings.candidate_max_tokens,
            ),
        ).run(
            question,
            trace=trace,
            trusted_context=trusted_context,
            candidate_prefix=candidate_prefix,
        )

    @staticmethod
    def _targeted_analysis(
        analysis: LawyerAOutput,
        snapshot: RiskSnapshot,
        matrix: AuditedEvidenceMatrix | None,
    ) -> tuple[LawyerAOutput, dict[str, tuple[str, ...]]]:
        """Deterministically turn verifier failures and evidence gaps into queries."""
        gaps: dict[str, list[str]] = {
            option: [] for option in analysis.option_claims
        }
        for signal in snapshot.candidates:
            report = signal.verification_report
            if report is None:
                continue
            option_gaps = gaps[signal.candidate.answer]
            for step in report.failed_steps:
                detail = report.dimensions[step]
                error = detail.error_type or "VERIFICATION_FAILURE"
                option_gaps.append(
                    f"failed {step} verification: {error}; {detail.reason}"
                )

        if matrix is None:
            for option_gaps in gaps.values():
                option_gaps.append("no audited evidence matrix is available")
        else:
            for option, audit in matrix.options.items():
                for lane in ("support", "refute", "exception"):
                    if not getattr(audit, lane):
                        gaps[option].append(f"missing audited {lane} evidence")
                if audit.temporal is not TemporalStatus.VALID:
                    gaps[option].append(
                        f"temporal evidence status is {audit.temporal.value}"
                    )

        normalized_gaps = {
            option: tuple(dict.fromkeys(items))
            for option, items in gaps.items()
        }
        claims: dict[str, OptionClaim] = {}
        for option, claim in analysis.option_claims.items():
            option_gaps = normalized_gaps[option]
            exception_gaps = tuple(
                item for item in option_gaps if "exception" in item.lower()
            )
            claims[option] = replace(
                claim,
                elements=tuple(dict.fromkeys((*claim.elements, *option_gaps))),
                possible_exceptions=tuple(
                    dict.fromkeys((*claim.possible_exceptions, *exception_gaps))
                ),
            )
        all_gaps = tuple(
            f"{option}: {gap}"
            for option, option_gaps in normalized_gaps.items()
            for gap in option_gaps
        )
        return (
            replace(
                analysis,
                question_focus=(
                    f"{analysis.question_focus}; targeted verification and "
                    "evidence-gap retrieval"
                ),
                option_claims=claims,
                unknowns=tuple(dict.fromkeys((*analysis.unknowns, *all_gaps))),
            ),
            normalized_gaps,
        )

    def _counterfactual_observations(
        self,
        question: str,
        *,
        original_answer: str,
        initial_cape: CapeVResult,
        trusted_context: Mapping[str, Any],
        trace: RunTrace,
        evaluation_model: ChatModel,
    ) -> tuple[CounterfactualObservation, ...]:
        settings = self.config.lgagent_plus.cape_v
        reorder_count = max(1, settings.permutation_count)
        transformations = option_reorder_transformations(
            question,
            original_answer=original_answer,
            count=reorder_count,
            seed=self.config.lgagent_plus.seed,
        )
        observations: list[CounterfactualObservation] = []
        reused = min(
            len(transformations),
            len(initial_cape.permutation_candidates),
        )
        for transformation, candidate in zip(
            transformations[:reused],
            initial_cape.permutation_candidates[:reused],
        ):
            observations.append(
                CounterfactualObservation(
                    transformation,
                    candidate.presented_answer,
                )
            )

        for index, transformation in enumerate(
            transformations[reused:],
            start=reused + 1,
        ):
            transformed_context = to_jsonable(trusted_context)
            assert isinstance(transformed_context, dict)
            matrix = transformed_context.get("audited_evidence_matrix")
            if isinstance(matrix, Mapping) and isinstance(
                matrix.get("options"), Mapping
            ):
                remapped_matrix = dict(matrix)
                options = matrix["options"]
                remapped_matrix["options"] = {
                    displayed: options.get(original, {})
                    for displayed, original in (
                        transformation.displayed_to_original.items()
                    )
                }
                transformed_context["audited_evidence_matrix"] = remapped_matrix
            result = self._cape_run(
                transformation.transformed_question,
                candidate_count=1,
                permutation_count=0,
                trusted_context=transformed_context,
                trace=trace,
                evaluation_model=evaluation_model,
                candidate_prefix=f"counterfactual-reorder-{index}-",
            )
            observations.append(
                CounterfactualObservation(
                    transformation,
                    result.candidates[0].presented_answer,
                )
            )

        normalization = harmless_text_normalization(
            question,
            original_answer=original_answer,
        )
        normalized = self._cape_run(
            normalization.transformed_question,
            candidate_count=1,
            permutation_count=0,
            trusted_context=trusted_context,
            trace=trace,
            evaluation_model=evaluation_model,
            candidate_prefix="counterfactual-normalization-",
        )
        observations.append(
            CounterfactualObservation(
                normalization,
                normalized.candidates[0].presented_answer,
            )
        )
        trace.add_route(
            "counterfactual-complete",
            "trusted label-preserving counterfactual observations completed",
            {"count": len(observations), "reused_permutations": reused},
        )
        return tuple(
            observation
            for observation in observations
            if (
                observation.transformation.expectation
                is TransformationExpectation.PRESERVE_LABEL
                or observation.transformation.is_metric_eligible
            )
        )

    def run(self, question: str) -> SingleQuestionResult:
        trace = RunTrace()
        plus = self.config.lgagent_plus
        captured_matrix: AuditedEvidenceMatrix | None = None
        budget = ExecutionBudget(
            max_calls=plus.risk.budget.max_calls,
            max_tokens=plus.risk.budget.max_tokens,
            max_seconds=plus.risk.budget.max_seconds,
            clock=self.clock,
            on_exhausted=lambda reason, details: trace.add_route(
                "budget-exhausted",
                f"global execution budget exhausted before model call: {reason}",
                details,
            ),
        )
        wrapped_models: dict[int, BudgetedSeededChatModel] = {}

        def wrap(model: ChatModel) -> BudgetedSeededChatModel:
            key = id(model)
            if key not in wrapped_models:
                wrapped_models[key] = BudgetedSeededChatModel(
                    model,
                    budget,
                    base_seed=plus.seed,
                )
            return wrapped_models[key]

        reasoning_model = wrap(self.reasoning_model)
        evaluation_model = wrap(self.evaluation_model)
        verifier_model = wrap(self.verifier_model)

        def provide_evidence(
            analysis: LawyerAOutput,
            run_trace: RunTrace,
        ) -> AuditedEvidenceMatrix:
            nonlocal captured_matrix
            captured_matrix = self._evidence_provider(
                analysis,
                run_trace,
                reasoning_model,
            )
            return captured_matrix

        baseline = SingleQuestionRunner(
            reasoning_model,
            evaluation_model,
            self.config.generation,
            self.evaluation_config,
            # CAPE-V replaces the legacy clarification loop with measured,
            # budgeted expansion. Running both would duplicate model calls.
            max_dialogue_rounds=(
                0
                if plus.enabled and plus.cape_v.enabled
                else self.max_dialogue_rounds
            ),
        ).run(
            question,
            evidence_provider=provide_evidence
            if plus.enabled and plus.oath_rag.enabled
            else None,
            trace=trace,
        )
        if not plus.enabled or not plus.cape_v.enabled:
            diagnostics = dict(baseline.diagnostics)
            diagnostics["pipeline_route"] = (
                "oath-only"
                if plus.enabled and plus.oath_rag.enabled
                else "corrected_baseline"
            )
            diagnostics["evidence_matrix"] = (
                captured_matrix.as_dict() if captured_matrix is not None else {}
            )
            diagnostics["execution_budget"] = budget.snapshot()
            diagnostics["reproduction"] = {
                "seed_requested": plus.seed,
                "seed_derivation": "sha256(base_seed,metadata,occurrence)-31bit-v1",
                "provider_seed_guarantee": "requested_not_guaranteed",
            }
            return replace(baseline, diagnostics=diagnostics)

        analysis = LawyerAOutput.from_text(baseline.lawyer_a_output)
        trusted_context: dict[str, Any] = {
            "lawyer_a": analysis,
            "judge": baseline.judge_output,
            "initial_b1": baseline.diagnostics,
        }
        if captured_matrix is not None:
            trusted_context["audited_evidence_matrix"] = captured_matrix
        trace.add_route(
            "cape-v",
            "generate candidates after initial B1 candidate",
            {"audited_context": captured_matrix is not None},
        )
        initial_cape = self._cape_run(
            question,
            candidate_count=plus.cape_v.initial_candidates,
            permutation_count=plus.cape_v.permutation_count,
            trusted_context=trusted_context,
            trace=trace,
            evaluation_model=evaluation_model,
        )
        # Permutation probes measure position stability only. Including them in
        # the vote would count the same semantic candidate multiple times.
        initial_candidates = initial_cape.candidates
        permutation_answers = tuple(
            item.answer for item in initial_cape.permutation_candidates
        )
        initial_metrics = calculate_cape_v_metrics(
            [item.answer for item in initial_candidates],
            permutation_answers,
        )
        signals = self._verify(
            question,
            analysis,
            initial_candidates,
            captured_matrix,
            initial_metrics.permutation_consistency,
            trace,
            verifier_model,
        )
        reports_by_id = {
            item.candidate.candidate_id: item.verification_report
            for item in signals
        }
        active_analysis = analysis
        round_number = 0
        targeted_retrievals: list[dict[str, Any]] = []

        def append_candidates(
            snapshot: RiskSnapshot,
            guard: Any,
            run_trace: RunTrace,
        ) -> RiskSnapshot:
            del guard
            nonlocal round_number
            round_number += 1
            more = self._cape_run(
                question,
                candidate_count=plus.cape_v.slow_path_candidates,
                permutation_count=0,
                trusted_context=trusted_context,
                trace=run_trace,
                evaluation_model=evaluation_model,
                candidate_prefix=f"slow-{round_number}-",
            )
            more_signals = self._verify(
                question,
                active_analysis,
                more.candidates,
                captured_matrix,
                initial_metrics.permutation_consistency,
                run_trace,
                verifier_model,
            )
            reports_by_id.update(
                {
                    item.candidate.candidate_id: item.verification_report
                    for item in more_signals
                }
            )
            combined = (*snapshot.candidates, *more_signals)
            return RiskSnapshot(
                candidates=combined,
                cape_metrics=calculate_cape_v_metrics(
                    [item.candidate.answer for item in combined],
                    permutation_answers,
                ),
                evidence_matrix=(
                    captured_matrix.as_dict()["options"]
                    if captured_matrix is not None
                    else None
                ),
            )

        def retrieve_and_reason(
            snapshot: RiskSnapshot,
            guard: Any,
            run_trace: RunTrace,
        ) -> RiskSnapshot:
            nonlocal active_analysis, captured_matrix
            if not plus.oath_rag.enabled:
                run_trace.add_route(
                    "retrieve-and-reason-degraded",
                    "CAPE-only route has no OATH-RAG retriever; appending candidates",
                    {"fallback": RiskRoute.VERIFY_MORE.value},
                )
                return append_candidates(snapshot, guard, run_trace)

            targeted, gaps = self._targeted_analysis(
                analysis,
                snapshot,
                captured_matrix,
            )
            captured_matrix = self._evidence_provider(
                targeted,
                run_trace,
                reasoning_model,
            )
            active_analysis = targeted
            trusted_context["lawyer_a"] = targeted
            trusted_context["targeted_analysis"] = targeted
            trusted_context["audited_evidence_matrix"] = captured_matrix
            targeted_retrievals.append(
                {
                    "round": len(targeted_retrievals) + 1,
                    "gaps": gaps,
                    "retrieval_status": captured_matrix.retrieval_status.value,
                }
            )
            run_trace.add_route(
                "targeted-retrieval-complete",
                "refreshed evidence from verifier failures and evidence gaps",
                {
                    "gap_count": sum(len(items) for items in gaps.values()),
                    "retrieval_status": captured_matrix.retrieval_status.value,
                },
            )
            return append_candidates(snapshot, guard, run_trace)

        verifier_count = sum(plus.cape_v.verifiers.values())
        slow_count = plus.cape_v.slow_path_candidates
        verify_more_cost = BudgetCost(
            calls=slow_count * (1 + verifier_count),
            tokens=slow_count
            * (
                plus.cape_v.candidate_max_tokens
                + verifier_count * plus.cape_v.verifier.max_tokens
            ),
        )
        evidence_audit_calls = 4 if plus.oath_rag.enabled else 0
        retrieve_and_reason_cost = BudgetCost(
            calls=verify_more_cost.calls + evidence_audit_calls,
            tokens=(
                verify_more_cost.tokens
                + evidence_audit_calls
                * min(self.config.generation.max_tokens, 2048)
            ),
        )
        adaptive = AdaptiveRiskOrchestrator(
            weights=plus.risk.weights,
            thresholds=plus.risk.thresholds,
            budget_limits=plus.risk.budget,
            action_costs={
                RiskRoute.VERIFY_MORE: verify_more_cost,
                RiskRoute.RETRIEVE_AND_REASON: retrieve_and_reason_cost,
            },
            clock=self.clock,
        ).run(
            RiskSnapshot(
                candidates=signals,
                cape_metrics=initial_metrics,
                evidence_matrix=(
                    captured_matrix.as_dict()["options"]
                    if captured_matrix is not None
                    else None
                ),
            ),
            actions={
                RiskRoute.VERIFY_MORE: append_candidates,
                RiskRoute.RETRIEVE_AND_REASON: retrieve_and_reason,
            },
            trace=trace,
            fixed_compute=self.fixed_compute,
        )
        counterfactual_observations: tuple[CounterfactualObservation, ...] = ()
        if plus.cape_v.enable_counterfactual:
            counterfactual_observations = self._counterfactual_observations(
                question,
                original_answer=adaptive.selected.answer,
                initial_cape=initial_cape,
                trusted_context=trusted_context,
                trace=trace,
                evaluation_model=evaluation_model,
            )
        counterfactual_metrics = calculate_counterfactual_metrics(
            counterfactual_observations
        )
        diagnostics = dict(baseline.diagnostics)
        diagnostics.update(
            {
                "pipeline_route": (
                    "joint" if captured_matrix is not None else "cape-only"
                ),
                "evidence_matrix": (
                    captured_matrix.as_dict()
                    if captured_matrix is not None
                    else {}
                ),
                "initial_b1_answer": baseline.final_answer,
                "candidates": [
                    to_jsonable(item.candidate) for item in adaptive.aggregation.ranked
                ],
                "verification_reports": [
                    reports_by_id[item.candidate.candidate_id].as_dict()
                    for item in adaptive.aggregation.ranked
                    if reports_by_id[item.candidate.candidate_id] is not None
                ],
                "permutation_predictions": list(permutation_answers),
                "counterfactual_observations": [
                    {
                        "transformation": to_jsonable(item.transformation),
                        "predicted_answer": item.predicted_answer,
                        "consistent": item.consistent,
                        "eligible": item.transformation.is_metric_eligible,
                        "expected_change": (
                            item.transformation.expectation
                            is TransformationExpectation.CHANGE_LABEL
                        ),
                        "observed_change": (
                            item.predicted_answer
                            != item.transformation.original_answer
                        ),
                    }
                    for item in counterfactual_observations
                ],
                "counterfactual_metrics": to_jsonable(counterfactual_metrics),
                "targeted_retrievals": targeted_retrievals,
                "risk_score": adaptive.risk,
                "risk_routing": {
                    "route": adaptive.route.value,
                    "rounds": adaptive.rounds,
                    "budget_exhausted": adaptive.budget_exhausted,
                    "budget_exhausted_reason": adaptive.budget_exhausted_reason,
                    "selected_candidate_id": adaptive.selected.candidate_id,
                    "aggregation_reason": adaptive.aggregation.reason,
                },
                "execution_budget": budget.snapshot(),
                "reproduction": {
                    "seed_requested": plus.seed,
                    "seed_derivation": (
                        "sha256(base_seed,metadata,occurrence)-31bit-v1"
                    ),
                    "provider_seed_guarantee": "requested_not_guaranteed",
                },
            }
        )
        return replace(
            baseline,
            final_answer=adaptive.selected.answer,
            diagnostics=diagnostics,
        )
