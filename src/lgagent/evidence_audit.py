"""Evidence-only option audit and serializable Evidence Matrix contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from .config import ModelConfig
from .corpus import LegalEvidence
from .model import (
    BudgetExceededError,
    ChatMessage,
    ChatModel,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from .oath_rag import EvidenceMatrix, RetrievedEvidence
from .protocol import StructuredOutputError, extract_json_object
from .serialization import dumps_json, to_jsonable
from .trace import ModelCallTrace, RunTrace, utc_now


class AuditLabel(str, Enum):
    SUPPORT = "SUPPORT"
    REFUTE = "REFUTE"
    EXCEPTION = "EXCEPTION"
    IRRELEVANT = "IRRELEVANT"
    TEMPORALLY_INVALID = "TEMPORALLY_INVALID"


class TemporalStatus(str, Enum):
    VALID = "VALID"
    MIXED = "MIXED"
    INVALID = "INVALID"
    NO_EVIDENCE = "NO_EVIDENCE"


class RetrievalStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FALLBACK_EMPTY = "FALLBACK_EMPTY"


class RetrievalFailureMode(str, Enum):
    FAIL_CLOSED = "fail_closed"
    EMPTY_EVIDENCE = "empty_evidence"


class EvidenceAuditError(ValueError):
    """Raised when an audit response is not grounded in supplied evidence."""


class EvidenceRetrievalError(RuntimeError):
    """Raised when evidence retrieval fails in fail-closed mode."""


class OptionEvidenceRetriever(Protocol):
    def retrieve(
        self,
        analysis: Mapping[str, Any] | object,
        *,
        jurisdiction: str | None = None,
        case_date: date | str | None = None,
    ) -> EvidenceMatrix:
        ...


@dataclass(frozen=True)
class EvidenceAuditConfig:
    authority_ceiling: int = 5
    retrieval_failure_mode: RetrievalFailureMode = RetrievalFailureMode.FAIL_CLOSED
    max_tokens: int = 2048
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.authority_ceiling <= 0:
            raise ValueError("authority_ceiling must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if isinstance(self.retrieval_failure_mode, str):
            object.__setattr__(
                self,
                "retrieval_failure_mode",
                RetrievalFailureMode(self.retrieval_failure_mode),
            )


@dataclass(frozen=True)
class AuditedEvidence:
    evidence_id: str
    label: AuditLabel
    exact_span: str
    retrieval_lanes: tuple[str, ...]
    source_type: str
    law_name: str
    article: str
    clause: str | None
    authority_level: int
    effective_from: date | None
    effective_to: date | None
    source_uri: str


@dataclass(frozen=True)
class OptionEvidenceAudit:
    support: tuple[AuditedEvidence, ...] = ()
    refute: tuple[AuditedEvidence, ...] = ()
    exception: tuple[AuditedEvidence, ...] = ()
    irrelevant: tuple[AuditedEvidence, ...] = ()
    temporally_invalid: tuple[AuditedEvidence, ...] = ()
    coverage: float = 0.0
    conflict: float = 0.0
    authority: float = 0.0
    temporal: TemporalStatus = TemporalStatus.NO_EVIDENCE

    @property
    def evidence(self) -> tuple[AuditedEvidence, ...]:
        return (
            *self.support,
            *self.refute,
            *self.exception,
            *self.irrelevant,
            *self.temporally_invalid,
        )


@dataclass(frozen=True)
class AuditedEvidenceMatrix:
    options: dict[str, OptionEvidenceAudit]
    retrieval_status: RetrievalStatus = RetrievalStatus.SUCCESS
    retrieval_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = to_jsonable(self)
        assert isinstance(value, dict)
        return value


def _field(value: Mapping[str, Any] | object, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _case_date(value: date | str | None) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("case_date must be an ISO date or date object")
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("case_date must be an ISO date (YYYY-MM-DD)") from exc


def is_temporally_valid(evidence: LegalEvidence, case_date: date | None) -> bool:
    """Return whether an evidence version is effective at the requested date."""
    if case_date is None:
        return True
    if evidence.effective_from is not None and evidence.effective_from > case_date:
        return False
    if evidence.effective_to is not None and case_date > evidence.effective_to:
        return False
    return True


def calculate_option_evidence_status(
    evidence: Sequence[AuditedEvidence],
    *,
    authority_ceiling: int = 5,
) -> tuple[float, float, float, TemporalStatus]:
    """Calculate deterministic coverage, conflict, authority, and temporal state."""
    if authority_ceiling <= 0:
        raise ValueError("authority_ceiling must be positive")
    if not evidence:
        return 0.0, 0.0, 0.0, TemporalStatus.NO_EVIDENCE

    labels = {item.label for item in evidence}
    represented = sum(
        label in labels
        for label in (AuditLabel.SUPPORT, AuditLabel.REFUTE, AuditLabel.EXCEPTION)
    )
    coverage = represented / 3.0

    support_weight = sum(
        item.authority_level
        for item in evidence
        if item.label is AuditLabel.SUPPORT
    )
    opposing_weight = sum(
        item.authority_level
        for item in evidence
        if item.label in (AuditLabel.REFUTE, AuditLabel.EXCEPTION)
    )
    conflict = (
        min(support_weight, opposing_weight) / max(support_weight, opposing_weight)
        if support_weight and opposing_weight
        else 0.0
    )

    relevant_valid = [
        item
        for item in evidence
        if item.label
        in (AuditLabel.SUPPORT, AuditLabel.REFUTE, AuditLabel.EXCEPTION)
    ]
    authority = (
        min(
            1.0,
            max(item.authority_level for item in relevant_valid)
            / authority_ceiling,
        )
        if relevant_valid
        else 0.0
    )

    invalid_count = sum(
        item.label is AuditLabel.TEMPORALLY_INVALID for item in evidence
    )
    if invalid_count == len(evidence):
        temporal = TemporalStatus.INVALID
    elif invalid_count:
        temporal = TemporalStatus.MIXED
    else:
        temporal = TemporalStatus.VALID
    return coverage, conflict, authority, temporal


def build_option_evidence_audit(
    evidence: Sequence[AuditedEvidence],
    *,
    authority_ceiling: int = 5,
) -> OptionEvidenceAudit:
    ordered = tuple(sorted(evidence, key=lambda item: item.evidence_id))
    coverage, conflict, authority, temporal = calculate_option_evidence_status(
        ordered,
        authority_ceiling=authority_ceiling,
    )

    def items(label: AuditLabel) -> tuple[AuditedEvidence, ...]:
        return tuple(item for item in ordered if item.label is label)

    return OptionEvidenceAudit(
        support=items(AuditLabel.SUPPORT),
        refute=items(AuditLabel.REFUTE),
        exception=items(AuditLabel.EXCEPTION),
        irrelevant=items(AuditLabel.IRRELEVANT),
        temporally_invalid=items(AuditLabel.TEMPORALLY_INVALID),
        coverage=coverage,
        conflict=conflict,
        authority=authority,
        temporal=temporal,
    )


_AUDITOR_PROMPT = """你是 Evidence Auditor。你只能使用用户消息中提供的选项主张和证据原文，不得使用常识、参数知识或外部法律知识。
逐一判断每个 option-evidence pair，标签只能是 SUPPORT、REFUTE、EXCEPTION、IRRELEVANT。
每项必须原样复制证据 text 中一个非空连续片段作为 exact_span，并原样返回 evidence_id。
必须且只能输出一次每个输入 evidence_id。仅输出严格 JSON：
{"audits":[{"evidence_id":"...","label":"SUPPORT","exact_span":"证据原文连续片段"}]}"""


class EvidenceAuditor:
    """Model-assisted auditor with deterministic provenance validation."""

    def __init__(
        self,
        model: ChatModel,
        model_config: ModelConfig | Mapping[str, Any],
        *,
        config: EvidenceAuditConfig | None = None,
        current_date: date | None = None,
    ) -> None:
        self.model = model
        self.model_config = (
            model_config
            if isinstance(model_config, ModelConfig)
            else ModelConfig.from_mapping(model_config)
        )
        self.config = config or EvidenceAuditConfig()
        self.current_date = current_date or date.today()

    @staticmethod
    def _option_claims(
        analysis: Mapping[str, Any] | object,
    ) -> dict[str, str]:
        raw = _field(analysis, "option_claims")
        if not isinstance(raw, Mapping) or not raw:
            raise EvidenceAuditError(
                "analysis.option_claims must be a non-empty mapping"
            )
        claims: dict[str, str] = {}
        for key, value in raw.items():
            option = str(key).strip().upper()
            if isinstance(value, Mapping):
                value = value.get("claim")
            elif hasattr(value, "claim"):
                value = getattr(value, "claim")
            claim = str(value or "").strip()
            if not option or not claim:
                raise EvidenceAuditError("option labels and claims must not be empty")
            claims[option] = claim
        return dict(sorted(claims.items()))

    @staticmethod
    def _candidates(
        lanes: Mapping[str, Sequence[RetrievedEvidence]],
    ) -> dict[str, tuple[LegalEvidence, tuple[str, ...]]]:
        collected: dict[str, tuple[LegalEvidence, set[str]]] = {}
        for lane, retrieved in sorted(lanes.items()):
            for item in retrieved:
                evidence_id = item.evidence_id
                previous = collected.get(evidence_id)
                if previous is not None and previous[0] != item.evidence:
                    raise EvidenceAuditError(
                        f"evidence_id {evidence_id!r} maps to conflicting records"
                    )
                if previous is None:
                    collected[evidence_id] = (item.evidence, {str(lane)})
                else:
                    previous[1].add(str(lane))
        return {
            evidence_id: (item, tuple(sorted(item_lanes)))
            for evidence_id, (item, item_lanes) in sorted(collected.items())
        }

    @staticmethod
    def _record(
        evidence: LegalEvidence,
        *,
        label: AuditLabel,
        exact_span: str,
        lanes: tuple[str, ...],
    ) -> AuditedEvidence:
        return AuditedEvidence(
            evidence_id=evidence.evidence_id,
            label=label,
            exact_span=exact_span,
            retrieval_lanes=lanes,
            source_type=evidence.source_type,
            law_name=evidence.law_name,
            article=evidence.article,
            clause=evidence.clause,
            authority_level=evidence.authority_level,
            effective_from=evidence.effective_from,
            effective_to=evidence.effective_to,
            source_uri=evidence.source_uri,
        )

    @staticmethod
    def _parse_response(
        text: str,
        candidates: Mapping[str, tuple[LegalEvidence, tuple[str, ...]]],
    ) -> tuple[AuditedEvidence, ...]:
        try:
            payload = extract_json_object(text)
        except StructuredOutputError as exc:
            raise EvidenceAuditError(str(exc)) from exc
        if set(payload) != {"audits"} or not isinstance(payload["audits"], list):
            raise EvidenceAuditError("audit output must contain only an audits array")

        records: list[AuditedEvidence] = []
        seen: set[str] = set()
        for index, raw in enumerate(payload["audits"]):
            if not isinstance(raw, Mapping):
                raise EvidenceAuditError(f"audits[{index}] must be an object")
            if set(raw) != {"evidence_id", "label", "exact_span"}:
                raise EvidenceAuditError(
                    f"audits[{index}] must contain evidence_id, label, exact_span"
                )
            evidence_id = raw["evidence_id"]
            if not isinstance(evidence_id, str) or evidence_id not in candidates:
                raise EvidenceAuditError(
                    f"audits[{index}] cites unknown evidence_id {evidence_id!r}"
                )
            if evidence_id in seen:
                raise EvidenceAuditError(
                    f"evidence_id {evidence_id!r} is audited more than once"
                )
            try:
                label = AuditLabel(str(raw["label"]).strip().upper())
            except ValueError as exc:
                raise EvidenceAuditError(
                    f"audits[{index}].label is invalid"
                ) from exc
            if label is AuditLabel.TEMPORALLY_INVALID:
                raise EvidenceAuditError(
                    "TEMPORALLY_INVALID is assigned only by the deterministic gate"
                )
            exact_span = raw["exact_span"]
            evidence, lanes = candidates[evidence_id]
            if (
                not isinstance(exact_span, str)
                or not exact_span
                or exact_span not in evidence.text
            ):
                # #region debug-point C:exact-span-mismatch
                try:
                    import json as _debug_json
                    import urllib.request as _debug_urlrequest
                    with open(".dbg/joint-budget-failures.env", encoding="utf-8") as _debug_stream:
                        _debug_url = next(line.split("=", 1)[1] for line in _debug_stream.read().splitlines() if line.startswith("DEBUG_SERVER_URL="))
                    _debug_payload = _debug_json.dumps({"sessionId": "joint-budget-failures", "runId": "post-fix", "hypothesisId": "C", "location": "evidence_audit.py:_parse_response:exact_span", "msg": "[DEBUG] Evidence exact span mismatch", "data": {"evidence_id": evidence_id, "audit_index": index, "span_length": len(exact_span) if isinstance(exact_span, str) else None, "evidence_length": len(evidence.text), "stripped_match": isinstance(exact_span, str) and exact_span.strip() in evidence.text, "normalized_match": isinstance(exact_span, str) and " ".join(exact_span.split()) in " ".join(evidence.text.split())}}).encode("utf-8")
                    _debug_urlrequest.urlopen(_debug_urlrequest.Request(_debug_url, data=_debug_payload, headers={"Content-Type": "application/json"}), timeout=0.2).read()
                except Exception:
                    pass
                # #endregion
                raise EvidenceAuditError(
                    f"audits[{index}].exact_span is not an exact evidence substring"
                )
            seen.add(evidence_id)
            records.append(
                EvidenceAuditor._record(
                    evidence,
                    label=label,
                    exact_span=exact_span,
                    lanes=lanes,
                )
            )
        missing = set(candidates) - seen
        if missing:
            # #region debug-point C:omitted-evidence
            try:
                import json as _debug_json
                import urllib.request as _debug_urlrequest
                with open(".dbg/joint-budget-failures.env", encoding="utf-8") as _debug_stream:
                    _debug_url = next(line.split("=", 1)[1] for line in _debug_stream.read().splitlines() if line.startswith("DEBUG_SERVER_URL="))
                _debug_payload = _debug_json.dumps({"sessionId": "joint-budget-failures", "runId": "post-fix", "hypothesisId": "C", "location": "evidence_audit.py:_parse_response:missing", "msg": "[DEBUG] Evidence audit omitted IDs", "data": {"candidate_count": len(candidates), "seen_count": len(seen), "missing_count": len(missing), "response_length": len(text)}}).encode("utf-8")
                _debug_urlrequest.urlopen(_debug_urlrequest.Request(_debug_url, data=_debug_payload, headers={"Content-Type": "application/json"}), timeout=0.2).read()
            except Exception:
                pass
            # #endregion
            raise EvidenceAuditError(
                f"audit output omitted evidence IDs: {sorted(missing)}"
            )
        return tuple(records)

    def audit(
        self,
        analysis: Mapping[str, Any] | object,
        retrieved: EvidenceMatrix,
        *,
        case_date: date | str | None = None,
        trace: RunTrace | None = None,
    ) -> AuditedEvidenceMatrix:
        claims = self._option_claims(analysis)
        requested_date = _case_date(
            case_date if case_date is not None else _field(analysis, "case_date")
        ) or self.current_date
        if set(retrieved) != set(claims):
            raise EvidenceAuditError(
                "retrieved matrix options must exactly match analysis.option_claims"
            )

        options: dict[str, OptionEvidenceAudit] = {}
        for option, claim in claims.items():
            candidates = self._candidates(retrieved[option])
            valid: dict[str, tuple[LegalEvidence, tuple[str, ...]]] = {}
            records: list[AuditedEvidence] = []
            for evidence_id, (evidence, lanes) in candidates.items():
                if is_temporally_valid(evidence, requested_date):
                    valid[evidence_id] = (evidence, lanes)
                else:
                    records.append(
                        self._record(
                            evidence,
                            label=AuditLabel.TEMPORALLY_INVALID,
                            exact_span=evidence.text,
                            lanes=lanes,
                        )
                    )

            if valid:
                evidence_payload = [
                    {
                        "evidence_id": evidence_id,
                        "text": evidence.text,
                        "retrieval_lanes": list(lanes),
                    }
                    for evidence_id, (evidence, lanes) in valid.items()
                ]
                messages = [
                    ChatMessage("system", _AUDITOR_PROMPT),
                    ChatMessage(
                        "user",
                        dumps_json(
                            {
                                "option": option,
                                "claim": claim,
                                "evidence": evidence_payload,
                            },
                            indent=None,
                        ),
                    ),
                ]
                last_error: BaseException | None = None
                for attempt in range(1, self.config.max_attempts + 1):
                    request = ModelRequest(
                        model=self.model_config.model,
                        messages=tuple(messages),
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=min(
                            self.model_config.max_tokens,
                            self.config.max_tokens,
                        ),
                        metadata={
                            "agent": "evidence_auditor",
                            "option": option,
                            "attempt": attempt,
                        },
                    )
                    started_at = utc_now()
                    started = perf_counter()
                    response: ModelResponse | None = None
                    error: BaseException | None = None
                    parsed: tuple[AuditedEvidence, ...] | None = None
                    try:
                        response = self.model.complete(request)
                        parsed = self._parse_response(response.content, valid)
                    except BaseException as exc:
                        error = exc
                        last_error = exc
                    if trace is not None:
                        message = str(error) if error is not None else None
                        if message and self.model_config.api_key:
                            message = message.replace(
                                self.model_config.api_key, "[REDACTED]"
                            )
                        trace.add_call(
                            ModelCallTrace(
                                call_id=uuid4().hex,
                                agent="evidence_auditor",
                                model=self.model_config.model,
                                attempt=attempt,
                                started_at=started_at,
                                duration_ms=(perf_counter() - started) * 1000,
                                usage=(
                                    response.usage
                                    if response
                                    else getattr(error, "usage", TokenUsage())
                                ),
                                request_id=(
                                    response.request_id if response else None
                                ),
                                seed_requested=(
                                    response.seed_requested
                                    if response
                                    else getattr(error, "diagnostics", {}).get(
                                        "seed_requested"
                                    )
                                ),
                                provider_seed_guarantee=(
                                    response.provider_seed_guarantee
                                    if response
                                    else getattr(error, "diagnostics", {}).get(
                                        "provider_seed_guarantee"
                                    )
                                ),
                                error_type=(
                                    type(error).__name__ if error else None
                                ),
                                error_message=message,
                            )
                        )
                        if error is not None:
                            trace.add_error(
                                "evidence_auditor",
                                error,
                                message=message or "",
                            )
                    if error is None:
                        assert parsed is not None
                        records.extend(parsed)
                        break
                    if isinstance(error, BudgetExceededError):
                        error.diagnostics.setdefault(
                            "trace",
                            trace.as_dict() if trace is not None else {},
                        )
                        raise error
                    if attempt < self.config.max_attempts:
                        if response is not None and response.content.strip():
                            messages.append(
                                ChatMessage("assistant", response.content.strip())
                            )
                        messages.append(
                            ChatMessage(
                                "user",
                                "上一响应未通过严格证据审计："
                                f"{error}。必须逐一返回输入中的每个 evidence_id，"
                                "exact_span 必须逐字复制对应 evidence.text 的连续子串。"
                                "请仅重新输出完整 JSON。",
                            )
                        )
                else:
                    assert last_error is not None
                    raise last_error
            options[option] = build_option_evidence_audit(
                records,
                authority_ceiling=self.config.authority_ceiling,
            )
        return AuditedEvidenceMatrix(options=options)

    def empty_matrix(
        self,
        analysis: Mapping[str, Any] | object,
        *,
        retrieval_error: str,
    ) -> AuditedEvidenceMatrix:
        claims = self._option_claims(analysis)
        return AuditedEvidenceMatrix(
            options={option: OptionEvidenceAudit() for option in claims},
            retrieval_status=RetrievalStatus.FALLBACK_EMPTY,
            retrieval_error=retrieval_error,
        )


class EvidenceAuditPipeline:
    """Retrieve and audit evidence with an explicit failure policy."""

    def __init__(
        self,
        retriever: OptionEvidenceRetriever,
        auditor: EvidenceAuditor,
    ) -> None:
        self.retriever = retriever
        self.auditor = auditor

    def run(
        self,
        analysis: Mapping[str, Any] | object,
        *,
        jurisdiction: str | None = None,
        case_date: date | str | None = None,
        trace: RunTrace | None = None,
    ) -> AuditedEvidenceMatrix:
        try:
            retrieved = self.retriever.retrieve(
                analysis,
                jurisdiction=jurisdiction,
                case_date=case_date,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if (
                self.auditor.config.retrieval_failure_mode
                is RetrievalFailureMode.FAIL_CLOSED
            ):
                raise EvidenceRetrievalError(
                    f"evidence retrieval failed: {message}"
                ) from exc
            return self.auditor.empty_matrix(
                analysis,
                retrieval_error=message,
            )
        return self.auditor.audit(
            analysis,
            retrieved,
            case_date=case_date,
                trace=trace,
        )
