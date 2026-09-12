"""Single-question LGAgent domain workflow."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, TypeVar
from uuid import uuid4

from .config import ModelConfig
from .evidence_audit import AuditedEvidenceMatrix
from .model import (
    BudgetExceededError,
    ChatMessage,
    ChatModel,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from .protocol import (
    B0Output,
    B1Output,
    JudgeOutput,
    LawyerAOutput,
    StructuredOutputError,
    should_request_clarification,
)
from .serialization import dumps_json
from .trace import ModelCallTrace, RunTrace, utc_now

SchemaT = TypeVar("SchemaT")

LAWYER_A_PROMPT = """你是律师A。请保持答案中立，仅输出严格JSON：
{"task_type":"single_choice","question_focus":"...","legal_domain":"...",
"jurisdiction":"CN","case_date":"YYYY-MM-DD或null",
"facts":[{"fact_id":"F1","text":"...","legally_relevant":true}],
"option_claims":{
"A":{"claim":"...","elements":["..."],"possible_exceptions":["..."]},
"B":{"claim":"...","elements":["..."],"possible_exceptions":["..."]},
"C":{"claim":"...","elements":["..."],"possible_exceptions":["..."]},
"D":{"claim":"...","elements":["..."],"possible_exceptions":["..."]}},
"option_keywords":{"A":["..."],"B":["..."],"C":["..."],"D":["..."]},
"trap_signals":["..."],"unknowns":["..."]}
无法从题目确定 jurisdiction 或 case_date 时分别输出空字符串或 null，不得猜测。"""

JUDGE_PROMPT = """你是法官，只规划检索和核验，不得给出答案。仅输出严格JSON：
{"need_retrieval":false,"global_query":"...",
"option_queries":{"A":"...","B":"...","C":"...","D":"..."},
"evidence_requirements":{"A":{"support":"...","refute":"..."},
"B":{"support":"...","refute":"..."},"C":{"support":"...","refute":"..."},
"D":{"support":"...","refute":"..."}},"counterfactual_focus":"...","stop_rule":"..."}"""

B0_PROMPT = """你是律师B的盲答阶段，只能看到原题。仅输出严格JSON：
{"initial_answer":"A","confidence":0.0}"""

B1_PROMPT = """你是律师B的核验阶段。逐项核验后仅输出严格JSON：
{"final_answer":"A","verification":{
"A":{"status":"SUPPORT","score":0.0,"reason":"..."},
"B":{"status":"REFUTE","score":0.0,"reason":"..."},
"C":{"status":"NEI","score":0.0,"reason":"..."},
"D":{"status":"REFUTE","score":0.0,"reason":"..."}},
"initial_answer":"A","initial_confidence":0.0,"reasoning":"..."}"""


@dataclass(frozen=True)
class SingleQuestionResult:
    final_answer: str
    lawyer_a_output: str
    judge_output: str
    diagnostics: dict[str, Any]
    trace: RunTrace

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "lawyer_a_output": self.lawyer_a_output,
            "judge_output": self.judge_output,
            "diagnostics": self.diagnostics,
            "trace": self.trace.as_dict(),
        }


class SingleQuestionRunner:
    """Corrected baseline orchestration over replaceable model clients."""

    def __init__(
        self,
        reasoning_model: ChatModel,
        evaluation_model: ChatModel,
        reasoning_config: ModelConfig | Mapping[str, Any],
        evaluation_config: ModelConfig | Mapping[str, Any],
        *,
        max_dialogue_rounds: int = 2,
        max_attempts: int = 2,
    ) -> None:
        self.reasoning_model = reasoning_model
        self.evaluation_model = evaluation_model
        self.reasoning_config = self._config(reasoning_config)
        self.evaluation_config = self._config(evaluation_config)
        self.max_dialogue_rounds = max_dialogue_rounds
        self.max_attempts = max_attempts
        if max_dialogue_rounds < 0:
            raise ValueError("max_dialogue_rounds cannot be negative")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    @staticmethod
    def _config(value: ModelConfig | Mapping[str, Any]) -> ModelConfig:
        return value if isinstance(value, ModelConfig) else ModelConfig.from_mapping(value)

    def _invoke(
        self,
        model_client: ChatModel,
        config: ModelConfig,
        agent: str,
        messages: list[ChatMessage],
        schema: type[SchemaT],
        trace: RunTrace,
        *,
        max_tokens: int,
    ) -> tuple[str, SchemaT]:
        working_messages = list(messages)
        last_output = ""
        last_error: BaseException | None = None
        last_error_message = ""

        for attempt in range(1, self.max_attempts + 1):
            started_at = utc_now()
            started = perf_counter()
            response: ModelResponse | None = None
            error: Exception | None = None
            parsed: SchemaT | None = None
            try:
                response = model_client.complete(
                    ModelRequest(
                        model=config.model,
                        messages=tuple(working_messages),
                        temperature=config.temperature,
                        top_p=config.top_p,
                        max_tokens=min(config.max_tokens, max_tokens),
                        metadata={"agent": agent, "attempt": attempt},
                    )
                )
                last_output = response.content.strip()
                parsed = schema.from_text(last_output)  # type: ignore[attr-defined]
            except Exception as exc:
                error = exc
                last_error = exc
                last_error_message = str(exc)
                if config.api_key:
                    last_error_message = last_error_message.replace(
                        config.api_key, "[REDACTED]"
                    )

            trace.add_call(
                ModelCallTrace(
                    call_id=uuid4().hex,
                    agent=agent,
                    model=config.model,
                    attempt=attempt,
                    started_at=started_at,
                    duration_ms=(perf_counter() - started) * 1000,
                    usage=(
                        response.usage
                        if response
                        else getattr(error, "usage", TokenUsage())
                    ),
                    request_id=response.request_id if response else None,
                    seed_requested=(
                        response.seed_requested
                        if response
                        else getattr(error, "diagnostics", {}).get("seed_requested")
                    ),
                    provider_seed_guarantee=(
                        response.provider_seed_guarantee
                        if response
                        else getattr(error, "diagnostics", {}).get(
                            "provider_seed_guarantee"
                        )
                    ),
                    error_type=type(error).__name__ if error else None,
                    error_message=last_error_message if error else None,
                )
            )
            if error is None:
                assert parsed is not None
                return last_output, parsed

            trace.add_error(agent, error, message=last_error_message)
            if isinstance(error, BudgetExceededError):
                error.diagnostics.setdefault("trace", trace.as_dict())
                raise error
            if attempt < self.max_attempts:
                if last_output:
                    working_messages.append(ChatMessage("assistant", last_output))
                working_messages.append(
                    ChatMessage(
                        "user",
                        f"上一响应不符合严格JSON协议：{error}。"
                        "请仅重新输出完整、合法且字段齐全的JSON对象。",
                    )
                )

        raise StructuredOutputError(
            agent,
            last_error_message or "model call failed",
            attempts=self.max_attempts,
            raw_output=last_output,
        ) from last_error

    @staticmethod
    def _evidence_text(retrieved_docs: list[str] | None) -> str:
        if not retrieved_docs:
            return "（无检索证据）"
        return "\n\n".join(
            f"[文档{index + 1}]\n{str(document).strip()[:300]}"
            for index, document in enumerate(retrieved_docs[:3])
        )

    @staticmethod
    def _b1_evidence_text(
        retrieved_docs: list[str] | None,
        evidence_matrix: AuditedEvidenceMatrix | None,
    ) -> str:
        if evidence_matrix is None:
            return SingleQuestionRunner._evidence_text(retrieved_docs)
        return (
            "以下 Evidence Matrix 是唯一允许使用的外部法律证据；"
            "不得补充矩阵之外的法律知识。引用时必须使用其中的 evidence_id "
            "和 exact_span。\n"
            f"{dumps_json(evidence_matrix)}"
        )

    def run(
        self,
        question: str,
        *,
        retrieved_docs: list[str] | None = None,
        retrieved_docs_provider: (
            Callable[
                [str, LawyerAOutput, JudgeOutput, B0Output, RunTrace],
                tuple[list[str], Mapping[str, Any]],
            ]
            | None
        ) = None,
        evidence_matrix: AuditedEvidenceMatrix | None = None,
        evidence_provider: (
            Callable[[LawyerAOutput, RunTrace], AuditedEvidenceMatrix] | None
        ) = None,
        trace: RunTrace | None = None,
    ) -> SingleQuestionResult:
        if not question.strip():
            raise ValueError("question cannot be empty")
        run_trace = trace or RunTrace()
        run_trace.add_route("corrected_baseline", "single-question entry")

        lawyer_a_text, lawyer_a = self._invoke(
            self.reasoning_model,
            self.reasoning_config,
            "lawyer_a",
            [
                ChatMessage("system", LAWYER_A_PROMPT),
                ChatMessage("user", f"请解析下面的法律问题：\n\n{question}"),
            ],
            LawyerAOutput,
            run_trace,
            max_tokens=1024,
        )
        if evidence_provider is not None:
            if evidence_matrix is not None:
                raise ValueError(
                    "evidence_matrix and evidence_provider are mutually exclusive"
                )
            evidence_matrix = evidence_provider(lawyer_a, run_trace)
        judge_text, judge = self._invoke(
            self.reasoning_model,
            self.reasoning_config,
            "judge",
            [
                ChatMessage("system", JUDGE_PROMPT),
                ChatMessage(
                    "user",
                    f"原始问题：\n{question}\n\n律师A结构：\n{lawyer_a_text}\n\n"
                    f"检索证据：\n{self._evidence_text(retrieved_docs)}",
                ),
            ],
            JudgeOutput,
            run_trace,
            max_tokens=1024,
        )
        _, b0 = self._invoke(
            self.evaluation_model,
            self.evaluation_config,
            "lawyer_b0",
            [
                ChatMessage("system", B0_PROMPT),
                ChatMessage("user", f"请盲答以下问题：\n\n{question}"),
            ],
            B0Output,
            run_trace,
            max_tokens=256,
        )
        retrieval_diagnostics: Mapping[str, Any] | None = None
        if retrieved_docs_provider is not None:
            if retrieved_docs is not None or evidence_matrix is not None:
                raise ValueError(
                    "retrieved_docs_provider cannot be combined with preloaded evidence"
                )
            retrieved_docs, retrieval_diagnostics = retrieved_docs_provider(
                question,
                lawyer_a,
                judge,
                b0,
                run_trace,
            )

        current_judge = judge_text
        _, b1 = self._run_b1(
            question,
            lawyer_a_text,
            current_judge,
            retrieved_docs,
            b0,
            run_trace,
            evidence_matrix=evidence_matrix,
        )
        dialogue_rounds = 0
        while (
            should_request_clarification(b1.verification, b0.confidence)
            and dialogue_rounds < self.max_dialogue_rounds
        ):
            run_trace.add_route(
                "dialogue",
                "NEI ratio or calibrated B0 confidence crossed threshold",
                {"round": dialogue_rounds + 1},
            )
            current_judge, _ = self._invoke(
                self.reasoning_model,
                self.reasoning_config,
                "judge",
                [
                    ChatMessage("system", JUDGE_PROMPT),
                    ChatMessage(
                        "user",
                        f"原始问题：\n{question}\n\n律师A结构：\n{lawyer_a_text}\n\n"
                        f"当前Judge输出：\n{current_judge}\n\n"
                        f"检索证据：\n{self._evidence_text(retrieved_docs)}\n\n"
                        f"上一轮核验：\n{dumps_json(b1.raw)}\n\n"
                        "请针对NEI或低置信问题更新完整核验指导。",
                    ),
                ],
                JudgeOutput,
                run_trace,
                max_tokens=1024,
            )
            _, b1 = self._run_b1(
                question,
                lawyer_a_text,
                current_judge,
                retrieved_docs,
                b0,
                run_trace,
                evidence_matrix=evidence_matrix,
            )
            dialogue_rounds += 1

        exhausted = should_request_clarification(b1.verification, b0.confidence)
        run_trace.add_route(
            "final",
            "dialogue budget exhausted" if exhausted else "verification complete",
            {"dialogue_rounds": dialogue_rounds, "budget_exhausted": exhausted},
        )
        diagnostics = dict(b1.raw)
        diagnostics.update(
            {
                "b0_blind_answer": b0.initial_answer,
                "b0_confidence": b0.confidence,
                "initial_b1_completed": True,
                "dialogue_rounds": dialogue_rounds,
                "dialogue_exhausted": exhausted,
            }
        )
        if evidence_matrix is not None:
            diagnostics["evidence_matrix"] = evidence_matrix.as_dict()
        if retrieval_diagnostics is not None:
            diagnostics["web_search"] = dict(retrieval_diagnostics)
        return SingleQuestionResult(
            final_answer=b1.final_answer,
            lawyer_a_output=lawyer_a_text,
            judge_output=current_judge,
            diagnostics=diagnostics,
            trace=run_trace,
        )

    def _run_b1(
        self,
        question: str,
        lawyer_a_text: str,
        judge_text: str,
        retrieved_docs: list[str] | None,
        b0: B0Output,
        trace: RunTrace,
        *,
        evidence_matrix: AuditedEvidenceMatrix | None = None,
    ) -> tuple[str, B1Output]:
        return self._invoke(
            self.evaluation_model,
            self.evaluation_config,
            "lawyer_b1",
            [
                ChatMessage("system", B1_PROMPT),
                ChatMessage(
                    "user",
                    f"原始问题：\n{question}\n\n律师A结构：\n{lawyer_a_text}\n\n"
                    f"Judge指导：\n{judge_text}\n\n"
                    "选项—证据审计结果：\n"
                    f"{self._b1_evidence_text(retrieved_docs, evidence_matrix)}\n\n"
                    f"B0盲答：{b0.initial_answer}，置信度：{b0.confidence}",
                ),
            ],
            B1Output,
            trace,
            max_tokens=1024,
        )


def run_single_question(
    question: str,
    *,
    reasoning_model: ChatModel,
    evaluation_model: ChatModel,
    reasoning_config: ModelConfig | Mapping[str, Any],
    evaluation_config: ModelConfig | Mapping[str, Any],
    retrieved_docs: list[str] | None = None,
    retrieved_docs_provider: (
        Callable[
            [str, LawyerAOutput, JudgeOutput, B0Output, RunTrace],
            tuple[list[str], Mapping[str, Any]],
        ]
        | None
    ) = None,
    evidence_matrix: AuditedEvidenceMatrix | None = None,
    evidence_provider: (
        Callable[[LawyerAOutput, RunTrace], AuditedEvidenceMatrix] | None
    ) = None,
    max_dialogue_rounds: int = 2,
) -> SingleQuestionResult:
    return SingleQuestionRunner(
        reasoning_model,
        evaluation_model,
        reasoning_config,
        evaluation_config,
        max_dialogue_rounds=max_dialogue_rounds,
    ).run(
        question,
        retrieved_docs=retrieved_docs,
        retrieved_docs_provider=retrieved_docs_provider,
        evidence_matrix=evidence_matrix,
        evidence_provider=evidence_provider,
    )
