"""Configuration-only routing shared by LGAgent entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .config import LGAgentConfig, LGAgentPlusConfig

ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class PipelineRoute:
    name: str
    oath_rag_enabled: bool
    cape_v_enabled: bool

    @property
    def corrected_baseline(self) -> bool:
        return self.name == "corrected_baseline"


@dataclass(frozen=True)
class LegacyEntrypointSettings:
    enable_lawyer_a: bool
    enable_judge: bool
    enable_dialogue: bool
    use_rag: bool


def resolve_pipeline_route(config: LGAgentConfig | LGAgentPlusConfig) -> PipelineRoute:
    plus = config.lgagent_plus if isinstance(config, LGAgentConfig) else config
    if not plus.enabled:
        return PipelineRoute(
            name="corrected_baseline",
            oath_rag_enabled=False,
            cape_v_enabled=False,
        )
    return PipelineRoute(
        name="lgagent_plus",
        oath_rag_enabled=plus.oath_rag.enabled,
        cape_v_enabled=plus.cape_v.enabled,
    )


def resolve_legacy_entrypoint_settings(
    route: PipelineRoute,
    *,
    enable_lawyer_a: bool,
    enable_judge: bool,
    enable_dialogue: bool,
    use_rag: bool,
) -> LegacyEntrypointSettings:
    """Keep legacy batch flags only while LGAgent++ routing is active."""
    if route.corrected_baseline:
        return LegacyEntrypointSettings(True, True, True, False)
    return LegacyEntrypointSettings(
        enable_lawyer_a,
        enable_judge,
        enable_dialogue,
        use_rag,
    )


class LGAgentPlusOrchestrator(Generic[ResultT]):
    """Apply only the feature stages selected by validated configuration."""

    def __init__(self, config: LGAgentConfig | LGAgentPlusConfig) -> None:
        self.route = resolve_pipeline_route(config)

    def run(
        self,
        corrected_baseline: Callable[[], ResultT],
        *,
        oath_rag: Callable[[ResultT], ResultT] | None = None,
        cape_v: Callable[[ResultT], ResultT] | None = None,
    ) -> ResultT:
        if not self.route.corrected_baseline:
            if self.route.oath_rag_enabled and oath_rag is None:
                raise RuntimeError("OATH-RAG is enabled but no stage was configured")
            if self.route.cape_v_enabled and cape_v is None:
                raise RuntimeError("CAPE-V is enabled but no stage was configured")

        result = corrected_baseline()
        if self.route.oath_rag_enabled:
            assert oath_rag is not None
            result = oath_rag(result)
        if self.route.cape_v_enabled:
            assert cape_v is not None
            result = cape_v(result)
        return result
