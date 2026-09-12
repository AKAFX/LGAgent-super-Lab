"""Role-aware client construction for the LegalMCQ workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..config import LGAgentConfig, ModelConfig
from ..model import ChatModel, OpenAIChatModel


class LegalModelClientError(RuntimeError):
    """Raised when a configured LegalMCQ role cannot create a model client."""


@dataclass(frozen=True)
class LegalRoleClients:
    controller: ChatModel
    solver: ChatModel
    solver_fallback: ChatModel | None
    verifier: ChatModel
    controller_config: ModelConfig
    solver_config: ModelConfig
    solver_fallback_config: ModelConfig | None
    verifier_config: ModelConfig


def resolve_role_configs(
    config: LGAgentConfig,
) -> tuple[ModelConfig, ModelConfig, ModelConfig]:
    settings = config.legal_mcq
    controller = settings.controller_model or config.generation
    solver = settings.solver_model or config.generation
    verifier = settings.verifier_model or controller
    return controller, solver, verifier


def build_openai_role_clients(
    config: LGAgentConfig,
    *,
    timeout: float = 60.0,
    max_retries: int = 0,
    client_factory: Callable[..., Any] | None = None,
) -> LegalRoleClients:
    """Create role clients without hidden SDK retries that bypass run budgets."""
    if not config.legal_mcq.enabled:
        raise LegalModelClientError("legal_mcq.enabled must be true")
    if client_factory is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LegalModelClientError(
                "openai is required for LegalMCQ execution"
            ) from exc
        client_factory = OpenAI

    controller_config, solver_config, verifier_config = resolve_role_configs(config)
    solver_fallback_config = config.legal_mcq.solver_fallback_model
    clients: dict[tuple[str, str], OpenAIChatModel] = {}

    def build(role: str, role_config: ModelConfig) -> OpenAIChatModel:
        if role_config.backend != "openai":
            raise LegalModelClientError(
                f"{role} backend must be openai-compatible, got "
                f"{role_config.backend!r}"
            )
        if not role_config.api_key:
            raise LegalModelClientError(f"{role} model has no API key")
        identity = (role_config.base_url, role_config.api_key)
        if identity not in clients:
            clients[identity] = OpenAIChatModel(
                client_factory(
                    api_key=role_config.api_key,
                    base_url=role_config.base_url,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )
        return clients[identity]

    return LegalRoleClients(
        controller=build("controller", controller_config),
        solver=build("solver", solver_config),
        solver_fallback=(
            build("solver_fallback", solver_fallback_config)
            if solver_fallback_config is not None
            else None
        ),
        verifier=build("verifier", verifier_config),
        controller_config=controller_config,
        solver_config=solver_config,
        solver_fallback_config=solver_fallback_config,
        verifier_config=verifier_config,
    )
