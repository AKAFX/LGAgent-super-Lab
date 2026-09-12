"""Deterministic, versioned skill catalog for the LegalMCQ workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .models import SolveMode

_KNOWN_TOOLS = frozenset(
    {
        "evaluate_legal_answer",
        "oath_evidence_provider",
        "prepare_legal_answer",
    }
)


@dataclass(frozen=True)
class SkillDescriptor:
    name: str
    display_name: str
    description: str
    reference: str
    version: str
    extra_tools: tuple[str, ...] = ()
    hidden_from_catalog: bool = False

    def prompt_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "reference": self.reference,
        }


class LegalSkillRegistry:
    """Load validated local skills without exposing evaluation capabilities."""

    def __init__(self, skills: Mapping[str, SkillDescriptor]) -> None:
        self._skills = dict(skills)
        if "legal-mcq-core" not in self._skills:
            raise ValueError("legal-mcq-core skill is required")

    @classmethod
    def load(cls, directory: str | Path | None = None) -> "LegalSkillRegistry":
        root = Path(directory or Path(__file__).with_name("skill_specs"))
        skills: dict[str, SkillDescriptor] = {}
        for path in sorted(root.glob("*.yaml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}: skill document must be an object")
            allowed = {
                "name",
                "display_name",
                "description",
                "reference",
                "version",
                "extra_tools",
                "hidden_from_catalog",
            }
            unknown = set(payload) - allowed
            if unknown:
                raise ValueError(f"{path}: unknown fields: {sorted(unknown)}")
            name = str(payload.get("name", "")).strip()
            if not name or name in skills:
                raise ValueError(f"{path}: skill name is empty or duplicated")
            if path.stem != name:
                raise ValueError(f"{path}: filename must match skill name {name!r}")
            tools = payload.get("extra_tools", [])
            if not isinstance(tools, list) or not all(
                isinstance(item, str) and item.strip() for item in tools
            ):
                raise ValueError(f"{path}: extra_tools must be a string array")
            unknown_tools = set(tools) - _KNOWN_TOOLS
            if unknown_tools:
                raise ValueError(f"{path}: unknown tools: {sorted(unknown_tools)}")
            skills[name] = SkillDescriptor(
                name=name,
                display_name=str(payload.get("display_name", name)).strip(),
                description=str(payload.get("description", "")).strip(),
                reference=str(payload.get("reference", "")).strip(),
                version=str(payload.get("version", "")).strip(),
                extra_tools=tuple(item.strip() for item in tools),
                hidden_from_catalog=bool(payload.get("hidden_from_catalog", False)),
            )
            if not skills[name].version or not skills[name].reference:
                raise ValueError(f"{path}: version and reference are required")
        return cls(skills)

    def production_catalog(self) -> tuple[SkillDescriptor, ...]:
        return tuple(
            skill
            for skill in self._skills.values()
            if not skill.hidden_from_catalog
        )

    def preload(self, mode: SolveMode) -> tuple[SkillDescriptor, ...]:
        names = ["legal-mcq-core"]
        if mode is SolveMode.OPEN_BOOK:
            names.append("legal-research")
        missing = [name for name in names if name not in self._skills]
        if missing:
            raise ValueError(f"required LegalMCQ skills are missing: {missing}")
        skills = tuple(self._skills[name] for name in names)
        if any(skill.hidden_from_catalog for skill in skills):
            raise ValueError("evaluation-only skills cannot enter production prompts")
        return skills
