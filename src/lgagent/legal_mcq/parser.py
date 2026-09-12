"""Deterministic LegalMCQ parsing before any model is called."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from datetime import date

from .models import (
    LegalAgentError,
    LegalAgentErrorCode,
    LegalOption,
    LegalQuestionRequest,
    ParsedLegalQuestion,
    QuestionType,
    SolveMode,
)

_OPTION_MARKER = re.compile(
    r"(?m)^[ \t]*([A-ZＡ-Ｚ])[ \t]*[.．、:：)）][ \t]*"
)
_INVERSE_STEM = re.compile(
    r"(不正确|错误|不属于|不包括|不符合|不能成立|说法有误|表述有误)"
)
_QUESTION_CUE = re.compile(
    r"(下列|以下|关于|对此|何者|哪(?:一|项|个|种|些)|"
    r"如何|是否|能否|应否|不正确的是|正确的是|错误的是|说法.{0,4}的是)"
)
_SENTENCE_BOUNDARY = re.compile(r"[。！？!?]+")
_MULTIPLE_CHOICE = re.compile(
    r"(多项选择|可多选|有几项|哪些.{0,8}(正确|错误|符合|不符合)|"
    r"所有正确|全部正确)"
)
_DATE_TOKEN = re.compile(
    r"(?<!\d)((?:19|20)\d{2})(?:[-/.年](\d{1,2})[-/.月](\d{1,2})日?|年)"
)
_HYPOTHETICAL_DATE_CUE = re.compile(r"(若|如果|假设|假定|设想|设定)")
_CURRENT_TIME_CUE = re.compile(r"(当代|现行|如今|目前|现在|当前|今日|今天)")
_CASE_DATE_CUE = re.compile(
    r"(案发|案件发生|行为发生|事故发生|合同(?:签订|订立)|"
    r"裁判作出|判决作出|发生于|发生在)"
)
_NAMED_LAW = re.compile(r"《([^》]{2,80})》")
_FULLWIDTH_OFFSET = ord("Ａ") - ord("A")


def _ascii_label(value: str) -> str:
    character = value.upper()
    if "Ａ" <= character <= "Ｚ":
        return chr(ord(character) - _FULLWIDTH_OFFSET)
    return character


def _final_question_clause(stem: str) -> str:
    clauses = [
        clause.strip()
        for clause in _SENTENCE_BOUNDARY.split(stem)
        if clause.strip()
    ]
    if clauses:
        return clauses[-1]
    return stem[-200:].strip()


def _question_directive(stem: str) -> str:
    """Return the final directive rather than negative words in case facts."""
    clause = re.sub(r"\s+", "", _final_question_clause(stem))
    cues = list(_QUESTION_CUE.finditer(clause))
    if cues:
        return clause[cues[-1].start() :].strip()
    return clause


def _asks_about_current_time(stem: str) -> bool:
    clauses = [c.strip() for c in _SENTENCE_BOUNDARY.split(stem) if c.strip()]
    if not clauses:
        return False
    if _CURRENT_TIME_CUE.search(clauses[-1]):
        return True
    return bool(
        len(clauses) > 1
        and re.match(r"(此时|这时|在此情形下|在此情况下)", clauses[-1])
        and _CURRENT_TIME_CUE.search(clauses[-2])
        and _HYPOTHETICAL_DATE_CUE.search(clauses[-2])
    )


def _date_candidates(value: str, warnings: list[str]) -> list[tuple[date, int, bool]]:
    candidates: list[tuple[date, int, bool]] = []
    for match in _DATE_TOKEN.finditer(value):
        year, month, day = match.groups()
        year_only = month is None
        try:
            parsed = (
                date(int(year), 12, 31)
                if year_only
                else date(int(year), int(month), int(day))
            )
        except ValueError:
            warnings.append("invalid_explicit_date")
            continue
        candidates.append((parsed, match.start(), year_only))
    return candidates


def _assumption_date(
    stem: str,
    candidates: list[tuple[date, int, bool]],
) -> tuple[date, bool] | None:
    for parsed, position, year_only in reversed(candidates):
        boundary = max(
            (stem.rfind(mark, 0, position) for mark in "。！？!?"),
            default=-1,
        )
        prefix = stem[boundary + 1 : position]
        if _HYPOTHETICAL_DATE_CUE.search(prefix):
            return parsed, year_only
    return None


def _candidate_in_text(
    text: str,
    warnings: list[str],
) -> tuple[date, bool] | None:
    candidates = _date_candidates(text, warnings)
    if not candidates:
        return None
    parsed, _, year_only = candidates[-1]
    return parsed, year_only


def _mark_year_only(warnings: list[str], year_only: bool) -> None:
    if year_only:
        warnings.append("year_only_date_assumed_end")


def parse_question_text(
    question: str,
    *,
    question_id: str | None = None,
    mode: SolveMode = SolveMode.CLOSED_BOOK,
    jurisdiction: str = "CN",
    as_of_date: date | None = None,
    question_type: QuestionType | None = None,
) -> LegalQuestionRequest:
    """Split a plain-text question into a production request without an LLM."""
    if not isinstance(question, str) or not question.strip():
        raise LegalAgentError(
            LegalAgentErrorCode.INVALID_QUESTION_FORMAT,
            "question text cannot be empty",
        )
    matches = list(_OPTION_MARKER.finditer(question))
    if len(matches) < 2:
        raise LegalAgentError(
            LegalAgentErrorCode.INVALID_QUESTION_FORMAT,
            "at least two labelled options are required",
        )

    labels = [_ascii_label(match.group(1)) for match in matches]
    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if labels != expected or len(set(labels)) != len(labels):
        raise LegalAgentError(
            LegalAgentErrorCode.INVALID_QUESTION_FORMAT,
            f"option labels must be unique and consecutive: {expected}",
            details={"observed_labels": labels},
        )

    stem = question[: matches[0].start()].strip()
    if not stem:
        raise LegalAgentError(
            LegalAgentErrorCode.INVALID_QUESTION_FORMAT,
            "question stem cannot be empty",
        )
    options = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        text = question[match.end() : end].strip()
        if not text:
            raise LegalAgentError(
                LegalAgentErrorCode.INVALID_QUESTION_FORMAT,
                f"option {labels[index]} cannot be empty",
            )
        options.append(LegalOption(labels[index], text))

    inferred_type = (
        QuestionType.MULTIPLE_CHOICE
        if _MULTIPLE_CHOICE.search(stem)
        else QuestionType.SINGLE_CHOICE
    )
    identifier = question_id or (
        "legal-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    )
    return LegalQuestionRequest(
        question_id=str(identifier),
        stem=stem,
        options=tuple(options),
        question_type=question_type or inferred_type,
        jurisdiction=jurisdiction,
        as_of_date=as_of_date,
        mode=mode,
    )


def parse_request(
    request: LegalQuestionRequest,
    *,
    runtime_date: date | None = None,
) -> ParsedLegalQuestion:
    """Add deterministic polarity and temporal metadata to a request."""
    today = runtime_date or date.today()
    warnings: list[str] = []
    stem = request.stem
    question_clause = _final_question_clause(stem)
    directive = _question_directive(stem)
    all_candidates = _date_candidates(stem, warnings)
    assumed = _assumption_date(stem, all_candidates)
    directive_date = _candidate_in_text(question_clause, warnings)

    if assumed is not None:
        resolved_date, year_only = assumed
        _mark_year_only(warnings, year_only)
        date_source = "question_hypothetical"
    elif directive_date is not None:
        resolved_date, year_only = directive_date
        _mark_year_only(warnings, year_only)
        date_source = "question_explicit"
    elif request.as_of_date is not None:
        resolved_date = request.as_of_date
        date_source = "request_metadata"
    elif _asks_about_current_time(stem):
        resolved_date = today
        date_source = "question_current"
    else:
        case_candidates = [
            item
            for item in all_candidates
            if _CASE_DATE_CUE.search(stem[max(0, item[1] - 40) : item[1]])
        ]
        selected = case_candidates[-1] if case_candidates else (
            all_candidates[0] if all_candidates else None
        )
        if selected is not None:
            resolved_date, _, year_only = selected
            _mark_year_only(warnings, year_only)
            if len(all_candidates) > 1 and not case_candidates:
                warnings.append("multiple_dates_first_assumed")
            date_source = "case_explicit"
        else:
            resolved_date = today
            date_source = "runtime_default"
            if re.search(r"(旧法|修订前|修法前|当时|彼时)", stem):
                warnings.append("possible_historical_law_without_date")

    normalized_request = replace(request, as_of_date=resolved_date)
    return ParsedLegalQuestion(
        request=normalized_request,
        asks_for_incorrect_option=bool(_INVERSE_STEM.search(directive)),
        date_source=date_source,
        named_laws=tuple(dict.fromkeys(_NAMED_LAW.findall(stem))),
        parse_warnings=tuple(dict.fromkeys(warnings)),
    )


def format_question(request: LegalQuestionRequest) -> str:
    lines = [request.stem, ""]
    lines.extend(f"{option.label}. {option.text}" for option in request.options)
    return "\n".join(lines)
