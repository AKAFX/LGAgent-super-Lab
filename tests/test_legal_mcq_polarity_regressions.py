from datetime import date
import json
from pathlib import Path

import pytest

from lgagent.legal_mcq.parser import parse_question_text, parse_request


@pytest.mark.parametrize("ending", [
    "\u4e0d\u6b63\u786e\u7684\u662f?",
    "\u4e0d \u6b63\u786e\u7684\u662f\uff1f",
    "\u4e0d\n\u6b63\u786e\u7684\u662f?",
    "\u9519\u8bef\u7684\u662f\uff1f",
    "\u4e0d\u7b26\u5408\u6cd5\u5f8b\u89c4\u5b9a\u7684\u662f?",
])
def test_final_negative_directive_keeps_negation(ending):
    stem = "\u4e0b\u5217\u6709\u5173\u8d23\u4efb\u627f\u62c5\u7684\u8bba\u8ff0\uff0c"
    parsed = parse_request(parse_question_text(stem + ending + "\nA. X\nB. Y"))
    assert parsed.asks_for_incorrect_option is True


def test_narrative_negative_word_does_not_reverse_positive_directive():
    stem = "\u6cd5\u9662\u8ba4\u4e3a\u7532\u7684\u5904\u7406\u4e0d\u6b63\u786e\uff0c\u4e0b\u5217\u6b63\u786e\u7684\u662f?"
    parsed = parse_request(parse_question_text(stem + "\nA. X\nB. Y"))
    assert parsed.asks_for_incorrect_option is False


def test_current_hypothesis_takes_precedence_over_old_case_but_not_metadata():
    stem = "1992\u5e74\u53d1\u751f\u672c\u6848\u3002\u5728\u5f53\u4ee3\uff0c\u82e5\u672c\u6848\u91cd\u73b0\uff0c\u54ea\u9879\u89c4\u5219\u9002\u7528?"
    request = parse_question_text(stem + "\nA. X\nB. Y")
    now = date(2026, 9, 11)
    parsed = parse_request(request, runtime_date=now)
    assert parsed.request.as_of_date == now
    assert parsed.date_source == "question_current"

    request = parse_question_text(stem + "\nA. X\nB. Y", as_of_date=date(2024, 1, 1))
    assert parse_request(request, runtime_date=now).date_source == "request_metadata"


def test_lexgenius_observed_parser_failures():
    dataset = Path(__file__).resolve().parents[1] / "data/LexGenius.jsonl"
    rows = {r["id"]: r for r in map(json.loads, dataset.read_text().splitlines())}
    now = date(2026, 9, 11)
    parsed = {
        key: parse_request(parse_question_text(rows[key]["question"]), runtime_date=now)
        for key in (5, 12, 20, 25, 68, 85)
    }
    assert all(not parsed[key].asks_for_incorrect_option for key in (5, 12, 25))
    assert parsed[85].asks_for_incorrect_option is True
    assert parsed[20].request.as_of_date == date(2023, 12, 31)
    assert parsed[68].request.as_of_date == now
