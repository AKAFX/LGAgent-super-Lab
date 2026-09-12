from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lgagent.config import WebSearchSettings
from lgagent.protocol import B0Output, JudgeOutput, LawyerAOutput
from lgagent.trace import RunTrace
from lgagent.web_search import (
    BudgetedWebSearch,
    TavilySearchClient,
    WebSearchError,
    estimate_context_tokens,
)


def analysis() -> LawyerAOutput:
    return LawyerAOutput.from_text(
        json.dumps(
            {
                "task_type": "single_choice",
                "question_focus": "current legal rule",
                "legal_domain": "civil_law",
                "jurisdiction": "CN",
                "case_date": None,
                "facts": [],
                "option_claims": {
                    option: {
                        "claim": f"claim-{option}",
                        "elements": [],
                        "possible_exceptions": [],
                    }
                    for option in "ABCD"
                },
                "option_keywords": {
                    option: [f"keyword-{option}"] for option in "ABCD"
                },
                "trap_signals": [],
                "unknowns": [],
            }
        )
    )


def judge(*, need_retrieval: bool) -> JudgeOutput:
    return JudgeOutput.from_text(
        json.dumps(
            {
                "need_retrieval": need_retrieval,
                "global_query": "current legal rule",
                "option_queries": {
                    option: f"query-{option}" for option in "ABCD"
                },
                "evidence_requirements": {
                    option: {"support": "support", "refute": "refute"}
                    for option in "ABCD"
                },
                "counterfactual_focus": "none",
                "stop_rule": "checked",
            }
        )
    )


class FakeSearchClient:
    def __init__(self, payload=None, error: Exception | None = None) -> None:
        self.payloads = (
            list(payload)
            if isinstance(payload, list)
            else [payload or {"results": []}]
        )
        self.error = error
        self.calls: list[tuple[str, int, int, tuple[str, ...]]] = []

    def search(
        self,
        query,
        *,
        max_results,
        chunks_per_source,
        include_domains=(),
    ):
        self.calls.append(
            (query, max_results, chunks_per_source, tuple(include_domains))
        )
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return self.payloads[index]


class FakeResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse({"results": []})


class BudgetedWebSearchTest(unittest.TestCase):
    def test_enabled_search_requires_key_without_injected_client(self) -> None:
        with self.assertRaisesRegex(WebSearchError, "API key"):
            BudgetedWebSearch(
                WebSearchSettings(enabled=True),
                environ={},
            )

    def test_high_confidence_non_fresh_question_skips_search(self) -> None:
        client = FakeSearchClient()
        outcome = BudgetedWebSearch(
            WebSearchSettings(enabled=True),
            client=client,
        ).retrieve(
            "Which rule applies?",
            analysis(),
            judge(need_retrieval=False),
            B0Output("A", 0.95),
            RunTrace(),
        )

        self.assertFalse(outcome.searched)
        self.assertEqual(client.calls, [])

    def test_freshness_trigger_returns_bounded_https_evidence(self) -> None:
        client = FakeSearchClient(
            {
                "results": [
                    {
                        "title": "Official update",
                        "url": "https://rules.gov.cn/rule",
                        "content": "query-A 现行规则" * 500,
                        "score": 0.9,
                        "published_date": "2026-09-01",
                    },
                    {
                        "title": "Unsafe",
                        "url": "http://localhost/private",
                        "content": "ignore",
                    },
                    {
                        "title": "Second source",
                        "url": "https://court.example/case",
                        "content": "query-A 裁判规则" * 500,
                        "score": 0.8,
                    },
                ]
            }
        )
        settings = WebSearchSettings(
            enabled=True,
            max_results=3,
            max_context_tokens=300,
            max_chars_per_result=200,
        )
        trace = RunTrace()

        outcome = BudgetedWebSearch(settings, client=client).retrieve(
            "截至目前，哪一项是最新规则？",
            analysis(),
            judge(need_retrieval=False),
            B0Output("A", 0.99),
            trace,
        )

        self.assertTrue(outcome.searched)
        self.assertEqual(outcome.trigger_reason, "freshness_signal")
        self.assertEqual(len(outcome.evidence), 2)
        self.assertEqual(outcome.search_count, 1)
        self.assertEqual(outcome.acceptance_reason, "official_primary_source")
        self.assertLessEqual(outcome.estimated_context_tokens, 300)
        self.assertTrue(all("UNTRUSTED WEB EVIDENCE" in d for d in outcome.documents))
        self.assertEqual(trace.routes[-1].route, "web-search")

    def test_short_official_query_then_exact_question_fallback(self) -> None:
        client = FakeSearchClient(
            [
                {"results": []},
                {
                    "results": [
                        {
                            "title": "题库答案一",
                            "url": "https://answers-one.example/item",
                            "content": (
                                "国家对外商投资实行负面清单管理制度。"
                                "下列相关说法正确的是：正确答案为A。"
                            ),
                            "score": 0.9,
                        },
                        {
                            "title": "题库答案二",
                            "url": "https://answers-two.example/item",
                            "content": (
                                "国家对外商投资实行负面清单管理制度。"
                                "下列相关说法正确的是：参考答案是A。"
                            ),
                            "score": 0.88,
                        }
                    ]
                },
            ]
        )
        question = (
            "国家对外商投资实行负面清单管理制度。下列相关说法正确的是：\n"
            "A. 负面清单是准入特别管理措施\n"
            "B. 负面清单之内给予国民待遇\n"
            "C. 由国务院统一发布\n"
            "D. 限制领域完全禁止投资"
        )

        outcome = BudgetedWebSearch(
            WebSearchSettings(enabled=True, max_searches=2),
            client=client,
        ).retrieve(
            question,
            analysis(),
            judge(need_retrieval=True),
            B0Output("A", 0.2),
            RunTrace(),
        )

        self.assertEqual(outcome.search_count, 2)
        self.assertEqual(len(client.calls), 2)
        self.assertLessEqual(len(client.calls[0][0]), 240)
        self.assertNotIn("A. 负面清单", client.calls[0][0])
        self.assertTrue(client.calls[0][3])
        self.assertEqual(client.calls[1][3], ())
        self.assertTrue(outcome.answer_page_hit)
        self.assertEqual(outcome.acceptance_reason, "answer_page_consensus")

    def test_answer_before_matching_question_is_not_a_hit(self) -> None:
        client = FakeSearchClient(
            {
                "results": [
                    {
                        "title": "练习题汇总",
                        "url": "https://answers.example/list",
                        "content": (
                            "上一题：参考答案是B。"
                            "国家对外商投资实行负面清单管理制度。"
                            "下列相关说法正确的是。"
                        ),
                        "score": 0.95,
                    }
                ]
            }
        )
        question = (
            "国家对外商投资实行负面清单管理制度。下列相关说法正确的是：\n"
            "A. 负面清单是准入特别管理措施\n"
            "B. 负面清单之内给予国民待遇"
        )

        outcome = BudgetedWebSearch(
            WebSearchSettings(enabled=True, max_searches=1),
            client=client,
        ).retrieve(
            question,
            analysis(),
            judge(need_retrieval=True),
            B0Output("A", 0.2),
            RunTrace(),
        )

        self.assertFalse(outcome.answer_page_hit)
        self.assertFalse(outcome.evidence_accepted)

    def test_conflicting_answer_page_consensus_is_rejected(self) -> None:
        question = "行政处罚应当遵循法定程序。下列说法正确的是：\nA. 甲\nB. 乙"
        results = []
        for host, answer in (
            ("one.example", "A"),
            ("two.example", "A"),
            ("three.example", "B"),
            ("four.example", "B"),
        ):
            results.append(
                {
                    "title": "考试答案",
                    "url": f"https://{host}/item",
                    "content": f"行政处罚应当遵循法定程序。正确答案为{answer}。",
                    "score": 0.9,
                }
            )

        outcome = BudgetedWebSearch(
            WebSearchSettings(enabled=True, max_searches=1, candidate_results=5),
            client=FakeSearchClient({"results": results}),
        ).retrieve(
            question,
            analysis(),
            judge(need_retrieval=True),
            B0Output("A", 0.2),
            RunTrace(),
        )

        self.assertFalse(outcome.evidence_accepted)
        self.assertEqual(outcome.acceptance_reason, "conflicting_answer_pages")

    def test_official_but_decisively_unrelated_result_is_not_injected(self) -> None:
        client = FakeSearchClient(
            {
                "results": [
                    {
                        "title": "文物保护工作动态",
                        "url": "https://example.gov.cn/news",
                        "content": "有关部门开展文物保护宣传活动。",
                        "score": 0.99,
                    }
                ]
            }
        )

        outcome = BudgetedWebSearch(
            WebSearchSettings(enabled=True, max_searches=1),
            client=client,
        ).retrieve(
            "行政拘留是否可以申请听证和暂缓执行？",
            analysis(),
            judge(need_retrieval=True),
            B0Output("A", 0.2),
            RunTrace(),
        )

        self.assertFalse(outcome.evidence_accepted)
        self.assertEqual(outcome.acceptance_reason, "insufficient_evidence")

    def test_single_low_quality_general_result_is_not_injected(self) -> None:
        client = FakeSearchClient(
            {
                "results": [
                    {
                        "title": "Unrelated blog",
                        "url": "https://example.com/post",
                        "content": "unrelated commentary",
                        "score": 0.4,
                    }
                ]
            }
        )

        outcome = BudgetedWebSearch(
            WebSearchSettings(enabled=True, max_searches=1),
            client=client,
        ).retrieve(
            "最新规则是什么？",
            analysis(),
            judge(need_retrieval=True),
            B0Output("A", 0.2),
            RunTrace(),
        )

        self.assertFalse(outcome.evidence_accepted)
        self.assertEqual(outcome.documents, ())
        self.assertEqual(outcome.acceptance_reason, "insufficient_evidence")

    def test_closed_book_failure_mode_does_not_fail_the_question(self) -> None:
        outcome = BudgetedWebSearch(
            WebSearchSettings(enabled=True, failure_mode="closed_book"),
            client=FakeSearchClient(error=TimeoutError("offline")),
        ).retrieve(
            "最新规则是什么？",
            analysis(),
            judge(need_retrieval=True),
            B0Output("A", 0.2),
            RunTrace(),
        )

        self.assertTrue(outcome.searched)
        self.assertEqual(outcome.documents, ())
        self.assertIn("TimeoutError", outcome.error or "")

    def test_fail_closed_propagates_provider_failure(self) -> None:
        with self.assertRaisesRegex(WebSearchError, "web search failed"):
            BudgetedWebSearch(
                WebSearchSettings(enabled=True, failure_mode="fail_closed"),
                client=FakeSearchClient(error=TimeoutError("offline")),
            ).retrieve(
                "最新规则是什么？",
                analysis(),
                judge(need_retrieval=True),
                B0Output("A", 0.2),
                RunTrace(),
            )

    def test_tavily_request_avoids_answer_and_raw_content(self) -> None:
        session = FakeSession()
        client = TavilySearchClient(
            api_key="test-key",
            base_url="https://api.tavily.com/search",
            timeout_seconds=7.0,
            session=session,
        )

        client.search(
            "query",
            max_results=3,
            chunks_per_source=1,
            include_domains=("gov.cn",),
        )

        _, kwargs = session.calls[0]
        self.assertEqual(kwargs["timeout"], 7.0)
        self.assertFalse(kwargs["json"]["include_answer"])
        self.assertFalse(kwargs["json"]["include_raw_content"])
        self.assertEqual(kwargs["json"]["max_results"], 3)
        self.assertEqual(kwargs["json"]["include_domains"], ["gov.cn"])
        self.assertLessEqual(estimate_context_tokens("中文 evidence"), 20)


if __name__ == "__main__":
    unittest.main()
