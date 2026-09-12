# CommandCode JSON Schema 独立 dev 验证

## 1. 结论

CommandCode 在本次运行中原生接受了 OpenAI-compatible
`response_format=json_schema` 请求：

- 9 / 9 次 Schema 请求未被网关拒绝；
- 0 次 `unsupported`、`unknown parameter` 或 `invalid schema`；
- 0 次 `legal-mcq-structured-output-fallback`；
- 6 / 9 次调用直接返回并通过严格 Solver v2 本地解析，调用级成功率 66.7%；
- 5 / 5 道题最终都至少取得一次有效的主 Solver v2 决策。

这证明 CommandCode 接受该请求参数，且可返回符合 Schema 的结果。API 没有
提供“服务端 constrained decoding 已启用”的独立证明，因此不能仅凭响应
反推出 Schema 是由服务端强制生成，还是由模型按 Prompt 遵循。

## 2. 样本与配置

- 数据集：`data/LexGenius.jsonl`，SHA-256
  `9033614d7115e1f3054c99dbcbe7295e06f35b5a3256fca3f9952741eea67d1e`
- 冻结划分：dev，split seed `20260826`
- 原始索引：`32, 34, 40, 43, 46`
- 与旧 30 题索引 `0-29` 的重叠：0
- seed：42
- concurrency：1
- Controller / Verifier：`Qwen/Qwen3.7-Flash`
- Solver：`meituan/LongCat-2.0:free`
- 模式：`solver_structured_output_mode=auto`
- 题级预算：7 calls、32768 tokens、180 seconds
- SDK 自动重试：0

样本、配置和哈希位于
`output/legal_mcq/schema-dev5-20260911-seed42/manifest.json`。

## 3. Schema 指标

| 指标 | 结果 |
|---|---:|
| Schema 请求未被供应商拒绝 | 9 / 9，100% |
| auto prompt-only fallback | 0 / 9，0% |
| 调用级 Schema 解析成功 | 6 / 9，66.7% |
| 首次主 Solver 成功 | 4 / 5，80% |
| 两次尝试内主 Solver 成功 | 5 / 5，100% |
| `finish_reason=stop` | 6 / 9 |
| `finish_reason=length` | 3 / 9 |

3 次失败调用都恰好使用 2400 completion tokens：

- ID 34 Revision attempt 1：`length + content=null`；
- ID 34 Revision attempt 2：`length + 190 chars`，JSON 截断；
- ID 40 Solver attempt 1：`length + 1037 chars`，JSON 截断。

因此当前主要问题不是 CommandCode 拒绝 JSON Schema，而是 LongCat 的隐藏
reasoning 与正文共享 2400-token 输出上限。紧凑 v2 降低了风险，但尚未完全
消除复杂题和修订轮次的截断。

## 4. 逐题结果

| ID | 主 Solver | Revision | 最终结果 |
|---|---|---|---|
| 32 | 首次 Schema 成功 | 未触发 | completed，答案正确 |
| 34 | 首次 Schema 成功 | 两次均 `length` 截断 | 工程失败 |
| 40 | 首次截断，第二次 Schema 成功 | 未进入 Solver Revision | Verifier 先 520、后超时，工程失败 |
| 43 | 首次 Schema 成功 | 未触发 | completed，答案正确 |
| 46 | 首次 Schema 成功 | 首次 Schema 成功 | completed，答案正确 |

端到端成功 3 / 5（60%），总体分母准确率 3 / 5（60%）。成功返回的 3 题
均与标签一致，但样本过小，不能用于证明准确率提升或与旧 30 题直接比较。

## 5. 用量与稳定性

- 逻辑模型调用：21；
- 已观察 total tokens：72470；
- 已观察 reasoning tokens：38060，占 52.5%；
- usage 未知调用：2；
- 总题级耗时：717.94 秒；
- 中位题级耗时：161.02 秒；
- 预算前阻断：0。

除 Solver 截断外，还观察到一次 CommandCode HTTP 520 和一次随后超时。
这说明 JSON Schema 接入已经可用，但完整三角色链路仍未达到稳定批量运行
标准。下一优先级应是控制 Solver reasoning/输出上限分配，并减少
`CONTROLLER_CLAIM_COVERAGE_MISMATCH` 引发的高成本全量修订。

该优先项的后续实现与工程重放见
[隐藏推理预算优化](legal_mcq_reasoning_budget_optimization_20260911.md)。

## 6. 工件

- `output/legal_mcq/schema-dev5-20260911-seed42/validation.summary.json`
- `output/legal_mcq/schema-dev5-20260911-seed42/probe/results.calls.jsonl`
- `output/legal_mcq/schema-dev5-20260911-seed42/remaining4/results.calls.jsonl`

调用日志保存了 request ID、`response_format_type`、Schema 名、finish reason、
usage、正文状态和错误类型。模型可见正文仅保存在本地日志中，不包含 API key
或隐藏 reasoning 正文。
