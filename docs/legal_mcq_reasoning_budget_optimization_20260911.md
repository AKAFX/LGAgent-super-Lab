# LegalMCQ 隐藏推理预算优化

## 1. 结论

本轮加入角色级 `reasoning_effort`，默认不发送，CommandCode 配置使用
`low`。该参数能显著降低 Qwen Controller/Verifier 的隐藏推理，但对
LongCat 是软约束，不是稳定的 token 硬上限。

在保持各角色 `max_tokens` 不变的 ID 34/40 工程重放中：

- 两题 Controller reasoning 合计从 6191 降至 2048 tokens，减少 66.9%；
- ID 34 全链路已观察 reasoning 从 10472 降至 6844，减少 34.6%；
- ID 34 Schema 调用成功率从 1 / 3 提升至 2 / 3；
- ID 34 从无结果变为 `partial`，预测与标签一致；
- ID 40 Controller reasoning 从 3399 降至 1024，但 Solver 请求达到
  60 秒调用上限，未返回 usage，不能用于比较 Solver token。

因此优化有效，但不能声称已彻底关闭 LongCat thinking。

## 2. 供应商能力验证

LongCat 官方 Chat Completions 文档声明原生接口支持：

```json
{"thinking": {"type": "disabled"}}
```

参考：

- https://longcat.ai/platform/docs/api/chat
- https://longcat.ai/platform/docs/open-code

CommandCode 文档声明其 Provider API 使用 OpenAI Chat Completions 结构：

- https://commandcode.ai/docs/provider

实际通过 CommandCode 的探针结果：

| 参数 | 结果 |
|---|---|
| `thinking.type=disabled` | 请求被接受，但 LongCat reasoning 413 -> 390，未关闭 |
| `enable_thinking=false` | Qwen reasoning 未关闭 |
| `reasoning.effort=none` | 请求被接受，但 LongCat reasoning 未关闭 |
| `reasoning_effort=minimal` | CommandCode 返回 400，只接受 low/medium/high/xhigh/max |
| `reasoning_effort=low` | 被明确接受；复杂 Qwen 调用稳定降至 1024，LongCat 部分调用明显下降 |

这说明 CommandCode 会静默忽略部分上游专用参数。实现不能仅凭 HTTP 200
断言 thinking 已关闭，必须同时记录请求参数和响应 `reasoning_tokens`。

## 3. 实现

`ModelConfig` 和 `ModelRequest` 新增可选 `reasoning_effort`。未配置时不发送
任何字段，保持旧行为；配置后通过 OpenAI SDK 顶层参数发送。

```yaml
legal_mcq:
  controller_model:
    reasoning_effort: low
  solver_model:
    reasoning_effort: low
  verifier_model:
    reasoning_effort: low
```

允许值为 `provider_default`、`minimal`、`low`、`medium`、`high`、
`xhigh` 和 `max`。`provider_default` 会归一化为 `None`，请求中省略参数。

每次调用 Trace 新增：

- `reasoning_effort_requested`；
- 已有的 `reasoning_tokens`；
- 已有的 `finish_reason`、`content_chars` 和 `outcome`。

数据集汇总新增：

- `reasoning_efforts`；
- `observed_reasoning_token_rate`。

这使“已请求低推理”和“供应商实际减少推理”可以分别审计。

## 4. 真实重放

重放使用此前失败的独立 dev ID 34、40。这是工程回放，不是新的独立准确率
实验，不能用于论文效果显著性结论。

### ID 34

| 指标 | provider default | low effort |
|---|---:|---:|
| 全链路 reasoning tokens | 10472 | 6844 |
| 全链路 observed total tokens | 18920 | 16580 |
| `finish_reason=length` | 2 | 1 |
| Schema 成功 | 1 / 3 | 2 / 3 |
| 最终结果 | 工程失败 | `partial`，标签一致 |
| 耗时 | 176.39s | 166.35s |

首次 low-effort Solver 仍消耗 2335 reasoning tokens 并截断，证明 LongCat
不保证低推理硬上限；第二次主 Solver为 1063，Revision 为 374，均成功。

### ID 40

Controller reasoning 从 3399 降到 1024，耗时从 47.22s 降到 28.98s。
随后 Solver 在 60 秒内未返回，hard timeout 正常生效。该调用 usage 未知，
所以不将其计入 token 降幅结论，也不重跑替换。

## 5. 决策

1. 保留 `reasoning_effort=low` 作为 CommandCode 当前最佳可用控制。
2. 不发送已证明被静默忽略的 `thinking` 或 `enable_thinking`。
3. 不降低 Solver 的 2400-token 上限，避免减少可见 JSON 空间。
4. 不把 `low` 当作硬约束；继续通过 `finish_reason` 和
   `reasoning_tokens` 识别供应商未遵循请求的情况。
5. 正式准确率实验必须使用新的独立样本，不能复用本轮工程重放题。

机器可读对照：
`output/legal_mcq/reasoning-low-replay-20260911/comparison.summary.json`。
