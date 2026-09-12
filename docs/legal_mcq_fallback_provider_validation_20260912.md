# LegalMCQ fallback 真实供应商验证

## 结论

2026-09-12 在 CommandCode 真实供应商环境完成两组串行验证。备用 Solver
`Qwen/Qwen3.7-Flash` 的 API 返回与 Solver v3 Schema 成功率为 `5/5`
（100%，Wilson 95% CI 56.6%-100%）；严格恢复为 `completed` 的成功率为
`4/5`（80%，Wilson 95% CI 37.6%-96.4%）。

因此，fallback 的供应商调用、Schema、共享预算和熔断路径已经真实打通，
但不能把小样本的 100% API 成功率解释为端到端 100% 恢复率。

## 样本与配置

- 数据源：`data/LexGenius.jsonl`；
- 独立样本 ID：103、481、259、162、67；
- 选择种子：`20260912`；
- 样本 SHA-256：
  `e912eddbe81547c3ff1ea825055bbe0895ce427d21eb97a3da8a69529cc96ddf`；
- 串行运行，失败不替换、不重跑；
- Controller、fallback、Verifier：`Qwen/Qwen3.7-Flash`；
- 主 Solver：`meituan/LongCat-2.0:free`；
- SDK 自动重试：0；
- 题级预算：7 calls、32768 tokens、180 秒。

样本未用于此前 v2/v3 的协议和超时定位。故障注入批次复用同一冻结样本，
只验证恢复路径，不作为独立准确率实验。

## 生产策略运行

生产配置保持主 Solver 70 秒、fallback 45 秒、连续两次主 Solver timeout
后熔断：

| 指标 | 结果 |
|---|---:|
| 题目 | 5 |
| 初始主 Solver 成功 | 5/5 |
| 初始主 Solver timeout | 0/5 |
| Revision Solver timeout | 1/1 |
| fallback API + Schema 成功 | 1/1 |
| fallback 严格 completed 恢复 | 0/1 |
| 返回答案 | 5/5 |
| completed | 4/5 |
| exact match | 5/5 |
| Verifier length | 0 |
| structured-output 降级 | 0 |
| 总耗时 | 424.43 秒 |

唯一 fallback 在题 259 的 Revision Solver 超时后触发，Qwen 在 16.76 秒
返回 `stop` 且通过 Solver v3 Schema。结果选项 C 正确，但没有完整覆盖
Controller 分配的 Claim ID，最终被确定性校验降为 `partial`。

## 故障注入运行

测试专用配置将主 Solver timeout 设为 1 秒，生产默认配置未修改。前两题
触发真实 timeout，达到阈值后，后三题由熔断器直接绕过 LongCat：

| 指标 | 结果 |
|---|---:|
| 主 Solver timeout 路由 | 2 |
| 熔断后直接绕过主模型 | 3 |
| fallback 调用 | 5 |
| fallback `stop` | 5/5 |
| fallback Solver v3 Schema 合法 | 5/5 |
| fallback 严格 completed 恢复 | 4/5 |
| 返回答案 | 5/5 |
| exact match | 5/5 |
| Verifier length | 0 |
| structured-output 降级 | 0 |
| 总耗时 | 255.22 秒 |

fallback 延迟为 17.22-25.28 秒，均值 19.78 秒，中位数 17.83 秒。首个
1 秒超时由 SDK 包装为 `provider_error/ModelCallError`，第二个记录为
`deadline_exceeded/BudgetExceededError`；路由层均正确识别为 timeout，
没有同模型重试。

## 原始未闭合问题

题 259 在两组运行中都为 `partial`。Controller 将选项 C 的第二个语义命题
错误放入选项 D，导致后续 Solver 很难同时满足语义正确性与精确 Claim ID
覆盖。确定性校验正确拦截了结果，但这说明当前严格恢复率受 Controller
claim 划分影响，不是 fallback 供应商不可用。

当前可得结论：

1. fallback API、JSON Schema、预算和熔断链路真实可用；
2. 小样本供应商成功率为 100%，样本量不足以外推稳定 SLA；
3. 严格端到端恢复率为 80%，不能宣称 fallback 已达到 100%；
4. 下一工程瓶颈是 Controller claim 与原始选项的确定性对齐。

## Controller v3 修复重放

后续将链路升级为 `legal-mcq-three-role-v5`。Controller v3 不再生成
`option_claims`；代码直接从原始选项文本绑定唯一的 A1、B1、C1、D1。
旧 Controller v1/v2 输出仍可读取，但其 Claims 不再进入下游。

使用题 259 和相同的 1 秒主 Solver 故障注入进行真实重放：

| 指标 | 修复前 | Controller v3 |
|---|---:|---:|
| Controller Claim 错误 | 2 | 0 |
| fallback Schema 合法 | 是 | 是 |
| Verifier accepted | 否 | 是 |
| 状态 | partial | completed |
| needs_review | 是 | 否 |
| 选项 | C | C |

重放共 4 次逻辑调用，46.88 秒，Qwen fallback 在 16.34 秒返回。Controller、
fallback 和 Verifier 都由 CommandCode 真实供应商执行，未发生 length 或
prompt-only Schema 降级。这证明已观察到的 Claim 跨选项划分故障被修复，
不代表单题重放可以证明总体准确率提升。

## 工件

- 生产策略配置与结果：
  `output/legal_mcq/fallback-dev5-20260912-seed20260912/`
- 故障注入配置与结果：
  `output/legal_mcq/fallback-fault-dev5-20260912-seed20260912/`
- 机器汇总：
  `output/legal_mcq/fallback-fault-dev5-20260912-seed20260912/validation.summary.json`
- Controller v3 题 259 重放：
  `output/legal_mcq/claim-binding-q259-20260912/`
- 当前冻结 ID：`1931b3df9f45caf8f1c3ddc8`

所有调用日志均经过凭据脱敏，未写入 API key。
