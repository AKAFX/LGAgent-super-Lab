# LegalMCQ v3 同题真实截断对照

## 结论

修复后的链路显著降低了已观察到的 `finish_reason=length`，但尚未达到健康
批量运行标准。主要故障由“响应返回但 JSON 截断”转为“Solver 在 60 秒内
没有返回”。因此可以确认输出空间修复有效，不能确认所有超时调用最终也不会
截断，更不能据此证明准确率提升。

## 设计

- 同题重放 ID：59、62、68、84、85。
- 样本 SHA-256：
  `ed8b72a6b6489d46fc9d06fc7cfecb677e9f615df683947f6fbb8ea34203a250`。
- baseline：`legal-mcq-three-role-v2`，Solver 上限 2400。
- repaired：`legal-mcq-three-role-v3`，Controller v2 Schema；
  Solver 首次 4096，length 恢复硬上限 6144。
- 两组均为 seed 42、串行、每题 7 calls / 32768 tokens / 180 秒，
  单调用 60 秒、SDK 自动重试 0、`reasoning_effort=low`。
- 失败题保留在分母，不替换、不追加重跑。

这是同题工程对照，题目已用于定位问题，不是独立准确率实验。

## 截断率

| 指标 | 修复前 | v3 | 变化 |
|---|---:|---:|---:|
| 全角色 length / 调用 | 10/16，62.5% | 1/15，6.7% | -55.8 个百分点 |
| Controller length / 调用 | 2/8，25.0% | 0/5，0% | -25.0 个百分点 |
| Solver length / 调用 | 8/8，100% | 0/7，0% | -100 个百分点 |
| 发生任意 length 的题目 | 5/5 | 1/5 | -4 题 |
| 发生 Solver length 的题目 | 5/5 | 0/5 | -5 题 |

总体观测 length 率相对下降 89.3%。但是 v3 的 7 次 Solver 调用中：

- 2 次 `stop` 且通过 Solver v2 校验；
- 5 次达到单调用超时，没有 finish reason、usage 或正文；
- 0 次报告 `length`。

因此 Solver 的确定性结论只能写成：完整返回的 2 次均未截断；其余 5 次是
右删失样本，无法知道供应商继续运行后会 `stop` 还是 `length`。

唯一一次 v3 `length` 来自 ID 68 的 Verifier。其第一次调用使用 768
visible tokens，reasoning 为 1024，JSON 截断；第二次同额度重试成功。
这暴露了辅助核验角色仍缺少与 Solver 类似的正文余量。

## Controller 与 Solver

Controller v2 的结果：

- 5/5 首次成功；
- 5/5 原生接受 `legal_mcq_controller_v2` Schema；
- 0 次 prompt-only fallback；
- 0 次 length 或格式错误。

baseline Controller 为 4/8 成功、4 次格式错误，其中 2 次 length。紧凑
协议与 Schema 在这 5 题上消除了 Controller 重试。

Solver 的结果：

- ID 68：首次 4096，reasoning 1635，正文 3000 字符，`stop`；
- ID 84：首次 4096，reasoning 1432，正文 2013 字符，`stop`；
- ID 59：首次调用 60 秒 hard timeout；
- ID 62：两次调用均 60 秒超时；
- ID 85：两次调用均 60 秒超时。

没有 Solver 返回 length，所以 6144 的自适应 length 恢复分支没有被真实
触发。两次成功均发生在首次 4096 上限。

## 端到端结果

| 指标 | 修复前 | v3 |
|---|---:|---:|
| 返回答案 | 0/5 | 2/5 |
| completed | 0/5 | 2/5 |
| 总体分母正确 | 0/5 | 1/5 |
| Solver 超时题 | 0/5 | 3/5 |
| 总耗时 | 536.97s | 553.62s |

- ID 68：completed，预测 C，标签 D，不一致；
- ID 84：completed，预测 C，与标签一致；
- ID 59、62、85：Solver 超时，无答案。

准确率不是本轮主指标。同题已参与修复，样本只有 5 题；ID 68 还涉及“当代
类似案件”时点修复，不能把结果与 v2 当作纯模型质量对照。

## 用量边界

v3 已观察到 27703 total tokens，baseline 为 54479。但 v3 有 5 次 Solver
调用未返回 usage，不能据此声称成本下降。v3 的总耗时反而增加 16.65 秒，
说明更大输出上限降低截断的同时放大了免费 Solver 的延迟风险。

15 个 started/finished 事件一一对应，无重复，无 SDK 隐式重试，无
structured-output fallback。所有题均保留在结果文件中。

## 判断

1. **截断修复有效**：Controller 格式稳定，两个完整 Solver 响应有足够正文。
2. **截断问题未完全关闭**：Verifier 仍有 1 次 length；5 次 Solver 超时使
   Solver 截断率估计存在删失偏差。
3. **链路仍不健康**：只完成 2/5，主要瓶颈变为 LongCat 免费端点延迟。
4. **下一优先级**：先处理 Solver 超时策略和 Verifier 紧凑协议/正文余量，
   再用新 dev 样本验证，不能通过重跑本 5 题继续调参后宣称准确率提升。

上述工程项已在 v4 实现，详见
[Solver 超时与 Verifier 截断优化](legal_mcq_timeout_verifier_optimization_20260912.md)；
尚未进行新的真实供应商验证。

## 工件

- [冻结清单](../output/legal_mcq/io-v3-replay5-20260912/manifest.json)
- [运行配置](../output/legal_mcq/io-v3-replay5-20260912/config.yaml)
- [结果](../output/legal_mcq/io-v3-replay5-20260912/results.jsonl)
- [逐调用日志](../output/legal_mcq/io-v3-replay5-20260912/results.calls.jsonl)
- [机器对照汇总](../output/legal_mcq/io-v3-replay5-20260912/comparison.summary.json)
