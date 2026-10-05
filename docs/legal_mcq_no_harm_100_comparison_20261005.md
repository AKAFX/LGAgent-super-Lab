# LegalMCQ No-Harm 100 题配对比较

- 日期：2026-10-05
- 数据集：LexGenius 前 100 条 interleaved records
- 数据哈希：`9033614d7115e1f3054c99dbcbe7295e06f35b5a3256fca3f9952741eea67d1e`
- 样本成员哈希：`225dc740ff13873c1d43fc4e983c5eee93b9022c50622c26ebbd4bf62eb5ea76`
- 源码提交：`778004c500b113bf0f5f254a0218b06e3fe08a53`
- Seed：42
- 并发：4
- 失败处理：保留在固定 100 题分母中并计错，不补跑、不替换

## 1. 结论

本轮数值最高的是当前 no-harm 最终答案：

| 方法 | 正确数 | 全分母准确率 | 95% Wilson CI |
|---|---:|---:|---:|
| 当前 no-harm 最终答案 | 60/100 | **60%** | 50.20%-69.06% |
| 同次未门控 Chain 候选 | 58/100 | 58% | 48.21%-67.20% |
| 历史 Qwen3-8B Original | 52/100 | 52% | 42.32%-61.54% |
| 历史 Qwen3-8B Web | 49/100 | 49% | 39.43%-58.65% |

直接回答“哪个准确率高”：当前 no-harm 最终答案最高，为 60%，比历史
Original 高 8 个百分点，比历史 Web 高 11 个百分点。

但这还不能证明方法稳定提升：

- 当前最终答案 vs 历史 Original：Corrected=18，Harmed=10，McNemar
  `p=0.1849`；
- 当前最终答案 vs 同次 Chain：Corrected=3，Harmed=1，McNemar
  `p=0.625`；
- 95% 置信区间明显重叠；
- 当前使用 GPT-5.6 Sol，历史基线使用 Qwen3-8B，模型升级与协议升级共同构成
  变化，不能把 8 个百分点全部归因于 no-harm gate。

因此严谨结论是：**当前配置在这 100 道同题上数值更高，但未达到统计显著，
no-harm gate 只贡献了观测上的净 +2 题。**

## 2. 实验口径

历史可比基线为：

- `output/qwen3_8b_lexgenius_100_final_baseline`
- 样本：LexGenius indices 0-99
- Original：52/100
- Web：49/100

本轮逐题核验了历史结果与 `data/LexGenius.jsonl` 前 100 条：

- 题目文本 100/100 一致；
- golden labels 100/100 一致；
- 数据集哈希一致；
- 七维度分布为 15、15、14、14、14、14、14。

本轮模型：

| 角色 | 模型 |
|---|---|
| Direct anchor | `gpt-5.6-sol` |
| Controller | `Qwen/Qwen3.7-Flash` |
| Strong Solver | `gpt-5.6-sol` |
| Solver fallback | `Qwen/Qwen3.8-Max-0902` |
| Verifier | `Qwen/Qwen3.7-Flash` |

闭卷 no-harm gate 开启。每道成功样本同时保留：

1. Direct anchor；
2. 未门控的最终 Chain candidate；
3. Gate 之后的最终答案。

因此 Gate 与 Chain 的比较是同次运行、同题、同模型条件下的配对比较，不需要
额外调用另一套 baseline。

## 3. Gate 实际行为

在 97 道成功返回题中：

| Gate action | 数量 |
|---|---:|
| `matched` | 76 |
| `preserve_anchor` | 7 |
| Anchor unavailable，降级旧链路 | 14 |

Gate 真正改变 7 道题：

| ID | Chain | Anchor/Final | 标签 | 结果变化 |
|---|---|---|---|---|
| 38 | A | C | D | 错到错 |
| 49 | A | C | C | 错到对 |
| 53 | B | C | C | 错到对 |
| 56 | B | A | C | 错到错 |
| 81 | C | D | C | **对到错** |
| 82 | D | B | B | 错到对 |
| 99 | D | A | C | 错到错 |

汇总：

- Corrected：3
- Harmed：1
- 不改变对错：3
- 净变化：+2

这验证了一个重要边界：no-harm 是“链路不得无证据覆盖 Direct anchor”的工程
规则，不是“相对真实标签绝不伤害”的准确率保证。Anchor 自身答错时，Gate
仍可能伤害 benchmark 准确率，ID 81 即为实际反例。

## 4. 工程健康度

| 指标 | 当前结果 | 历史 Original |
|---|---:|---:|
| 成功返回 | 97/100 | 100/100 |
| 题级失败 | 3 | 0 |
| 模型调用 | 454 | 703 |
| 总 tokens | 952,218 | 1,005,917 |
| Prompt tokens | 446,505 | 756,225 |
| Completion tokens | 505,713 | 249,692 |
| Reasoning tokens | 428,978 | 未单列 |
| 批次墙钟时间 | 1,878.25 秒 | 未记录为同口径 |

当前调用数减少 35.42%，总 tokens 减少 5.34%。但 GPT-5.6 Sol 与 Qwen3-8B
单价不同，不能仅凭 token 数认定当前成本更低。

调用结果：

- success：431
- schema error：19
- deadline exceeded：4
- fallback：1 次，成功
- 平均题级延迟：72.56 秒
- 中位题级延迟：52.83 秒
- P95：168.65 秒
- 最大：262.24 秒

最终状态：

- `completed`：77
- `partial`：20
- failed：3

结果文件中的 option coverage 为 90%。其中 7 题是 Gate 保留 Anchor 后主动清空
与最终答案冲突的 Chain option assessments，另有 3 题失败；原始 Solver candidate
在 97 道返回题上均保留于 diagnostics。

## 5. 三道失败

| ID | 标签 | 失败阶段 | 原因 | 历史 Original |
|---|---|---|---|---|
| 14 | C | Solver | 两次均输出空或非法 `selected_options` | C，正确 |
| 54 | D | Solver Revision | Revision 输出空选择，预算不足以继续重试 | D，正确 |
| 75 | A | Controller | Anchor 后 Controller 达到 360 秒全局 wall budget | B，错误 |

三题在失败前均已得到可解析的 Direct anchor，但当前实现只在 Anchor 自身不可用时
fail-open；下游 Chain 失败时不会返回已经存在的 Anchor。这降低了返回率。该问题
不影响本轮准确率：三道失败的 Anchor 分别为 D、B、B，均与标签不符；如果回退，
返回率会从 97% 升到 100%，正确数仍为 60。

## 6. Anchor 健康度

成功返回题中有 14 次 Anchor 不可用于 Gate：

- 12 次单选题返回多个选项，被 `NO_HARM_ANCHOR_CARDINALITY` 拒绝；
- 2 次 Anchor 达到 120 秒 timeout。

Gate 实际完成比较 83 题，其中 76 题一致、7 题分歧。三道下游失败题也都已获得
有效 Anchor，因此 100 次 Anchor 调用中共有 86 次形成有效单选结果。

这说明下一轮优化应优先处理：

1. 在 Anchor 输入中显式传递 `question_type=single_choice`，要求严格只选一项；
2. 下游 Chain 失败且 Anchor 有效时，返回 Anchor 并标记 `partial`；
3. 继续保持失败计入分母，不用重跑结果覆盖失败记录；
4. 修复后必须使用新的冻结样本验证，不能在本 100 题上调参后宣称提升。

## 7. 领域结果

| 领域 | 当前 Final | 同次 Chain | 历史 Original |
|---|---:|---:|---:|
| Legal Understanding | 12/15 | 11/15 | 7/15 |
| Legal Reasoning | 10/15 | 10/15 | 8/15 |
| Legal Application | 6/14 | 6/14 | 6/14 |
| Legal Ethics | 11/14 | 11/14 | 9/14 |
| Legal Language | 10/14 | 10/14 | 8/14 |
| Law and Society | 6/14 | 5/14 | 9/14 |
| Judicial Practice | 5/14 | 5/14 | 5/14 |

样本量不足以据此对领域能力排序。值得关注的是 Law and Society 比历史低 3 题，
而 Legal Understanding 高 5 题，说明总体增益并不均匀。

## 8. 结论边界

本轮可以证实：

- 当前配置在同一 100 题上的观测准确率最高；
- Gate 相对同次 Chain 净纠正 2 题；
- 当前协议使用更少调用和略少总 tokens；
- 真实批量运行仍有 3% 题级失败和 14% Anchor 不可用问题。

本轮不能证实：

- 8 个百分点由 no-harm gate 单独导致；
- 当前方法在独立测试集上稳定优于历史方法；
- Gate 对真实正确性严格无害；
- 当前成本低于历史 Qwen3-8B。

基于这些证据，`no_harm_gate_enabled` 应继续默认关闭。应先修复 Anchor 基数和
下游失败回退，再在新的冻结 100 题上验证 `Harmed=0`、返回率和统计显著性。

## 9. 产物

- 预运行清单：
  `output/legal_mcq/no-harm-lexgenius100-first100-20261005/manifest.pre_run.json`
- 完成清单：
  `output/legal_mcq/no-harm-lexgenius100-first100-20261005/manifest.json`
- 逐题结果：
  `output/legal_mcq/no-harm-lexgenius100-first100-20261005/results.jsonl`
- 逐调用日志：
  `output/legal_mcq/no-harm-lexgenius100-first100-20261005/results.calls.jsonl`
- CLI 汇总：
  `output/legal_mcq/no-harm-lexgenius100-first100-20261005/results.summary.json`
- 配对统计：
  `output/legal_mcq/no-harm-lexgenius100-first100-20261005/paired_analysis.summary.json`
- Git 可追踪统计副本：
  `docs/legal_mcq_no_harm_100_paired_summary_20261005.json`
