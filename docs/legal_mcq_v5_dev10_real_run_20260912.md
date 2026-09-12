# LegalMCQ v5 LexGenius dev10 真实运行报告

## 1. 结论

本轮在 CommandCode 真实供应商环境运行 LegalMCQ v5：

- 10/10 返回答案；
- 9/10 为 `completed`，1/10 为 `partial`；
- 5/10 与数据标签完全一致，Exact Match 为 50%；
- Controller v3、fallback Solver v3、Verifier v2 均为 10/10 Schema 成功；
- 0 次 `finish_reason=length`；
- 0 次 Schema error；
- 0 次 prompt-only structured-output 降级；
- 0 次 Controller Claim 错误；
- 主 LongCat 前两题均超时，随后熔断；
- 10/10 题最终均由 Qwen fallback 作答。

因此当前链路的工程完成率和协议稳定性已经显著改善，但这批结果没有显示
高于历史 Original 约 50%-52% 的准确率。由于没有在完全相同 10 题上同步
运行 Original，本轮不能作为方法优于基线的配对证据。

## 2. 实验设计

| 项目 | 设置 |
|---|---|
| 数据源 | `data/LexGenius.jsonl` |
| 冻结划分 | dev |
| 冻结 ID | `64161b2c27f126a3ffe6d73c` |
| 题目数 | 10 |
| 题号 | 23、69、88、113、122、127、156、227、235、265 |
| Prompt | `legal-mcq-three-role-v5` |
| Controller | Qwen/Qwen3.7-Flash |
| 主 Solver | meituan/LongCat-2.0:free |
| fallback Solver | Qwen/Qwen3.7-Flash |
| Verifier | Qwen/Qwen3.7-Flash |
| 并发 | 1 |
| 失败处理 | 保留分母，不替换、不追加重跑 |

预算：

- 题级最多 7 calls；
- 题级最多 32768 tokens；
- 题级最多 180 秒；
- Controller 30 秒；
- 主 Solver 70 秒；
- fallback Solver 45 秒；
- Verifier 30 秒；
- 连续 2 次主 Solver timeout 后熔断；
- SDK 自动重试为 0。

## 3. 样本审计

10 题均来自当前冻结 LexGenius dev。运行后复核发现 ID 23 曾出现在
`lexgenius-30-20260908-seed42` 的早期失败批次中；当时没有单独 sample
文件，预运行排除脚本未识别这一记录。

因此准确描述为：

- 9 道此前 LegalMCQ 运行未出现的题；
- 1 道历史失败回归题；
- ID 23 保留在本轮分母中；
- 没有因其答错而替换或重跑。

本轮可作为工程和小样本质量 pilot，不能称为 10 道完全独立的新测试。

## 4. 总体结果

| 指标 | 结果 |
|---|---:|
| 总题数 | 10 |
| 返回答案 | 10，100% |
| 异常失败 | 0 |
| completed | 9，90% |
| partial | 1，10% |
| Exact Match | 5，50% |
| Exact Match Wilson 95% CI | 23.7%-76.3% |
| completed 内准确率 | 4/9，44.4% |
| Verifier accepted | 9/10，90% |
| needs_review | 1/10，10% |
| Option coverage | 10/10，100% |
| 总耗时 | 714.79 秒 |
| 平均单题 | 71.48 秒 |
| 中位单题 | 56.59 秒 |
| 最快/最慢 | 50.66 / 136.98 秒 |

## 5. 逐题结果

| ID | 维度 | 标签 | 预测 | 状态 | Verifier | 耗时 |
|---|---|---|---|---|---|---:|
| 23 | Legal Application | C | A | completed | accepted | 123.77s |
| 69 | Judicial Practice | D | C | completed | accepted | 136.98s |
| 88 | Legal Language | D | D | completed | accepted | 51.53s |
| 113 | Legal Reasoning | B | B | completed | accepted | 50.66s |
| 122 | Legal Ethics | B | B | completed | accepted | 55.39s |
| 127 | Legal Reasoning | A | C | completed | accepted | 57.03s |
| 156 | Legal Application | B | A | completed | accepted | 64.99s |
| 227 | Legal Ethics | B | B | partial | rejected | 62.24s |
| 235 | Legal Language | B | C | completed | accepted | 56.03s |
| 265 | Judicial Practice | D | D | completed | accepted | 56.15s |

按本批覆盖维度：

| 维度 | 正确/总数 |
|---|---:|
| Legal Application | 0/2 |
| Legal Ethics | 2/2 |
| Legal Language | 1/2 |
| Legal Reasoning | 1/2 |
| Judicial Practice | 1/2 |

本批没有覆盖 Legal Understanding 和 Law and Society，不能据此推断七维表现。

## 6. fallback 与熔断

| 指标 | 结果 |
|---|---:|
| 主 Solver 实际调用 | 2 |
| 主 Solver timeout | 2/2 |
| fallback 调用 | 10 |
| fallback API + Schema 成功 | 10/10 |
| fallback 成功率 Wilson 95% CI | 72.2%-100% |
| 熔断直接绕过主 Solver | 8 |
| fallback 平均耗时 | 22.86 秒 |
| fallback 中位耗时 | 21.47 秒 |

前两题分别在 70 秒达到主 Solver timeout。熔断打开后，后八题没有继续等待
LongCat，直接使用 Qwen fallback。前两题耗时 123.77 和 136.98 秒；熔断后
八题平均约 56 秒，说明批次级熔断有效限制了持续超时造成的吞吐损失。

但也意味着本轮没有得到 LongCat 的有效法律答案，50% 准确率主要反映
Qwen fallback 在该样本上的表现。

## 7. 协议与可观测性

总逻辑调用 32 次：

- Controller：10；
- 主 Solver：2；
- fallback Solver：10；
- Verifier：10。

调用结果：

- success：30；
- provider error：1；
- deadline exceeded：1；
- length：0；
- schema error：0；
- budget blocked：0。

Schema：

- `legal_mcq_controller_v3`：10/10；
- fallback `legal_mcq_solver_v3`：10/10；
- `legal_mcq_verifier_v2`：10/10；
- structured-output prompt fallback：0。

Controller v3 的确定性 Claim 绑定在 10 题上均成功，未出现：

- `CONTROLLER_CLAIM_COVERAGE_MISMATCH`；
- `CONTROLLER_CLAIM_ID_MISMATCH`；
- 跨选项 Claim；
- 选项覆盖遗漏。

这说明 v5 修复在真实数据集批次中保持稳定。

## 8. Token 使用

| 指标 | 结果 |
|---|---:|
| Prompt tokens | 33,455 |
| Completion tokens | 39,281 |
| Total tokens | 72,736 |
| Reasoning tokens | 30,720 |
| Reasoning / completion | 78.2% |
| Usage 未知调用 | 2 |

两个 timeout 调用没有完整 usage，因此 72,736 是已观察值，不是严格账单总量。
尽管请求使用 `reasoning_effort=low`，reasoning 仍占 completion 的 78.2%，
再次证明 CommandCode/模型把该参数视为软约束。

## 9. 法律回答质量分析

### 9.1 Verifier 不能作为正确率代理

5 个错误预测中，ID 23、69、127、156、235 均为 `completed`；其中模型
Verifier 全部接受。说明同源 Qwen fallback 和 Qwen Verifier 存在较强错误
相关性。

结论：

- `Verifier accepted` 只能证明协议审查通过；
- 不能把 90% 接受率解释为 90% 法律正确率；
- 若要提高法律正确率，应优先引入异构 Verifier 或权威证据，而不是继续增加
  同模型自审轮数。

### 9.2 partial 机制避免错误覆盖

ID 227：

- fallback Solver 选择 B；
- 数据标签也是 B；
- Verifier 认为 C 更合理并拒绝；
- 由于该题已使用 fallback，系统没有继续无限 Revision；
- 最终保留 B，但标记 `partial` 和 `needs_review=true`。

这说明 Verifier 没有权力直接覆盖 Solver。虽然状态不再是 completed，但系统
避免把错误的 Verifier 建议 C 强制写成最终答案。

### 9.3 Parser 警告

- ID 69、122：`year_only_date_assumed_end`；
- ID 235：`multiple_dates_first_assumed`。

这些警告不会自动改变答案，但提示涉及历史背景和多日期的题目仍适合进一步
法源时点审计。

## 10. 与历史结果的关系

历史 Qwen3-8B LexGenius 100 Original 约为 50%-52%。本轮 v5 为 50%，但：

- 不是同一批 10 题；
- 主 Solver LongCat 没有成功输出；
- 最终答案全部来自 Qwen3.7-Flash fallback；
- 样本只有 10 题；
- 未执行配对 Original；
- ID 23 是历史失败回归题。

因此不能说 v5 提升或降低了准确率。当前只能确认：

1. 工程返回率达到 100%；
2. 严格 completed 达到 90%；
3. Controller Claim 错误为 0；
4. fallback 和熔断真实有效；
5. 当前 fallback 法律准确率仍有明显提升空间。

## 11. 下一步建议

优先级从高到低：

1. 在完全相同 10 题上运行 Original，形成配对 Corrected/Harmed 对照；
2. 更换与 fallback 不同家族的 Verifier，减少错误相关性；
3. 评估是否将稳定的 Qwen 直接设为主 Solver，避免每批先损失 140 秒；
4. 对 5 个错误题进行法源和标签独立审计，不根据模型答案直接改标签；
5. 稳定后扩展到至少 50 道全新 dev；
6. 最终冻结模型和参数，在 test split 使用 42/43/44 三个种子。

## 12. 工件

- `output/legal_mcq/v5-dev10-20260912-seed42/sample.jsonl`
- `output/legal_mcq/v5-dev10-20260912-seed42/config.yaml`
- `output/legal_mcq/v5-dev10-20260912-seed42/manifest.json`
- `output/legal_mcq/v5-dev10-20260912-seed42/results.jsonl`
- `output/legal_mcq/v5-dev10-20260912-seed42/results.calls.jsonl`
- `output/legal_mcq/v5-dev10-20260912-seed42/results.summary.json`
- `output/legal_mcq/v5-dev10-20260912-seed42/analysis.summary.json`

所有模型正文日志均经过凭据脱敏，检查未发现 API key。
