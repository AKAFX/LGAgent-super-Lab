# LegalMCQ v5 独立 dev29 有效性门控报告

## 1. 结论

本轮使用冻结 LexGenius dev 中剩余全部 29 道、此前从未进入 LegalMCQ
样本或结果工件的题目，比较：

1. LegalMCQ v5 三角色链路；
2. 同一个 `gpt-5.6-sol` 的单模型 Direct 基线。

结果：

| 方法 | 正确 | 准确率 | 返回率 | Calls | Tokens | 墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| LegalMCQ v5 | 18/29 | 62.07% | 29/29 | 96 | 219,026 | 595.65s |
| Direct GPT-5.6 Sol | 19/29 | 65.52% | 29/29 | 29 | 24,395 | 85.63s |

配对结果：

- 两者都正确：18；
- 只有 LegalMCQ 正确：0；
- 只有 Direct 正确：1；
- 两者都错误：10；
- Exact McNemar：`p=1.0`；
- LegalMCQ 准确率差：`-3.45` 个百分点。

因此，本轮能够证明 **LegalMCQ 工程链路稳定、可审计且能恢复协议错误**，
但不能证明它比同模型 Direct 提高答案准确率。按当前结果继续宣称准确率创新
是不严谨的。

## 2. 预注册设计

### 2.1 样本

- 数据集：`data/LexGenius.jsonl`；
- 冻结 ID：`d3175602655a56632c3d1bb3`；
- 划分：dev；
- 样本数：29；
- 选择规则：冻结 dev 中没有出现在任何历史 LegalMCQ sample/result 工件的
  全部剩余样本；
- 不挑题；
- 不替换失败；
- 不重跑失败；
- 所有失败计入分母。

样本 ID：

```text
276, 278, 280, 287, 290, 299, 313, 315, 327, 362,
366, 371, 376, 381, 402, 405, 418, 419, 427, 444,
453, 468, 469, 514, 518, 522, 525, 546, 547
```

领域分布：

| 领域 | 数量 |
|---|---:|
| Judicial Practice | 3 |
| Law and Society | 8 |
| Legal Application | 1 |
| Legal Ethics | 6 |
| Legal Language | 1 |
| Legal Reasoning | 1 |
| Legal Understanding | 9 |

### 2.2 LegalMCQ 配置

- Controller：Qwen/Qwen3.7-Flash；
- Strong Solver：gpt-5.6-sol；
- fallback：Qwen/Qwen3.8-Max-0902；
- Verifier：Qwen/Qwen3.7-Flash；
- Controller timeout：60 秒；
- Solver timeout：120 秒；
- fallback timeout：90 秒；
- Verifier timeout：45 秒；
- 题级上限：7 calls、65,536 tokens、360 秒；
- 并发：3；
- Prompt：`legal-mcq-three-role-v5`；
- Closed-book；
- 失败保留在分母。

### 2.3 Direct 基线

Direct 使用与链路主 Solver 相同的：

- 模型：`gpt-5.6-sol`；
- temperature：0.1；
- top_p：1；
- reasoning effort：high；
- 初始 max tokens：8192；
- 最多一次格式/length 恢复，硬上限 12288；
- timeout：120 秒；
- 并发：3。

Direct 只接收原始题目，不接收 Controller、Verifier、Oracle 或链路输出，
仅返回：

```json
{
  "protocol_version": "direct-v1",
  "selected_options": ["A"],
  "confidence": 0.9
}
```

## 3. LegalMCQ 绝对结果

### 3.1 答案质量

| 指标 | 结果 |
|---|---:|
| 总题数 | 29 |
| 正确 | 18 |
| Exact Match | 62.07% |
| Wilson 95% CI | 44.00%-77.31% |
| 返回答案 | 29/29 |
| 题级失败 | 0 |
| completed | 25 |
| partial | 4 |
| Verifier accepted | 25 |

### 3.2 分领域

| 领域 | LegalMCQ | Direct |
|---|---:|---:|
| Judicial Practice | 3/3 | 3/3 |
| Law and Society | 6/8 | 7/8 |
| Legal Application | 0/1 | 0/1 |
| Legal Ethics | 4/6 | 4/6 |
| Legal Language | 0/1 | 0/1 |
| Legal Reasoning | 0/1 | 0/1 |
| Legal Understanding | 5/9 | 5/9 |

各领域样本量很小，不应据此排序领域能力。

## 4. 配对检验

| | Direct 正确 | Direct 错误 |
|---|---:|---:|
| LegalMCQ 正确 | 18 | 0 |
| LegalMCQ 错误 | 1 | 10 |

只有一个 discordant pair，ID 313：

- Direct：B，与 benchmark 标签一致；
- LegalMCQ：D，与 benchmark 标签不一致；
- 链路 Verifier 认为 B 的法条引用错误，建议 D；
- Revision 后仍选择 D，并标记 `partial`。

Exact McNemar 双侧 `p=1.0`，不存在准确率提升证据。

两种方法最终预测完全一致 26/29；按“是否正确”比较则 28/29 一致。这说明
当前 Controller 与 Verifier 大多数时候没有改变强 Solver 的最终选择。

## 5. ID 313：唯一差异题

问题涉及超越经营范围订立合同的效力。Benchmark 标签为 B。

Direct 选择 B：

```json
{
  "selected_options": ["B"],
  "confidence": 0.9
}
```

LegalMCQ 选择 D，理由是：

```text
选项 B 将“超越经营范围不得作为合同无效的唯一依据”归因于司法解释第16条，
而链路认为该规则直接规定于《民法典》第505条；D 所述区分效力性强制规范
与管理性规范更能体现裁判理念演进。
```

Verifier 两轮均拒绝接受 B 的引用并建议 D。最终：

```json
{
  "selected_options": ["D"],
  "status": "partial",
  "needs_review": true,
  "warnings": ["INCORRECT_OPTION_CITATION"]
}
```

按现有 benchmark 标签，LegalMCQ 在这题属于 `Harmed`。但该题同时暴露了
可能的题目引用或标签争议，应进行独立法源审计。在审计完成前，统计结果仍严格
按原标签计算，不做事后改标。

后续审计已于 2026-10-03 完成：B 存在可验证的法源错引，D 的规范路径更严谨，
但题目不足以支持无争议的唯一答案。数据暂不改标，ID 313 标记为高置信争议题。
详见 `docs/legal_mcq_id313_no_harm_audit_20261003.md`。

## 6. 工程健康度

### 6.1 调用结果

| 指标 | 结果 |
|---|---:|
| 逻辑调用 | 96 |
| 直接成功 | 92 |
| Schema error | 4 |
| 最终恢复 | 4/4 |
| length | 1 |
| Controller timeout | 0 |
| Solver timeout | 0 |
| fallback 调用 | 0 |
| Revision Solver 调用 | 5 |
| budget blocked | 0 |
| 题级异常失败 | 0 |

四次协议异常：

1. ID 278：Solver `selected_options` 为空，重试恢复；
2. ID 525：Solver `selected_options` 为空，重试恢复；
3. ID 522：Verifier note 超过 160 字，重试恢复；
4. ID 518：Final Verifier JSON length 截断，重试恢复。

### 6.2 60 秒 Controller

- 29/29 Controller 调用成功；
- 最大耗时：24.69 秒；
- 未再次出现原 ID 472 的 30 秒硬超时；
- 说明调整到 60 秒为当前供应商波动提供了足够余量。

### 6.3 状态与 Revision

- completed：25；
- partial：4；
- Revision Solver：5 次；
- 2 个 partial 与标签一致；
- 2 个 partial 与标签不一致。

`partial` 不等于错误，而是表示最后仍存在确定性或模型核验争议。

## 7. Verifier 有效性

Verifier accepted 25 题，其中 16 题与标签一致，9 题与标签不一致。
4 个 partial 中有 2 题与标签一致。

因此：

- Verifier 能发现协议、引用、极性或基数风险；
- Verifier acceptance 不能作为答案正确率的替代指标；
- 当前 Verifier 没有在这批数据上带来净 Corrected；
- 同源或相关错误仍然存在。

## 8. 成本

| 指标 | LegalMCQ | Direct | 倍率 |
|---|---:|---:|---:|
| Calls | 96 | 29 | 3.31x |
| Tokens | 219,026 | 24,395 | 8.98x |
| Reasoning tokens | 86,904 | 10,154 | 8.56x |
| 墙钟时间 | 595.65s | 85.63s | 6.96x |

LegalMCQ 平均每题约 7,553 个已观察 tokens。当前额外成本没有换来 benchmark
准确率提升。

## 9. 错误与数据质量

两种方法共同答错 10 题：

```text
278, 287, 290, 366, 427, 469, 514, 522, 525, 547
```

其中部分题目值得独立法源或标签审计，例如：

- ID 469：题干“9年《宪法修正案》”存在年份缺失，模型选择 A，标签为 B；
- ID 287：模型认为唐代中央司法审判权由大理寺而非刑部行使，选择 D，
  标签为 C；
- ID 547：模型依据行政诉讼被告不得在诉讼中自行向原告和证人收集证据，
  选择 A，标签为 C；
- ID 366：题干使用“哪些选项”，但解析和标签基数可能存在争议。

这些只能列为审计候选，不能在看到模型答案后直接修改标签。

## 10. 能够成立的结论

### 已得到证据支持

1. Controller 60 秒配置在 29 题上实现 29/29 成功；
2. 全链路 29/29 返回，无题级异常失败；
3. 4 次结构化输出故障全部在预算内恢复；
4. 选项覆盖率 100%；
5. Oracle 隔离、调用日志、预算和状态机均正常工作；
6. 系统能够把存在争议的结果标为 partial，而不是全部伪装成 completed。

### 未得到证据支持

1. LegalMCQ 提高 benchmark 准确率；
2. Controller/Verifier 对 GPT-5.6 Sol 产生净纠错收益；
3. 额外约 9 倍 tokens 具有准确率成本效益；
4. 当前结果足以进入正式 test 并宣称方法有效。

## 11. 决策建议

当前版本应定义为：

```text
工程链路有效：是
协议与恢复机制有效：是
可审计性提升：是
准确率提升：否，当前 dev 证据不支持
```

在正式 test 之前，应优先：

1. 对 ID 313、287、469、547、366 做独立法源与标签审计；
2. 增加 no-harm gate：Controller/Verifier 若要改变 Direct 锚点，必须提供
   可验证法源或更高证据等级；
3. 将 Direct GPT 作为候选之一，而不是默认假设多 Agent 一定更好；
4. 在新的独立 dev 上重新进行配对门控；
5. 只有 Corrected 明显多于 Harmed 且 McNemar/Bootstrap 支持时，才冻结并
   进入 test split。

## 12. 工件

- Chain 样本：`output/legal_mcq/effectiveness-dev29-20260928-seed42/sample.jsonl`
- Chain 配置：`output/legal_mcq/effectiveness-dev29-20260928-seed42/config.yaml`
- Chain 结果：`output/legal_mcq/effectiveness-dev29-20260928-seed42/results.jsonl`
- Chain 调用日志：`output/legal_mcq/effectiveness-dev29-20260928-seed42/results.calls.jsonl`
- Direct 结果：`output/legal_mcq/effectiveness-dev29-20260928-seed42/direct-baseline/results.jsonl`
- 配对汇总：`output/legal_mcq/effectiveness-dev29-20260928-seed42/paired_effectiveness.summary.json`

模型日志和工件已检查，未发现 API key。
