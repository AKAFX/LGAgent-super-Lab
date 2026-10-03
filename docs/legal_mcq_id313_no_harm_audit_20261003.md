# LegalMCQ ID 313 标签审计与 No-Harm Gate 验证报告

- 日期：2026-10-03
- 审计对象：LexGenius ID 313
- 历史冻结批次：`effectiveness-dev29-20260928-seed42`
- 当前源码冻结：`d507b835fb987cbff1c11136`
- 当前结论：保留原始数据不改标，将 ID 313 标记为高置信争议题
- 功能状态：no-harm gate 已实现但默认关闭，等待新的独立冻结 dev 集验证

## 1. 执行摘要

ID 313 的 benchmark 标签是 B。历史配对实验中，Direct GPT-5.6 Sol
选择 B，三角色链路选择 D，形成该批次唯一一组“正确性改变”：Direct 正确，
Chain 错误。

法源审计确认题目本身存在实质瑕疵：

1. B 的效力结论与现行法方向一致，但将规则错误归因于合同编通则司法解释
   第 16 条；直接规则其实是《民法典》第 505 条。
2. C 与《民法典》第 505 条直接冲突。
3. D 的“区分效力性与管理性强制性规定”具有规范依据，在四个选项中法律论证
   更严谨；但其主要援引《九民纪要》第 30 条，而纪要不是司法解释，且民法典
   施行后第 505 条应当是更直接的裁判依据。
4. 因题干要求选择“最能体现社会变革中商事裁判理念演进”的选项，价值判断成分
   较强。B 有错误法源归因，D 又不是最直接的现行法表达，因此不能把 benchmark B
   继续当作无争议 ground truth，也不宜仅凭一次模型审计直接改为 D。

工程侧已增加可选 no-harm gate。闭卷时，结构化链路与独立 Direct 锚点不一致，
链路不得覆盖锚点；开卷时，只有候选通过确定性校验和独立 Verifier，并且每个
差异选项都有权威、时点有效、非搜索摘要的全文证据，才允许覆盖。

## 2. 原题争议

与争议直接相关的选项是：

- B：依据《民法典》第 143 条认定有效，并称合同编通则司法解释第 16 条明确
  “超越经营范围不影响合同效力”。
- C：依据《民法典》第 505 条认定无效。
- D：依据《民法典》第 153 条和《九民纪要》第 30 条，区分效力性规定与
  管理性规定。

### 2.1 《民法典》第 505 条

第 505 条规定：

> 当事人超越经营范围订立的合同的效力，应当依照本法第一编第六章第三节和
> 本编的有关规定确定，不得仅以超越经营范围确认合同无效。

因此：

- C 的“依据第 505 条认定无效”与法条文本相反；
- B 所表达的“不因超越经营范围当然无效”方向正确；
- 但该规则的直接法源是第 505 条，不是 B 所称的司法解释第 16 条。

官方来源：
[《中华人民共和国民法典》第三编 合同](https://www.spp.gov.cn/spp/ssmfdyflvdtpgz/202008/t20200831_478413.shtml)。

### 2.2 合同编通则司法解释第 16 条

《最高人民法院关于适用〈中华人民共和国民法典〉合同编通则若干问题的解释》
第 16 条处理的是合同违反法律、行政法规强制性规定时，行政责任或者刑事责任
足以实现规范目的等情况下，可依第 153 条第一款认定合同不因违反强制性规定而
无效。其文本没有明确写出“超越经营范围不影响合同效力”。

因此 B 存在可客观验证的法源错引，不只是表述风格差异。

官方来源：
[法释〔2023〕13号全文](http://gongbao.court.gov.cn/Details/f4722cf61c92a585f04b2ecd334f5b.html)。

### 2.3 《九民纪要》第 30 条

第 30 条说明，经营范围、交易时间、交易数量等行政管理性质的强制性规定，
一般应认定为管理性强制性规定。这支持 D 的区分方法。

但最高人民法院在印发通知中同时明确：纪要不是司法解释，不能作为裁判依据
援引，只能在裁判说理中参考。因此 D 的方法论合理，却不能替代第 505 条这一
直接法律依据。

官方来源：
[《全国法院民商事审判工作会议纪要》](https://www.court.gov.cn/zixun/xiangqing/199691.html)。

## 3. 标签处置

本次不修改 `sample.jsonl` 或 `data/LexGenius.jsonl` 中的 B 标签，原因是：

- 事后根据模型输出改标会污染冻结评测；
- B 的核心效力结论正确，只是规范依据错引；
- D 的法理路径更严谨，但题干的“最能体现”不足以形成唯一、客观答案。

建议后续数据治理采用：

```json
{
  "question_id": "313",
  "label_status": "high_confidence_disputed",
  "benchmark_label": ["B"],
  "plausible_alternative": ["D"],
  "issue_codes": [
    "MISATTRIBUTED_AUTHORITY",
    "NON_UNIQUE_BEST_OPTION",
    "SOURCE_HIERARCHY_AMBIGUITY"
  ],
  "exclude_from_primary_accuracy": true
}
```

在建立正式争议题清单前，现有统计仍按原始 B 标签计算，并单独报告
“含争议题准确率”和“剔除争议题准确率”。

## 4. No-Harm Gate 规则

设：

- `A`：Controller 之前，由同一 Strong Solver 仅看原题得到的 Direct anchor；
- `C`：Controller、Solver、Verifier 链路产生的最终候选；
- `D = A △ C`：两个答案集合的对称差。

### 4.1 闭卷

```text
if A == C:
    final = C
else:
    final = A
    status = partial
    needs_review = true
```

闭卷没有外部权威证据，因此链路分歧不能覆盖锚点。其直接含义是：开启 gate 后，
闭卷最终选项恒等于当次 Direct anchor；链路只提供结构化审查、分歧检测和审计
价值，不再承诺相对 Direct 的准确率提升。

### 4.2 开卷

只有同时满足以下条件才允许 `final = C`：

1. 候选通过 `DecisionValidator`；
2. 独立 Verifier 接受候选；
3. 对每个 `D` 中的选项，候选 assessment 至少引用一条已存在证据；
4. 该证据 `authority_level >= min_authority_level`；
5. 该证据覆盖题目法律时点；
6. `source_type` 不是搜索摘要或 snippet；
7. `LegalEvidence` 已验证 quote 非空及 content hash。

任一条件不满足，保留 `A` 并标记人工复核。保锚点时清空候选
`option_assessments` 和 citations，避免输出与最终选项矛盾的解释或法源。

### 4.3 Anchor 不可用

Anchor 超时、供应商错误、Schema 错误或预算不足时，系统 fail-open 到原链路，
并记录 `NO_HARM_ANCHOR_UNAVAILABLE`。此路径保证可用性，不提供 no-harm 保证，
因此必须通过 warning 和 Trace 进入监控。

## 5. 实现位置

- `src/lgagent/legal_mcq/models.py`
  - `DirectAnchorDecision`
  - `direct-anchor-v1` Schema 与本地严格校验
- `src/lgagent/legal_mcq/no_harm.py`
  - `NoHarmGate`
  - 闭卷保锚点与开卷证据覆盖规则
- `src/lgagent/legal_mcq/agent.py`
  - Controller 前独立 anchor 调用
  - 最终决策前 gate
  - `partial`、warnings、Trace 和审计字段
- `src/lgagent/config.py`
  - `no_harm_gate_enabled`
  - `no_harm_anchor_max_tokens`
  - 最小调用预算校验
- `src/lgagent/runner.py`
  - 配置透传
- `tools/run_legal_mcq.py`
  - dry-run 成本与策略展示
- `tests/test_legal_mcq_no_harm.py`
  - 协议、规则、集成、降级和预算测试

正式示例配置保持：

```yaml
no_harm_gate_enabled: false
no_harm_anchor_max_tokens: 4096
```

## 6. 真实 ID 313 重放

临时配置使用：

```yaml
no_harm_gate_enabled: true
max_model_calls: 8
```

有效重放结果：

| 项目 | 结果 |
|---|---|
| Direct anchor | B，confidence 0.98 |
| Chain candidate | D，confidence 0.91 |
| Gate action | `preserve_anchor` |
| Final | B |
| Status | `partial` |
| Needs review | `true` |
| 调用 | 4/8 |
| 总 tokens | 11,264 |
| 总耗时 | 84.28 秒 |
| Schema | 4/4 成功 |

产物：

- `output/legal_mcq/no-harm-id313-20261003/replay-2.jsonl`
- `output/legal_mcq/no-harm-id313-20261003/replay-2.calls.jsonl`

第一次重放发现供应商不接受 JSON Schema 的 `uniqueItems` 关键字。系统按设计记录
`NO_HARM_ANCHOR_UNAVAILABLE` 并完成原链路；随后将唯一性与非空约束保留在本地
解析器中，移除供应商不支持的 Schema 关键字。第二次重放四次调用全部成功。

## 7. 历史 Dev29 零调用反事实

输入：

- Chain：`results.jsonl`
- Direct：`direct-baseline/results.jsonl`

闭卷规则按历史 Direct 预测作 anchor，结果为：

| 指标 | 结果 |
|---|---:|
| 总题数 | 29 |
| 预测一致 `matched` | 26 |
| 分歧并保锚点 | 3 |
| 允许覆盖 | 0 |
| 反事实正确数 | 19 |
| 反事实准确率 | 65.52% |

三道预测分歧：

| ID | Chain | Direct | Benchmark | Gate 后 |
|---|---|---|---|---|
| 290 | A | B | C | B，仍错 |
| 313 | D | B | B | B，由错变对 |
| 427 | B | B、D | C | B、D，仍错 |

因此历史反事实从 Chain 的 `18/29` 恢复为 Direct 的 `19/29`，Corrected=1、
Harmed=0。这里的“Corrected”只表示相对历史 Chain 避免一次损害，不证明真实法律
正确性，也不是独立测试集上的因果效果。

机器可读结果：
`docs/legal_mcq_no_harm_counterfactual_20261003.json`。运行目录中也保留了一份副本。

## 8. 验证与限制

当前验证：

- no-harm 专项：16 项；
- LegalMCQ、配置、输出恢复、CLI 相关回归：140 项；
- 全仓测试：337 项通过；
- 真实供应商：anchor、Controller、Solver、Verifier 全部成功；
- `compileall`：通过；
- `git diff --check`：通过。

仍需遵守以下结论边界：

1. Dev29 是同一历史开发集上的事后反事实，不能用于宣称泛化提升。
2. 闭卷 gate 的最终答案恒等于 Direct anchor，因此不会优于该锚点；代价是额外
   一次强模型调用和完整链路成本。
3. Anchor 与 Chain Solver 使用同一模型和供应商，只是上下文隔离，并非模型层面
   独立。
4. 开卷证据的来源、时效、全文和覆盖范围由代码校验；语义支持关系仍依赖
   Verifier，尚不是形式化证明。
5. 只有在新的、未使用冻结 dev 集上达到 `Harmed=0`，且失败率、延迟和成本在
   预设阈值内，才应考虑默认开启。
