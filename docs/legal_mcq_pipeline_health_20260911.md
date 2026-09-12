# LegalMCQ 当前链路健康检查

## 结论

当前链路是可运行、可观测的工程原型，尚未成熟，不满足稳定批量使用标准。
不能保证“最优回答”，也不能将此前局部 reasoning 降幅视为端到端质量提升。
本轮没有修改核心代码、Prompt、预算或旧结果，没有替换失败题。

## 实验范围

- 版本：Task16 freeze `2dc44ee693640c4f2371d6a0`；本轮 manifest 额外记录
  50 个源码/运行脚本文件哈希，覆盖 LegalMCQ 协议、解析和判题代码。
- 数据：LexGenius 冻结 dev，split seed `20260826`。
- ID：`59, 62, 68, 84, 85`；按索引顺序取未在历史 LegalMCQ 批次出现的
  五个不同维度样本，不按标签或正确率筛选。
- 维度：法律伦理、司法实践、法律与社会、法律理解、法律推理。
- Controller/Verifier：`Qwen/Qwen3.7-Flash`；
  Solver：`meituan/LongCat-2.0:free`，经 CommandCode 调用。
- 三角色 `reasoning_effort=low`，Solver `response_format=json_schema`。
- seed 42，串行；每题最多 7 calls、32768 tokens、180 秒；单次 60 秒，
  SDK 自动重试为 0。输出上限仍为 Controller 1200、Solver 2400、Verifier 768。
- 新样本仅指相对历史 LegalMCQ 运行无重叠，不声称整个项目从未接触这些题。
  五题未覆盖所有七维，且没有对照组，不是正式独立准确率或最优性证明。

## 真实结果

| 指标 | 结果 |
|---|---:|
| 返回答案 | 0 / 5 |
| completed / partial / exception | 0 / 0 / 5 |
| 总体分母准确率（失败计错） | 0 / 5，0% |
| Controller 本地解析成功 | 4 / 8 次 |
| Solver v2 本地解析成功 | 0 / 8 次 |
| Verifier 调用 | 0 次 |
| 全部调用成功 / schema_error / provider_error | 4 / 6 / 6 次 |
| finish_reason=length | 10 / 16 次 |
| HTTP/网络错误、超时、调用预算阻断 | 均未观察到 |
| 已观察总 tokens / reasoning tokens | 54479 / 25224 |
| reasoning 占 completion tokens | 72.38% |
| usage 未知 | 0 次 |
| 总耗时 / 平均单题 / 最慢单题 | 536.96s / 107.39s / 132.30s |

这里的 0% 是工程失败计错后的结果，不表示已测得模型法律能力为零。
`provider_error` 的六次均为生成达到长度上限后的空正文，不是六次网络故障。
本轮没有任何题进入 Verifier，无法据此验证后半段实际表现。

## 逐题故障

| ID | 维度 | 最终失败位置 | 证据 |
|---|---|---|---|
| 59 | 法律伦理 | Solver | 两次 length，3208/3251 字符 JSON 均不完整 |
| 62 | 司法实践 | Solver | 两次 length + content=null；每次 reasoning=2399、completion=2400 |
| 68 | 法律与社会 | Controller | 两次 length，2255/2259 字符 JSON 均不完整 |
| 84 | 法律理解 | Solver | Controller 首次 stop 但 JSON 非法，重试修复；Solver 两次 length + 空正文 |
| 85 | 法律推理 | Solver | Controller 首次 stop 但 JSON 非法，重试修复；Solver 两次 length + 空正文 |

八次 Solver 请求均发送了 Schema、都收到 length 响应，无显式 Schema
不支持错误，无 prompt-only 回退。接受请求参数不代表服务端强制解码或完整
JSON 保证。低 reasoning 强度也不是 LongCat 的可靠上限。

Qwen 的八次 reasoning 计数均为 1024，但仍有四次 Controller 解析失败。
所以“reasoning 减少”与“格式成功、答案正确”必须分开统计。

## 额外解析缺陷

ID 85 的确定性复现：

```text
最终提问：下列有关谁应承担丁之治疗费的论述，不正确的是?
实际提取：正确的是
asks_for_incorrect_option: false
预期：true
```

`src/lgagent/legal_mcq/parser.py:26-28` 的 `_QUESTION_CUE` 能匹配“不正确”
内部的“正确的是”，`:65-70` 又从最后一次匹配起截取，导致否定词被丢弃。
即使后续模型输出完整，该错误也可能影响选项极性校验。

ID 68 从 1992 年案情转到“当代……若类似案件发生”的假设，解析器仍选择
`1992-12-31 / case_explicit`。这是相对时间与历史背景冲突的待处理风险。
这两点不是本轮 JSON 截断的已证原因，需独立修复、回归；原标签不改动。

## 成熟度判断

- 离线回归：本轮重新运行 `273 passed`，仅第三方 Authlib 弃用警告。
  它只证明已覆盖行为，不能证明供应商稳定返回合法输出。
- 观测与工件完整性：16 对 started/finished 一一对应，全部有 usage；
  日志、Trace 与结果一致，50 个源码、2 个输入、27 个历史工件哈希未变化。
- 输入与协议：存在确定性极性错误，以及 Controller/Solver 大量格式失败。
- 预算：本轮运行在配置限制内；但相同上限下重复生成仍可能连续截断，
  有限重试只能限制开销，不能保证恢复成功。
- 回答质量：本轮没有最终答案，无法评价逐项法律论证或证明准确率增益；
  旧法时点、题干歧义和标签质量也尚未完成独立审计。
- 最优性：多角色和 JSON Schema 都不构成正确性证明，更不保证全局最优答案。

## 下一步

1. 修复“不正确”否定词截取及相对时间处理，增加真实题干回归。
2. 收敛 Controller 协议并加入结构化约束；校验约束不能省略。
3. 针对 LongCat 明确区分隐藏推理与正文预算，验证真正可控的推理参数/端点；
   改进 length 重试策略，避免相同预算重复失败，不直接扩大付费批次。
4. 修复后先做脱敏回放与新小样本；稳定后再冻结配置、比较同题同预算基线、
   用独立测试集和种子 42/43/44 验证准确率。

## 工件

- [冻结清单](../output/legal_mcq/health-dev5-low-20260911/manifest.json)
- [完整结果](../output/legal_mcq/health-dev5-low-20260911/results.jsonl)
- [逐调用日志](../output/legal_mcq/health-dev5-low-20260911/results.calls.jsonl)
- [健康指标](../output/legal_mcq/health-dev5-low-20260911/health.summary.json)
- [可复核脚本](../output/legal_mcq/health-dev5-low-20260911/audit.py)

日志包含脱敏后的可见正文，不保存隐藏 reasoning 正文。模型日志应仅本地
使用；观测到的 tokens 不是实际账单金额，未推算或声称零费用。
