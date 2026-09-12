# LegalMCQ 输入输出修复

## 范围与结论

针对健康检查中极性丢失、Controller 非法 JSON、Solver reasoning 挤占正文
的问题，更新三角色 Prompt 版本为 `legal-mcq-three-role-v3`。
本轮只做代码修复、离线回放和 MockTransport 验证，没有真实供应商调用。
历史 30 题、dev5 和 low-effort 重放结果均不覆盖，也不回填新成绩。

## 极性与时点

- 提取提问时先规范化空白，完整匹配“不正确的是”，不再从中间的
  “正确的是”开始截取。仍只检查最终提问，避免案情否定词污染。
- 最终提问明确采用“当代/现行/目前”等时点时，优先于旧案日期。
- “当代……若类似案件发生。此时……”的两句结构也能识别。
- 明确假设日期、明确提问日期和请求 metadata 的原有优先级不变。
- ID 85 恢复为反向题；ID 68 不再绑定 1992 年；ID 5/12/20/25 回归通过。

## Controller v2

```json
{
  "protocol_version": "controller-v2",
  "facts": ["关键事实摘要"],
  "issues": [{"description": "待核验争点", "options": ["A", "B"]}],
  "option_claims": {"A": ["必要命题"], "B": ["必要命题"]},
  "checks": ["需要核验的条件"]
}
```

不再要求重复领域、法条、逐条 ID 和多套核验列表。ID 由代码按输出顺序生成：
F1、I1、A1、A2 等；下游使用规范化后的 Prompt 数据而非未编号原始输出。
v2 下 Solver 必须覆盖这些 claim ID；不能用同数量的其他 ID 替代。

Schema 名为 `legal_mcq_controller_v2`，每个 object 均 required 且禁止额外键，
option_claims 的键按当前题目选项固定。Schema 只使用公共关键字子集；
本地继续拒绝空 facts/issues/claims、非法标签和答案字段。
旧无版本 Controller 与 Solver v1 仍可读取，Solver 主协议保持 v2。

`controller_structured_output_mode` 支持 auto/json_schema/prompt_only，
明确能力拒绝才允许 auto 回退，且计入调用预算。格式失败不能降为直接接受。

## Solver 输出额度

示例新配置：

```yaml
legal_mcq:
  controller_structured_output_mode: auto
  solver_structured_output_mode: auto
  solver_visible_output_tokens: 2048
  solver_reasoning_allowance_tokens: 2048
  auxiliary_reasoning_reserve_tokens: 1024
  solver_model:
    max_tokens: 6144
```

- 首次总 completion 上限为 `min(model.max_tokens, visible + reasoning)`，
  示例为 4096，不再固定截成 2400。
- 只在失败响应报告 length 时规划一次恢复：
  `min(cap, max(previous_limit + 1024, observed_reasoning + visible_target + 256))`。
- 空正文且 length 使用同一恢复路径；缺失 reasoning 计数时仍只作有界扩额。
- 复用原始可信输入，附紧凑重生成指令；不把截断 JSON 拼到下一次上下文。
- 模型硬上限已满、恢复已使用或全局剩余预算不足时，跳过重复请求并记录原因。
- Revision Solver 从同样的首次额度开始，业务修订仍最多一次。
- Controller/Verifier 请求额外计入辅助 reasoning 预留；实际 usage 仍全额结算。

题级 7 次调用、32768 tokens、180 秒，单调用 60 秒均不增加。
设置更大的模型 cap 不代表每次使用 cap；实际请求值在 Trace/dry-run 可查。
旧独立配置中显式写的 2400 不会被偷偷覆盖，要使用新策略需在新运行配置中
显式更新模型 cap。总上限增长可能增加单次生成费用，但不增加题级预算。

重要边界：visible/reasoning 是预算规划目标，不是供应商硬分区；low effort
也仍是软参数。此修复消除了代码中固定 2400 和相同预算重复截断的机制，
不能保证供应商一定完成思考并给出正文。

## 验证

最终全仓结果为 `305 passed`，仅一条第三方 Authlib 弃用警告；compileall 与
`git diff --check` 通过。核对的 33 个历史工件哈希无变化。新冻结 ID 为
`22372a3fef72237c9dfe70bb`，旧 ID `2dc44ee693640c4f2371d6a0` 已归档。

- 极性、相对时间、旧题干回归。
- Controller v2 严格字段、动态选项、ID 分配、旧协议读取。
- 真实 OpenAI SDK + httpx.MockTransport，验证 Controller/Solver Schema 与
  有效请求上限经过预算包装后抵达 HTTP 请求体；无真实网络。
- 空正文和截断 JSON 的有界恢复、硬上限不重复、三次尝试配置下最多一次
  length 恢复、调用或 token 额度不足时跳过。
- Controller 格式恢复 + Solver length 恢复 + 一次业务修订 + 最终 Verifier，
  共七次调用的完整组合路径。
- 历史健康检查的 12 条失败可见输出仍全部被拒绝，没有放宽协议掩盖错误。

运行：

```bash
.venv/bin/python -m pytest -q
```

后续需要使用新输出目录、固定配置做受限的供应商 smoke test，才能判断
真实 Controller Schema 支持度及截断率是否改善。离线通过不等于生产成熟。

后续同题真实重放已完成：观测 length 从 10/16 降至 1/15，Controller
首次成功 5/5，Solver 有 2 次完整 `stop`，但另 5 次调用在 60 秒超时。
所以截断显著改善，端到端链路仍不健康。详见
[v3 同题真实截断对照](legal_mcq_io_v3_real_replay_20260912.md)。

## 回滚

总开关 `legal_mcq.enabled: false` 保持原 LGAgent 路由。Controller 的
prompt_only 可省略 Schema 参数但仍要求 v2；它不是恢复旧 Prompt。
旧实验完整复现需使用旧源码及其冻结配置，不能仅修改 prompt_version 标签。
