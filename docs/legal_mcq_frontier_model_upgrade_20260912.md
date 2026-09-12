# LegalMCQ 前沿模型角色升级

## 1. 目标

将最高能力模型放在唯一有答案决定权的 Strong Solver，而不是让 Controller
提前引导选项。Controller 继续只做中立结构，Verifier 继续只做核验。

最终角色配置：

| 角色 | 模型 | 原因 |
|---|---|---|
| Controller | Qwen/Qwen3.7-Flash | 已在 v5 dev10 中 10/10 Schema 成功 |
| Strong Solver | gpt-5.6-sol | 当前账户可访问的高能力模型，负责唯一主答案 |
| Solver fallback | Qwen/Qwen3.8-Max-0902 | 中文能力较强，与主 Solver 异构 |
| Verifier | Qwen/Qwen3.7-Flash | 紧凑 Verifier v2 输出稳定，10/10 成功 |

## 2. 模型发现与选择

2026-09-12 调用 CommandCode `/provider/v1/models`，返回 69 个模型。供应商
模型页显示 GPT-5.6 Sol 的 Intelligence 指标为 60.9，Qwen3.8 Max 为 58.1。
当前账户短探针结果：

| 模型 | HTTP | 短输出结果 | 决策 |
|---|---:|---|---|
| gpt-5.6-sol | 200 | stop，有正文 | 主 Solver |
| Qwen/Qwen3.8-Max-0902 | 200 | stop，有正文 | fallback |
| Qwen/Qwen3.8-Flash | 200 | stop，有正文 | 可用，但不替换已验证辅助模型 |
| deepseek/deepseek-v4-pro | 200 | stop，有正文 | 实际 Verifier Prompt 不稳定 |
| zai-org/GLM-5.3 | 200 | length，空正文 | 不用作紧凑 Verifier |
| google/gemini-3.8-flash | 200 | length，空正文 | 不用作紧凑 Verifier |
| meta/muse-spark-1.2 | 200 | length，空正文 | 不用作紧凑 Verifier |
| claude-fable-5-1 | 400 | unsupported_model | 当前账户不可用 |
| claude-opus-5 | 400 | unsupported_model | 当前账户不可用 |

模型列表只证明端点枚举；因此又执行了真实 Solver v3、Verifier v2 和
fallback 路径探针。

外部参考：

- CommandCode 模型与价格：<https://commandcode.ai/models>
- CommandCode Provider API：<https://commandcode.ai/docs/provider>
- Vals AI LegalBench：<https://www.vals.ai/benchmarks/legal_bench>
- Vals AI Legal Research：<https://www.vals.ai/benchmarks/legal_research>

外部榜单只用于提出候选，最终选择仍以当前账户可访问性和本项目协议实测为准。

## 3. 真实角色探针

### 3.1 GPT-5.6 Sol

两次完整 Solver v3 调用均成功：

- ID 127：约 52.7 秒，Schema 合法；
- ID 156：约 103.7 秒，Schema 合法；
- 扩大首次预算后 ID 156：约 114.5 秒，Schema 合法。

GPT-5.6 Sol 的隐藏 reasoning 占比较高，因此最终配置使用：

```yaml
solver_visible_output_tokens: 2048
solver_reasoning_allowance_tokens: 6144
solver_model:
  model_name: gpt-5.6-sol
  max_tokens: 12288
  reasoning_effort: high
solver_call_timeout_seconds: 120
```

正常首次请求为 8192 tokens，12288 仅用于一次明确 length 恢复。

### 3.2 Qwen3.8 Max fallback

题 259 使用 1 秒主 Solver 故障注入：

- GPT 主调用按预期 timeout；
- Qwen3.8 Max 在约 16.9 秒返回 Solver v3；
- Schema 合法；
- 选项 C 与标签一致；
- 后续 Verifier 可以完成；
- 最终 `completed`。

### 3.3 Verifier 候选

DeepSeek V4 Pro 在简单 32-token 探针中返回正常，但真实 Verifier Prompt
表现不稳定：

- 384 tokens：全部用于 reasoning，空正文 length；
- 512 tokens：全部用于 reasoning，空正文 length；
- 1024 tokens：题 259 可成功；
- 题 156 在 1024 和 1536 tokens 下仍连续空正文 length。

这说明简单 ping 不能代表真实角色适配。继续使用 DeepSeek 会显著扩大
Verifier token 和延迟，并仍不能保证短 JSON，因此没有纳入正式配置。

Qwen3.7 Flash 已在 v5 dev10 中完成 10/10 Verifier v2 Schema 响应，故保留
为辅助核验模型。主答案来自 GPT，避免形成 Qwen fallback 主答时的必然同模型
自审；fallback 场景仍可能存在同家族相关性，需要在后续实验中单独报告。

## 4. 编排修复

探针还发现 fallback 已使用后，第一轮 Verifier 仍错误预留了完整 Revision
的三个调用位。由于设计规定 fallback 后不允许业务 Revision，这个预留没有
意义，并可能阻止 Verifier 的 length 恢复。

修复后：

```text
solver_fallback_used = true
  -> revision_allowed = false
  -> Verifier reserve_calls_after = 0
  -> Verifier 可在剩余预算内执行一次 length 恢复
```

新增离线测试覆盖：

- primary timeout；
- fallback 成功；
- Verifier 首次 length；
- Verifier 384→512 恢复；
- 总调用数为 5；
- 最终 completed。

## 5. 最终配置变化

| 配置 | 旧值 | 新值 |
|---|---:|---:|
| 主 Solver | LongCat 2.0 free | GPT-5.6 Sol |
| fallback | Qwen3.7 Flash | Qwen3.8 Max 0902 |
| max_total_tokens | 32768 | 65536 |
| max_wall_time_seconds | 180 | 360 |
| solver timeout | 70s | 120s |
| fallback timeout | 45s | 90s |
| verifier timeout | 30s | 45s |
| Solver reasoning allowance | 2048 | 6144 |
| Solver hard cap | 6144 | 12288 |
| fallback hard cap | 4096 | 8192 |
| auxiliary reasoning reserve | 1024 | 2048 |

调用数上限仍为 7，SDK 自动重试仍为 0，最多一次业务 Revision。

## 6. 能证明与不能证明的内容

已经证明：

- 当前账户可调用 GPT-5.6 Sol；
- GPT 能输出合法 Solver v3；
- Qwen3.8 Max 能执行合法 fallback；
- 推荐角色组合可完成 Controller → GPT Solver → Qwen Verifier；
- fallback 后的 Verifier 恢复预算不再被无效 Revision 预留阻断。

尚未证明：

- GPT-5.6 Sol 在 LexGenius 上显著优于 Qwen；
- 新组合能稳定提高总体准确率；
- GPT 高 reasoning 在全部题型上具有最佳成本收益；
- 当前模型榜单能直接迁移到中文法律选择题。

ID 127 的推荐链路完成但与数据标签不一致；ID 156 的 GPT 输出也未提供明确
准确率改善证据。这些题可以用于 dev 模型选择，不能据此宣传提升。

## 7. 后续验证

下一步应在同一批冻结 dev 题上运行：

1. 旧 Qwen fallback 配置；
2. 新 GPT-5.6 Sol 主答配置；
3. 相同 Parser、Controller v3、Verifier v2 和预算口径；
4. 报告 Corrected、Harmed、completed、partial、timeout、tokens 和延迟；
5. 选择后冻结配置，再进入 test split 和 seeds 42/43/44。

## 8. 最终验证状态

- 推荐组合真实主路径：3/3 调用 `stop + Schema valid`；
- GPT-5.6 Sol Solver v3：成功；
- Qwen3.8 Max fallback Solver v3：成功；
- Qwen3.7 Flash Controller v3 / Verifier v2：成功；
- fallback 后 Verifier length-retry 预算回归：通过；
- 全仓离线测试：`321 passed`；
- 当前冻结 ID：`1931b3df9f45caf8f1c3ddc8`。

推荐链路 ID 127 返回 `completed`，但预测 B 与数据标签 A 不一致。因此本轮
结论仅为模型和协议兼容性通过，不是准确率提升结论。

机器汇总：
`output/legal_mcq/frontier-model-probe-20260912/validation.summary.json`。
