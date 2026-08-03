# Agentic Workload 性能/精度评估方法 — 六系统横向对比

> 汇总自六份独立调查报告：`vllm_agentic_evaluation_investigate.md`、`sglang_agentic_evaluation_investigate.md`、`tensorrt_llm_agentic_evaluation_investigate.md`、`llama_cpp_agentic_evaluation_investigate.md`、`ollama_agentic_evaluation_investigate.md`、`openvino_genai_agentic_evaluation_investigate.md`
> 调查方法一致：不看关键词命中数量，逐个打开源码验证"机制是否真的做了该做的事"

## 一、总览矩阵

| 维度 | vLLM | SGLang | TensorRT-LLM | llama.cpp | Ollama | OpenVINO GenAI |
|---|---|---|---|---|---|---|
| 定位 | 多租户 serving | 多租户 serving | 多租户 serving（企业级） | 单机推理引擎 | 单机产品（vendored llama.cpp + 自研 MLX） | Intel 硬件单机推理引擎 + 多语言 SDK |
| 官方性能 CI 负载形状 | 单轮 ShareGPT + 泊松到达 | 单轮 sharegpt/random/mmmu | 单轮固定 ISL/OSL 合成 | **无**（旗舰 Benchmark CI 已停用） | 单请求微基准 | 单/批量 prompt 微基准，**无官方回归 CI** |
| 仓库内多轮/agentic 压测能力 | 有，**两套互补**（Python 版 `benchmarks/multi_turn/` 强在多并发会话；Rust 版强在前缀共享比例+per-turn 指标），均未接入 CI | 有（原生集成于主 bench 模块，Mooncake trace 支持） | 有（trtllm-bench 原生支持，可用真实 MT-Bench 数据） | 无 | 无 | 无 |
| 多并发会话（多 Agent 负载形状）建模 | **最完整**：`--num-clients` × `--max-active-conversations` + 会话亲和 | 有：`gsp_num_groups` × `gsp_prompts_per_group`（多会话共享 system prompt） | 无专门建模 | 无 | 无 | 无 |
| 缓存命中率指标 | 客户端**近似值** `approx_cached_percent`（假设历史全命中） | **最强**：`cache_hit_rate_pct` + `device_/host_/storage_cached_tokens` 三层拆解 | 只有配置项（`kv_cache_reuse` bool），**无命中率指标** | 无 | 无 | 无（`cacheviz` 仅可视化） |
| 标准 agentic 精度基准接入 | **有现成 BFCL multi-turn 评测脚本**（`run-bfcl-eval.sh`），但零 pipeline 引用、零阈值判定 → 手动工具 | 无 | 无（仅 examples 里的 SWE-bench Coder，未接入日常 CI） | 无 | 无 | 无 |
| 该能力是否接入官方性能 CI | 否 | 否（仅接入一条独立的"功能正确性"nightly，非性能报告） | 否 | — | — | — |
| 真实两轮工具调用测试（依赖上一轮真实推理输出） | 否（fixture 伪多轮） | 写对了，但唯一调用点被注释掉，**从未跑过** | **有，且确认在 L0 CI 活跃跑** | 有真实模型，但仍是 fixture 伪多轮 | 仅单轮（21 模型矩阵） | 无（parser 单测用手工构造 delta，`react_sample` 测试只比对跨语言一致性） |
| 该工具调用测试是否在默认 PR 门禁跑 | 是 | 否（死代码） | 是 | 否（标记 `slow`，仅 schedule/手动） | 否（需 `-tags=integration`，未见于公开 CI） | 是（`test_react_sample_refs` 在常规 CI 跑，但不验证任务正确性） |
| 工具调用测试时前缀缓存/KV复用默认状态 | **11/12 配置显式关闭** | 默认开启（未显式关闭） | 默认开启（未显式关闭） | 未特别处理 | N/A（产品层不直接控制） | N/A（工具调用测试未涉及 CB/前缀缓存路径） |
| 量化/优化 × 精度 × 缓存复用 三者同框测试 | 无（2/38 kv_fp8 配置动机是"加速评测"） | 有 KV FP8 精度测试，但**显式关闭** radix cache | **有**：21 处显式开启 block reuse 配合量化/投机解码/guided decoding 做精度评测，且有 reuse/no-reuse 显式 A/B | 无自动化精度回归 | 无（精度评估完全委托上游） | **有，且是六者里方法论最严谨的一处**：同一断言里同时要求"相似度阈值"和"缓存压缩比阈值"都达标，覆盖 SnapKV/KVCrush/AdaptiveRKV |
| 性能指标独有亮点 | `moving_avg_ttft/tpot`（观测缓存填满引起的时延漂移） | 分层缓存 token 统计 | `output_throughput_per_user`（多会话下唯一正确的吞吐视角）、`acceptance_rate/length`（投机解码收益量化） | — | — | `ipot`（区分推理时延与端到端 TPOT）、`grammar_compile_time`（约束解码开销） |
| 多 Agent **协作正确性**测试 | 无 | 无 | 有编排原语（`ParallelProcess`、`MCTSController`/`TOTController`），但测试用 `DummyTask`/mock worker | 无 | 无 | 无 |
| 隔离性/公平性指标（一个会话是否拖累其他会话） | 无（有 `num_preemptions` Prometheus 指标但未进压测报告） | 无 | 无 | 无 | 无 | 无 |
| 精度评估方法论本身 | lm-eval-harness，38 配置，单轮 gsm8k/mmlu | lm_eval_configs，4 配置，单轮 gsm8k | 224 个测试，GSM8K/MMLU/JsonModeEval/CnnDailymail | 仅困惑度 + KL 散度，纯统计非任务型 | 无 | WWB：11 类评测器（文本/图像/视频/语音/多模态），任务类型覆盖面六者最广，但无 agentic 类型 |
| 精度评估是否接入 CI | 是 | 是 | 是 | **否**（零命中，纯人工） | 无自身精度评估 | 是（WWB 矩阵化接入 `linux.yml`） |
| MCP 协议原生支持 | 无 | 无 | 有（`tensorrt_llm.scaffolding`，编排层） | **有（推理服务器本体自带）** | 有（产品 Agent 层，`agent/tools`） | 无 |
| 真实可运行的 Agent 框架/产品 | 无 | 无（`benchmark/react` 明确声明"不是真实 agent"） | 有（`examples/scaffolding`，含真实 SWE-bench Coder），但活在 examples/，未接入 CI | 无 | **有，是主二进制自带的产品功能**（`agent/`） | 无（`react_sample` 只是样例脚本，非产品功能） |
| 针对"多轮上下文持续增长"的产品级方案 | 无（仅系统层 KV cache 驱逐） | 无 | 无 | 无 | **有：自动上下文压缩**（`compactor.go`，80% 阈值触发） | 无（KV cache eviction 是系统层，非会话层方案） |
| 独特工程决策 | — | — | — | — | — | **直接复用 vLLM 的 tool-call parser 生态**（`VLLMParserWrapper`），不重复造轮子 |
| "看似完整 agent 循环"测试打开后发现的真相 | 写死的三消息 fixture | 死代码（从未执行） | `DummyWorker` 按轮数硬编码返回 | 写死的 fixture（但真模型） | `fakeClient` 按调用次数返回预设响应 | `test_react_sample_refs` 真推理但只测跨语言输出一致性，不测任务对错 |

## 二、跨系统重复验证的四条规律

这六次独立调查里，同一类问题在完全不同的代码库里反复出现，说明这不是某一家的疏漏，而是这个领域当前的系统性状态。

### 规律 1：任何"看起来像完整 agent 循环"的测试，打开生成侧必然是假 LLM，或者验证目标另有所指

vLLM 的 fixture、SGLang 的死代码、TRT-LLM 的 `DummyWorker`、llama.cpp 的写死历史、Ollama 的 `fakeClient`——六次调查里，前五次直接印证。第六次（OpenVINO GenAI 的 `test_react_sample_refs`）是这条规律的一个变体：它确实用了真实模型做真实推理，看起来最接近"终于有一个真的"，但打开断言一看，验证目标压根不是"任务有没有做对"，而是"两种语言绑定的输出是否逐字节一致"——**换了一种方式得到同一个结论：编排/集成测试普遍不验证"真实模型是否做出了正确的 agentic 决策"**。

### 规律 2：量化/缓存类精度回归，几乎总是在关掉相关优化的前提下做的（但也有两个层次不同的例外）

vLLM（`--no-enable-prefix-caching`，11/12 工具调用配置）、SGLang（KV FP8 精度测试 `--disable-radix-cache`）都是这个模式；**TensorRT-LLM 和 OpenVINO GenAI 是六者中的两个例外，但深度不同**——TRT-LLM 把"缓存复用开启"当成精度测试矩阵里的一个显式变量（21 处 A/B），OpenVINO GenAI 更进一步：把"精度阈值"和"性能收益阈值"写进同一条断言里联合校验，是唯一做到"既要开着优化、又要求收益达标、还要求精度不掉"三者合一的系统。这条规律的例外正好说明：这不是技术上做不到，只是大多数项目没有把"缓存复用开启"当成精度测试矩阵里需要显式覆盖的一个维度。

### 规律 3：多轮/agentic 压测能力普遍"仓库里有，官方指标里没有"——三家单机引擎则是压根没有这个能力

vLLM 的 Rust bench 工具、SGLang 的 Mooncake/GSP 数据集、TRT-LLM 的 trtllm-bench 多轮支持——三家云端 runtime 都具备原生或半原生的多轮压测能力，但官方追踪、用来判定 PR 是否引入性能回归的指标，清一色是单轮固定长度或 ShareGPT 泊松到达。SGLang 稍微好一点：把 GSP 多轮接入了一条独立的"功能正确性"nightly CI（验证前缀增长的结构对不对），但这条线依然不产出、不追踪性能数字。**三家单机引擎/产品（llama.cpp、Ollama、OpenVINO GenAI）连"仓库里有"这一步都没有**——它们的官方 bench 工具（`llama-bench`/`ollama bench`/`llm_bench`）都是单请求微基准，从设计上就没打算支持并发/数据集/多轮，这和"多租户 serving vs 单机推理"的产品定位差异直接相关。

### 规律 4：真实、有价值的测试存在，但被排除在"日常必跑"的门禁之外

TRT-LLM 的 SWE-bench Coder（`examples/`，纯手动）、llama.cpp 的 `test_calc_result`（标记 `slow`，仅 schedule 触发）、Ollama 的 21-模型真实工具调用矩阵（需要 `-tags=integration`，未见于公开 CI）——最贴近真实场景、最有说服力的测试，普遍因为"太慢/太贵/需要下载模型或占用大量磁盘"而被移出了每次 PR 都会跑的路径，只在 nightly、手动触发或私有基础设施上运行。OpenVINO GenAI 的 `test_kv_cache_eviction` 反而是这条规律的正面例外——它确实跑真实模型、真实 LongBench 数据，却依然稳定地接在 Linux/Windows 的常规 CI 里（只在 Mac 上被注释掉，原因是平台兼容性问题而非"太慢"）。

## 三、单项最强 / 最弱一览

| 单项 | 最强 | 说明 |
|---|---|---|
| 真实两轮工具调用 CI 覆盖 | **TensorRT-LLM** | 唯一一处确认活跃在 L0 门禁里的真两轮（依赖上一轮真实推理）测试 |
| 优化 × 精度（联合断言）严谨程度 | **OpenVINO GenAI** | 唯一把"性能收益阈值"和"精度阈值"写进同一条断言、且设计了"不该优化时验证没瞎优化"边界场景的系统 |
| 缓存复用 × 精度 交叉覆盖广度 | **TensorRT-LLM** | 21 处显式测试，含 reuse/no-reuse A/B，覆盖量化/投机解码/guided decoding 多种组合 |
| 真实模型工具调用覆盖广度 | **Ollama** | 21 个真实模型矩阵，但只测单轮且未接入公开 CI |
| 多轮压测负载真实感/原生集成度 | **SGLang** | Mooncake trace（真实 hash_ids 前缀共享结构）+ GSP，原生在主 bench 模块 |
| 多轮会话结构正确性的活跃功能测试 | **SGLang** | `test_gsp_multi_turn`：真实解析服务端日志校验前缀包含关系 |
| MCP 协议集成深度 | **llama.cpp** | 唯一把 MCP 直接内建在推理服务器本体的系统 |
| 真实可运行 Agent 产品 | **Ollama** | 唯一把 Agent 框架做成主二进制自带功能而非 example 的系统 |
| 针对 agentic 上下文增长的产品方案 | **Ollama** | 唯一的自动上下文压缩机制 |
| 前缀缓存功能测试的严谨程度（机制层面） | **llama.cpp** | 唯一验证"可观测效果"（`prompt_n` 真实变少）而非仅内部数据结构的测试 |
| 自带精度工具的任务类型覆盖面 | **OpenVINO GenAI（WWB）** | 11 类评测器横跨文本/图像/视频/语音/多模态，但同样没有 agentic 类型 |
| 工程复用意识 | **OpenVINO GenAI** | 唯一直接复用别家（vLLM）tool-call parser 生态而非重复实现的系统 |
| 性能压测工具原始程度（越原始越差） | **Ollama ≈ llama.cpp ≈ OpenVINO GenAI** | 三家单机引擎/产品的官方 bench 工具都是单请求微基准，无并发/数据集/多轮建模 |
| 精度评估自动化程度（越弱越差） | **llama.cpp（最弱）> Ollama（无）** | llama.cpp 好歹有困惑度但零 CI 化；Ollama 干脆没有自己的精度评估 |
| 官方性能回归 CI 存续状态（越差越差） | **llama.cpp（最差）** | 旗舰 Benchmark 工作流已停用近两年，无任何自动化性能回归；OpenVINO GenAI 次之（从未存在过对应机制，而非"曾有后停用"） |

## 四、最终结论

把六份报告叠在一起看，得到一个跨越所有六个系统、没有例外的结论：

> **没有一个系统，把"多轮会话 + 工具调用 + 主流性能优化组合开启（前缀缓存/KV量化/投机解码）"作为一个整体，纳入过日常自动化的 CI 精度回归。**

每家都在某个局部维度做得比其他家扎实（TRT-LLM 的缓存复用×精度矩阵、SGLang 的多轮结构校验、Ollama 的产品级上下文压缩、llama.cpp 的 MCP 原生集成、vLLM 的 BFCL 负载真实感、OpenVINO GenAI 的性能-精度联合断言方法论），但这些扎实的局部实践从未在任何一家被拼接成一条完整的端到端验证链路，也没有一家互相借鉴——六个系统各自独立地在不同的局部维度上"发明"了自己的一小块最佳实践，却没有哪一块被推广成行业common practice。这不是资源不够或某个团队疏忽，而是当前整个领域对"agentic serving"的评测方法论——无论是性能基准还是精度回归——都还处于早期、碎片化的阶段：负载建模能力（多轮/前缀共享/工具调用）的成熟度，明显领先于"如何系统性地验证这些能力组合在一起时是否仍然正确"这件事的成熟度。

对 Runtime 研发的启示：这恰恰是一个还没有被充分标准化、值得投入建设的方向——具体抓手包括：

1. 把 OpenVINO GenAI 的"性能阈值 + 精度阈值联合断言"方法论,从长文本 QA 场景**移植到工具调用/多轮 agentic 场景**（例如：前缀缓存开启后,工具调用准确率必须维持在 X%,同时 TTFT 必须降低至少 Y%,同一断言里两者都要过）；
2. 参考 TensorRT-LLM 的 reuse/no-reuse A/B 设计,推广到 KV 量化、投机解码等其他优化项；
3. 参考 SGLang 的日志级结构校验方法（真实解析服务端日志验证前缀增长关系）,把它用在验证"压缩/驱逐之后,历史信息保真度是否足够支撑任务继续"这个 Ollama compactor 目前的评测盲区上；
4. 把这些局部实践系统化成一套可以直接套用到自己 Runtime 上的标准评测矩阵，而不是继续让每家各自摸索。
