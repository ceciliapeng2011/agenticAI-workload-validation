# 面向管理层的技术报告：OpenVINO（含 OpenVINO GenAI）应如何构建 Agentic AI Workload 测试体系

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：**OpenVINO 管理层报告** · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

> 提交对象：OpenVINO / OpenVINO GenAI 管理层
> 依据材料：对 vLLM、SGLang、TensorRT-LLM、llama.cpp、Ollama、OpenVINO GenAI 六个推理引擎的独立源码级调查（`*_agentic_evaluation_investigate.md`），及在此基础上产出的横向对比、测试设计方案、能力×严谨度矩阵、基准测试详解四份分析文档
> 调查范围声明：本报告的实测证据全部来自 `~/openvino.genai` 仓库，**不包含 OVMS（OpenVINO Model Server）**；OVMS 若有独立的 serving 层测试资产，需另行盘点，本报告的现状评估和结论不覆盖它。

## 摘要

六个业界主流推理引擎的源码级调查得到一条没有例外的结论：**没有一个系统把"多轮会话 + 工具调用 + 主流性能优化（前缀缓存/KV量化/投机解码）组合开启"作为一个整体，纳入过日常自动化的 CI 精度回归。** 这不是 OpenVINO GenAI 一家的短板，而是整个行业当前的系统性空白——这既是坏消息（说明这件事确实难），也是好消息（说明谁先做出来，谁就定义了这个方向的行业实践）。

OpenVINO GenAI 目前的独特优势和短板都很鲜明：`test_kv_cache_eviction` 的"性能阈值 + 精度阈值同断言"方法论，是六个系统里**方法论最严谨**的一处，值得作为全公司测试设计的范本向外推广；但这套严谨方法论目前只用在了长文本 QA 这个相对基础的能力级别上，完全没有触及工具调用、多轮任务完成度这些更核心的 agentic 能力。同时，OpenVINO GenAI 在"多轮/并发压测能力"上是六者中最原始的一档（`llm_bench` 无并发、无数据集、无多轮参数），也没有官方性能回归 CI。

本报告的核心建议：**不是从零发明新测试方法论，而是把"OpenVINO GenAI 自己已经证明可行的联合断言方法论"，向上嫁接到工具调用/多轮 agentic 场景，同时借鉴其他五家已经分别做对的局部实践**（TensorRT-LLM 的真实两轮编排、llama.cpp 的终态数值判定、vLLM 的多会话负载建模、SGLang 的缓存命中日志核验）。报告第四节给出按优先级排列的 5 类测试（A-E），每类都标注了可参考的现成实现、需要补的缺口，以及建议的 CI/Nightly/Weekly 归属。

---

## 一、Agentic Workload 的特点，及每个特点应该由哪类测试覆盖

在讨论"该建什么测试"之前，先明确 agentic workload 相对传统单轮 QA 负载，到底多出了哪些系统层面必须应对的特点——这是后面所有测试设计的出发点，每一条特点都对应报告第四节里的具体测试类别。

| # | Agentic Workload 特点 | 对系统的挑战 | 对应的测试类别（见第四节） |
|---|---|---|---|
| 1 | **多轮上下文持续累积** | 单个会话的 KV cache 占用随轮次线性增长，长会话可能挤占其他会话资源 | C（agentic 性能）、E（KV eviction 类基础设施） |
| 2 | **高前缀重复率**（system prompt、工具 schema 在同一会话/多会话间反复出现） | 前缀缓存/RadixAttention 的收益高度依赖这类负载，但"配置开了"≠"真的命中了" | C（缓存命中率验证）、B（配置敏感性） |
| 3 | **长输入、短输出**（工具调用结果、长历史 vs 模型只需给出简短的下一步决策） | 与传统"长输出"压测的性能特征相反，ISL/OSL 比例失衡场景常被默认压测忽略 | C（负载形状建模） |
| 4 | **工具执行期间的空闲间隙**（调用外部 API/沙箱执行等待） | 调度器需要正确处理"请求挂起等待外部结果再续接"，不是纯粹的 token-by-token 生成 | A（编排真实性）、C（多会话资源竞争） |
| 5 | **依赖驱动到达，而非泊松过程**（第二轮请求的时机和内容依赖第一轮的真实输出） | 默认压测的到达模型（ShareGPT+泊松）无法模拟这种"因果链"负载 | A（编排真实性）、C（负载形状建模） |
| 6 | **工具调用是结构化输出，需要被下游解析**（而非自由文本） | 结构化/约束解码（xgrammar 等）在高并发、复杂 schema 下可能有性能陷阱；解析器本身需要单独校验 | A（工具调用正确性）、D（对抗性鲁棒性） |
| 7 | **多 Agent / 并行子任务**（fan-out 到多个子 Agent，结果汇总） | 子 Agent 间共享前缀的复用效率、协作后的终态正确性，都是单会话测试测不到的 | C（多会话）、新增的多 Agent 协作正确性测试 |
| 8 | **轨迹级正确性，而非单次输出正确性**（任务是否最终完成，取决于一整条多轮决策链） | 传统"这一次生成的 token 对不对"的判分方式失效，需要终态判定 + 可靠性（pass^k）度量 | A/B（终态判定）、任务集设计整体 |

这八条特点里，第 1/2/3/7 条是"负载形状"问题（测试需要真实还原这种负载），第 4/5/6/8 条是"判分方法论"问题（测试需要用对的方式判断对错）——第四节的测试清单同时覆盖这两类。

---

## 二、OpenVINO GenAI 现状盘点：已有资产 vs 空白

### 2.1 已有的、值得公司范围内推广的资产

- **`test_kv_cache_eviction_1/2.py` 的联合断言方法论**（`tests/python_tests/test_kv_cache_eviction/`）：同一个测试函数里，`similarity_metric > similarity_threshold`（精度）和 `max_optimization_ratio >= max_cache_usage_optimization_ratio`（性能）两类断言同时要求通过，且设计了"prompt 短到不触发驱逐"的边界对照场景（此时精度阈值反而提高到 0.98，明确写"no improvement expected"）。覆盖 SnapKV/KVCrush/AdaptiveRKV 多种驱逐算法，用真实 LongBench 数据（`samsum`/`trec`/`qasper`）和标准评分函数（ROUGE/分类准确率），确认活跃在 Linux/Windows CI（`linux.yml:597,601`；`windows.yml:684,688`；仅 Mac 因平台兼容性问题注释掉）。**这是六个系统调查以来，唯一一处把"性能收益阈值"和"精度阈值"写进同一条断言的实现**，对比之下 TensorRT-LLM 只把"缓存复用开启"当精度实验变量做 A/B，但不要求"性能收益必须达到多少倍"；vLLM/SGLang 的对应测试普遍直接关掉优化项，不测权衡。
- **`PerfMetrics` 结构已具备较细粒度的性能指标**（`src/cpp/include/openvino/genai/perf_metrics.hpp`）：`ttft`/`tpot`/`ipot`（区分推理时延与端到端 TPOT，六者中独有）/`throughput`，以及 `grammar_compiler_init_times` + `get_grammar_compile_time()`（约束解码编译开销，六者中唯一显式度量这项开销的系统）。这些字段已经就位，缺的是"在 agentic 负载形状下采集并断言"这一层。
- **`VLLMParserWrapper`**（`tests/python_tests/test_vllm_parsers_wrapper.py`）：直接复用 vLLM 生态已维护的 tool-call parser，而非像其余五家各自独立维护一套 per-model parser。这是六个系统里唯一一处"不重复造轮子、直接对接社区已有生态"的工程决策，测试文件版权头都保留了"Copyright contributors to the vLLM project"——继续沿这条路线走（例如未来对接 BFCL 评测脚本时，优先看能否复用 vLLM/SGLang 已经写好的评测适配层），比自建一整套更省成本。
- **WWB（who_what_benchmark）的可扩展性**：`@register_evaluator` 注册机制覆盖 11 类任务（文本/图像/视频/语音/多模态），是六个系统自带精度工具里覆盖面最广的，且真正矩阵化接入 `linux.yml` 日常 CI。新增一个 agentic/工具调用类型的 evaluator，是在现有框架内扩展，不需要另起炉灶。
- **已有 `@pytest.mark.agent` marker**（`tests/python_tests/samples/test_react_sample.py`）：六个系统里唯一带专属 agent 标记的测试，且是真实推理（真实转换 TinyLlama 并跑 ReAct 样例）。目前断言目标是"Python 版和 JS 版输出逐字节一致"（跨语言一致性，非任务正确性），但这个 marker 和真实推理管线已经搭好，改造断言逻辑的成本远低于从零新建。
- **`cacheviz`**：前缀缓存可视化工具，可作为"验证缓存真的被命中"这类测试的调试/取证基础设施。

### 2.2 确认的空白

- **没有真实的多轮工具调用正确性测试**：`test_parsers.py` 系列测的是解析器单测（手工构造 delta 片段），`test_react_sample_refs` 虽真实推理但判分目标错位（跨语言一致性 ≠ 任务对错）。六个系统里，这个模式反复出现——"看起来像完整 agent 循环"的测试，打开后要么生成侧是假 LLM，要么判分目标另有所指；OpenVINO GenAI 属于后一种。
- **没有接入任何标准 agentic 精度基准**（BFCL/τ-bench/AgentBench），WWB 的 11 类评测器里没有工具调用/agentic 类型。
- **`llm_bench` 是单/批量 prompt 微基准**：核心参数只有 `-p/-bs/-d`，没有并发、没有数据集后端、没有多轮/session 参数——和 llama.cpp 的 `llama-bench`、Ollama 的 `cmd/bench` 同属最原始一档，**完全没有面向 agentic 场景（多轮、并发会话、前缀共享）的性能评测手段**。
- **没有官方性能回归 CI**：`.github/workflows/shared/*.yml` 里对 perf/benchmark/throughput/latency 关键字零命中，没有类似 TensorRT-LLM `base_perf_pytorch.csv` 那样的基线追踪机制。
- **前缀缓存 + 投机解码组合测试只测"不崩"**（`test_eagle3_prefix_caching_no_crash`），不比较两次输出内容是否一致或精度是否保持——和第三节的 cache eviction 测试形成鲜明对照，说明公司内部已有的严谨方法论并未被推广到其它优化组合上。
- **`test_kv_cache_eviction` 的覆盖范围局限在长文本 QA/摘要/分类**，完全没有延伸到工具调用输出正确性、多轮任务完成度——这两项恰恰是 agentic workload 最核心的能力维度。

一句话总结现状：**OpenVINO GenAI 已经在"如何设计一个不作假的性能-精度联合测试"这个方法论问题上，拿出了六个系统里最好的范本，但还没有把这个范本用在刀刃上（工具调用/多轮任务），性能侧的 agentic 负载建模能力也是六者中最薄弱的一档之一。** 这既是差距，也意味着补齐的技术路径是清楚的——不需要发明新方法论，只需要把已经验证过的方法论迁移应用。

---

## 三、优先级测试清单：A-E 五类，每类标注参考实现与 CI 归属

### A 类：工具调用编排正确性（对应特点 4/5/6/8）——补最大的空白

**要测什么**：真实两轮（或循环到模型自己结束）的工具调用编排——第二轮请求必须依赖第一轮真实推理产出的 `tool_call`，工具本身真实/仿真执行（不是把结果预置进请求里），最终判定任务终态是否正确（不是判定"格式合不合法"或"有没有崩"）。

**为什么重要**：这是六次调查里反复出现的"最大假象"——vLLM 的三消息写死 fixture、SGLang 写对了但唯一调用点被注释掉、TensorRT-LLM 的 `DummyWorker` 按轮数硬编码返回、llama.cpp 工具结果仍写死注入、Ollama 的 `fakeClient`、OpenVINO GenAI 自己的 `test_react_sample_refs`（真推理但判分目标错位）——**没有一个系统把"模型真实决策"和"任务终态正确"两件事同时测到**。

**谁做对了哪一块，可以直接借鉴**：
- **编排真实性（G1）最佳实践：TensorRT-LLM** `tests/unittest/llmapi/apps/_test_openai_chat_harmony.py::test_tool_calls`——唯一一处确认活跃在 L0 CI 里的真两轮测试，第二轮真实依赖第一轮输出。缺口：仅覆盖 Harmony 格式，需扩展到通用 tool-call 协议，且需要从"固定两轮"扩展到"循环到模型自己收尾"。
- **判分正确性（G2）最佳实践：llama.cpp** `test_calc_result`——六者中唯一断言具体数值结果（0.56）而非格式合法性的工具调用测试。缺口：该测试标记 `slow`，默认 CI 门禁不跑；且工具结果仍是写死注入，需要改造成真实/仿真执行。
- 具体任务集设计（覆盖单工具/多工具依赖/反例场景"不该调用工具时不能瞎调用"/长历史工具结果召回）和编排代码框架，已在 `agentic_test_design_proposal.md` 第 3.1-3.2 节给出可直接落地的 Python 实现，可直接参考改造。

**CI 归属**：精简任务集（3-5 个代表性任务，1 个小模型）放 **PR 门禁**；完整任务集 + 长历史召回场景放 **Nightly**；`pass^k`（k≥4）稳定性验证放 **Release 前**（详见 3.4 节 τ-bench 方法论：同一任务重复跑 k 次，全部成功才算过，用来暴露"运气好跑通一次"的假阳性）。

### B 类：优化配置 × 精度的联合测试（对应特点 2）——把公司已有方法论用到刀刃上

**要测什么**：把 A 类的任务集，分别跑在"优化全关"和"优化组合开启"（前缀缓存 + KV 量化 + 投机解码同时开）两种配置下，且判分标准在两种配置下都要过——直接沿用 `test_kv_cache_eviction` 已经验证过的"同一断言里性能阈值和精度阈值都要满足"的写法，只是把断言对象从"LongBench 相似度"换成"工具调用任务成功率"。

**为什么重要**：这是全公司现有最强方法论资产的自然延伸，成本增量主要在"设计新的判分逻辑"，不需要重新发明测试框架。同时这个组合（缓存+量化+投机解码同时开、且要求 agentic 任务成功率不掉）**是六次调查里没有任何一个系统测过的**——先做出来就是行业领先。

**可参考实现**：TensorRT-LLM 的 21 处 `enable_block_reuse=True` vs 59 False 精度测试，是缓存复用×精度覆盖面最广的（含量化/投机解码/guided decoding 多种组合的 reuse/no-reuse A/B），可以参考其"如何搭建 A/B 矩阵"，但要补上 OpenVINO GenAI 自己已经做到的"性能收益量化阈值"这一层——TRT-LLM 目前只断言精度过线，不断言收益倍数。

**CI 归属**：`baseline` + 1 个"全优化开启"档放 **PR 门禁**；完整配置矩阵（4 档：baseline/仅缓存/缓存+量化/缓存+量化+投机解码）放 **Nightly**。

### C 类：Agentic 性能压测（对应特点 1/2/3/7）——补最原始的一环

**要测什么**：分四层递进，建议按顺序建设：
1. **负载形状建模**：多轮、高前缀重复率、长输入短输出的合成/真实负载生成器（目前 `llm_bench` 完全没有这一层）。具体设计需要把三个维度当成独立可组合的旋钮：
   - **多轮**：优先用"真实驱动"而非"回放录制对话"——起真实 server，Turn 1 真推理拿到输出，回填后驱动 Turn 2，历史增长曲线才反映"这个配置下真实会发生的样子"（参考 vLLM `rust/src/bench/`、SGLang `serving.py::wrap_multi_turn_request_func`），而不是像 vLLM Python 版 `benchmarks/multi_turn/` 那样基于 ShareGPT 录制对话回放（真实感强但历史增长和被测模型的真实行为脱节）。
   - **高前缀重复率**：用分组结构直接控制共享比例（参考 SGLang GSP 数据集的 `gsp_num_groups × gsp_prompts_per_group`），或直接给比例旋钮（vLLM rust bench 的 `--multi-turn-prefix-global-ratio`），比人工猜一个"前缀缓存开着就行"更可控；有条件时优先用真实生产 trace（SGLang 的 Mooncake trace，真实 `hash_ids`）而非人工合成分布。
   - **长输入短输出**：复用真实长上下文 agent 任务的 prompt 分布，例如 SGLang FlexKV 用的 SWE-bench_Lite_oracle（真实观测 p50=7088、max=27961 token）,或 OpenVINO GenAI 自己 `test_kv_cache_eviction` 已经在用的 LongBench 子任务（`samsum`/`trec`/`qasper`）——后者的 ISL≫OSL 结构和评分函数可以直接复用，只需把评分函数换成工具调用任务终态判定。
   - **多长算长**：六个系统里大多把"多长"当运行时可配置参数，没有统一的官方上限；唯一有硬编码数字的是 Ollama `agent/compactor.go`（`defaultCompactionContextWindowTokens=32768`，80% 阈值触发压缩，即约 26214 token）——但这是压缩触发点，不是能力上限。建议测试矩阵里显式加入"session 长度逼近模型 context window 上限时会发生什么"这一档，目前全行业没有人测过。
2. **缓存命中率的真实验证**：不能只信任"配置开了前缀缓存"这个开关本身，要解析服务端真实日志/指标，确认 agentic 会话过程中缓存确实被命中——否则"任务成功率没掉"这个结论可能只是因为缓存压根没被命中（比如工具调用改变了 prompt 结构导致前缀失配）。
3. **多会话场景下的正确性能判据**：单会话时代的四个惯用指标在多会话下会系统性说谎，必须换成对应的多会话版本：

| 指标 | 单会话做法（会说谎） | 多会话正确做法 | 谁已经做到，可参考 |
|---|---|---|---|
| 吞吐 | 聚合吞吐（高并发会推高，但每个 Agent 都变慢了） | **per-user 吞吐**：N 个并发会话下单会话吞吐不低于阈值 | TensorRT-LLM `output_throughput_per_user_tok_s` |
| 缓存命中率 | 请求加权/客户端近似估算 | **token 加权 + 分层拆解**（GPU/host/storage 命中对 TTFT 影响差一个数量级） | SGLang `cache_hit_rate_pct` + `device_/host_/storage_cached_tokens` |
| 隔离性/公平性 | 完全不测 | P99/P50 时延比、被抢占/重算次数、会话间吞吐方差（三者具体含义及为何要一起看，见 [`agentic_test_design_proposal.md`](agentic_test_design_proposal.md) 3.7.1 节） | **全场空白**（vLLM 有 `num_preemptions` 指标但未进压测报告） |
| 时延 | 全局均值（被早期轮次拉平） | **按轮次拆解** per-turn TTFT/TPOT/ITL/E2EL | vLLM Rust 版 `per_turn_metrics` |

4. **多轮会话下的内存增长/泄漏验证**：这一层此前被判断为全行业空白，但重新检索后发现两个系统各自有半个正确实现，可以合并借鉴——**vLLM** `tests/models/multimodal/generation/test_memory_leak.py`：预热 2 轮 + 正式测量 16 轮，每轮重跑同一批请求，用 `gc.collect()` 后采样 GPU 显存和 CPU 峰值 RSS，断言预热后增长**零容忍**，已确认作为独立 CI job 运行；**局限是场景为多模态图像对话，不是多轮 agentic 会话**。**TensorRT-LLM** `tests/integration/defs/stress_test/stress_test.py`：`stress-test-with-accuracy` 模式在持续并发压力（`concurrency_list=[8,16,32,64,128,256]`，`stress_time` 180-300 秒可配置更长）前后各跑一次 GSM8K 精度评测比较是否稳定，每个模型配置显式声明 `memory_requirement`，已确认接入 L0 CI 及独立 QA 清单；**局限是压力场景为通用持续并发吞吐，不针对多轮会话累积，也未采样内存增长曲线本身**。建议把 vLLM 的"预热+多轮零容忍内存增长断言"方法论，和第 1 点的"真实驱动多轮+高前缀共享"负载形状结合，同时借鉴 TRT-LLM 的"压力前后精度对比"设计——这正是当前的一处具体空白，而不是无从下手。

**为什么重要**：这是 OpenVINO GenAI 目前和 llama.cpp、Ollama 并列六者中最薄弱的一环——单机推理引擎的定位不代表可以完全不管"多个 agentic 会话同时打进来"这个场景，尤其 NPU/GPU 上部署多 Agent 应用是明确的产品方向。

**可参考的完整负载建模实现**：vLLM `benchmarks/multi_turn/benchmark_serving_multi_turn.py`（`--num-clients` × `--max-active-conversations` + 会话亲和，五家里对多并发会话建模最完整，但只产出时延/吞吐，不产出隔离性判据，且未接入 CI——需要照抄负载生成部分，自己补隔离性判据）；SGLang `gsp_num_groups` × `gsp_prompts_per_group`（多会话共享 system prompt 组）用来验证共享前缀复用效率。

**CI 归属**：小规模（8 并发会话）隔离性测试放 **Nightly**；大规模（32+ 并发×长会话）放 **Weekly**；子 Agent 共享前缀复用效率验证放 **Nightly**。

### D 类：对抗性输入下的鲁棒性（对应特点 6）——当前公司和全行业都是零

**要测什么**：工具 schema 本身可能来自不完全可信的第三方（MCP server），可能是恶意/畸形的（巨大枚举、灾难性回溯正则、深层嵌套/自引用 JSON Schema）。测试要验证的不是"任务做没做对"，而是"面对这类输入，系统会不会被拖垮"——真实引擎（不能 mock）+ 真实并发（不能只测单条请求）+ 验证一个恶意请求不会拖累同批次正常请求的延迟（隔离性）。

**为什么重要**：这是一个和"精度"完全独立的可用性维度——一个系统可以精度很高、正常负载吞吐很好，同时被恶意输入轻易拖垮。agentic/MCP 场景下工具 schema 的可信边界比传统单轮 QA 更模糊，这个风险是 agentic workload 特有的。

**现状**：六个系统里只有 vLLM 沾边，且是两个从未拼在一起的碎片——`test_regex_compilation_timeout.py`（对应真实安全公告 GHSA-rwxx-mrjm-wc2m 的 ReDoS 防护单测，但用 mock 顶替了真实 xgrammar 编译器）、`benchmark_serving_structured_output.py`（真实并发压测，但代码里显式过滤掉了复杂/极端 schema）。**OpenVINO GenAI 目前和其余四家一样是零命中**，但这也意味着建这类测试没有"追赶谁"的压力，是全行业公认的空白，先做出来就是唯一。

**建设方式**：把 vLLM 的两个碎片拼起来并补齐（去 mock、换真实引擎；不过滤、专测极端 case；新增隔离性验证），`agentic_test_design_proposal.md` 3.6 节给出了具体的对抗 schema 集合样例（灾难性回溯正则、5 万项枚举、深度 500 的嵌套 JSON、自引用 schema）和测试代码框架，可直接参考。

**CI 归属**：单条 schema 的超时/崩溃防护足够便宜，放 **PR 门禁**；完整对抗 schema 集合 × 真实并发 × 隔离性验证，成本接近一次小型压测，放 **Nightly**。

### E 类：基础设施型增强（对应特点 1，是 A-D 的地基）

- **把 `test_kv_cache_eviction` 的联合断言方法论显式文档化为公司内部测试设计规范**——不需要新代码，是把已经验证有效的模式制度化，供 A/B/C 类新测试直接套用，避免每个团队各自摸索判分逻辑。
- **`llm_bench` 增加并发/数据集后端/多轮参数**——这是 C 类性能测试的前提设施，建议作为独立的工具增强项，不与具体测试用例的开发绑定排期。
- **WWB 新增 agentic/工具调用类型的 evaluator**——在已有 `@register_evaluator` 框架内扩展，用于对接后续可能引入的 BFCL/τ-bench 风格评测。
- **`@pytest.mark.agent` + `test_react_sample` 的推理管线复用**——改造现有测试的判分逻辑（从"跨语言一致性"改为"任务终态正确性"），比新建一套推理管线成本低得多，建议作为 A 类测试的第一个落地点。

**CI 归属**：文档规范类工作无 CI 归属；工具增强（`llm_bench`、WWB evaluator）是一次性开发投入，完成后其产出的具体测试用例按 A-D 类归属分层。

---

## 四、CI / Nightly / Weekly 汇总分层表

| 测试类别 | PR 门禁（必跑） | Nightly | Weekly / Release 前 |
|---|---|---|---|
| A. 工具调用编排正确性 | 精简任务集（3-5 任务）+ 1 个小模型 | 完整任务集（含长历史召回） | pass^k 稳定性验证（k≥4）+ 多模型规模 |
| B. 优化配置×精度联合测试 | baseline + 1 个全优化档 | 完整配置矩阵（4 档） | — |
| C. Agentic 性能压测 — 隔离性（小规模，8并发） | — | ✓ | — |
| C. Agentic 性能压测 — 隔离性（大规模，32+并发） | — | — | ✓ |
| C. 子 Agent 共享前缀复用效率 | — | ✓ | — |
| C. 多 Agent 协作正确性 | — | 精简任务集 | 完整任务集 + pass^k |
| D. 对抗性鲁棒性 — 单条 schema 超时/崩溃防护 | ✓ | — | — |
| D. 对抗性鲁棒性 — 完整对抗集×并发×隔离性 | — | ✓ | — |
| E. 基础设施增强 | 一次性开发投入，非常规 CI 归属 | | |

**分层设计的核心原则**（源自六次调查里反复印证的"规律 4"）：最有价值、最贴近真实场景的测试，恰恰最容易因为"太慢/太贵"被移出日常必跑路径——llama.cpp 的 `test_calc_result` 标记 `slow`、Ollama 21-模型矩阵需要 `-tags=integration`、TensorRT-LLM 的 SWE-bench Coder 活在 examples 里从未接入 CI，都是同一个模式的重复出现。**PR 门禁层必须设计得足够便宜，才能保证新测试不会重蹈覆辙被静默移出日常路径。** 建议在设计每一类新测试时，第一步就明确"哪个精简子集必须能塞进 PR 门禁"，而不是先做完整版本再事后拆分层级。

---

## 五、Roadmap 建议

1. **第一阶段（低成本，建议优先）**：E 类的文档化工作（把 `test_kv_cache_eviction` 方法论写成内部规范）+ 改造 `test_react_sample` 的判分逻辑（复用现有推理管线，只换断言）。这两项几乎不需要新增测试框架，是验证"投入产出比"的最快落地点。
2. **第二阶段**：A 类工具调用编排测试的 PR 门禁精简版 + B 类的 baseline/全优化档对比——直接复用 A 类的任务集，边际成本低。
3. **第三阶段**：D 类对抗性鲁棒性的 PR 门禁子集（单条 schema 超时防护）——这是全行业空白，风险收益比高，且实现成本相对可控（可以先从最小可行的 1-2 个对抗 case 开始）。
4. **第四阶段（investment 较重）**：`llm_bench` 的并发/数据集/多轮能力增强，这是 C 类全部测试的前提设施，工作量接近重写压测工具的核心逻辑，建议单独立项评估。
5. **第五阶段**：C 类 Nightly/Weekly 测试、D 类完整版本、多 Agent 协作正确性（目前全行业空白，既没有权威基准可借鉴，也没有其他系统的参考实现，需要从任务集设计开始自建，是投入最重的一块，建议放在其余基础打好之后再启动）。

各阶段之间存在依赖关系：A 类任务集是 B 类的输入；C 类依赖 `llm_bench` 增强；D 类可独立并行推进，不阻塞其他阶段。

---

## 六、局限说明

- 本报告的现状评估仅基于 `~/openvino.genai` 仓库的源码级调查，**不包含 OVMS**；若 OVMS 已有独立的 serving 层压测/精度回归资产，应单独评估后再决定是否需要为 OpenVINO GenAI 补齐同等能力，避免重复建设。
- A-D 类测试里"工具真实/仿真执行"这部分，涉及文件系统、网络请求类工具时需要考虑沙箱隔离与结果可重复性，建议先从纯计算型工具（无副作用、结果确定）开始覆盖，再逐步扩展到有状态工具。
- "终态判定逻辑"本身需要为每个任务单独设计，不像文本相似度那样可以复用一套通用打分器——这是本报告建议的方案比现有"相似度评分"类方法更扎实、但也更需要前期设计投入的地方。
- 本报告聚焦"Runtime 层能验证的工具调用正确性"，不覆盖"整个 agent 任务是否最终完成"这类需要完整 agent 框架/沙箱的评测（如 SWE-bench）——那是更上层的验证责任，超出 Runtime 自测的合理范围，但两者应该衔接：Runtime 层测试通过，是上层 agent 评测有意义的前提。
- 多 Agent 协作正确性（第四节 C 类最后一项）目前是全行业"两个坐标轴同时挂零"的区域——横轴上没有权威基准可借鉴，纵轴上没有任何系统验证过。这意味着这部分建设成本和不确定性都高于其它几类，建议作为独立的中长期方向单独评估投入产出比，不与本报告其余建议的时间线绑定。
