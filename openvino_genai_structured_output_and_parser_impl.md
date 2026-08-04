# OpenVINO GenAI 源码拆解：约束解码（Structured Output）与 Tool-Call Parser 是怎么实现的

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · **约束解码与 Parser 源码拆解**
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

> 
> 承接：[tool_calling_mcp_primer.md](tool_calling_mcp_primer.md) 第 4 节提到的两块 Runtime 内部机制——"约束解码（保证语法合法）"和"Tool-call Parser（原始输出→结构化）"。本文档打开 `~/openvino.genai` 源码，逐一说明这两套子系统各自怎么实现、代码在哪、怎么被顶层管线调用，并在最后指出一个容易被忽略的架构事实：**这两套子系统目前是相互独立、没有自动关联的**。
> 调查方法：与本系列其余文档一致——不满足于"存在这个功能"，要打开代码确认具体机制。

## 一、两套子系统各自对应文件总览

```
约束解码（第一部分）                              Tool-Call Parser（第二部分）
src/cpp/src/sampling/structured_output/           src/cpp/include/openvino/genai/parsers.hpp
  ├── structured_output_controller.{hpp,cpp}      src/cpp/src/parsers.cpp
  └── xgrammar_backend.{hpp,cpp}                  src/cpp/include/openvino/genai/text_streamer.hpp
src/cpp/include/openvino/genai/                     src/cpp/src/text_streamer.cpp
  └── generation_config.hpp (StructuredOutputConfig) src/python/py_parsers.cpp（VLLMParserWrapper 绑定）
src/cpp/src/sampling/logit_processor.hpp           src/cpp/src/llm/pipeline.cpp（run_generate_with_parsers 编排点）
src/cpp/src/sampling/sampler.cpp
```

## 二、约束解码：怎么"保证语法合法"

### 2.1 架构总览（源码里自带的注释图，`structured_output_controller.hpp:20-35`）

```
+---------------------------+    uses   +--------------------------+   implements  +-----------------------+
| XGrammarLogitsTransformer |---------->| XGrammarStructuredOutput |-------------->| IStructuredOutputImpl |
+---------------------------+           +--------------------------+               +-----------------------+
                                                                                            ↑
                                                                                            | holds/used by
                                                                                            |
                                                                                +----------------------------+
                                                                                | StructuredOutputController |
                                                                                +----------------------------+
```

这是一个**后端可插拔**的设计：`StructuredOutputController` 只认 `IStructuredOutputImpl` 这个接口，具体实现（目前唯一注册的是 `xgrammar`）通过静态注册表接入——[`structured_output_controller.cpp:9-30`](../openvino.genai/src/cpp/src/sampling/structured_output/structured_output_controller.cpp) 里的 `register_backend`/`get_backend_registry` 就是这套注册机制，`xgrammar_backend.cpp:93-103` 在文件加载时用一个静态变量触发 `registerXGrammarBackend()` 把自己注册进去。这意味着未来接入 outlines/llguidance 之类的其它后端，理论上不需要改 `StructuredOutputController` 本身。

### 2.2 从用户配置到 logits 被屏蔽：完整调用链

```
用户代码（Python/C++）
  config.structured_output_config = StructuredOutputConfig(json_schema=...)
        │  ← generation_config.hpp:94 StructuredOutputConfig 类定义
        ▼
GenerationConfig.is_structured_output_generation() == true
        │  ← generation_config.hpp:723
        ▼
LogitProcessor 构造函数（每个请求一个实例）
  if (sampling_params.is_structured_output_generation() && structured_output_controller != nullptr) {
      auto transformer = structured_output_controller->get_logits_transformer(sampling_params);
      m_logit_transformers.push_back(transformer);
  }
        │  ← logit_processor.hpp:52-57
        ▼
StructuredOutputController::get_logits_transformer()
  1. 按 backend 名字(默认 "xgrammar")取出/创建对应的 IStructuredOutputImpl 实例
  2. 调用 backend->get_logits_transformer(sampling_parameters)
        │  ← structured_output_controller.cpp:64-74
        ▼
XGrammarStructuredOutput::get_logits_transformer()
  1. create_grammar(): 按 config 里到底填的是 json_schema / regex / grammar(EBNF) / structural_tags_config
     调用 xgrammar::Grammar::FromJSONSchema / FromRegex / FromEBNF / FromStructuralTag 之一
  2. m_grammar_compiler->CompileGrammar(grammar) —— 这一步就是"编译 schema"，耗时被计入 grammar_compile_time
  3. 返回一个新建的 XGrammarLogitsTransformer，持有编译好的 CompiledGrammar
        │  ← xgrammar_backend.cpp:121-131
        ▼
每个 decoding step：Sampler 调用 LogitProcessor 上挂的每个 transformer 的 apply(logits)
XGrammarLogitsTransformer::apply()
  1. m_grammar_matcher.FillNextTokenBitmask(bitmask)  —— 问 xgrammar："当前语法状态下，哪些 token 合法？"
  2. xgrammar::ApplyTokenBitmaskInplaceCPU(logits, bitmask) —— 把不合法 token 的 logit 置为 -inf
        │  ← xgrammar_backend.cpp:187-209
        ▼
XGrammarLogitsTransformer::accept_tokens()
  真正采样出 token 之后，回调这个方法让 m_grammar_matcher 前进到下一个语法状态
        │  ← xgrammar_backend.cpp:178-185
```

**关键机制**：xgrammar 把 JSON Schema/regex/EBNF 都统一编译成一个状态机（`CompiledGrammar`），每一步解码前先问它"当前状态下哪些 token 会破坏语法"，用一个**位掩码（bitmask）**把词表里不合法的 token 全部屏蔽掉再采样——这正是上一份文档里说的"逐 token 掩码计算开销"的具体实现：`FillNextTokenBitmask` 和 `ApplyTokenBitmaskInplaceCPU` 这两步，每个 decoding step 都要跑一次（[`xgrammar_backend.cpp:187-209`](../openvino.genai/src/cpp/src/sampling/structured_output/xgrammar_backend.cpp)）。

### 2.3 支持的 grammar 类型（`generation_config.hpp` 内 `StructuredOutputConfig`）

`create_grammar()`（[`xgrammar_backend.cpp:66-115`](../openvino.genai/src/cpp/src/sampling/structured_output/xgrammar_backend.cpp)）按优先级依次检查配置里填了哪种约束：

| 字段 | 对应 xgrammar 调用 | 典型用途 |
|---|---|---|
| `json_schema` | `xgrammar::Grammar::FromJSONSchema` | 约束输出为符合某个 JSON Schema 的对象——**工具调用参数校验、结构化抽取的最常用形式** |
| `regex` | `xgrammar::Grammar::FromRegex` | 约束输出匹配某个正则 |
| `grammar`（EBNF） | `xgrammar::Grammar::FromEBNF` | 自定义上下文无关语法，比 JSON Schema 更灵活 |
| `structural_tags_config` / `compound_grammar` | `xgrammar::Grammar::FromStructuralTag` | 混合结构——比如"一段自由文本 + 中间夹一段必须合法的 JSON 片段"，`Tag`/`TriggeredTags`/`Union`/`Concat` 这些组合子都是为了描述这种混合场景（`generation_config.hpp:278-304`） |

`compound_grammar` 字段已被标记为 deprecated（[`xgrammar_backend.cpp:108-111`](../openvino.genai/src/cpp/src/sampling/structured_output/xgrammar_backend.cpp) 会打印 `GENAI_WARN`），推荐迁移到 `structural_tags_config`。

### 2.4 性能计量：编译耗时是被显式追踪的

`StructuredOutputController` 内部维护两组计时（[`structured_output_controller.hpp:87-88`](../openvino.genai/src/cpp/src/sampling/structured_output/structured_output_controller.hpp)）：

- `m_init_grammar_compiler_times`：**创建 `GrammarCompiler` 实例**（构造 `xgrammar::TokenizerInfo` 并建立词表相关的编译上下文）这一步的耗时——只在每个 backend 第一次被用到时发生一次（`get_backend()` 里用 `std::chrono` 打点，见 [`structured_output_controller.cpp:47-53`](../openvino.genai/src/cpp/src/sampling/structured_output/structured_output_controller.cpp)）
- `m_grammar_compile_times`：**每次调用 `get_logits_transformer()` 编译一个具体 grammar**（比如某一次请求传入的具体 json_schema）的耗时——每次请求都会记一条（[`structured_output_controller.cpp:64-74`](../openvino.genai/src/cpp/src/sampling/structured_output/structured_output_controller.cpp)）

这两组数据通过 `Sampler::get_structured_output_times()`（[`sampler.cpp:243-246`](../openvino.genai/src/cpp/src/sampling/sampler.cpp)）向上传递，最终对应到 `PerfMetrics.grammar_compiler_init_times`（`map<backend_name, float>`）和 `PerfMetrics::get_grammar_compile_time()`（对多次编译耗时做统计）这两个此前在 OpenVINO GenAI 调查报告里提到过的字段——**现在可以确认这两个字段背后具体测的就是上一节"schema 编译开销"和"逐请求语法编译开销"这两件事**，不是笼统的"结构化输出耗时"。

### 2.5 一个重要的架构事实：约束解码和 Tool Calling 目前没有代码层面的自动关联

搜索 `parsers.cpp`/`parsers.hpp` 全文，**没有任何一处引用 `StructuredOutputConfig` 或 `structured_output`**；反过来搜索 `generation_config.hpp`，也**没有 `tools` 字段**——也就是说，OpenVINO GenAI 目前没有"传入 OpenAI 风格的 `tools` 列表，自动把工具 schema 转换成 `StructuredOutputConfig` 并强制约束解码"这样的自动化管线。官方样例（[`samples/python/text_generation/structured_output_generation.py`](../openvino.genai/samples/python/text_generation/structured_output_generation.py)）演示的是通用场景——用 Pydantic model 的 `model_json_schema()` 直接构造 `StructuredOutputConfig`，和"工具调用"这个具体应用场景没有绑定。

这意味着：**如果开发者想要"保证工具调用参数一定是合法 JSON"，需要自己手动把工具的参数 schema 转成 `json_schema` 传给 `StructuredOutputConfig`，这是一个需要应用层自己拼装的组合，不是内置的开箱功能**。对照 `tool_calling_mcp_primer.md` 第 8 节的边界表，这一点提醒我们：即便同属 Runtime scope，"约束解码"和"tool-call parser"目前在 OpenVINO GenAI 里也是两个**互相不知道对方存在**的独立子系统，要不要把它们接起来，责任在应用层。

## 三、Tool-Call Parser：怎么把"原始输出"变成"结构化 tool_calls"

### 3.1 两条并存的路径：批量 Parser vs 流式 IncrementalParser

`parsers.hpp` 定义了两套完全独立的类体系（[`parsers.hpp:17-31`](../openvino.genai/src/cpp/include/openvino/genai/parsers.hpp) 与 [`parsers.hpp:163-206`](../openvino.genai/src/cpp/include/openvino/genai/parsers.hpp)）：

```
Parser（批量，生成结束后对完整文本跑一次）              IncrementalParser（流式，每个 chunk 到达时跑一次）
├── ReasoningParser                                    ├── ReasoningIncrementalParser
│     ├── DeepSeekR1ReasoningParser                    │     ├── DeepSeekR1ReasoningIncrementalParser
│     └── Phi4ReasoningParser                          │     └── Phi4ReasoningIncrementalParser
├── Llama3PythonicToolParser                           （目前没有 IncrementalParser 版本的 tool parser——
├── Llama3JsonToolParser                                工具调用参数通常需要完整 JSON 才有意义，
└── VLLMParserWrapper                                   增量提取的价值不如"推理内容实时展示"）
```

**两条路径解决不同的展示需求**：推理内容（`<think>...</think>`）适合流式展示给用户（用户想实时看到模型在"想什么"），所以有专门的 `IncrementalParser` 版本；工具调用参数在拿到完整 JSON 之前基本没法用（不能拿半个 JSON 去调用一个函数），所以目前只有批量版本。

### 3.2 批量 Tool Parser 的具体实现：正则提取，不是语义理解

`Llama3PythonicToolParser`（对应 Llama3 的 Pythonic 风格输出，如 `[get_weather(location='NY')]`）和 `Llama3JsonToolParser`（对应 JSON 风格输出）都是**纯正则/字符串扫描**，没有调用任何模型或语义分析：

- `Llama3PythonicToolParserImpl::parse`（[`parsers.cpp:219-253`](../openvino.genai/src/cpp/src/parsers.cpp)）：先用 `std::regex R"(\[(.*?)\])"` 抓出 `[...]` 括起来的部分，再手动切出函数名（`(` 之前）和参数（`(...)` 内部），最后用另一个正则 `R"((\w+)\s*=\s*\"([^"]*)\")"` 逐个抓 `key="value"` 键值对，拼成 `JsonContainer` 写入 `message["tool_calls"]`。
- `Llama3JsonToolParserImpl::parse`（[`parsers.cpp:265-280`](../openvino.genai/src/cpp/src/parsers.cpp)）：直接找 `content` 里**第一个 `{` 到最后一个 `}`**之间的子串，当作一段 JSON 用 `JsonContainer::from_json_string` 解析，塞进 `tool_calls`。

这一实现方式解释了它的**能力边界**：只要模型输出严格符合预期的 Pythonic/JSON 格式，就能正确抽取；但如果模型输出里夹杂了多个 `{...}` 块、或者格式稍有走样（比如用单引号而不是双引号），这种"找第一个 `{` 和最后一个 `}`"的简单策略容易抽取错误——这也是为什么第二节讨论的约束解码在实践中有价值：如果用约束解码保证模型输出**严格**符合 schema，这里的正则提取就会更可靠。

### 3.3 VLLMParserWrapper：不是重新实现，是真的在跑 vLLM 的 Python 代码

`VLLMParserWrapper` 定义在 pybind11 绑定文件里（[`py_parsers.cpp:85-133`](../openvino.genai/src/python/py_parsers.cpp)），不在核心 C++ 库 `parsers.cpp` 里——因为它的实现本质是一层**桥接**：

```cpp
VLLMParserWrapper(py::object py_parser) {
    if (py::hasattr(py_parser, "extract_tool_calls")) {
        m_parsers.push_back([py_parser](const std::string& content) -> JsonContainer {
            py::object parsed = py_parser.attr("extract_tool_calls")(content, py::none());
            // parsed.model_dump_json() → JsonContainer::from_json_string(...)
        });
    }
    if (py::hasattr(py_parser, "extract_reasoning")) { ... }
}
```

它接收一个**真实的 Python 对象**（`py_parser`），通过 pybind11 反射检查这个对象是否有 `extract_tool_calls`/`extract_reasoning` 方法，然后**直接调用**这两个方法。对应的测试文件（[`tests/python_tests/test_vllm_parsers_wrapper.py:47,52`](../openvino.genai/tests/python_tests/test_vllm_parsers_wrapper.py)）证实了 `py_parser` 传进来的就是：

```python
from vllm.entrypoints.openai.tool_parsers.llama_tool_parser import Llama3JsonToolParser
parser = Llama3JsonToolParser(AutoTokenizer.from_pretrained(model_cached))
```

——**这是 vLLM 这个 pip 包里真实的类**，OpenVINO GenAI 没有拷贝或重写这段逻辑，而是让用户 `pip install vllm` 之后把 vLLM 的解析器实例直接传进 `VLLMParserWrapper`，运行时通过 Python 互操作层调用它。这是此前调查报告里"六个系统里唯一一处直接对接社区已有解析器生态"这一结论的具体代码依据。

### 3.4 流式路径：`TextParserStreamer` 怎么处理"半个 tag 被切在 chunk 边界"这种情况

流式场景下，`delta_text` 是一个个 token 解码出来的文本片段，**标签可能被切断**——比如 `<think>` 这五个字符可能被分在两个 chunk 里（第一个 chunk 结尾是 `<th`，第二个 chunk开头是 `ink>`）。`ReasoningIncrementalParser` 专门处理了这个问题：

```cpp
// parsers.cpp:40-50
size_t find_close_tag_prefix_length(std::string_view text) const {
    // 找 text 末尾有没有和 close_tag 开头重叠的部分
    // 例如 text 以 "</th" 结尾，m_close_tag 是 "</think>"，返回 4
}
```

`handle_inside_reasoning()`（[`parsers.cpp:114-134`](../openvino.genai/src/cpp/src/parsers.cpp)）用这个函数判断"当前 chunk 末尾是不是可能是下一个标签的开头"，如果是，就把这部分**暂存在 `m_text_cache` 里，先不吐出去**，等下一个 chunk 到达再拼起来重新判断——这正是上一份文档提到的"喂手工构造的增量片段测状态机"里要测的核心逻辑：`test_parsers.py::test_stop_invoked_by_tool_call` 里手工构造 `delta_text == "{"` 这类切片，测的就是"标签/JSON 被硬切在 chunk 边界时，状态机有没有正确处理"。

调用链上，`TextParserStreamer::write(std::string delta_text)`（[`text_streamer.cpp:154-210`](../openvino.genai/src/cpp/src/text_streamer.cpp)）是驱动这一切的入口——每次 `TextStreamer` 解码出新文本，就依次把 `delta_text` 喂给 `m_pimpl->m_parsers` 列表里的每个 `IncrementalParser`，每个 parser 可以修改 `delta_text`（比如把 `<think>` 标签内的内容过滤掉再往下传）、往 `delta_message` 里写字段（比如 `reasoning_content`），并可以通过 `set_status()` 提前终止流式输出（`StreamingStatus`）。

### 3.5 顶层编排：`run_generate_with_parsers` 把两条路径粘合在一起

真正决定"这次 `generate()` 调用要不要跑 parser、跑哪些"的编排逻辑，在 [`llm/pipeline.cpp:59-106`](../openvino.genai/src/cpp/src/llm/pipeline.cpp) 的 `run_generate_with_parsers()`：

```
1. 如果传入的 streamer 是 TextParserStreamer 实例 → 生成前先 reset()，生成后取出 get_parsed_message()
   （这条路径覆盖的是"流式 IncrementalParser"，逐 chunk 已经在生成过程中处理完了）

2. 无论有没有走流式路径，只要 generation_config.parsers 非空 → 
   对每条结果文本，依次跑一遍所有批量 Parser（Llama3JsonToolParser / VLLMParserWrapper / ...)
   （这条路径覆盖"批量 Parser"，在生成完全结束、拿到完整文本之后才跑）
```

**这里有一个容易被忽视的设计细节**：两条路径不是互斥的，可以同时生效——比如同时配置一个 `ReasoningIncrementalParser`（流式展示推理过程）和一个 `Llama3JsonToolParser`（生成结束后批量抽取工具调用）。批量 `Parser` 是在流式路径处理完之后**追加执行**的（源码注释原话："Apply Base parsers sequentially even if IncrementalParser has run"），而且是对 `res.parsed[i]["content"]` 继续处理，不是重新处理原始模型输出。

## 四、一张图总结这两套子系统的关系

```
                     用户配置
                        │
          ┌─────────────┴─────────────┐
          ▼                            ▼
  config.structured_output_config   config.parsers = [...]（批量）
  （json_schema/regex/grammar/...）  streamer = TextParserStreamer([...])（流式）
          │                            │
          ▼                            │
┌─────────────────────┐               │        ★ 二者目前互不知道对方存在 ★
│ StructuredOutputCtrl  │               │        没有代码路径把 tools 列表自动
│  → xgrammar 编译      │               │        转成 json_schema 喂给左边，
│  → 逐 token 掩码解码   │               │        也没有 parser 主动查询左边
└──────────┬───────────┘               │        "这次生成是不是被约束过"
           │                            │
           ▼                            ▼
    保证输出语法合法              run_generate_with_parsers()
   （token 一定构成合法 JSON）    （生成结束后/流式过程中，
                                 把原始文本解析成结构化字段）
```

**结论**：OpenVINO GenAI 已经把"保证语法合法"（约束解码）和"原始输出解析成结构化"（tool-call parser）这两块能力都做得相当完整——约束解码支持可插拔后端、四种 grammar 来源、显式的编译耗时追踪；tool-call parser 覆盖批量/流式两条路径，且流式路径专门处理了 chunk 边界切分问题，还独创了直接复用 vLLM 生态解析器的 `VLLMParserWrapper`。但**这两块目前是并列存在、没有被自动串联起来的两个零件**——如果要保证"工具调用输出既语法合法、又能被正确解析"，需要应用层开发者自己手动把工具 schema 接进 `StructuredOutputConfig`，这是一处值得关注的可改进点（也是 `agentic_test_design_proposal.md` G3 提到的"约束解码在高并发/复杂 schema 下的性能陷阱"，如果引入这条自动化管线之后，需要额外考虑的一个新维度）。

## 五、文件与关键符号速查表

| 关注点 | 文件 | 关键符号 |
|---|---|---|
| 约束解码后端注册机制 | `sampling/structured_output/structured_output_controller.{hpp,cpp}` | `StructuredOutputController::register_backend/get_backend` |
| xgrammar 具体实现 | `sampling/structured_output/xgrammar_backend.{hpp,cpp}` | `XGrammarStructuredOutput::create_grammar/get_logits_transformer`、`XGrammarLogitsTransformer::apply/accept_tokens` |
| 约束配置的用户接口 | `include/openvino/genai/generation_config.hpp` | `StructuredOutputConfig`（`json_schema`/`regex`/`grammar`/`structural_tags_config`） |
| 约束解码接入采样循环 | `sampling/logit_processor.hpp` | `LogitProcessor` 构造函数第 52-57 行 |
| 编译耗时统计 | `structured_output_controller.{hpp,cpp}`、`sampler.cpp` | `m_init_grammar_compiler_times`、`m_grammar_compile_times`、`Sampler::get_structured_output_times` |
| 批量 tool-call parser 基类与实现 | `include/openvino/genai/parsers.hpp`、`src/parsers.cpp` | `Parser`、`Llama3JsonToolParser`、`Llama3PythonicToolParser` |
| 复用 vLLM 生态解析器 | `src/python/py_parsers.cpp` | `VLLMParserWrapper` |
| 流式 incremental parser 基类与实现 | `parsers.hpp`、`parsers.cpp` | `IncrementalParser`、`ReasoningIncrementalParser`、`find_close_tag_prefix_length` |
| 流式驱动入口 | `include/openvino/genai/text_streamer.hpp`、`src/text_streamer.cpp` | `TextParserStreamer::write`、`TextParserStreamerImpl` |
| 顶层编排（两条路径粘合点） | `src/llm/pipeline.cpp` | `run_generate_with_parsers` |
| 官方使用样例 | `samples/python/text_generation/structured_output_generation.py` | — |
| 测试证据（正确性） | `tests/python_tests/test_parsers.py`、`tests/python_tests/test_vllm_parsers_wrapper.py` | 手工构造 delta 片段测状态机 |
