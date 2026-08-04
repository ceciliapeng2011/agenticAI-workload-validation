# Tool Calling / Function Calling / MCP 关系全景：给刚接触 Agentic 系统的工程师

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · [Ollama](ollama_agentic_evaluation_investigate.md) · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · **Tool Calling/MCP 概念全景** · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

> 
> 背景：这份文档回答一个新手最容易卡住的问题——"工具 schema、function call、tool call、OpenAI Chat Completions API、MCP server，这些词到底是什么关系？"
> 定位：这是一份**概念地图**，不是调查报告——本系列其余文档（`*_agentic_evaluation_investigate.md`、`agentic_test_design_proposal.md`）里出现的具体代码（`VLLMParserWrapper`、`real_tool_impl`、`--mcp-servers-json` 等）都会在这里被安放到对应的位置上，方便对照理解。

## 0. 先说结论：这些术语分属两条不同的协议，解决两个不同的问题

```
问题 A："模型怎么用人类能理解、代码能解析的方式，表达'我要调用一个工具'这个意图？"
       → 解决方案：Function Calling / Tool Calling（一套 模型 ↔ 推理引擎 之间的协议）

问题 B："成千上万个外部工具（发邮件、查数据库、控制浏览器……），怎么用统一的方式接入任意一个 Agent，
        而不用每接一个工具就写一段定制对接代码？"
       → 解决方案：MCP，Model Context Protocol（一套 Agent ↔ 工具提供方 之间的协议）
```

**这两套协议是叠加关系，不是替代关系**：MCP 负责"发现工具、描述工具、执行工具"，最终还是要把工具的 schema 塞进 Function Calling 协议的 `tools` 字段里，才能让模型看到、决定要不要调用。后面每一节都会回到这张图上。

## 1. 起点：模型本身只会吐文本，其它一切都是"外挂"出来的

大语言模型的原始能力只有一件事：给定一段 token 序列，预测下一个 token。它不会"调用函数"，也不知道什么是"工具"——它能做的，最多是**在文本里生成一段看起来像调用意图的字符串**，比如：

```
用户: 帮我查一下上海今天的气温
模型原始输出: 好的，我需要查询天气。<tool_call>{"name": "get_weather", "arguments": {"location": "上海"}}</tool_call>
```

这段 `<tool_call>...</tool_call>` 只是**模型吐出来的普通文本**，本质上和它说"你好"没有区别——真正让这段文本变成一次"工具调用"，需要外面一整套机制去约定格式、解析内容、执行动作、把结果喂回去。这套机制分成几层，下面按顺序拆开讲。

## 2. 第一层：Tool Schema——怎么让模型"知道"有什么工具能用

模型不会凭空知道存在一个叫 `get_weather` 的工具，必须在请求里明确告诉它。这份"告知"就是 **Tool Schema**——一段声明式的接口描述（通常是 JSON Schema），只写"这个工具叫什么、干什么、需要什么参数"，不包含任何执行逻辑：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询指定城市的实时气温",
    "parameters": {
      "type": "object",
      "properties": { "location": {"type": "string", "description": "城市名"} },
      "required": ["location"]
    }
  }
}
```

对应到上一次讨论的代码：`agentic_test_design_proposal.md` 里 `"tools": [get_weather]` 这一行，`get_weather` 就是这样一份 schema 对象——**模型只能看到这个，看不到、也不需要知道它背后到底怎么实现**。

## 3. 第二层：OpenAI Chat Completions 风格 API——一套事实标准的"传话协议"

有了 schema，还需要一个**大家都认的请求/响应格式**，规定"schema 放在请求体的哪个字段""模型的调用意图在响应体的哪个字段"。OpenAI 最早定义了这套格式（`tools` 请求字段 + 响应里的 `tool_calls` 字段），因为影响力足够大，六个被调查的推理引擎（vLLM、SGLang、TensorRT-LLM、llama.cpp、Ollama、OpenVINO GenAI）全部选择了兼容这套格式，而不是各自发明一套——这样任何写好的客户端代码，理论上换个 base_url 就能指向任意一个后端。

```
请求（客户端 → Runtime）：
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "帮我查一下上海今天的气温"}],
  "tools": [ <上一节那份 get_weather schema> ]
}

响应（Runtime → 客户端）：
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc123",
        "function": { "name": "get_weather", "arguments": "{\"location\": \"上海\"}" }
      }]
    }
  }]
}
```

**这份协议只规定"长什么样"，不规定"谁负责什么"**——这正是新手最容易搞混的地方，第 8 节会专门画一张边界图说清楚。

## 4. 第三层：从"模型原始输出"到"响应里那个干净的 tool_calls 字段"，中间发生了什么

第 1 节看到的模型原始输出是一段夹杂了普通文本和某种自定义标记的字符串（`<tool_call>{"name":...}</tool_call>`，或者 Llama3 家族用别的写法，Hermes 家族又用另一种写法——**每个模型家族的"自定义标记"长得都不一样**）。要把这段原始文本，变成第 3 节那种干净的、跨模型统一的 `tool_calls` JSON 字段，Runtime 内部要做两件事：

1. **（可选）约束解码，保证输出语法合法**：用 xgrammar/outlines/llguidance 这类约束解码后端，把 schema 编译成状态机，解码时屏蔽掉会破坏 JSON 格式的 token——这是上一轮讨论过的内容，保证的是"字符串本身合法"，不保证"内容对不对"。
2. **Tool-call parser，把原始文本解析成结构化字段**：这是一段**per-model-family** 的解析代码——因为不同模型家族的原始输出格式不一样，Llama3 用 `Llama3JsonToolParser`，Hermes 家族用专门的 detector，OpenVINO GenAI 甚至直接复用了 vLLM 生态维护的解析器（`VLLMParserWrapper`）。这一层纯粹是软件解析逻辑，和模型有没有"决策对"是两个独立的问题——上上轮讨论过，这也是为什么六个系统里都能看到专门喂手工构造的增量片段去测这层解析器状态机对不对。

```
模型原始 token 流
    │
    ▼
┌─────────────────────────┐
│ (可选) 约束解码           │  ← xgrammar/outlines/llguidance：保证语法合法
│ 屏蔽非法 token           │
└─────────────┬───────────┘
              ▼
┌─────────────────────────┐
│ Tool-call Parser         │  ← per-model-family：Llama3JsonToolParser / Hermes detector / VLLMParserWrapper
│ 原始文本 → 结构化字段     │
└─────────────┬───────────┘
              ▼
   干净的 tool_calls 字段（放进 HTTP 响应体，返回给客户端）
```

**到这里为止，全部发生在 Runtime 内部**——这是关键分界点，下一节开始离开 Runtime。

## 5. 第四层：工具到底是谁执行的？——离开 Runtime，进入 Agent 层

Runtime 返回了 `tool_calls` 字段之后，它的任务**已经结束**——它不会去真的调用天气 API，不会自己决定"接下来该不该再问模型一次"。这些事全部由**客户端代码**（也就是"Agent"或者"Orchestrator"）负责：

```
Agent（客户端）拿到 tool_calls 之后，要做的事：
  1. 解析出 function.name = "get_weather"，function.arguments = {"location": "上海"}
  2. 真正执行——调用真实的天气 API，或者像测试代码里那样跑 real_tool_impl(location="上海")
  3. 拿到执行结果，包装成一条新的消息：{"role": "tool", "content": "{\"celsius\": 32}", "tool_call_id": "call_abc123"}
  4. 把这条消息追加进对话历史，决定要不要再发一次请求给 Runtime（继续下一轮，或者任务已经结束）
```

这正是上上轮解释过的 `real_tool_impl` 和 `run_agentic_task` 那个 for 循环——它们写在**测试代码/Agent 代码**里，而不是 Runtime 代码里，就是因为"执行工具、决定循环"这件事从架构上就不归 Runtime 管。

## 6. 第五层：MCP——解决"工具从哪来"这个更大的问题

前面四节讲的是"**一个**工具怎么被模型调用"，但没讲"**这个工具的 schema 和执行逻辑，一开始是怎么进到 Agent 代码里的"。最朴素的做法是像 `get_weather`/`real_tool_impl` 那样，工程师手写一个 Python 函数——这在工具很少时没问题，但如果你想接入几十上百个第三方工具（发邮件、查 Jira、控制浏览器、搜索网页……），每接一个都手写一段对接代码，成本会失控。

**MCP（Model Context Protocol）解决的正是这个"接入"问题**：它定义了一套标准协议，让"工具提供方"（**MCP Server**）可以用统一的方式，向任何"Agent 宿主"（**MCP Client**）声明自己有哪些工具、schema 是什么、怎么调用。有了 MCP，Agent 不需要为每个工具写定制代码，只需要实现一次"MCP 客户端"逻辑，就能接入任意遵循 MCP 协议的工具服务。

```
                     ┌──────────────────────────────┐
                     │     Agent / Orchestrator       │
                     │  (Claude Desktop / Cursor /     │
                     │   AutoGen / 你自己写的脚本)      │
                     │                                 │
                     │   ┌─────────────┐               │
                     │   │  MCP Client  │──发现/调用────┐│
                     │   └─────────────┘               ││
                     └──────────────────┬──────────────┘│
                                        │                │
                       ① 拉取工具 schema，塞进 tools 数组   │ ② 模型决定调用后，
                          发给 Runtime                    │   转发给对应 MCP Server 执行
                                        ▼                ▼
                     ┌─────────────────────┐   ┌──────────────────────┐
                     │      Runtime         │   │     MCP Server(s)     │
                     │  (vLLM/SGLang/...)   │   │  真实工具实现：         │
                     │  只认 tools/tool_calls │   │  天气 API / 文件系统 /  │
                     │  这套 Function-Calling │   │  网页搜索 / Shell ……   │
                     │  协议，不知道 MCP存在   │   └──────────────────────┘
                     └─────────────────────┘
```

关键理解：**MCP Server 不会绕开 Function-Calling 协议直接和模型对话**——它提供的工具 schema，最终还是要被 Agent 的 MCP Client 转换成第 2 节那种 JSON Schema，塞进第 3 节的 `tools` 请求字段里，Runtime 侧完全不知道这个工具是"手写的"还是"从 MCP Server 拉来的"，处理方式没有任何区别。MCP 解决的是**Agent 层"工具从哪来"**的问题，不是"模型怎么调用工具"的问题——后者从始至终都是 Function Calling 协议在管。

## 7. 全链路图：一次完整的"多轮 + 工具调用"往返

把前面几节串起来，看一次完整的两轮对话：

```
┌─────────────────────────────────────── Agent / Orchestrator 层 ───────────────────────────────────────┐
│                                                                                                          │
│  ① 准备 tools 列表                                                                                       │
│     - 手写 schema（如 get_weather），或                                                                   │
│     - 通过 MCP Client 从 MCP Server 拉取                                                                  │
│                                                                                                          │
│  ┌──────────────── Turn 1 ────────────────┐        ┌──────────────── Turn 2 ────────────────┐            │
│  │ messages=[user]                        │        │ messages=[user, assistant(tool_calls),  │            │
│  │ tools=[get_weather]                    │        │           tool(result)]                 │            │
│  └──────────────────┬──────────────────────┘        └──────────────────┬──────────────────────┘            │
│                     │ HTTP POST                                        │ HTTP POST                        │
└─────────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────┘
                       ▼                                                  ▼
┌──────────────────────────────────────── Runtime 层 ─────────────────────────────────────────────────────┐
│  模型推理 → (可选)约束解码 → tool-call parser                    模型推理（这次不需要工具）→ 直接吐文本      │
│  返回: tool_calls=[{name:"get_weather", args:{location:"上海"}}]  返回: content="上海32度，超过30度，       │
│                                                                            换算成华氏度是89.6°F"            │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────── Agent / Orchestrator 层 ───────────────────────────────────────┐
│  ② 真正执行工具（可能是本地函数，也可能转发给 MCP Server）                                                 │
│     real_tool_impl(location="上海") → {"celsius": 32}                                                    │
│  ③ 包装成 tool 消息，追加进历史，发起 Turn 2 请求                                                          │
│  ④ Turn 2 拿到 content，判断任务是否结束（本例结束了，不再发 Turn 3）                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 8. Scope 边界表：哪些是 Agent 的活，哪些是 Runtime 的活

这是最容易混淆、也是最需要记住的一张表：

| 职责 | 属于谁的 Scope | 具体例子 |
|---|---|---|
| 决定这一轮给模型看哪些工具（`tools` 数组的内容） | **Agent** | 要不要把"发邮件"这个工具暴露给模型，是产品/业务逻辑决定的 |
| 把工具 schema 组装进请求体 | **Agent**（构造）+ Runtime（消费） | Agent 拼出 `tools=[...]`，Runtime 只是把它塞进 prompt 模板 |
| 真实模型推理，决定要不要调用、调用哪个、填什么参数 | **Runtime** | vLLM/SGLang/OpenVINO GenAI 内部的一次 forward pass |
| 保证输出的 tool_call 语法合法（约束解码） | **Runtime** | xgrammar / outlines / llguidance |
| 把模型原始 token 流解析成结构化 `tool_calls` 字段 | **Runtime** | `Llama3JsonToolParser`、Hermes detector、`VLLMParserWrapper` |
| 真正执行工具（调真实 API / 跑代码 / 查数据库） | **Agent**（或转发给 MCP Server 执行） | `real_tool_impl`；一个真实的天气 MCP Server |
| 发现/管理有哪些第三方工具可用，标准化对接 | **Agent**（通过 MCP Client） | Claude Desktop 里配置的 MCP servers 列表 |
| 把工具执行结果重新组装进对话历史，决定要不要发起下一轮 | **Agent** | `run_agentic_task` 里的那个 for 循环 |
| 管理多轮上下文膨胀 | **两层都有，解决的问题不同** | Agent 侧：语义级上下文压缩（Ollama `compactor.go`）；Runtime 侧：系统级 KV cache 驱逐/offload（`test_kv_cache_eviction`） |

## 9. 真实案例对照：六个系统分别站在这张图的什么位置上

- **vLLM / SGLang / TensorRT-LLM 的 tool-call parser 代码**（`tests/tool_use/`、`test/registered/.../function_call/`、`test_tool_parsers.py`）——都是第 4 节"Runtime 内部把原始输出解析成 tool_calls"这一层，纯 Runtime scope。
- **OpenVINO GenAI 的 `VLLMParserWrapper`**——同样是第 4 节这一层，只是工程实现上直接复用了 vLLM 维护的解析器，而不是自己重写一套。
- **`agentic_test_design_proposal.md` 里的 `real_tool_impl` / `run_agentic_task`**——是第 5 节"Agent 层真正执行工具、维护多轮循环"，纯 Agent scope，这也是为什么这套测试设计要在 Runtime 的测试代码库之外，额外补一层——因为 Runtime 自己的测试范围压根不覆盖这一层。
- **Ollama 的 `agent/` 包**（`bash.go`/`file.go`/`web.go`/`compactor.go`）——Ollama 比较特殊，它把 Runtime（推理引擎）和 Agent（工具执行、上下文压缩）**打包进了同一个产品**，但代码架构上这两层依然是分开的模块，`agent/` 包整体属于第 5/6 节的 Agent scope，不属于它自己的推理引擎那一层。
- **llama.cpp 的 `--mcp-servers-json`**——这是一个例外情况：正常来说 MCP Client 应该活在 Agent 层（第 6 节），但 llama.cpp **直接把 MCP Client 功能内建进了推理服务器本体**（`llama-server`），是六个系统里唯一这样做的——这也是为什么之前的调查报告会把这一点单独标注出来："这是 vLLM/SGLang/TRT-LLM 的推理服务器本体都没有的能力"，本质上是这套系统主动打破了第 8 节那张边界表的默认分工。
- **TensorRT-LLM 的 `examples/scaffolding`**——是一套独立的 Agent 编排框架（含自己的 MCP client 支持、`ParallelProcess`、`MCTSController`），架构上属于第 5/6 节的 Agent scope，只是这套代码托管在推理引擎的同一个仓库里，容易被误认为是 Runtime 的一部分。

## 10. 常见误解澄清

- **"MCP 是不是取代了 function calling？"**——不是。MCP 解决"工具从哪来、怎么标准化接入"，function calling（`tools`/`tool_calls` 协议）解决"模型怎么表达调用意图"。一个 MCP 工具最终还是要走 function calling 协议才能被模型看到、调用。
- **"Runtime 支持了 tool calling，是不是就等于支持了 Agent？"**——不是。Runtime 只保证"能不能正确生成/解析一次工具调用意图"，多轮循环、真实工具执行、上下文管理这些"Agent 能力"，全部在 Runtime 之外的那一层。这正是本系列六份调查报告反复强调的一点：Runtime 层的工具调用测试再扎实，也回答不了"整个 agent 任务有没有真正做对"这个问题。
- **"约束解码是不是保证了工具调用一定对？"**——不是，它只保证**语法合法**（这是一段能被解析成 JSON 的字符串），不保证**内容正确**（调用的函数是不是模型该调的、参数填的是不是对的）——这是两个独立的正确性维度。
- **"OpenAI 的 API 格式是不是唯一标准？"**——不是官方强制标准，只是因为使用者最多、生态最成熟，成了事实标准（de facto standard）——六个被调查的 Runtime 全部选择兼容它，而不是各自发明格式，纯粹是生态/工程决策，不是技术上必须如此。

## 11. 一图总览

```
                          ┌───────────────────────────────────────┐
                          │              Agent Scope                │
                          │  • 准备/发现工具（手写 或 MCP Client）    │
                          │  • 真正执行工具                          │
                          │  • 维护多轮循环、上下文管理                │
                          └──────────────────┬────────────────────┘
                                             │
                          ═════════ Function-Calling 协议 ═════════
                          （OpenAI 风格 tools 请求 / tool_calls 响应，
                             双方都要遵守，是两层之间的"契约"）
                                             │
                          ┌──────────────────▼────────────────────┐
                          │             Runtime Scope                │
                          │  • 模型推理                              │
                          │  • 约束解码（保证语法合法）                │
                          │  • Tool-call parser（原始输出→结构化）    │
                          └───────────────────────────────────────┘

MCP 是一个挂在 Agent Scope 里的"子协议"，专门解决"工具从哪来"，不改变上面这张图的主结构。
```

## 术语速查表

| 术语 | 一句话解释 | 属于哪层 |
|---|---|---|
| Tool Schema / Function 定义 | 描述工具名字、用途、参数格式的声明式 JSON，模型只看得到这个 | 契约（双方共享） |
| Function Calling / Tool Calling | "模型用结构化方式表达调用意图"这件事的统称 | 契约 + Runtime 实现 |
| OpenAI Chat Completions API（`tools`/`tool_calls`） | 事实标准的请求/响应 wire format | 契约 |
| Tool-call Parser | 把模型原始输出解析成标准 `tool_calls` 字段的 Runtime 内部代码 | Runtime |
| 约束/引导解码（xgrammar 等） | 解码时屏蔽非法 token，保证输出语法合法 | Runtime |
| Agent / Orchestrator | 发起请求、执行工具、维护多轮循环的客户端代码 | Agent |
| MCP（Model Context Protocol） | 工具提供方和 Agent 之间的标准化"即插即用"协议 | Agent（子协议） |
| MCP Server | 真正实现并暴露工具的进程 | 工具提供方 |
| MCP Client | Agent 里负责发现/调用 MCP Server 的模块 | Agent |
