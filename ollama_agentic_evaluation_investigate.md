# Ollama Agentic Workload 性能 / 精度评估方法 — 调查报告

> **文档导航**（完整索引见 [README.md](README.md)）
>
> **调查报告**：[vLLM](vllm_agentic_evaluation_investigate.md) · [SGLang](sglang_agentic_evaluation_investigate.md) · [TensorRT-LLM](tensorrt_llm_agentic_evaluation_investigate.md) · [llama.cpp](llama_cpp_agentic_evaluation_investigate.md) · **Ollama** · [OpenVINO GenAI](openvino_genai_agentic_evaluation_investigate.md)
>
> **横向分析**：[六系统横向对比](cross_comparison_agentic_evaluation.md) · [能力×严谨度矩阵](capability_x_systems_rigor_matrix.md) · [基准全景对比](benchmark_landscape_comparison.md) · [测试设计方案](agentic_test_design_proposal.md)
>
> **管理层报告 / 概念科普**：[OpenVINO 管理层报告](openvino_management_technical_report.md) · [Tool Calling/MCP 概念全景](tool_calling_mcp_primer.md) · [约束解码与 Parser 源码拆解](openvino_genai_structured_output_and_parser_impl.md)
>
> **方法论 / 早期产物**：[方法论笔记](agentic_workload_research.md) · [脚本3人工检查点记录](vllm_investigation.md)
>
> **审计脚本**：[详细说明](AUDIT_README.md) · [5分钟上手](QUICKSTART.md)

> 
> 仓库路径：`~/agenticAI/ollama`
> 方法：与 vLLM / SGLang / TensorRT-LLM / llama.cpp 调查一致——不依赖关键词命中数量,逐个打开源码读实现逻辑
> 说明：Ollama 的定位和前四者都不同——它不是一个裸的 Runtime/推理服务器,而是**面向单机用户的产品**,底层引擎是 vendored 的 llama.cpp（`LLAMA_CPP_VERSION` 文件锁定版本为 `b10091`）+ 自研的 MLX 后端,同时最上层还有一个**自己的 Agent 产品层**（详见第二节）。这个双层结构决定了调查要分两条线看。

## 一、Tool-Call 测试:同样的"假 LLM"规律第四次复现,但真实模型矩阵覆盖面最广

涉及文件:

- `agent/session_test.go`(编排层单测,mock LLM)
- `integration/tools_test.go` / `integration/tools_stress_test.go`(真实模型端到端)

### Agent 编排层单测:`fakeClient` 模式,和另外三个 runtime 里发现的规律完全一致

```
cd ~/agenticAI/ollama && sed -n '390,410p' agent/session_test.go
```

```go
func TestSessionRunsToolLoop(t *testing.T) {
    client := &fakeClient{
        responses: [][]api.ChatResponse{
            {{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
                Function: api.ToolCallFunction{Name: "echo_tool", Arguments: args}}}}}},
            {{Message: api.Message{Role: "assistant", Content: "done"}}},
        },
    }
    ...
}
```

`fakeClient` 按调用次数返回预先写好的固定响应序列,从头到尾没有调用真实模型。这是本次系列调查里**第四次**在"看起来最像完整 agent 循环"的测试里发现同一个模式(TRT-LLM 的 `DummyWorker`、vLLM 的 fixture、SGLang 的死代码是前三次)。**这条规律现在可以确认是行业普遍做法,不是某一家的疏漏**——编排逻辑(会不会正确触发第二轮工具调用、会不会正确拼接历史)确实应该用可控的假响应去单测,这本身没问题;问题在于**没有任何一家把这类单测和"真实模型会不会正确决策"的验证串起来**。

Ollama 这套单测本身质量很高(`agent/session_test.go` 一个文件里有 30+ 个测试函数,覆盖工具循环轮数上限、取消/中断时的部分结果处理、大工具结果截断、上下文不足时的降级等边界情况),是四次调查里编排层单测覆盖最细的一个,但覆盖细≠验证了真实模型行为。

### 真实模型端到端测试:模型矩阵覆盖面是五个系统里最广的

```
cd ~/agenticAI/ollama && sed -n '30,50p' integration/tools_test.go
```

```go
var toolsMinVRAM = map[string]uint64{
    "gemma4": 8, "lfm2.5": 6, "granite4.1:3b": 4, "granite4.1:8b": 6,
    "nemotron3:33b": 32, "qwen3.5:2b": 4, "qwen3.6:27b": 20, "qwen3-vl": 16,
    "gpt-oss:20b": 16, "gpt-oss:120b": 70, "qwen3": 6, "llama3.1": 8,
    "llama3.2": 4, "mistral": 6, "qwen2.5": 6, "qwen2": 6, "ministral-3": 20,
    "mistral-nemo": 9, "mistral-small": 16, "mixtral:8x22b": 80, "qwq": 20,
    "granite3.3": 7,
}
```

`runAPIToolCallingModelWithClient` 对每个模型真实起服务、真实推理,验证真的能触发 `get_weather` 工具调用——**21 个真实模型**,比 vLLM 的 12 个(`tests/tool_use/utils.py`)、SGLang 的 detector 矩阵覆盖更广。但同样是**单轮**测试,没有把工具结果喂回去验证第二轮——覆盖面全场最广,但深度和 vLLM 处于同一档(TRT-LLM 的 Harmony 真两轮测试仍是唯一深度案例)。

### 核心发现:这批测试覆盖面最广、最真实,却完全没有接入公开 CI

```
cd ~/agenticAI/ollama && grep -n "integration" .github/workflows/*.yaml
```

```
(唯一命中是 release.yaml 里一个不相关的 URL: "kms-integrations")
```

```
cd ~/agenticAI/ollama && cat .github/workflows/test.yaml | grep "go test"
```

```
go test -count=1 -benchtime=1x ./...              # 只跑普通单测(含 agent/ 的 fakeClient 单测)
go test -count=1 -tags updater_live ./app/...
```

`integration/tools_test.go` 需要 `//go:build integration` 编译标签才会被编译进测试二进制,`integration/README.md` 里写明要手动加 `-tags=integration,fast/release/library` 才能跑,且 `library` 档需要约 2.5 TiB 磁盘。**公开的 GitHub Actions 工作流里,没有一处引用这个编译标签。** 也就是说:五个系统调查以来,这是"覆盖面最广的真实模型工具调用测试"和"完全确认没有接入公开 CI"这两个极端同时出现在一处的案例——大概率是 Ollama 在私有发布前基础设施上手动/定期跑的(`release` 档描述写的是"release regression coverage"),但公开仓库看不到这条流水线。

## 二、独有能力:产品层的 Agent 框架,直接在应用层解决了"agentic 上下文增长"问题——这是另外四个系统都没有的东西

`agent/` 包(不是 example,是主二进制自带的功能)包含 `bash.go`(真实 shell 执行)、`file.go`(真实文件读写)、`web.go`(真实网页抓取)、`skill.go`(SKILL.md 技能加载约定)、`approval.go`(高风险操作前的人工确认)、`compactor.go`(自动上下文压缩)。这套东西的完整度和产品化程度,超过 TRT-LLM 那个还停留在 `examples/` 目录里的 Scaffolding 框架。

### 最值得记录的一处:自动上下文压缩(Compaction)——直接对应"agentic 上下文持续增长"这个系统特点

```
cd ~/agenticAI/ollama && sed -n '25,35p' agent/compactor.go
```

```go
const (
    defaultCompactionContextWindowTokens = 32768
    defaultCompactionKeepUserTurns       = 3
    defaultCompactionThreshold           = 0.8    // 上下文用到 80% 触发压缩
    compactOnlySummaryContextTokens      = 16000
)
```

压缩逻辑:保留最近 3 轮用户对话原文,更早的历史用另一次 LLM 调用总结成一段摘要,替换掉原始历史。摘要 prompt 明确要求:

```go
compactionSystemPrompt = "Summarize the archived part of an Ollama agent conversation. " +
    "Preserve user goals, decisions, files, commands, tool results, and unresolved tasks " +
    "needed to continue. Omit private reasoning and return only the summary."
```

**这是五个系统里唯一一处,在"多轮会话历史会不断增长"这个 agentic 核心特点上,给出了产品级解决方案的地方。** vLLM/SGLang/TRT-LLM 的 KV cache 驱逐/offload 解决的是"显存不够了怎么办"(系统层),Ollama 这里解决的是"上下文窗口不够了、又不想让 agent 忘记关键信息怎么办"(应用层)——是完全不同、但同样关键的一层问题,而且只有 Ollama 做了。

### 测试质量:细致但同样是"验证机制,不验证摘要质量"

```
cd ~/agenticAI/ollama && grep -n "^func Test" agent/compactor_test.go | wc -l
```

**19 个测试函数**,覆盖:触发阈值判断是否正确(`TestSimpleCompactorSkipsBelowThreshold`)、保留轮数是否正确(`TestSimpleCompactorDefaultsToKeepingThreeUserTurns`)、超长摘要截断(`TestSimpleCompactorTruncatesOversizedSummary`)、摘要生成失败时的重试策略、甚至专门验证了"压缩用的 prompt 不应该泄漏模型的隐藏推理内容"(`TestSimpleCompactorSummarizesOldMessages` 断言 `!strings.Contains(request, "hidden")`)。

但同样用的是 `fakeClient` 返回写死的 `"summary"` 字符串——**没有任何测试验证"压缩之后的摘要是否真的保留了足够信息让 agent 能继续完成任务"**,即压缩机制的编排逻辑测得很细,但压缩产出的实际质量(信息保真度)完全没有被评测覆盖。这是这个功能目前最大的验证盲区,也是**这套本可以直接拿去做"agentic 精度评测"的机制**(比较压缩前后模型完成同一任务的成功率)目前完全没被用上的地方。

## 三、性能基准:比 llama.cpp 更简陋,单请求微基准,不存在任何 serving/并发/多轮建模

```
cd ~/agenticAI/ollama && grep -n 'flag\.' cmd/bench/bench.go | grep -v _test
```

```go
models:       flag.String("model", "", "Model to benchmark")
epochs:       flag.Int("epochs", 6, "Number of epochs (iterations) per model")
maxTokens:    flag.Int("max-tokens", 200, ...)
prompt:       flag.String("p", DefaultPrompt, "Prompt to use")
warmup:       flag.Int("warmup", 1, ...)
promptTokens: flag.Int("prompt-tokens", 0, ...)
```

`ollama bench`:单模型、单固定 prompt、顺序跑 N 个 epoch,报 benchstat/csv 格式的统计量。**没有 `--concurrency`,没有数据集,没有多轮。** 这符合 Ollama"单机单用户"的产品定位——它确实不需要像 vLLM/SGLang 那样测多租户吞吐——但结果是:**Ollama 在"评估 agentic workload 性能"这件事上,不但没有比底层 llama.cpp 更进一步,反而因为产品定位更简单,连 llama.cpp 的 `llama-bench` 那种批处理式微基准都不需要,直接更原始。** 前面第二节提到的 Agent 编排层(工具循环、压缩)所引入的额外推理调用次数、额外 token 消耗,目前没有任何配套的性能测量工具。

## 四、精度评估:全仓库零命中,完全依赖上游

```
cd ~/agenticAI/ollama && grep -rliE "gsm8k|mmlu|accuracy.*threshold|lm.eval|bfcl|gorilla|tau.?bench|agentbench|swe.?bench" --include="*.go" .
```

**零命中。** Ollama 不做任何自己的模型精度回归测试——原因也合理:它不训练模型、通常也不自己做量化(多数模型是社区/官方发布的 GGUF,量化精度损失的责任在上游),所以"模型准不准"这件事被完全委托给了 llama.cpp(困惑度,但如第一份报告所述,那也是纯人工、非任务型的)和模型发布者。**这意味着 Ollama 自己的 Agent 层(工具循环、上下文压缩)如果因为一次代码改动而降低了任务完成率,不会被任何精度信号捕捉到**——因为压缩摘要质量、工具调用决策质量这些都没有对应的评测,而模型权重本身的精度评测又不在 Ollama 的职责范围内。

## 五、综合结论

### 性能侧

1. `ollama bench` 是五个系统里最原始的性能工具——单请求、无并发、无数据集、无多轮,比它继承的 llama.cpp `llama-bench` 更简单;
2. Agent 层(工具循环、自动压缩)引入的额外推理开销,没有任何配套的性能测量。

### 精度侧

1. Agent 编排层(`agent/`)的单测覆盖非常细致(session 循环、压缩阈值、边界情况),但和另外四个系统里发现的规律完全一致——用 `fakeClient` 代替真实模型,只验证编排逻辑,不验证真实模型决策质量;
2. 真实模型工具调用测试(`integration/tools_test.go`)覆盖 21 个真实模型,是五个系统里模型矩阵最广的一个,但只测单轮,且**确认没有接入公开 GitHub Actions CI**,只能在私有/手动流程里跑;
3. **独有亮点**:`agent/compactor.go` 的自动上下文压缩机制,是五个系统里唯一在产品层面直接回应"agentic 会话上下文持续增长"这一核心特点的实现——但其压缩质量(摘要是否保真到足以让任务继续)同样没有被任何评测覆盖,是这个亮点目前最大的短板,也是最值得补的一块;
4. 模型精度评估完全委托给上游(llama.cpp 的困惑度方法论 + 模型发布者),自身零覆盖。

### 五个系统横向定位(结合 llama.cpp 报告)

如果把"广度"(模型/场景覆盖面)和"深度"(测试是否触及真实模型行为、是否接入日常 CI)当作两个轴:

- **TRT-LLM**:深度最高(唯一活跃的真两轮工具调用 CI、唯一把缓存复用当精度变量),广度较窄;
- **Ollama**:真实模型工具调用测试**广度最高**(21 个模型),但深度不够(单轮、且不在公开 CI);同时是唯一在**产品层**（而非 runtime 层）解决 agentic 上下文增长问题的系统,这是一个另外四家都没有覆盖的独特维度;
- **vLLM / SGLang**:广度和深度都居中,vLLM 多一步"用 BFCL 做负载生成器",SGLang 多一步"用真实日志校验多轮前缀增长的结构正确性";
- **llama.cpp**:作为 Ollama(及很多其他项目)的底层引擎,性能/精度评估工具链是五者里最原始/最不自动化的(旗舰性能 CI 停用、精度方法论只有困惑度且零 CI 化)——这个下限直接传导给了 Ollama。

**跨五个系统重复验证的最终规律**:任何"看起来像完整 agentic 循环"的测试,一旦打开生成侧的实现,几乎必然发现是 mock/fixture/dummy worker 在扮演 LLM;任何"缓存/量化优化对精度的影响"测试,几乎必然是在关掉缓存优化或不涉及工具调用/多轮场景的条件下做的。到目前为止调查的五个系统,没有一个把"多轮会话 + 工具调用 + 主流性能优化组合开启"作为一个整体,纳入过日常自动化的精度回归。
