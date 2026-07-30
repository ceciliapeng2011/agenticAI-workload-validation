# SGLang Agentic Workload 性能 / 精度评估方法 — 调查报告

> 
> 仓库路径：`~/agenticAI/sglang`
> 方法：与 vLLM 调查一致——不依赖关键词命中数量，逐个打开源码读实现逻辑，找"机制是否真的做了该做的事"

## 一、Tool-Call 精度测试：`test/registered/**/function_call/`

涉及文件：

- `test/registered/unit/function_call/`（纯解析器单测，如 `test_hermes_detector.py`、`test_function_call_parser.py` 等，逐模型一个文件）
- `test/registered/openai_server/function_call/test_openai_function_calling.py`（起真实 server 的端到端测试）

### 单元测试层

与 vLLM 的 `tests/tool_parsers/` 结构完全对应：喂固定字符串给各家 detector（Hermes、Mistral、MiniMax-M3、Kimi-K2、GLM4-MoE 等），断言解析出的 JSON 结构正确。**不涉及真实模型推理**。

### 端到端测试层：发现一处"写了但从未被执行"的真实两轮测试

检索：

```
cd ~/agenticAI/sglang && grep -n "_test_function_calling_multiturn()" test/registered/openai_server/function_call/test_openai_function_calling.py
```

输出：

```
950:#         self._test_function_calling_multiturn()
```

只有**一处调用**，而且这一行是注释掉的。往回看这个辅助方法的真实实现（117 行起）：

```python
def _test_function_calling_multiturn(self):
    ...
    response = client.chat.completions.create(..., tools=tools)   # 第 1 轮：真实推理
    tool_call = response.choices[0].message.tool_calls[0]
    ...
    messages.append(response.choices[0].message)
    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                      "content": "8", "name": function_name})
    final_response = client.chat.completions.create(              # 第 2 轮：真实推理，依赖第 1 轮的真实输出
        model=self.model, messages=messages, tools=tools, ...
    )
    assert "8" in final_response.choices[0].message.content
```

### 核心重大发现

> 
> 这段实现**本身是对的**：第 2 轮请求真实依赖第 1 轮的 `tool_call.id` / `function_name`（真实模型输出），不是像 vLLM 那样用写死的三消息 fixture 一次性验证——这是两次真实、有依赖关系的推理调用，是名副其实的"多轮"。
> 
> 但往下看唯一的调用点：

```python
# Skip for ci test
# class TestGLM45ServerFunctionCalling(TestOpenAIServerFunctionCalling):
#     ...
#     def test_function_calling_multiturn(self):
#         self._test_function_calling_multiturn()
```

**这个方法所在的整个测试类都被注释掉了**（原因写在类上方："Skip for ci test"，且该类需要 TP=8 起 GLM-4.5，成本较高）。也就是说：**SGLang 代码库里写出了正确的真实多轮工具调用测试实现，但它是死代码——没有任何活跃的 CI 路径会执行它**。

对比 vLLM：vLLM 的问题是"测试实现方式本身有缺陷"（fixture 伪多轮）；SGLang 的问题是"测试实现方式是对的，但被搁置、从未激活"。**殊途同归——两边最终都没有活跃 CI 在验证真实多轮工具调用的正确性**，只是踩坑的位置不同,这点值得你以后审计任何 runtime 时都留意:「写没写对」和「写没写完 / 有没有跑」是两个独立要检查的维度。

### Prefix Caching 关键结论（与 vLLM 相反的发现）

检索：

```
cd ~/agenticAI/sglang && grep -n "disable_radix_cache" python/sglang/test/test_utils.py
```

输出：

```
1861:    disable_radix_cache=False,
1893:    disable_radix_cache=False,
```

`popen_launch_server` 的默认值是 `disable_radix_cache=False`（即**默认开启前缀缓存**）。翻查 `test_openai_function_calling.py` 里所有 `other_args`,没有一处传入 `--disable-radix-cache`。

✅ 推论（与 vLLM 结论相反）：
**SGLang 的工具调用端到端测试是在前缀缓存（RadixAttention）开启的默认状态下跑的。** 这一点上 SGLang 比 vLLM 更扎实——至少"工具调用 + 前缀缓存"这个组合在 CI 里被隐式覆盖到了(尽管没有专门断言"缓存命中前后输出一致")。

## 二、性能基准：官方 CI vs 仓库内高级能力（结构与 vLLM 相同,细节不同）

### 官方性能 CI 范围

检索：

```
cd ~/agenticAI/sglang && grep -n "def test_\|dataset_name" test/registered/perf/test_bench_serving_1gpu_part1.py test/registered/perf/test_bench_serving_1gpu_part2.py test/registered/perf/test_bench_serving_2gpu.py
```

输出摘要（节选）：

```
test_offline_throughput_default
test_offline_throughput_non_stream_small_batch_size   → dataset_name="sharegpt"
test_offline_throughput_without_radix_cache
test_offline_throughput_without_chunked_prefill
test_online_latency_default
test_online_lora_latency
test_vlm_offline_throughput                           → dataset_name="mmmu"
test_moe_offline_throughput_without_radix_cache
test_pp_long_context_prefill                          → dataset_name="random"
```

### 官方 CI 局限

- 全部是**单轮独立请求**：`sharegpt` / `random` / `mmmu` 数据集,没有一个用到下面第三节会提到的 `mooncake` 或 `generated-shared-prefix`（GSP）数据集；
- 有意思的是,这里专门有 `test_offline_throughput_without_radix_cache` / `test_moe_offline_throughput_without_radix_cache` 这类"关掉前缀缓存"的对照测试——说明 SGLang 关心"前缀缓存对吞吐的增益",但衡量的仍是**单轮重复 prompt** 场景下的收益,不是真实多轮会话场景；
- 指标同样是标准的 throughput / latency,没有任务级指标、没有 cache 命中率指标。

### 仓库内真正支持多轮会话的性能压测能力（比 vLLM 更"原生集成"）

路径：`python/sglang/benchmark/serving.py`（新路径,`bench_serving.py` 已是废弃的转发壳）

检索：

```
cd ~/agenticAI/sglang && grep -n "dataset_name\|multi.turn\|generated.shared.prefix\|mooncake" python/sglang/benchmark/serving.py | head -20
```

关键发现:

1. **真实多轮请求循环**（不是 vLLM 那种独立工具搁置的 Rust 项目,而是主 benchmark 模块自带的功能）：

```python
def wrap_multi_turn_request_func(request_func, backend):
    async def f(request_func_input, pbar=None):
        prev_messages = []
        for round_index in range(len(prompts)):
            ...
            inner_input = replace(..., prompt=prev_messages)
            output = await request_func(inner_input, ...)   # 真实推理,逐轮累积历史
            prev_messages.append({"role": "assistant", "content": output.generated_text})
        return outputs
```

这是**真实执行的多轮会话回放**——历史是真实累积的模型输出,不是预先写好的静态文本。

2. **Mooncake trace 集成**（业界少见,直接用带 `hash_ids` 的真实前缀共享 trace）：

```python
if args.dataset_name == "mooncake":
    hash_ids = warmup_record.get("hash_ids", [])
    prompt_text = ""
    for hash_id in hash_ids:
        prompt_text += f"{hash_id}" + " ".join(["hi"] * 512)
```

还支持 `use_trace_timestamps`（按 trace 里的真实时间戳回放到达节奏,而不是泊松近似）、`mooncake_slowdown_factor`、`mooncake_num_rounds`。

3. **Generated-Shared-Prefix（GSP）数据集**：专门建模"多个会话共享同一个系统提示 group"的场景,参数包括 `gsp_num_groups`、`gsp_prompts_per_group`、`gsp_num_turns`(>1 时启用"真·多轮会话压测")。

### 关键结论：这项能力被一个"功能测试"验证了,但从未出现在"性能报告"里

检索：

```
cd ~/agenticAI/sglang && cat test/registered/bench_fn/test_bench_serving_functionality.py
```

发现 `TestBenchServingFunctionality::test_gsp_multi_turn`——**这是一个已注册进 nightly CI 的活跃测试**（`register_cuda_ci(..., suite="nightly-1-gpu", nightly=True)`）,用 `dataset_name="generated-shared-prefix"` 真跑 4 个会话 × 3 轮:

```python
res = run_benchmark(args)
self.assertEqual(res["completed"], NUM_CONVERSATIONS * NUM_TURNS)
...
self._verify_multi_turn_logs(logs)   # 解析服务端请求日志,校验后轮请求文本确实以前轮文本为前缀
```

**这是我在两次调查里读到的最扎实的一处"多轮会话结构正确性"验证**——它不只是"跑完没崩",而是解析真实服务端日志、逐对比较,断言"至少有 `NUM_CONVERSATIONS × (NUM_TURNS - 1)` 对请求满足真实前缀包含关系",即**真正验证了多轮会话的上下文累积机制在服务端确实按预期工作**。

但注意:
- 这个测试归类在 `test/registered/bench_fn/`(functionality,功能正确性),**不在** `test/registered/perf/`(performance,性能报告)——也就是说,和第 2.1 节列出的官方性能基准是**两条平行的 CI 流水线**,`test_gsp_multi_turn` 只验证"能不能正确工作",不产出、不追踪吞吐/延迟数字。
- 它验证的是"请求结构对不对",**不验证模型输出内容的正确性**(它没有检查每一轮的回答是否符合任务要求,只检查了请求文本的前缀关系和完成数量)。

现状:与 vLLM 一样是"能力有,但两条流水线没打通"——只是 SGLang 这边多轮压测能力更原生集成、且有一个专门的功能正确性测试在守着结构层面(vLLM 那边连这一层功能测试都没有)。

## 三、Tool-Calling / Agentic 负载真实感来源:三个 benchmark 的坦白说明

### `benchmark/react/`——文档里直接写明"这不是真的 Agent"

```
cd ~/agenticAI/sglang && cat benchmark/react/README.md | head -5
```

```
NOTE: This is an implementation for replaying a given trace for throughput/latency
benchmark purposes. It is not an actual ReAct agent implementation.
```

**一手证据,比 vLLM 那边任何注释都直白**:SGLang 官方文档主动承认,这个挂着 "ReAct" 名字的 benchmark 只是**回放一段固定 trace 来测吞吐/延迟**,不是一个会真正做决策、调工具的 agent。用的数据是 `hotpotqa_100.jsonl`——多跳问答数据集,被用作"prompt 形状来源",而不是"正确性评判标准"。

### `benchmark/multi_turn_chat/`——用随机 token 拼出的"对话"

```python
def gen_prompt(tokenizer, token_num):
    all_available_tokens = list(tokenizer.get_vocab().values())
    selected_tokens = random.choices(all_available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)
```

多轮的"内容"是随机 token 序列,不是真实语言。这个 benchmark 的价值仅限于:验证 SGLang 前端 DSL(`sgl.function` + `s += qa["prompt"]; s += sgl.gen(...)`)能不能正确利用 RadixAttention 做跨轮前缀复用来测吞吐——**和"这一轮回答得对不对"完全无关**,连"看起来像不像真实对话"都不追求。

### `benchmark/generative_agents/`——衍生自论文的 trace,但仍是纯性能向

```
README.md: "Ensure that this benchmark is run in a serial manner (using --parallel 1)
            to preserve any potential dependencies between requests."
```

这个 benchmark 数据源是 Stanford《Generative Agents》论文里的真实 agent 调用序列(`agent_calls.jsonl`),比 `multi_turn_chat` 更贴近真实 agent 行为——请求间存在真实依赖关系,所以要求串行执行以保留依赖顺序(这一点比 vLLM 官方 CI 默认的泊松独立到达更贴近 agentic 特点⑤)。但同样,`bench_sglang.py` 只测耗时,不校验任何输出内容对不对。

### 额外发现:`FlexKV`(分层 KV 存储)拿 SWE-bench 提示做性能验证

```
cd ~/agenticAI/sglang && grep -n "SWE-bench" -B10 python/sglang/srt/mem_cache/storage/flexkv/README.md
```

```
Workload: 120 prompts sampled from princeton-nlp/SWE-bench_Lite_oracle
with input length ≤ 28k tokens (p50 = 7088, max = 27961).
...
| baseline                    | TTFT p50 8.04s  | ...
| --enable-hierarchical-cache | TTFT p50 0.04s  | ...
```

用真实的长上下文编码 agent 提示(SWE-bench)做分层缓存特性的 TTFT 收益验证——是三者中"负载最贴近真实 agentic 场景"的一个,但**依然只测性能收益(TTFT/吞吐),不评估 SWE-bench 任务本身有没有被正确解决**。

### 全局仓库检索结论

```
cd ~/agenticAI/sglang && grep -rniE "bfcl|gorilla|tau.?bench|agentbench|toolbench" --include="*.py" --include="*.md" . | grep -v "\.git/"
```

**零命中**(除 SWE-bench 的一处性能向引用外,连 τ-bench、AgentBench、ToolBench 都没有任何引用)。和 vLLM 的结论一致:**没有把任何标准 agentic 精度基准接入自身评测体系**——vLLM 至少还接了 BFCL 数据集(哪怕只用于性能压测),SGLang 连这一步都没有。

## 四、精度 CI 与 KV 量化 / 缓存驱逐测试覆盖现状

### lm_eval 精度回归规模

```
cd ~/agenticAI/sglang && ls test/lm_eval_configs/ | wc -l
```

输出:**4 个 yaml**(远少于 vLLM 的 38 个),全部是 NVIDIA-Nemotron-3-Nano / Qwen3.5 系列,任务清一色 `gsm8k`(`exact_match,strict-match` / `flexible-extract`),和 vLLM 一样**没有工具调用或多轮任务**。

### 一处"名不副实"的发现:`test_radix_cache_slru_accuracy.py`

文件名带 "accuracy",容易望文生义以为是"模型精度测试"。打开一看:

```python
class TestSLRUAccuracy(unittest.TestCase):
    def setUp(self):
        self.kv_cache = MHATokenToKVPool(size=8, ...)   # 极小的合成 KV pool,专为触发驱逐设计
    def test_eviction_mechanism(self):
        """Test that SLRU eviction mechanism works correctly"""
        frequent_key = RadixKey(array("q", [1, 2]))
        ...
```

这里的 "accuracy" 指的是**"SLRU 驱逐算法是否正确保留高频 key、驱逐低频 key"**——是纯算法/数据结构单测(CPU 上跑,用假造的 token id),和"模型输出准不准"毫无关系。**这是我在两次调查里遇到的第二处"multi_round/accuracy 这类词面容易造成误判"的例子**(第一处是 vLLM 的 `test_ec_connector_with_partial_cache_hit_multi_round`),再次印证:光靠关键词搜索会被自然语言的歧义坑,必须打开代码读语义。

### 一处真正的"量化 + 精度"联合验证——但重演了 vLLM 同款陷阱

```
cd ~/agenticAI/sglang && cat test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py | head -80
```

`test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`:MXFP4 权重量化 + `--kv-cache-dtype fp8_e4m3` KV 量化,在真实 8×MI35x 硬件上跑 GSM8K few-shot,设了 `accuracy_threshold`(如 0.93),注册为 nightly CI(`nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-kv-fp8`,预估耗时 3600 秒/60 分钟)——**这是一处认真做的、真实硬件上的量化精度回归**,比 vLLM 那种"顺手加个 fp8 只为跑得快"的动机更扎实。

但往下看 `other_args`:

```python
other_args=[
    "--attention-backend", "aiter",
    "--chunked-prefill-size", "131072",
    "--disable-radix-cache",     # ← 同样显式关闭了前缀缓存
    "--mem-fraction-static", "0.85",
    ...
]
```

**和 vLLM 的 `--no-enable-prefix-caching` 完全是同一个模式**:KV 量化的精度验证,是在**前缀缓存关闭**的状态下做的。也就是说:"KV cache FP8 量化 + RadixAttention 前缀复用 同时开启时,精度是否仍然保持"——这个组合在 SGLang 里同样**没有任何 CI 覆盖**。而且任务依然是单轮 GSM8K,不含工具调用或多轮场景。

## 五、综合结论

### 性能侧现状

1. 官方性能 CI(`test/registered/perf/`)和 vLLM 一样,是**单轮独立请求**(sharegpt/random/mmmu),额外做了"有无 radix cache"的 A/B 对照,但对照场景仍是单轮重复 prompt,不是真实多轮会话;
2. `python/sglang/benchmark/serving.py` **原生内置**(不像 vLLM 那样另起一个独立 Rust 工具)了真实多轮会话回放、Mooncake trace(含 `hash_ids` 前缀共享结构、真实时间戳到达)、Generated-Shared-Prefix 数据集——建模能力覆盖了 agentic 特点①②④⑤;
3. 这项能力被 `test/registered/bench_fn/test_gsp_multi_turn` 接入了**nightly 功能正确性 CI**,并且做了真正的"多轮请求前缀包含关系"校验——这是我在两个 runtime 里见过最扎实的结构层面验证;
4. 但这条"功能正确性"流水线与"性能报告"流水线是**两条不相交的线**——官方吞吐/延迟报告依然完全不用多轮/Mooncake/GSP 数据集;
5. `benchmark/react`、`benchmark/multi_turn_chat`、`benchmark/generative_agents`、FlexKV 的 SWE-bench 压测,负载真实感依次递增,但全部只测性能,不评估任务正确性;`benchmark/react` 的 README 甚至主动声明自己不是真实 agent。

### 精度侧现状

1. Tool-call 端到端测试**默认开启前缀缓存**(优于 vLLM)——但唯一写对了"真实两轮推理依赖"的多轮工具调用测试(`_test_function_calling_multiturn`),其调用点整个类都被注释掉,**从未在 CI 里跑过**(能力有,执行链路断了——和 vLLM"能力本身有缺陷"是不同性质的同一个结果);
2. lm_eval 精度回归规模小于 vLLM(4 vs 38 个配置),同样清一色单轮 gsm8k,不覆盖工具调用/多轮;
3. 存在一处认真做的 KV FP8 量化 + GSM8K 精度回归(真实硬件、真实阈值),但和 vLLM 一样,**显式关闭了前缀缓存**,导致"量化 + RadixAttention 同时开启"这个生产环境真实状态从未被精度 CI 验证过;
4. 文件命名容易误导人:`test_ec_connector_with_partial_cache_hit_multi_round`(vLLM)、`test_radix_cache_slru_accuracy.py`(SGLang)都不是字面意思暗示的"多轮会话"/"模型精度",而是各自系统内部数据结构的单测——审计任何 runtime 时都要打开代码验证词面。

### 一句话总览,以及与 vLLM 的对照

SGLang 在**结构层面**比 vLLM 领先一步:多轮/Mooncake/GSP 负载建模能力是原生集成的(不是搁置的独立工具),且有一个专门的、活跃的 nightly 功能测试在校验"多轮会话的前缀累积机制是否真的按预期工作";工具调用端到端测试也默认在前缀缓存开启状态下跑。

但落到**"精度是否被 agentic 场景下的优化组合保护"**这个最终问题上,两者殊途同归:

- 都没有接入任何标准 agentic 精度基准(BFCL/τ-bench/AgentBench 等)——vLLM 好歹拿 BFCL 当性能负载源,SGLang 连这一步都没做;
- 都存在"量化精度回归 vs 前缀缓存"的组合盲区——两边的 KV 量化精度测试都显式关闭了前缀缓存;
- 真正意义上的"多轮工具调用"测试,vLLM 是实现方式有缺陷,SGLang 是实现方式对但从未被执行——**都不构成一条能在日常 CI 里守住"多轮 + 工具调用 + 主流优化组合开启"这一整套 agentic 场景精度的自动化防线**。
