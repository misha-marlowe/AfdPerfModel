# AFD decode performance model: 8 × MI355X

This accounting follows [FastAFD](https://haoailab.com/blogs/fastafd/#where-the-speedup-comes-from).

Compare colocated serving and attention–FFN disaggregation (AFD) on the **same eight GPUs**, after prefill, generating one token per active request per step. This is an analytical model, not a measured MI355X speedup. Each MI355X has 288 GB HBM; GB below means $10^9$ bytes. [AMD specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)

## 1. Throughput and sources of speedup

Let $N=N_A+N_F=8$, where $N_A$ GPUs run attention and $N_F$ run FFN/MoE. Define:

- $B_{\mathrm{col}}$: resident requests **per colocated GPU**.
- $B_{\mathrm{afd}}$: resident requests **per attention GPU**.
- $T_{\mathrm{col}},T_{\mathrm{afd}}$: wall-clock latency of a complete decode step, across all layers and microbatches, at each system's own batch size.

Define $\mathrm{tput}$ as output-token throughput per GPU, measured in **output tokens/s/GPU**. Since each decode step generates one token per active request, **per-GPU throughput is the global batch size divided by the total decode-step latency, then divided by the total number of allocated GPUs**, including FFN GPUs:

$$
\mathrm{tput}
=\frac{\text{global batch size}}{\text{total decode-step latency (s)}}
\cdot\frac{1}{\text{total GPU count}}.
$$

The global batch counts requests across all request-hosting GPUs: $N \cdot B_{\mathrm{col}}$ for colocated serving and $N_A \cdot B_{\mathrm{afd}}$ for AFD. Substituting these batches, with step latencies expressed in seconds:

$$
\mathrm{tput}_{\mathrm{col}}=\frac{N \cdot B_{\mathrm{col}}}{N \cdot T_{\mathrm{col}}}=\frac{B_{\mathrm{col}}}{T_{\mathrm{col}}},\qquad
\mathrm{tput}_{\mathrm{afd}}=\frac{N_A \cdot B_{\mathrm{afd}}}{(N_A+N_F) \cdot T_{\mathrm{afd}}}.
$$

Therefore:

$$
\boxed{S=\frac{\mathrm{tput}_{\mathrm{afd}}}{\mathrm{tput}_{\mathrm{col}}}
=\begin{array}{ccccc}
\displaystyle\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}
&\cdot&\displaystyle\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}
&\cdot&\displaystyle\frac{N_A}{N}\\
\textbf{batch expansion}&&\textbf{step-latency ratio}&&\textbf{attention GPU fraction}
\end{array}}
$$

Let’s break down these ratios:

- **$r=\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}$ — batch expansion:** Moving expert weights off attention GPUs frees memory for KV cache, allowing more resident requests per attention GPU.
- **$\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}$ — step-latency ratio:** **Aggregation** means combining routed tokens from multiple attention GPUs into larger batches for each FFN expert, which can make its matrix multiplications more efficient. Overlap lets attention on one microbatch run while another uses the FFN pool. These mechanisms can reduce AFD step latency, while larger batches and repeated microbatch work can increase it.
- **$p=\frac{N_A}{N}$ — attention GPU fraction:** This factor counts the FFN GPUs in the total GPU budget even though they host no requests; for a 6:2 split, $p=6/8=0.75$.

Both latencies depend on batch, context, placement, and kernels. AFD wins only when $T_{\mathrm{afd}}<r \cdot p \cdot T_{\mathrm{col}}$; memory capacity alone does not establish a speedup.

## 2. Deriving the batch advantage

Use request-parallel attention (TP=1) with expert weights sharded across all eight colocated GPUs, or across $N_F$ AFD GPUs. Non-expert weights are replicated on request-hosting GPUs. Each request runs attention on one GPU and keeps its KV cache there. The memory formulas below assume this placement.

Let $H$ be usable HBM capacity per GPU, $E$ the total offloaded expert-weight bytes, and $K(CL)$ the per-request cache/state bytes at context length $CL$.

**We neglect non-expert weight memory, including shared experts, relative to HBM capacity.** This simplifies capacity estimates; their execution time remains part of the latency model. Assuming the colocated expert shard fits, $E/N<H$:

$$
B_{\mathrm{col}} \cdot K(CL)+E/N\le H,
\qquad B_{\mathrm{afd}} \cdot K(CL)\le H,
$$

$$
B_{\mathrm{col}}^{\max}\approx\frac{H-E/N}{K(CL)},\qquad
B_{\mathrm{afd}}^{\max}\approx\frac{H}{K(CL)}.
$$

Round these estimates down to whole requests for concrete capacities. There is **no extra division of $B_{\mathrm{col}}$ by $N$**: $B_{\mathrm{col}}$ is already per GPU. FFN placement must separately satisfy $E/N_F+R_F\le H$, where $R_F$ includes FFN buffers and other resident state. Choose $H$ after reserving memory for request-side runtime buffers; reduce it if larger batches require more workspace.

Ignoring integer rounding, the batch expansion is:

$$
r\approx\frac{H}{H-E/N}.
$$

Longer context lowers both capacity-limited batches through $K(CL)$. In this simplified equal-cache-layout model, $K(CL)$ cancels from their ratio: **$r$ need not grow with context**. Admission limits, different cache layouts, buffer growth, or rounding can change it. Actual batches may be below these memory ceilings.

## 3. Concrete capacity examples

### DeepSeek-V4.1-Flash

The [official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) specifies 40 backbone layers, hidden size 5,120, expert intermediate size 2,304, 384 routed experts per layer, and FP4 expert weights. Each SwiGLU expert has three matrices, giving:

$$
P_{\rm routed}=40 \cdot 384 \cdot 3 \cdot 5120 \cdot 2304
=543{,}581{,}798{,}400,
$$

or **271.79 GB of raw four-bit routed weights**, before scales and padding. Keep the shared expert on the attention side in this example.

The [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) reports 890 bytes per token for the global KV cache and a separate 196B-parameter Engram component. The [technical report](https://arxiv.org/html/2609.19969v1) describes host-memory Engram prefetch and bounded sliding-window KV storage. Do not treat the headline backbone parameter count as the full resident memory footprint.

Use these **explicit sizing assumptions**, rather than claiming a measured allocation:

| Input | Assumption |
|---|---|
| Usable HBM $H$ | 244.8 GB/GPU, assuming 85% of the physical 288 GB is usable |
| Routed weights $E$ | 290 GB including an assumed allowance for scales/padding |
| Non-expert weight memory | Neglected, including shared experts; actual request capacity will be lower |
| Engram | Host-resident in both layouts; host capacity and lookup throughput must be provisioned |
| Request state | $K(CL)=890 \cdot CL+4 \cdot 2^{20}$ bytes; 4 MiB is an assumed SWA/metadata allowance |
| AFD placement | $N_A=6$, $N_F=2$; text decode, speculation disabled |

The resulting approximate cache budgets are $244.8-290/8=208.55$ GB for colocated GPUs and $244.8$ GB for attention GPUs:

| Context tokens | Cache/state MB per request | $B_{\mathrm{col}}^{\max}$ | $B_{\mathrm{afd}}^{\max}$ | $B_{\mathrm{afd}}/B_{\mathrm{col}}$ |
|---:|---:|---:|---:|---:|
| 32,768 | 33.36 | 6,251 | 7,338 | 1.174 |
| 131,072 | 120.85 | 1,725 | 2,025 | 1.174 |
| 1,048,576 | 937.43 | 222 | 261 | 1.176 |

These are optimistic memory ceilings, not recommended serving batches. At 128K, total capacity is **13,800 colocated requests versus 12,150 AFD requests**: the 17.4% per-attention-GPU advantage is outweighed by dedicating two GPUs to FFN. AFD needs shorter steps to compensate. Two FFN GPUs each hold 145 GB of routed weights, leaving 99.8 GB within the usable HBM budget for other FFN allocations; one FFN GPU cannot fit the assumed 290 GB.

This example assumes host Engram access remains covered by the runtime. If it stalls, include that delay in step latency. DeepSeek's sparse attention also means stored KV bytes are not automatically bytes read on every step.

### MiMo-V2.6-Flash

Use the official **MiMo-V2.6-Flash-RL** checkpoint as the reference. Its [model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL#4-model-architecture) specifies 9 global-attention layers and 39 sliding-window layers. Global attention has 4 KV heads; sliding-window attention has 8. Keys have 192 channels and values 128; the sliding window retains 128 tokens. The [configuration](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/main/config.json) specifies 47 MoE layers, 256 routed experts per layer, hidden size 4,096, and expert intermediate size 2,048.

Keep $N=8$, $H=244.8$ GB/GPU, attention TP=1, and the 6:2 AFD split. Assume **FP8 KV cache and FP8-resident routed weights**, text decode, and no speculative decoder. The [vLLM recipe](https://recipes.vllm.ai/XiaomiMiMo/MiMo-V2.6-Flash-RL) describes FP8 computation with MXFP4 checkpoint storage and uses FP8 KV cache in its AMD example. The checkpoint's disk size is not the assumed resident weight size here.

Routed expert parameters and the memory budget are:

$$
P_{\rm routed}=47 \cdot 256 \cdot 3 \cdot 4096 \cdot 2048
=302{,}795{,}194{,}368
$$

At one byte per FP8 weight, this is 302.80 GB before scales. Use $E=303$ GB, rounding up to allow for block scales. The cache budgets are then $H-E/N=206.925$ GB for colocated GPUs and $H=244.8$ GB for attention GPUs.

With one byte per cached element, the per-request KV bytes are:

$$
K(CL)=9 \cdot 4 \cdot (192+128) \cdot CL
+39 \cdot 8 \cdot (192+128) \cdot \min(CL,128)
$$

For the contexts below, $K(CL)=11{,}520 \cdot CL+12{,}779{,}520$ bytes. The second term is the bounded sliding-window cache; it does not grow with context.

| Context tokens | Cache/state MB per request | $B_{\mathrm{col}}^{\max}$ | $B_{\mathrm{afd}}^{\max}$ | $B_{\mathrm{afd}}/B_{\mathrm{col}}$ |
|---:|---:|---:|---:|---:|
| 32,768 | 390.27 | 530 | 627 | 1.183 |
| 131,072 | 1,522.73 | 135 | 160 | 1.185 |
| 1,048,576 | 12,092.38 | 17 | 20 | 1.176 |

These are optimistic capacities: non-expert weights, per-request metadata, allocator padding, and extra retained KV blocks are omitted. The cache formula assumes separate 192-channel keys and 128-channel values, with expired sliding-window KV reclaimed. A backend that pads values or retains more history needs a larger $K(CL)$. FP8 cache scale overhead is also omitted. BF16 KV cache doubles the raw cache bytes; retaining packed MXFP4 experts instead changes $E$ and requires recalculation. Two FFN GPUs hold about 151.5 GB of expert weights each; fitting weights does not establish sufficient FFN throughput.

### Frontier-model comparison: DeepSeek V4, Kimi K3, and Qwen3.8

Source configurations checked **2026-10-04**. Keep the same eight MI355X GPUs, $H=244.8$ GB/GPU, request-parallel attention with TP=1, and speculation disabled. Compare the same resident expert precision in both layouts. Include GPT-OSS-120B and the source study's MiniMax-M2.5 and Qwen3-235B-A22B alongside the frontier models. The Qwen3.8 cases cover the large open text model, Flash-Next, and a separate dense-27B comparison below; multimodal encoders and generation are outside this analysis.

First compare the **capacity ratios and latency needed to win**. These do not require assuming a particular KV-cache implementation: $K(CL)$ cancels before rounding. Absolute batches still require each backend's cache/state bytes, including recurrent state for hybrid models. The cache budgets below can be substituted into $B^{\max}=\lfloor\text{cache bytes}/K(CL)\rfloor$ at any supported context. They are not measured request capacities.

#### Expert-memory inputs

For each model, multiply MoE layers, routed experts per layer, three expert matrices, expert input width, and intermediate width. Exclude shared experts and speculative layers, consistently with the earlier examples.

| Model and official configuration | Routed parameter calculation | Routed parameters, billions | Resident expert assumption | $E$, GB |
|---|---|---:|---|---:|
| [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | $36 \cdot 128 \cdot 3 \cdot 2880 \cdot 2880$ | 114.662 | MXFP4 + block scales + BF16 expert biases, rounded | 61.00 |
| [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/config.json) | $62 \cdot 256 \cdot 3 \cdot 3072 \cdot 1536$ | 224.680 | FP8 + assumed FP32 block scales | 224.74 |
| [Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8/blob/main/config.json) | $94 \cdot 128 \cdot 3 \cdot 4096 \cdot 1536$ | 227.096 | FP8 + assumed FP32 block scales | 227.15 |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) | $43 \cdot 256 \cdot 3 \cdot 4096 \cdot 2048$ | 277.025 | FP4 + block scales | 147.17 |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) | $61 \cdot 384 \cdot 3 \cdot 7168 \cdot 3072$ | 1,547.396 | FP4 + block scales | 822.05 |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) | $92 \cdot 896 \cdot 3 \cdot 3584 \cdot 3072$ | 2,722.741 | MXFP4 + block scales | 1,446.46 |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) | $48 \cdot 512 \cdot 3 \cdot 2560 \cdot 640$ | 120.796 | FP8 + rounded scale allowance / BF16 | 121.00 / 241.59 |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json) | $92 \cdot 512 \cdot 3 \cdot 8192 \cdot 2048$ | 2,370.822 | BF16 / FP8 before scales / hypothetical MXFP4 | 4,741.64 / 2,370.82 / 1,259.50 |

DeepSeek's [reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py) stores two FP4 weights per byte plus one one-byte scale per 32 weights. K3's configuration also specifies four-bit weights with one-byte scales per group of 32. Thus these estimates use $0.5+1/32=0.53125$ bytes per routed parameter, before backend padding or duplicate weight copies. K3's expert input width is its **3,584-dimensional latent MoE width**, not the 7,168-dimensional residual stream; its first layer is dense.

For GPT-OSS-120B, assume experts remain packed in MXFP4 at runtime. Its [weight decoder](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py) uses 32-weight blocks with one-byte scales, giving 60.914 GB for the expert matrices. The [expert bias shapes](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py) add $36\cdot128\cdot(2\cdot2880+2880)\cdot2=0.0796$ GB in BF16; round the total to $E=61$ GB. This is a packed-resident sizing assumption, not the allocation of the PyTorch reference, which expands expert matrices to BF16. The colocated comparison keeps experts sharded across all eight GPUs, as in the other MoE rows; eight independent full-model replicas would require different memory and latency accounting.

MiniMax-M2.5 and Qwen3-235B-A22B-FP8 are the two checkpoints evaluated in the source study. Both configurations specify FP8 weights with $128\times128$ blocks. Assume a four-byte scale per block, giving $1+4/(128\cdot128)$ bytes per routed parameter: $E=224.735330304$ GB and $227.151839232$ GB respectively, before backend padding. Calculations use these unrounded values; MiniMax's MTP layers are excluded. **These rows recalculate capacity for our eight-MI355X, colocated EP=8 setup.** The source study instead measured a 1.5 batch ratio on GB200 with a colocated EP=4 baseline and four GPUs per FFN node. Its batch ratio and measured throughput gains do not transfer directly to this table.

For Flash-Next, the [official FP8 configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/main/config.json) uses $128\times128$ weight blocks. FP32 block scales would add about 0.029 GB to its 120.796 GB of routed weights; $E=121$ GB covers that allowance. Assume its additional **51B n-gram embedding is host-resident in both layouts**, as supported by the [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next). Include host-lookup stalls in latency if they are exposed.

The large Qwen's [official FP8 version](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) already exceeds the node's **1,958.4 GB usable HBM in routed weights alone**. Neither BF16 nor FP8 fits this setup. Its MXFP4 row is a **conditional sizing exercise requiring quantization and kernel/quality validation**, not a claim about the official checkpoint or an available MI355X deployment.

#### Ratios and break-even latency

Choose the smallest FFN pool that fits routed weights alone:

$$
N_F^{\min}=\left\lceil\frac{E}{H}\right\rceil,\qquad
r\approx\frac{H}{H-E/8},\qquad
p=\frac{8-N_F}{8}.
$$

This is a **memory lower bound on $N_F$**, not a recommended split. FFN workspace, routing imbalance, supported expert partitions, and compute throughput can require more GPUs. In particular, a six-GPU pool does not evenly divide 896 or 512 experts; it needs a supported uneven or padded placement. All ratios below retain the simplified omission of non-expert weights; **K3 fails that approximation**, as quantified after the tables.

| Model / resident expert format | Colocated cache budget, GB/GPU | $r=B_{\mathrm{afd}}/B_{\mathrm{col}}$ | Minimum $N_A:N_F$ | $p$ | $r \cdot p$ | Step-latency reduction needed to win |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash, earlier 290 GB assumption | 208.55 | 1.174 | 6:2 | 0.750 | 0.880 | **>12.0%** |
| MiMo-V2.6-Flash, FP8 | 206.93 | 1.183 | 6:2 | 0.750 | 0.887 | **>11.3%** |
| GPT-OSS-120B, MXFP4 | 237.18 | 1.032 | 7:1 | 0.875 | 0.903 | **>9.7%** |
| MiniMax-M2.5, FP8 | 216.71 | 1.130 | 7:1 | 0.875 | 0.988 | **>1.2%** |
| Qwen3-235B-A22B, FP8 | 216.41 | 1.131 | 7:1 | 0.875 | 0.990 | **>1.0%** |
| DeepSeek-V4-Flash, FP4 | 226.40 | 1.081 | 7:1 | 0.875 | 0.946 | **>5.4%** |
| DeepSeek-V4-Pro, FP4 | 142.04 | 1.723 | 4:4 | 0.500 | 0.862 | **>13.8%** |
| Kimi K3, MXFP4 — formal only; TP=1 baseline does not fit | 63.99 | 3.825 | 2:6 | 0.250 | 0.956 | >4.4%, **not actionable for this layout** |
| Qwen3.8-Flash-Next, FP8 | 229.68 | 1.066 | 7:1 | 0.875 | 0.933 | **>6.7%** |
| Qwen3.8-Flash-Next, BF16 | 214.60 | 1.141 | 7:1 | 0.875 | 0.998 | **>0.2%**, with very little FFN workspace |
| Qwen3.8-2.4T-A95B, hypothetical MXFP4 | 87.36 | 2.802 | 2:6 | 0.250 | 0.701 | **>29.9%**, conditional on the omitted allocations fitting |

Every attention GPU has 244.8 GB of cache budget **under this approximation**. The last column is $100\cdot(1-r\cdot p)$, rounded; use the unrounded condition $T_{\mathrm{afd}}/T_{\mathrm{col}}<r\cdot p$ at the boundary. These are context-independent ratios before whole-request rounding, provided both systems use the same cache layout and reach their memory ceilings. Longer context changes absolute batches and stage timings, so it can still change the actual speedup substantially.

Adding one FFN GPU makes the required latency reduction much larger:

| Model / format | Minimum split | Space left per FFN GPU after routed weights, GB | Split with one extra FFN GPU | Required step-latency reduction at that split |
|---|---:|---:|---:|---:|
| GPT-OSS-120B, MXFP4 | 7:1 | 183.80 | 6:2 | >22.6% |
| MiniMax-M2.5, FP8 | 7:1 | 20.06 | 6:2 | >15.3% |
| Qwen3-235B-A22B, FP8 | 7:1 | 17.65 | 6:2 | >15.2% |
| DeepSeek-V4-Flash, FP4 | 7:1 | 97.63 | 6:2 | >18.9% |
| DeepSeek-V4-Pro, FP4 | 4:4 | 39.29 | 3:5 | >35.4% |
| Kimi K3, MXFP4 — formal only | 2:6 | 3.72 | 1:7 | >52.2% |
| Qwen3.8-Flash-Next, FP8 | 7:1 | 123.80 | 6:2 | >20.1% |
| Qwen3.8-Flash-Next, BF16 | 7:1 | 3.21 | 6:2 | >14.4% |
| Qwen3.8-2.4T-A95B, hypothetical MXFP4 | 2:6 | 34.88 | 1:7 | >65.0% |

These residual bytes are within $H$, after the same 15% physical-HBM reserve used throughout; they are available for additional FFN allocations, not proof that those allocations fit. Precision changes both systems: for example, FP8 makes the Flash-Next FFN pool much easier to fit, but also frees memory in the colocated baseline and lowers $r$.

#### Where the simplification fails for K3

K3 demonstrates why a large formal ratio is insufficient. Its [reference model](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py) has 69 KDA layers, each with full-width Q, K, V, output, and output-gate projections. Those five matrices alone require $69\cdot5\cdot7168\cdot(96\cdot128)\cdot2=60.78$ GB in BF16. Its two shared experts per MoE layer add $92\cdot3\cdot7168\cdot(2\cdot3072)\cdot2=24.31$ GB. The checkpoint excludes attention and shared experts from MXFP4 quantization.

At attention TP=1, each colocated GPU would therefore need at least **180.81 + 60.78 + 24.31 = 265.90 GB**, exceeding $H=244.8$ GB before MLA weights, embeddings, KV cache, or recurrent state. **The 3.825 ratio cannot describe a feasible K3 baseline under these assumptions.** K3 needs different attention sharding, non-expert precision, placement, or a larger GPU budget; then both batch definitions and memory accounting must be recalculated. This also means the formal 4.4% threshold is not evidence that K3 is the best AFD candidate. Other large hybrid models, particularly Qwen 2.4T, also need an explicit non-expert allocation audit before their simplified ratios are used for deployment.

#### Is there a chance to win?

Under the simplified MoE assumptions, allocating FFN GPUs cannot by itself increase the node's total request capacity. Ignoring rounding:

$$
r\cdot p=\frac{8\cdot H-N_F\cdot H}{8\cdot H-E}\le1,
\qquad\text{because }N_F\cdot H\ge E.
$$

Colocated serving already leaves $8\cdot H-E$ bytes for request caches across the node. AFD leaves $(8-N_F)\cdot H$ on attention GPUs; spare FFN memory is unavailable to requests in this placement. **A win therefore requires faster decode steps**, through more efficient expert execution and overlap. This conclusion specifically assumes sharded routed weights and omitted non-expert weights; replicated dense FFNs differ, as discussed below.

- **GPT-OSS-120B:** At 7:1, its 3.2% per-attention-GPU batch gain requires 9.7% shorter steps to offset the FFN GPU. One FFN GPU has ample weight capacity, but must sustain requests from seven attention GPUs. At 6:2, the required reduction rises to 22.6%. Compared with V4-Flash at the same 7:1 split, its smaller batch ratio makes the latency target harder.
- **MiniMax-M2.5 and Qwen3-235B-A22B, FP8:** Their minimum 7:1 splits have small latency hurdles: 1.2% and 1.0%, respectively. Each FFN GPU retains almost the entire usable weight budget, leaving about 20.06 GB and 17.65 GB for additional allocations. These are attractive capacity thresholds if one FFN GPU can fit the runtime and keep up; at 6:2 the required reductions rise to 15.3% and 15.2%. The source study's measured wins justify investigation, but do not validate these MI355X splits.
- **DeepSeek-V4-Flash:** A useful first candidate under this model: 7:1 needs only a 5.4% step reduction. The main question is whether one GPU can serve the experts for seven attention GPUs without exposing FFN time. If two FFN GPUs are needed, the target becomes 18.9%.
- **Qwen3.8-Flash-Next, FP8:** Another useful candidate: 7:1 needs 6.7% shorter steps and has substantial expert-side memory headroom. Measure recurrent-attention work, host n-gram lookup, and FFN service capacity. The BF16 row's 0.2% target is fragile because only 3.21 GB remains for FFN allocations.
- **DeepSeek-V4-Pro:** Worth profiling if the colocated step spends a large fraction in routed MoE. Its 72.3% per-attention-GPU batch expansion still needs 13.8% shorter steps with half the GPUs assigned to FFN. The larger batch also raises attention-side work.
- **Kimi K3:** Rework the placement before evaluating speedup. The memory-omission assumption fails for TP=1, and adding a seventh FFN GPU would sharply reduce request-hosting capacity even in the formal model.
- **Qwen3.8-2.4T-A95B:** Official BF16/FP8 weights rule out this eight-GPU setup. Even a hypothetical MXFP4 expert layout requires about 30% shorter steps at 2:6, plus a successful audit of non-expert allocations. It is a more demanding candidate for this node budget.

The latency hurdle is necessary, not sufficient. In Section 4's hidden-path approximation, winning also requires:

$$
\frac{T_{\rm att}}{T_{\mathrm{col}}}
+\frac{m}{r}\cdot\frac{T_{\rm dense}}{T_{\mathrm{col}}}<p.
$$

For example, with two microbatches and dense work equal to 10% of the colocated step, the removable MoE fraction must exceed about **21.0% for V4-Flash**, **21.3% for Flash-Next FP8**, or **51.6% for V4-Pro** at their minimum splits. These are hypothetical profile thresholds, not measured model timings. FFN and communication must also satisfy Section 5's exposed-stage bound; fewer FFN GPUs may improve the capacity accounting while making overlap impossible.

#### Dense Qwen3.8-27B is a different comparison

The [27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) uses dense FFNs: $64\cdot3\cdot5120\cdot17408=17.113$ billion FFN parameters, or 34.23 GB in BF16. With attention TP=1, colocated GPUs each retain a **full copy** of those FFNs, so their memory term is $E$, not $E/8$. If an AFD FFN GPU retains that same full copy:

$$
r\approx\frac{244.8}{244.8-34.22552064}=1.163,\qquad
r\cdot p\approx1.017\quad\text{at }7:1.
$$

This can increase total request capacity slightly by removing replicated weights, unlike the sharded-MoE cases. It says nothing about whether one FFN GPU can sustain the dense computation of seven attention GPUs. Additional FFN replicas or tensor sharding change that balance; dense AFD needs its own measured execution model.

Run `python3 capacity_comparison.py` to reproduce the MoE memory, ratio, and split-sensitivity tables. All figures are analytical; no AFD throughput measurements or MI355X kernel qualifications were performed.

## 4. Speedup when FFN and communication are hidden

Decompose the measured colocated step at $B_{\mathrm{col}}$:

$$
T_{\mathrm{col}}=T_{\rm att}+T_{\rm dense}+T_{\rm MoE}.
$$

Here $T_{\rm MoE}$ covers only the expert path that will move to FFN GPUs, including its dispatch/combine. $T_{\rm dense}$ collects remaining request-side work, including shared-expert execution if retained there. With $m$ microbatches, the blog's approximation is:

$$
A\equiv T_{\mathrm{afd}}^{\rm hidden}\approx r \cdot T_{\rm att}+m \cdot T_{\rm dense},
$$

$$
\boxed{S_{\rm hidden}\approx
r \cdot p \cdot
\frac{T_{\rm att}+T_{\rm dense}+T_{\rm MoE}}
{r \cdot T_{\rm att}+m \cdot T_{\rm dense}}}
$$

This assumes attention time scales with batch, small-kernel costs scale with microbatch count, and the remote path stays overlapped. The source study established this approximation on its GB200 workloads; **MI355X and V4.1 Flash require their own calibration**, especially for sparse attention and Engram.

Using the DeepSeek example's $r=244.8/208.55\approx1.17382$, the following are **hypothetical timing scenarios**, not benchmark results. Each baseline step is 100 ms; hidden-path feasibility is assumed in each row.

| Scenario | $N_A:N_F$ (GPUs) | $m$ | Baseline att/dense/MoE (ms) | $T_{\mathrm{afd}}^{\rm hidden}$ (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| Large removable MoE cost | 6:2 | 2 | 40 / 10 / 50 | 66.95 | **1.315×** |
| Attention dominates | 6:2 | 2 | 80 / 10 / 10 | 113.91 | **0.773×** |
| Dense overhead is substantial | 6:2 | 2 | 20 / 30 / 50 | 83.48 | **1.055×** |
| More GPUs dedicated to FFN | 4:4 | 2 | 40 / 10 / 50 | 66.95 | **0.877×** |
| Extra microbatches without extra hiding | 6:2 | 4 | 40 / 10 / 50 | 86.95 | **1.012×** |

At 6:2, $r \cdot p\approx0.8804$, so AFD must reduce step latency by more than **12.0%** just to break even. More FFN GPUs can make overlap feasible, but reduce the fraction hosting requests.

## 5. When FFN or communication becomes exposed

For a simplified uniform pipeline, let $F,D,R$ be whole-step FFN, dispatch, and return service times, respectively—not sums of GPU-seconds. A useful lower-bound approximation is:

$$
T_{\mathrm{afd}}\gtrsim\max\left(A,F,D,R,\frac{A+D+F+R}{m}\right),
\qquad S=r \cdot p \cdot \frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}.
$$

The last term covers microbatch dependency cycles. Add fill/drain, synchronization, and imbalance delays for a practical estimate. This extension assumes independent stage resources and no double-counting of fused work; if both transfer directions serialize on one resource, also include its $D+R$ service constraint.

The attention-limited conditions are $F,D,R\le A$ and $D+F+R\le(m-1) \cdot A$. Thus two microbatches are sufficient only if the other stages fit under one attention interval in this simplified model. $F,D,R$ depend on the global batch $N_A \cdot B_{\mathrm{afd}}$, the FFN count, microbatch size, routing, and the effective interconnect. More microbatches cannot remove an FFN throughput bottleneck.

For the first scenario, if $F=110$ ms and $D=R=2$ ms, the bound rises to 110 ms and the corresponding optimistic speedup falls to **0.800×** before pipeline overhead. Use measured stage timings or a schedule simulation to evaluate the exposed regime; these equations do not establish that a particular MI355X placement achieves overlap.
