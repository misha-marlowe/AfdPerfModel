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
