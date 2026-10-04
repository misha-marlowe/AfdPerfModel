# AFD decode performance model: 8 × MI355X

This accounting follows [FastAFD](https://haoailab.com/blogs/fastafd/#where-the-speedup-comes-from).

This model compares colocated serving with attention–FFN disaggregation (AFD) on **eight MI355X GPUs**. It covers steady-state decode after prefill, with one output token per active request per step. The throughput estimates are analytical. Each MI355X has 288 GB of HBM; GB means $10^9$ bytes throughout. [AMD specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)

## 1. Throughput and sources of speedup

AFD separates attention from the mixture-of-experts (MoE) feed-forward network (FFN). Let $N=N_A+N_F=8$, where $N_A$ GPUs run attention and $N_F$ run the experts. The subscripts $\mathrm{col}$ and $\mathrm{afd}$ denote colocated serving and AFD. Define:

- $B_{\mathrm{col}}$: resident requests **per colocated GPU**.
- $B_{\mathrm{afd}}$: resident requests **per attention GPU**.
- $T_{\mathrm{col}},T_{\mathrm{afd}}$: latency of a complete decode step across all layers and microbatches, at each system's own batch size.

Define $\mathrm{tput}$ as output-token throughput per GPU, in **output tokens/s/GPU**. Each step generates one token per active request. Therefore, **per-GPU throughput is the global batch size divided by the decode-step latency and the total allocated GPU count**, including FFN GPUs:

$$
\mathrm{tput}
=\frac{\text{global batch size}}{\text{total decode-step latency (s)}}
\cdot\frac{1}{\text{total GPU count}}.
$$

The global batch is $N\cdot B_{\mathrm{col}}$ for colocated serving and $N_A\cdot B_{\mathrm{afd}}$ for AFD. With step latencies in seconds:

$$
\mathrm{tput}_{\mathrm{col}}=\frac{N \cdot B_{\mathrm{col}}}{N \cdot T_{\mathrm{col}}}=\frac{B_{\mathrm{col}}}{T_{\mathrm{col}}},\qquad
\mathrm{tput}_{\mathrm{afd}}=\frac{N_A \cdot B_{\mathrm{afd}}}{(N_A+N_F) \cdot T_{\mathrm{afd}}}.
$$

Therefore:

$$
\boxed{S=\frac{\mathrm{tput}_{\mathrm{afd}}}{\mathrm{tput}_{\mathrm{col}}}
=\begin{array}{ccccc}
\underbrace{\displaystyle\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}}
&\cdot&\underbrace{\displaystyle\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}}
&\cdot&\underbrace{\displaystyle\frac{N_A}{N}}\\
\scriptstyle\textbf{batch expansion}&&\scriptstyle\textbf{step-latency ratio}&&\scriptstyle\textbf{FFN GPU penalty}
\end{array}}
$$

Let’s break down these ratios:

- **$r=\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}$ — batch expansion:** Removing expert weights from attention GPUs frees memory for KV cache and increases the number of resident requests per attention GPU.
- **$\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}$ — step-latency ratio:** The FFN pool combines routed tokens from several attention GPUs into larger expert batches. This **aggregation** can improve expert matrix-multiplication efficiency. Microbatch overlap can hide expert execution and communication, but larger attention batches and repeated request-side work can increase step latency.
- **$p=\frac{N_A}{N}$ — FFN GPU penalty:** This attention GPU fraction reduces AFD's per-GPU throughput because FFN GPUs count toward the GPU budget but host no requests. For a 6:2 split, $p=6/8=0.75$: the allocation imposes a 25% penalty before accounting for batch expansion and step latency.

Both latencies depend on batch size, context length, GPU placement, and kernels. AFD has higher throughput when $T_{\mathrm{afd}}<r\cdot p\cdot T_{\mathrm{col}}$.

## 2. Deriving the batch advantage

Each GPU runs attention for its own requests and keeps their KV cache locally; attention tensor parallelism is **TP=1**. The colocated GPUs form two independent four-GPU groups. Each group has four request workers (**DP=4**, data parallelism) and distributes its expert weights across those same four GPUs (**EP=4**, expert parallelism). AFD distributes one expert-weight copy across $N_F$ GPUs.

Let $H$ be usable HBM capacity per GPU, $E$ the total expert-weight bytes in one model copy, and $K(CL)$ the KV-cache bytes per request at context length $CL$.

**For the memory calculation, neglect non-expert weights, including any shared experts.** Their execution time remains in the latency model. Each colocated GPU holds $E/4$ expert bytes, so, provided $E/4<H$:

$$
B_{\mathrm{col}} \cdot K(CL)+E/4\le H,
\qquad B_{\mathrm{afd}} \cdot K(CL)\le H,
$$

$$
B_{\mathrm{col}}^{\max}\approx\frac{H-E/4}{K(CL)},\qquad
B_{\mathrm{afd}}^{\max}\approx\frac{H}{K(CL)}.
$$

Round capacities down to whole requests. Both batches are already defined per request-hosting GPU. The FFN pool must also fit its weights and buffers: $E/N_F+R_F\le H$, where $R_F$ is the additional memory required per FFN GPU. Choose $H$ after reserving memory for request-side runtime buffers.

When both systems fill their available KV cache, the batch expansion is approximately:

$$
r\approx\frac{H}{H-E/4}.
$$

At the same context length and with the same cache layout, $K(CL)$ cancels from the ratio. Longer context reduces both batches without changing $r$, apart from rounding. If either system runs below its memory capacity, use its actual batch size.

## 3. Concrete capacity examples

Use **Qwen3-235B-A22B-FP8** with the two colocated DP=4/EP=4 groups defined above. Each group stores one complete expert-weight copy, divided among four GPUs.

The [official configuration](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8/blob/main/config.json) specifies 94 MoE layers, 128 experts per layer, hidden size 4,096, and expert intermediate size 1,536. Each expert has gate, up, and down projection matrices. With one byte per FP8 weight and an assumed four-byte scale per $128\times128$ block:

$$
E=94\cdot128\cdot3\cdot4096\cdot1536\cdot\left(1+\frac{4}{128\cdot128}\right)
\approx227.15\text{ GB}.
$$

Use $H=244.8$ GB of usable HBM per GPU (85% of 288 GB), neglecting non-expert weights as in Section 2. Calculations retain full precision; displayed numbers have at most two decimal places. A colocated GPU has $244.8-227.15/4\approx188.01$ GB for cache; an AFD attention GPU has 244.8 GB. Therefore:

$$
r=\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}
\approx\frac{244.8}{244.8-227.15/4}
\approx1.30.
$$

Multiply the batch and GPU-allocation factor $r\cdot p$ by the step-latency ratio to estimate throughput speedup:

$$
S=r\cdot p\cdot\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}.
$$

| AFD attention:FFN GPUs | Batch ratio $r$ | FFN GPU penalty $p$ | Throughput speedup $S$ |
|---|---:|---:|---:|
| 7:1 | 1.30 | 7/8 | $1.14\cdot T_{\mathrm{col}}/T_{\mathrm{afd}}$ |
| 6:2 | 1.30 | 6/8 | $0.98\cdot T_{\mathrm{col}}/T_{\mathrm{afd}}$ |

At 6:2, AFD hosts requests on six GPUs instead of eight. Its total request count relative to colocated serving is:

$$
\frac{6\cdot B_{\mathrm{afd}}}{8\cdot B_{\mathrm{col}}}
=\frac{6}{8}\cdot r\approx97.65\%
$$

To match colocated throughput, AFD needs **2.35% lower step latency** (100% − 97.65% = 2.35%). For a 100 ms colocated step, AFD matches throughput at 97.65 ms. Shorter AFD steps give higher throughput; longer steps give lower throughput.

Actual speedup depends on $T_{\mathrm{col}}/T_{\mathrm{afd}}$. The next section estimates this ratio when AFD hides MoE and communication.

## 4. The step-latency ratio when MoE and communication are hidden

AFD overlaps expert execution and communication for one microbatch with attention for another. If the FFN pool keeps up and communication is fully overlapped, the attention side determines the decode-step latency.

Decompose the colocated step at $B_{\mathrm{col}}$. Each term is the time across all model layers:

$$
T_{\mathrm{col}}=T_{\rm att}+T_{\rm dense}+T_{\rm MoE}.
$$

- $T_{\rm att}$: attention-kernel time.
- $T_{\rm dense}$: other work on request-hosting GPUs, including projections, routing, norms, and KV updates.
- $T_{\rm MoE}$: the expert path moved to FFN GPUs, including activation, quantization, dispatch, and combine.

Let $m$ be the number of microbatches per decode step. With the FFN path fully hidden, estimate:

$$
T_{\mathrm{afd}}^{\rm hidden}\approx r\cdot T_{\rm att}+m\cdot T_{\rm dense}.
$$

This approximation scales attention time by $r$ because the larger batch reads more KV data. It scales the remaining work by $m$ because the small kernels run once per microbatch. Both assumptions require calibration on MI355X. The resulting latency ratio is:

$$
\boxed{\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}^{\rm hidden}}
\approx\frac{T_{\rm att}+T_{\rm dense}+T_{\rm MoE}}
{r\cdot T_{\rm att}+m\cdot T_{\rm dense}}}
$$

**Illustrative example:** use the Qwen 6:2 split, $r\approx1.30$, and $m=2$. Suppose the colocated step spends 40 ms on attention, 10 ms on other request-side work, and 50 ms on the expert path, for 100 ms total. If the FFN path is fully hidden:

$$
T_{\mathrm{afd}}^{\rm hidden}\approx r\cdot40+2\cdot10\approx72.08\text{ ms},
$$

$$
\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}^{\rm hidden}}
\approx\frac{100}{72.08}\approx1.39,
\qquad
S\approx r\cdot p\cdot\frac{100}{72.08}\approx1.35.
$$

These hypothetical timings give a **1.35× throughput speedup** despite the 6:2 split's smaller global batch. The hidden expert work saves more time than the larger attention batch and microbatch overhead add. In the article's measured runs, step latency remained close to the colocated baseline; most of the gain came from batch expansion.
