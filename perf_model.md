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
\underbrace{\displaystyle\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}}
&\cdot&\underbrace{\displaystyle\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}}
&\cdot&\underbrace{\displaystyle\frac{N_A}{N}}\\
\textbf{batch expansion}&&\textbf{step-latency ratio}&&\textbf{attention GPU fraction}
\end{array}}
$$

Let’s break down these ratios:

- **$r=\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}$ — batch expansion:** Moving expert weights off attention GPUs frees memory for KV cache, allowing more resident requests per attention GPU.
- **$\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}$ — step-latency ratio:** **Aggregation** means combining routed tokens from multiple attention GPUs into larger batches for each FFN expert, which can make its matrix multiplications more efficient. Overlap lets attention on one microbatch run while another uses the FFN pool. These mechanisms can reduce AFD step latency, while larger batches and repeated microbatch work can increase it.
- **$p=\frac{N_A}{N}$ — attention GPU fraction:** This factor counts the FFN GPUs in the total GPU budget even though they host no requests; for a 6:2 split, $p=6/8=0.75$.

Both latencies depend on batch, context, placement, and kernels. AFD wins only when $T_{\mathrm{afd}}<r \cdot p \cdot T_{\mathrm{col}}$; memory capacity alone does not establish a speedup.

## 2. Deriving the batch advantage

Use request-parallel attention (TP=1). The eight colocated GPUs form two independent groups, each with DP=4 and EP=4, so each GPU stores $E/4$ expert bytes. AFD instead shards one expert-weight copy across $N_F$ GPUs. Non-expert weights are replicated on request-hosting GPUs. Each request runs attention on one GPU and keeps its KV cache there.

Let $H$ be usable HBM capacity per GPU, $E$ the total offloaded expert-weight bytes, and $K(CL)$ the per-request cache/state bytes at context length $CL$.

**We neglect non-expert weight memory, including shared experts, relative to HBM capacity.** This simplifies capacity estimates; their execution time remains part of the latency model. Assuming the colocated expert shard fits, $E/4<H$:

$$
B_{\mathrm{col}} \cdot K(CL)+E/4\le H,
\qquad B_{\mathrm{afd}} \cdot K(CL)\le H,
$$

$$
B_{\mathrm{col}}^{\max}\approx\frac{H-E/4}{K(CL)},\qquad
B_{\mathrm{afd}}^{\max}\approx\frac{H}{K(CL)}.
$$

Round these estimates down to whole requests for concrete capacities. There is **no extra division of $B_{\mathrm{col}}$ by $N$**: $B_{\mathrm{col}}$ is already per GPU. FFN placement must separately satisfy $E/N_F+R_F\le H$, where $R_F$ includes FFN buffers and other resident state. Choose $H$ after reserving memory for request-side runtime buffers; reduce it if larger batches require more workspace.

Ignoring integer rounding, the batch expansion is:

$$
r\approx\frac{H}{H-E/4}.
$$

Longer context lowers both capacity-limited batches through $K(CL)$. In this simplified equal-cache-layout model, $K(CL)$ cancels from their ratio: **$r$ need not grow with context**. Admission limits, different cache layouts, buffer growth, or rounding can change it. Actual batches may be below these memory ceilings.

## 3. Concrete capacity examples

Use **Qwen3-235B-A22B-FP8**, one of the blog's models, on our eight MI355X GPUs. The colocated baseline consists of **two independent DP=4, EP=4 groups**, with attention TP=1. Each group holds one complete set of expert weights, sharded four ways.

The [official configuration](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8/blob/main/config.json) gives 94 MoE layers, 128 experts per layer, hidden size 4,096, and expert intermediate size 1,536. With three matrices per expert, one byte per FP8 weight, and an assumed four-byte scale per $128\times128$ block:

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

The batch ratio and GPU allocation give the factor $r\cdot p$. **Total speedup also includes the decode-step latency ratio**, so keep that term explicit:

$$
S=r\cdot p\cdot\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}.
$$

| AFD attention:FFN GPUs | Batch ratio $r$ | Attention fraction $p$ | Projected speedup including step latency |
|---|---:|---:|---:|
| 7:1 | 1.30 | 7/8 | $1.14\cdot T_{\mathrm{col}}/T_{\mathrm{afd}}$ |
| 6:2 | 1.30 | 6/8 | $0.98\cdot T_{\mathrm{col}}/T_{\mathrm{afd}}$ |

At 6:2, AFD hosts requests on six GPUs instead of eight. Its total request count relative to colocated serving is:

$$
\frac{6\cdot B_{\mathrm{afd}}}{8\cdot B_{\mathrm{col}}}
=\frac{6}{8}\cdot r\approx97.65\%
$$

To match throughput, AFD must finish each step in 97.65% of the time: **$100\%-97.65\%=2.35\%$ shorter steps**. For a 100 ms colocated step, AFD breaks even at 97.65 ms and wins below that.

Actual speedup depends on $T_{\mathrm{col}}/T_{\mathrm{afd}}$. The next section estimates this ratio when AFD hides MoE and communication.

## 4. The step-latency ratio when MoE and communication are hidden

AFD overlaps one microbatch's MoE and communication with another microbatch's attention. When this overlap fully hides the FFN path, that work no longer adds to the decode-step latency.

Start with the colocated step at $B_{\mathrm{col}}$:

$$
T_{\mathrm{col}}=T_{\rm att}+T_{\rm dense}+T_{\rm MoE}.
$$

- $T_{\rm att}$: attention-kernel time.
- $T_{\rm dense}$: remaining request-side work, including projections, routing, norms, KV updates, and any retained shared experts.
- $T_{\rm MoE}$: the expert path moved to FFN GPUs, including dispatch and combine.

With $m$ microbatches, estimate the remaining attention-side time:

$$
T_{\mathrm{afd}}^{\rm hidden}\approx r\cdot T_{\rm att}+m\cdot T_{\rm dense}.
$$

The factor $r$ accounts for the larger batch: this approximation assumes attention time grows in proportion to KV traffic. The factor $m$ accounts for repeating the small request-side kernels for each microbatch. Thus the latency ratio is:

$$
\boxed{\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}^{\rm hidden}}
\approx\frac{T_{\rm att}+T_{\rm dense}+T_{\rm MoE}}
{r\cdot T_{\rm att}+m\cdot T_{\rm dense}}}
$$

**Example:** keep the Qwen 6:2 split, $r\approx1.30$, and use two microbatches. Suppose a colocated step takes $40+10+50=100$ ms for attention, dense work, and MoE respectively. If MoE and communication are fully hidden:

$$
T_{\mathrm{afd}}^{\rm hidden}\approx r\cdot40+2\cdot10\approx72.08\text{ ms},
$$

$$
\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}^{\rm hidden}}
\approx\frac{100}{72.08}\approx1.39,
\qquad
S\approx r\cdot p\cdot\frac{100}{72.08}\approx1.35.
$$

The latency improvement turns the 6:2 split's 0.98 request-count factor into a **1.35× throughput speedup**. These timings are hypothetical; the approximation needs calibration on MI355X. AFD shortens steps when the hidden MoE cost outweighs the extra attention and microbatch work.
