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

Use $H=244.8$ GB of usable HBM per GPU (85% of 288 GB), neglecting non-expert weights as in Section 2. A colocated GPU has $244.8-227.15/4\approx188.01$ GB for cache; an AFD attention GPU has 244.8 GB. Therefore:

$$
r=\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}
\approx\frac{244.8}{244.8-227.15/4}
\approx1.302.
$$

The batch ratio and GPU allocation give the factor $r\cdot p$. **Total speedup also includes the decode-step latency ratio**, so keep that term explicit:

$$
S=r\cdot p\cdot\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}.
$$

| AFD attention:FFN GPUs | Batch ratio $r$ | Attention fraction $p$ | Projected speedup including step latency |
|---|---:|---:|---:|
| 7:1 | 1.302 | 7/8 | $1.139\cdot T_{\mathrm{col}}/T_{\mathrm{afd}}$ |
| 6:2 | 1.302 | 6/8 | $0.977\cdot T_{\mathrm{col}}/T_{\mathrm{afd}}$ |

**For the 6:2 split, start with the total request counts.** Six AFD attention GPUs host requests, compared with all eight colocated GPUs. Using the unrounded expert bytes above gives $r\approx1.302044$, so the ratio of total resident requests is:

$$
\frac{6\cdot B_{\mathrm{afd}}}{8\cdot B_{\mathrm{col}}}
=\frac{6}{8}\cdot r
\approx0.75\cdot1.302044
\approx0.976533.
$$

AFD therefore generates about 97.6533% as many tokens per step. To match colocated throughput, it must finish each step in 97.6533% of the time. Set $S=1$ to find this break-even point:

$$
1=0.976533\cdot\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}
\quad\Longrightarrow\quad
\frac{T_{\mathrm{afd}}}{T_{\mathrm{col}}}=0.976533.
$$

The required percentage reduction in step latency is therefore:

$$
\frac{T_{\mathrm{col}}-T_{\mathrm{afd}}}{T_{\mathrm{col}}}\cdot100\%
=(1-0.976533)\cdot100\%
\approx2.35\%.
$$

For a 100 ms colocated step, AFD breaks even at about **97.65 ms**; shorter AFD steps produce a speedup. The table rounds the request-count factor to 0.977; the 2.35% calculation uses the unrounded value.

For example, if hiding MoE and communication makes AFD steps 20% shorter, $T_{\mathrm{afd}}=0.8\cdot T_{\mathrm{col}}$, giving $S\approx0.976533/0.8\approx1.221$ — a **22.1% speedup**. This is an illustrative latency assumption, not a measurement.

At 7:1, AFD benefits from storing one expert-weight copy instead of the colocated baseline's two copies. Its single FFN GPU has about $244.8-227.15=17.65$ GB left within $H$ for additional allocations. Actual speedup depends on the FFN pool fitting the runtime and on the resulting step latency, including exposed communication. The blog's reported 1.5 batch ratio comes from its GB200 setup, not this MI355X calculation.

Run `python3 capacity_comparison.py` to reproduce this table. The calculation uses unrounded expert bytes and ignores whole-request rounding.

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

This assumes attention time scales with batch, small-kernel costs scale with microbatch count, and the remote path stays overlapped. The source study established this approximation on its GB200 workloads; **MI355X requires its own calibration**.

Using the Qwen example's $r\approx1.30204$, the following are **hypothetical timing scenarios**, not benchmark results. Each baseline step is 100 ms; hidden-path feasibility is assumed in each row.

| Scenario | $N_A:N_F$ (GPUs) | $m$ | Baseline att/dense/MoE (ms) | $T_{\mathrm{afd}}^{\rm hidden}$ (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| Large removable MoE cost | 7:1 | 2 | 40 / 10 / 50 | 72.08 | **1.581×** |
| Attention dominates | 7:1 | 2 | 80 / 10 / 10 | 124.16 | **0.918×** |
| Dense overhead is substantial | 7:1 | 2 | 20 / 30 / 50 | 86.04 | **1.324×** |
| More GPUs dedicated to FFN | 6:2 | 2 | 40 / 10 / 50 | 72.08 | **1.355×** |
| Extra microbatches without extra hiding | 7:1 | 4 | 40 / 10 / 50 | 92.08 | **1.237×** |

At 7:1, $r\cdot p\approx1.1393$, so AFD wins while $T_{\mathrm{afd}}<1.1393\cdot T_{\mathrm{col}}$. At 6:2, the break-even latency reduction is $(1-0.976533)\cdot100\%\approx2.35\%$, as derived in Section 3. More FFN GPUs can make overlap feasible, but reduce the fraction hosting requests.

## 5. When FFN or communication becomes exposed

For a simplified uniform pipeline, let $F,D,R$ be whole-step FFN, dispatch, and return service times, respectively—not sums of GPU-seconds. A useful lower-bound approximation is:

$$
T_{\mathrm{afd}}\gtrsim\max\left(A,F,D,R,\frac{A+D+F+R}{m}\right),
\qquad S=r \cdot p \cdot \frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}.
$$

The last term covers microbatch dependency cycles. Add fill/drain, synchronization, and imbalance delays for a practical estimate. This extension assumes independent stage resources and no double-counting of fused work; if both transfer directions serialize on one resource, also include its $D+R$ service constraint.

The attention-limited conditions are $F,D,R\le A$ and $D+F+R\le(m-1) \cdot A$. Thus two microbatches are sufficient only if the other stages fit under one attention interval in this simplified model. $F,D,R$ depend on the global batch $N_A \cdot B_{\mathrm{afd}}$, the FFN count, microbatch size, routing, and the effective interconnect. More microbatches cannot remove an FFN throughput bottleneck.

For the first scenario, if $F=130$ ms and $D=R=2$ ms, the bound rises to 130 ms and the corresponding optimistic speedup falls to **0.876×** before pipeline overhead. Use measured stage timings or a schedule simulation to evaluate the exposed regime; these equations do not establish that a particular MI355X placement achieves overlap.
