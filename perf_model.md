# AFD decode performance model: 8 × MI355X

Compare colocated serving and attention–FFN disaggregation (AFD) on the **same eight GPUs**, after prefill, generating one token per active request per step. This is an analytical model, not a measured MI355X speedup. Each MI355X has 288 GB HBM; GB below means $10^9$ bytes. [AMD specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)

## 1. Throughput and sources of speedup

Let $N=N_A+N_F=8$, where $N_A$ GPUs run attention and $N_F$ run FFN/MoE. Define:

- $B_{\mathrm{col}}$: resident requests **per colocated GPU**.
- $B_{\mathrm{afd}}$: resident requests **per attention GPU**.
- $T_{\mathrm{col}},T_{\mathrm{afd}}$: wall-clock latency of a complete decode step, across all layers and microbatches, at each system's own batch size.

The global batches are $N \cdot B_{\mathrm{col}}$ and $N_A \cdot B_{\mathrm{afd}}$. Per-GPU throughput, counting every allocated GPU, is:

$$
tp_{\mathrm{col}}=\frac{N \cdot B_{\mathrm{col}}}{N \cdot T_{\mathrm{col}}}=\frac{B_{\mathrm{col}}}{T_{\mathrm{col}}},\qquad
tp_{\mathrm{afd}}=\frac{N_A \cdot B_{\mathrm{afd}}}{(N_A+N_F) \cdot T_{\mathrm{afd}}}.
$$

Therefore:

$$
\boxed{S=\frac{tp_{\mathrm{afd}}}{tp_{\mathrm{col}}}
=\underbrace{\frac{B_{\mathrm{afd}}}{B_{\mathrm{col}}}}_{r:\ \text{batch expansion}}
\cdot \underbrace{\frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}}_{\text{step-latency ratio}}
\cdot \underbrace{\frac{N_A}{N}}_{p:\ \text{attention GPU fraction}}.}
$$

Removing expert weights can increase $r$. Aggregation and overlap can improve the latency ratio, although larger batches and microbatch overhead increase attention-side time. The fraction $p$ charges AFD for FFN GPUs that host no requests. This accounting follows [FastAFD](https://haoailab.com/blogs/fastafd/#where-the-speedup-comes-from).

Both latencies depend on batch, context, placement, and kernels. AFD wins only when $T_{\mathrm{afd}}<r \cdot p \cdot T_{\mathrm{col}}$; memory capacity alone does not establish a speedup.

## 2. Deriving the batch advantage

Use request-parallel attention (TP=1) with expert weights sharded across all eight colocated GPUs, or across $N_F$ AFD GPUs. Non-expert weights are replicated on request-hosting GPUs. This makes the per-GPU batch definition consistent; an attention-TP layout needs different weight and KV accounting.

Let $U=\eta \cdot H$ be usable HBM, $W_{\mathrm{col}},W_{\mathrm{afd}}$ the resident non-offloaded weights per request-hosting GPU, $E$ the total offloaded expert-weight bytes, and $K(CL)$ the per-request cache/state bytes at context length $CL$. Then:

$$
B_{\mathrm{col}} \cdot K(CL)+W_{\mathrm{col}}+E/N\le U,
\qquad B_{\mathrm{afd}} \cdot K(CL)+W_{\mathrm{afd}}\le U,
$$

$$
B_{\mathrm{col}}^{\max}=\max\!\left(0,\left\lfloor\frac{U-W_{\mathrm{col}}-E/N}{K(CL)}\right\rfloor\right),\quad
B_{\mathrm{afd}}^{\max}=\max\!\left(0,\left\lfloor\frac{U-W_{\mathrm{afd}}}{K(CL)}\right\rfloor\right).
$$

There is **no extra division of $B_{\mathrm{col}}$ by $N$**: $B_{\mathrm{col}}$ is already per GPU. FFN placement must separately satisfy $E/N_F+R_F\le U$, where $R_F$ includes FFN buffers and other resident state. The reserve $(1-\eta) \cdot H$ must cover request-side runtime buffers; increase it if larger batches require more workspace.

With equal $W_{\mathrm{col}}=W_{\mathrm{afd}}=W$ and ignoring integer rounding:

$$
r\approx\frac{U-W}{U-W-E/N}.
$$

Longer context lowers both capacity-limited batches through $K(CL)$. In this simplified equal-cache-layout model, $K(CL)$ cancels from their ratio: **$r$ need not grow with context**. Admission limits, different cache layouts, buffer growth, or rounding can change it. Actual batches may be below these memory ceilings.

## 3. Concrete capacity example: DeepSeek-V4.1-Flash

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
| Usable HBM | $\eta=0.85$, so $U=244.8$ GB/GPU |
| Routed weights $E$ | 290 GB including an assumed allowance for scales/padding |
| Other resident weights $W$ | 20 GB/request-hosting GPU, including shared experts |
| Engram | Host-resident in both layouts; host capacity and lookup throughput must be provisioned |
| Request state | $K(CL)=890 \cdot CL+4 \cdot 2^{20}$ bytes; 4 MiB is an assumed SWA/metadata allowance |
| AFD placement | $N_A=6$, $N_F=2$; text decode, speculation disabled |

The resulting cache budgets are $244.8-20-290/8=188.55$ GB for colocated GPUs and $244.8-20=224.8$ GB for attention GPUs:

| Context tokens | Cache/state MB per request | $B_{\mathrm{col}}^{\max}$ | $B_{\mathrm{afd}}^{\max}$ | $B_{\mathrm{afd}}/B_{\mathrm{col}}$ |
|---:|---:|---:|---:|---:|
| 32,768 | 33.36 | 5,652 | 6,739 | 1.192 |
| 131,072 | 120.85 | 1,560 | 1,860 | 1.192 |
| 1,048,576 | 937.43 | 201 | 239 | 1.189 |

These are memory ceilings, not recommended serving batches. At 128K, total capacity is **12,480 colocated requests versus 11,160 AFD requests**: the 19.2% per-attention-GPU advantage is outweighed by dedicating two GPUs to FFN. AFD needs shorter steps to compensate. Two FFN GPUs each hold 145 GB of routed weights, leaving 99.8 GB within $U$ for other FFN allocations; one FFN GPU cannot fit the assumed 290 GB.

This example assumes host Engram access remains covered by the runtime. If it stalls, include that delay in step latency. DeepSeek's sparse attention also means stored KV bytes are not automatically bytes read on every step.

## 4. Speedup when FFN and communication are hidden

Decompose the measured colocated step at $B_{\mathrm{col}}$:

$$
T_{\mathrm{col}}=T_{\rm att}+T_{\rm dense}+T_{\rm MoE}.
$$

Here $T_{\rm MoE}$ covers only the expert path that will move to FFN GPUs, including its dispatch/combine. $T_{\rm dense}$ collects remaining request-side work, including shared-expert execution if retained there. With $m$ microbatches, the blog's approximation is:

$$
A\equiv T_{\mathrm{afd}}^{\rm hidden}\approx r \cdot T_{\rm att}+m \cdot T_{\rm dense},
\qquad
\boxed{S_{\rm hidden}\approx
\frac{r \cdot p \cdot (T_{\rm att}+T_{\rm dense}+T_{\rm MoE})}
{r \cdot T_{\rm att}+m \cdot T_{\rm dense}}.}
$$

This assumes attention time scales with batch, small-kernel costs scale with microbatch count, and the remote path stays overlapped. FastAFD established this approximation on its GB200 workloads; **MI355X and V4.1 Flash require their own calibration**, especially for sparse attention and Engram. [FastAFD latency model](https://haoailab.com/blogs/fastafd/#predicting-the-gb200-speedup)

Using $r=224.8/188.55\approx1.19226$, the following are **hypothetical timing scenarios**, not benchmark results. Each baseline step is 100 ms; hidden-path feasibility is assumed in each row.

| Scenario | $N_A:N_F$ (GPUs) | $m$ | Baseline att/dense/MoE (ms) | $T_{\mathrm{afd}}^{\rm hidden}$ (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| Large removable MoE cost | 6:2 | 2 | 40 / 10 / 50 | 67.69 | **1.321×** |
| Attention dominates | 6:2 | 2 | 80 / 10 / 10 | 115.38 | **0.775×** |
| Dense overhead is substantial | 6:2 | 2 | 20 / 30 / 50 | 83.85 | **1.066×** |
| More GPUs dedicated to FFN | 4:4 | 2 | 40 / 10 / 50 | 67.69 | **0.881×** |
| Extra microbatches without extra hiding | 6:2 | 4 | 40 / 10 / 50 | 87.69 | **1.020×** |

At 6:2, $r \cdot p\approx0.8942$, so AFD must reduce step latency by more than **10.6%** just to break even. More FFN GPUs can make overlap feasible, but reduce the fraction hosting requests.

## 5. When FFN or communication becomes exposed

For a simplified uniform pipeline, let $F,D,R$ be whole-step FFN, dispatch, and return service times, respectively—not sums of GPU-seconds. A useful lower-bound approximation is:

$$
T_{\mathrm{afd}}\gtrsim\max\left(A,F,D,R,\frac{A+D+F+R}{m}\right),
\qquad S=r \cdot p \cdot \frac{T_{\mathrm{col}}}{T_{\mathrm{afd}}}.
$$

The last term covers microbatch dependency cycles. Add fill/drain, synchronization, and imbalance delays for a practical estimate. This extension assumes independent stage resources and no double-counting of fused work; if both transfer directions serialize on one resource, also include its $D+R$ service constraint.

The attention-limited conditions are $F,D,R\le A$ and $D+F+R\le(m-1) \cdot A$. Thus two microbatches are sufficient only if the other stages fit under one attention interval in this simplified model. $F,D,R$ depend on the global batch $N_A \cdot B_{\mathrm{afd}}$, the FFN count, microbatch size, routing, and the effective interconnect. More microbatches cannot remove an FFN throughput bottleneck.

For the first scenario, if $F=110$ ms and $D=R=2$ ms, the bound rises to 110 ms and the corresponding optimistic speedup falls to **0.813×** before pipeline overhead. Use measured stage timings or a schedule simulation to evaluate the exposed regime; these equations do not establish that a particular MI355X placement achieves overlap.
