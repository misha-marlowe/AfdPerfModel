"""Reproduce Section 3's Qwen3-235B-A22B-FP8 projection.

Eight MI355X GPUs; colocated serving uses two independent DP=4/EP=4
groups with attention TP=1. Assume equal decode-step latency in AFD.
Dimensions and source links are in perf_model.md. GB is decimal.
Non-expert weights, backend padding, and whole-request rounding are omitted.
Run: python3 capacity_comparison.py
"""

from fractions import Fraction


H = Fraction(2448, 10) * 10**9
N = 8
EP_COL = 4
FP8_BLOCK_BYTES = 1 + Fraction(4, 128 * 128)  # Assumed FP32 block scales.
EXPERT_BYTES = 94 * 128 * 3 * 4096 * 1536 * FP8_BLOCK_BYTES


def main():
    col_cache = H - EXPERT_BYTES / EP_COL
    r = H / col_cache
    print(f"Expert weights: {float(EXPERT_BYTES / 10**9):.9f} GB")
    print(f"Colocated cache per GPU: {float(col_cache / 10**9):.9f} GB")
    print(f"AFD cache per attention GPU: {float(H / 10**9):.1f} GB\n")
    print("| AFD attention:FFN GPUs | Batch ratio r | Attention fraction p | Projected speedup at equal step latency |")
    print("|---|---:|---:|---:|")
    for nf in (1, 2):
        p = Fraction(N - nf, N)
        speedup = r * p
        print(f"| {N-nf}:{nf} | {float(r):.3f} | {N-nf}/{N} "
              f"| {float(speedup):.3f}× ({float((speedup-1)*100):+.1f}%) |")


if __name__ == "__main__":
    main()
