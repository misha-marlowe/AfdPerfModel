"""Reproduce Section 3's Qwen3-235B-A22B-FP8 projection.

Eight MI355X GPUs; colocated serving uses two independent DP=4/EP=4
groups with attention TP=1. Keep the step-latency ratio explicit.
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
    print("Calculations retain full precision; displayed numbers have at most two decimal places.")
    print(f"Expert weights: {float(EXPERT_BYTES / 10**9):.2f} GB")
    print(f"Colocated cache per GPU: {float(col_cache / 10**9):.2f} GB")
    print(f"AFD cache per attention GPU: {float(H / 10**9):.1f} GB\n")
    print("| AFD attention:FFN GPUs | Batch ratio r | Attention fraction p | Projected speedup including step latency |")
    print("|---|---:|---:|---:|")
    for nf in (1, 2):
        p = Fraction(N - nf, N)
        capacity_factor = r * p
        print(f"| {N-nf}:{nf} | {float(r):.2f} | {N-nf}/{N} "
              f"| {float(capacity_factor):.2f} · T_col / T_afd |")
    factor_6_2 = r * Fraction(6, 8)
    print(f"\n6:2 total request ratio: r · 6/8 ≈ {float(factor_6_2*100):.2f}%")
    print(f"At S = 1, T_afd / T_col = r · 6/8 ≈ {float(factor_6_2*100):.2f}%")
    print(f"Break-even step reduction: (1 - r · 6/8) · 100% ≈ 100% - {float(factor_6_2*100):.2f}% = {float((1-factor_6_2)*100):.2f}%")
    print(f"For a 100 ms colocated step, AFD breaks even at {float(factor_6_2*100):.2f} ms.")
    print(f"6:2 speedup if AFD steps are 20% shorter: {float(factor_6_2 / Fraction(4, 5)):.2f}×")


if __name__ == "__main__":
    main()
