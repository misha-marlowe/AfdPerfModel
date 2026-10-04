"""Reproduce Section 3's analytical ratios; no measured performance inputs.

Run: python3 capacity_comparison.py
Dimensions, precision assumptions, and source links are in perf_model.md.
GB is decimal. Non-expert weights and FFN buffers are omitted in this screen.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import ceil


H = Fraction(2448, 10) * 10**9
N = 8
MXFP4_BYTES = Fraction(1, 2) + Fraction(1, 32)


@dataclass(frozen=True)
class Model:
    name: str
    expert_bytes: Fraction


def routed_bytes(layers, experts, hidden, intermediate, bytes_per_weight):
    return layers * experts * 3 * hidden * intermediate * Fraction(bytes_per_weight)


MODELS = [
    Model("DeepSeek-V4.1-Flash (existing assumption)", Fraction(290 * 10**9)),
    Model("MiMo-V2.6-Flash (FP8)", Fraction(303 * 10**9)),
    Model("DeepSeek-V4-Flash (FP4)", routed_bytes(43, 256, 4096, 2048, MXFP4_BYTES)),
    Model("DeepSeek-V4-Pro (FP4)", routed_bytes(61, 384, 7168, 3072, MXFP4_BYTES)),
    Model("Kimi K3 (MXFP4; formal only)", routed_bytes(92, 896, 3584, 3072, MXFP4_BYTES)),
    Model("Qwen3.8-Flash-Next (FP8; rounded)", Fraction(121 * 10**9)),
    Model("Qwen3.8-Flash-Next (BF16)", routed_bytes(48, 512, 2560, 640, 2)),
    Model("Qwen3.8-2.4T (hypothetical MXFP4)", routed_bytes(92, 512, 8192, 2048, MXFP4_BYTES)),
    Model("Qwen3.8-2.4T (FP8; before scales)", routed_bytes(92, 512, 8192, 2048, 1)),
    Model("Qwen3.8-2.4T (BF16)", routed_bytes(92, 512, 8192, 2048, 2)),
]


def main():
    print("| Model | E GB | Col cache GB | r | A:F | p | r · p | Required step reduction |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        e = model.expert_bytes
        nf = ceil(e / H)
        if e >= N * H or nf >= N:
            print(f"| {model.name} | {float(e / 10**9):,.2f} | — | — | No fit | — | — | — |")
            continue
        budget = H - e / N
        r = H / budget
        p = Fraction(N - nf, N)
        print(f"| {model.name} | {float(e / 10**9):,.2f} | {float(budget / 10**9):,.2f} "
              f"| {float(r):.3f} | {N-nf}:{nf} | {float(p):.3f} | {float(r*p):.3f} "
              f"| >{float((1-r*p)*100):.1f}% |")

    print("\n| Model | Minimum A:F | FFN GB remaining/GPU | One more FFN GPU | Required step reduction |")
    print("|---|---:|---:|---:|---:|")
    for model in MODELS:
        e = model.expert_bytes
        nf = ceil(e / H)
        if nf >= N - 1:
            continue
        r = H / (H - e / N)
        next_p = Fraction(N - nf - 1, N)
        print(f"| {model.name} | {N-nf}:{nf} | {float((H-e/nf)/10**9):.2f} "
              f"| {N-nf-1}:{nf+1} | >{float((1-r*next_p)*100):.1f}% |")


if __name__ == "__main__":
    main()
