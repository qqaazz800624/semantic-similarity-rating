"""Compare an SSR app CSV export against OpinionQA human distributions.

Metrics follow the SSR paper (Maier et al., arXiv:2510.08338):

- KS similarity per question: 1 - max|CDF_sim - CDF_human| over the 5-point
  scale. The paper reports KS similarity > 0.85 as "realistic distributions".
- Mean-score Pearson correlation across questions (the paper's correlation-
  attainment metric divides this by the human test-retest ceiling; without
  test-retest data we report the raw correlation).
- Mean absolute error of the per-question mean scores, as a plain-language
  companion number.

Usage:
  python compare_results.py ssr_results.csv
  python compare_results.py ssr_results.csv --reference opinionqa_human_reference.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCORES = np.arange(1, 6)


def ks_similarity(p: np.ndarray, q: np.ndarray) -> float:
    return 1.0 - float(np.max(np.abs(np.cumsum(p) - np.cumsum(q))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", help="CSV exported from the SSR web app")
    parser.add_argument("--reference",
                        default=str(Path(__file__).parent / "opinionqa_human_reference.csv"))
    args = parser.parse_args()

    sim = pd.read_csv(args.results_csv)
    ref = pd.read_csv(args.reference)

    pcols = [f"p_score_{s}" for s in SCORES]
    hcols = [f"human_p{s}" for s in SCORES]

    missing = [c for c in ["question", *pcols] if c not in sim.columns]
    if missing:
        raise SystemExit(f"Results CSV is missing columns {missing} — "
                         "is this the file exported by the app's 下載結果 CSV button?")

    # Aggregate per question: mean of per-respondent PMFs (the app's survey PMF),
    # skipping failed respondents (NaN probability rows).
    agg = sim.dropna(subset=pcols).groupby("question", sort=False)[pcols].mean()
    n_resp = sim.dropna(subset=pcols).groupby("question", sort=False).size()

    rows = []
    for _, r in ref.iterrows():
        q = r["question"]
        if q not in agg.index:
            print(f"!! not found in results, skipped: {q[:70]}")
            continue
        p_sim = agg.loc[q, pcols].to_numpy(dtype=float)
        p_sim = p_sim / p_sim.sum()
        p_hum = r[hcols].to_numpy(dtype=float)
        rows.append({
            "question": q,
            "wave": r["wave"],
            "n_sim": int(n_resp.loc[q]),
            "sim_mean": float((p_sim * SCORES).sum()),
            "human_mean": float((p_hum * SCORES).sum()),
            "ks_similarity": ks_similarity(p_sim, p_hum),
        })

    if not rows:
        raise SystemExit("No overlapping questions between results and reference.")

    df = pd.DataFrame(rows)
    df["mean_abs_err"] = (df["sim_mean"] - df["human_mean"]).abs()

    pd.set_option("display.width", 160)
    print("\nPer-question comparison")
    print("=" * 100)
    out = df.copy()
    out["question"] = out["question"].str.slice(0, 60)
    print(out.round(3).to_string(index=False))

    print("\nOverall metrics (SSR-paper style)")
    print("=" * 100)
    print(f"  Questions compared:            {len(df)}")
    print(f"  Mean KS similarity:            {df['ks_similarity'].mean():.3f}"
          f"   (paper benchmark: > 0.85)")
    print(f"  Min  KS similarity:            {df['ks_similarity'].min():.3f}")
    if len(df) >= 3:
        r = float(np.corrcoef(df["sim_mean"], df["human_mean"])[0, 1])
        print(f"  Mean-score Pearson r:          {r:.3f}")
    print(f"  Mean |Δ mean score|:           {df['mean_abs_err'].mean():.3f} (on the 1-5 scale)")


if __name__ == "__main__":
    main()
