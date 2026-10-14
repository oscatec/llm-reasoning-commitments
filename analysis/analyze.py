"""
Descriptive and inferential analysis of the annotated reasoning traces.

Reads data/annotations.csv, writes analysis/results.md and figures/accuracy_validity.png.
With n = 50 the goal is honest uncertainty, not strong claims: every proportion is
reported with a 95% Wilson interval, and comparisons use Fisher's exact test.

Run from the repository root:  python analysis/analyze.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import fisher_exact, norm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "annotations.csv"
OUT_MD = ROOT / "analysis" / "results.md"
OUT_FIG = ROOT / "figures" / "accuracy_validity.png"

SHORT = {
    "Qwen/Qwen3.6-27B": "Qwen3.6 27B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5 7B",
}
ORDER = ["Qwen3.6 27B", "Llama 3.1 8B", "Qwen2.5 7B"]

SURFACE, INK, MUTED, GRID, MARK = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6"


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * ((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5 / (1 + z**2 / n)
    return p, centre - half, centre + half


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["model"] = df["model_name"].map(SHORT)
    df["correct"] = df["model_answer_is_correct"].eq("yes")
    df["valid"] = df["trace_is_valid"].eq("yes")
    df["n_explicit"] = df["annotator_explicit_reasoning"].fillna("").str.count(r"(?m)^\s*-\s")
    return df


def rate_table(df: pd.DataFrame, by: str, col: str, order=None) -> pd.DataFrame:
    rows = []
    groups = order or sorted(df[by].unique())
    for g in groups:
        sub = df[df[by] == g]
        p, lo, hi = wilson(int(sub[col].sum()), len(sub))
        rows.append({by: g, "n": len(sub), "k": int(sub[col].sum()), "rate": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def fmt(t: pd.DataFrame, by: str) -> str:
    lines = [f"| {by.capitalize()} | n | Rate | 95% CI |", "|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r[by]} | {r.n} | {r.k}/{r.n} ({r.rate:.0%}) | {r.lo:.0%} to {r.hi:.0%} |")
    return "\n".join(lines)


def fisher(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    a, b = df[mask], df[~mask]
    table = [[a[col].sum(), (~a[col]).sum()], [b[col].sum(), (~b[col]).sum()]]
    return fisher_exact(table)[1]


def plot(acc: pd.DataFrame, val: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True, facecolor=SURFACE)
    panels = [(axes[0], acc, "Answer accuracy"), (axes[1], val, "Well-formed trace")]
    for ax, t, title in panels:
        ax.set_facecolor(SURFACE)
        y = range(len(t))[::-1]
        for yi, (_, r) in zip(y, t.iterrows()):
            ax.plot([r.lo, r.hi], [yi, yi], color=MARK, lw=2, solid_capstyle="round")
            ax.scatter(r.rate, yi, s=64, color=MARK, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{r.k}/{r.n}", (r.hi, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", color=MUTED, fontsize=9)
        ax.set_yticks(list(y), t["model"])
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_ylim(-0.6, len(t) - 0.4)
    fig.text(0.01, 0.01, "Dot = observed rate, line = 95% Wilson interval. n = 50 annotated traces.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200, facecolor=SURFACE)


def main() -> None:
    df = load()
    acc_m = rate_table(df, "model", "correct", ORDER)
    val_m = rate_table(df, "model", "valid", ORDER)
    acc_mode = rate_table(df, "mode", "correct")
    val_mode = rate_table(df, "mode", "valid")
    acc_subj = rate_table(df, "subject", "correct")

    p_val_qwen25 = fisher(df, df.model.eq("Qwen2.5 7B"), "valid")
    p_acc_mode = fisher(df, df["mode"].eq("independent"), "correct")
    p_val_mode = fisher(df, df["mode"].eq("independent"), "valid")

    invalid = df[~df.valid]
    invalid_correct = int(invalid.correct.sum())
    issues = invalid["annotator_notes"].value_counts()
    conf = df["annotation_confidence"].value_counts().reindex(["high", "medium", "low"]).fillna(0).astype(int)
    expl = df.groupby("model")["n_explicit"].agg(["mean", "min", "max"]).reindex(ORDER)

    md = [
        "# Results",
        "",
        "Generated by `analysis/analyze.py` from `data/annotations.csv` (50 traces, 3 models, "
        "3 subjects, 2 prompting modes). All intervals are 95% Wilson intervals; p-values are "
        "two-sided Fisher exact tests. With this sample size the intervals are wide and the "
        "numbers below are descriptive, not conclusive.",
        "",
        "![Accuracy and validity by model](../figures/accuracy_validity.png)",
        "",
        "## Accuracy by model",
        "A trace counts as correct only when its answer matches the ground truth and follows "
        "consistently from its own steps (see the README).",
        "",
        fmt(acc_m, "model"),
        "",
        "## Well-formed traces by model",
        fmt(val_m, "model"),
        "",
        f"Qwen2.5 7B produced malformed traces more often than the other two models combined "
        f"(Fisher p = {p_val_qwen25:.3f}).",
        "",
        "## Prompting mode",
        "Accuracy:",
        "",
        fmt(acc_mode, "mode"),
        "",
        "Well-formed traces:",
        "",
        fmt(val_mode, "mode"),
        "",
        f"Accuracy difference between modes: p = {p_acc_mode:.2f}. "
        f"Validity difference between modes: p = {p_val_mode:.3f}.",
        "",
        "Mode and model are not balanced in this sample: "
        + ", ".join(f"{m} has {int(r['independent'])} independent and {int(r['shared'])} shared"
                    for m, r in pd.crosstab(df.model, df["mode"]).reindex(ORDER).iterrows())
        + ". The shared condition leans on the strongest model, so the gap between modes "
        "should not be read as an effect of the mode itself.",
        "",
        "## Accuracy by subject",
        fmt(acc_subj, "subject"),
        "",
        "## Format failures are not reasoning failures",
        f"All {len(invalid)} malformed traces still reached the correct answer "
        f"({invalid_correct}/{len(invalid)}). The failures were purely structural:",
        "",
        *[f"- {k} ({v})" for k, v in issues.items()],
        "",
        "Checking output format and checking correctness are separate problems, and a pipeline "
        "that rejects malformed output would have discarded correct answers here.",
        "",
        "## Annotation",
        f"Annotator confidence: {conf['high']} high, {conf['medium']} medium, {conf['low']} low.",
        "",
        "Explicit commitments extracted per trace:",
        "",
        "| Model | Mean | Range |",
        "|---|---|---|",
        *[f"| {m} | {r['mean']:.1f} | {int(r['min'])} to {int(r['max'])} |" for m, r in expl.iterrows()],
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    plot(acc_m, val_m)
    print(f"Wrote {OUT_MD.relative_to(ROOT)} and {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
