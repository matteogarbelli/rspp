"""Every figure and every experimental number in the paper, from the database.

Run:  python -m experiments.analysis.figures
Writes PDF/PNG into experiments/figures/ and LaTeX macros + tables into
submission_cor/.
"""
from __future__ import annotations

import json
import os
import re
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import indicators as ind
from . import load as L
from .style import ARM_COLOR, ARM_LABEL, ARM_ORDER, GREY, VERMILLION, BLUE, ORANGE, apply, save

FIGDIR = "experiments/figures"
TEXDIR = "submission_cor"
MACROS: dict[str, str] = {}


_DIGIT_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
               "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro(name, value):
    """Register a LaTeX macro. Command names may only contain letters, so
    digits in generated names are spelled out."""
    clean = "".join(_DIGIT_WORD.get(c, c) for c in name if c.isalnum())
    MACROS[clean] = str(value)


def pct(x, digits=1):
    """Percentage for the manuscript.  A negative value gets a typeset minus
    rather than a hyphen; \\ensuremath keeps it valid in text, tables and math."""
    s = f"{x:.{digits}f}\\%"
    return "\\ensuremath{-}" + s[1:] if s.startswith("-") else s


def _pop_size(rows):
    """Population size actually used, read from a run's recorded config."""
    for r in rows:
        cfg = r.get("config") or {}
        if cfg.get("pop_size"):
            return int(cfg["pop_size"])
    return None


# ------------------------------------------------------------------ Figure: RQ1
def fig_mechanism(df, rows):
    """Does the mechanism steer selection?

    Both panels use the front-index change, which is free of random-number
    effects: constrained non-dominated sorting is deterministic given the
    population.  Panel (a) is the per-run average, panel (b) resolves it over
    the course of the run.
    """
    d = df[(df.study == "main") & df.arm.isin(["sim_lambda", "corridor"])]
    arms = ["sim_lambda", "corridor"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    vals = [d[d.arm == a].rank_change.dropna().values * 100 for a in arms]
    bp = ax.boxplot(vals, positions=[0, 1], widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2), showfliers=False)
    for patch, a in zip(bp["boxes"], arms):
        patch.set_facecolor(ARM_COLOR[a]); patch.set_alpha(0.75)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arms):
        v = d[d.arm == a].rank_change.dropna().values * 100
        ax.scatter(np.full(len(v), i) + rng.uniform(-.13, .13, len(v)), v,
                   s=3, color="black", alpha=0.25, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Adaptive weight", "Corridor\n(proposed)"])
    ax.set_ylabel("individuals changing front (%)")
    ax.set_title("(a) Effect on the ranking, per run")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    lam = d[d.arm == "sim_lambda"].rank_change.dropna().values * 100

    ax = axes[1]
    for a in arms:
        runs = [r for r in rows if r["study"] == "main" and r["arm"] == a]
        if not runs:
            continue
        gens = [h["gen"] for h in runs[0]["history"]]
        M = np.array([[(h.get("rank_change") or 0.0) for h in r["history"]]
                      for r in runs if len(r["history"]) == len(gens)]) * 100
        lo, hi = np.percentile(M, [25, 75], axis=0)
        ax.fill_between(gens, lo, hi, color=ARM_COLOR[a], alpha=0.20, lw=0)
        ax.plot(gens, np.median(M, axis=0), color=ARM_COLOR[a], lw=1.5,
                label=ARM_LABEL[a])
    ax.set_xlabel("generation")
    ax.set_ylabel("individuals changing front (%)")
    ax.set_title("(b) Effect on the ranking, over the run")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_ylim(bottom=-0.6)

    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_mechanism")

    macro("LAMBDARANK", pct(lam.mean(), 1))

    # Per-*generation* detail of the weight's residue, computed from the results
    # database rather than from a sidecar file: the run-level mean above hides
    # how the residue is distributed, and a stale sidecar would describe a
    # different campaign than the one being reported.
    per_gen, pop = [], None
    for r in rows:
        if r["study"] != "main" or r["arm"] != "sim_lambda":
            continue
        for h in r.get("history", []):
            if h.get("rank_change") is not None:
                per_gen.append(h["rank_change"])
    if per_gen:
        v = np.array(per_gen, float)
        pop = int(round(2 * main_pop)) if (main_pop := _pop_size(rows)) else None
        nz = v[v > 0]
        macro("LAMBDANMEAS", f"{len(v):,}".replace(",", "{,}"))
        macro("LAMBDANNONZERO", len(nz))
        macro("LAMBDAPCTNONZERO", pct(100.0 * len(nz) / len(v), 2))
        # The complement, so prose can say "unchanged in X" without a reader
        # having to subtract from 100 in their head.
        macro("LAMBDAPCTNONZEROCOMP", pct(100.0 * (1 - len(nz) / len(v)), 2))
        macro("LAMBDAMEANPCT", pct(100.0 * v.mean(), 3))
        if pop:
            macro("LAMBDAPOP", pop)
            renum = np.rint(nz * pop).astype(int)
            if len(renum):
                macro("LAMBDAMEDRENUM", int(np.median(renum)))
                macro("LAMBDAMAXRENUM", int(renum.max()))
    cor = d[d.arm == "corridor"].rank_change.dropna().values * 100
    macro("CORRRANK", pct(np.median(cor), 1))
    macro("CORRRANKIQR", f"{np.percentile(cor,25):.1f}--{np.percentile(cor,75):.1f}\\%")
    lt = d[d.arm == "sim_lambda"].flip_treat.dropna().values * 100
    ln_ = d[d.arm == "sim_lambda"].flip_noise.dropna().values * 100
    macro("LAMBDAFLIP", pct(np.median(lt), 1))
    macro("LAMBDANOISE", pct(np.median(ln_), 1))
    ct = d[d.arm == "corridor"].flip_treat.dropna().values * 100
    cn = d[d.arm == "corridor"].flip_noise.dropna().values * 100
    macro("CORRFLIP", pct(np.median(ct), 1))
    macro("CORRNOISE", pct(np.median(cn), 1))
    macro("NRUNSMECH", len(lam))

    # The ten-decade sweep (eleven values), run in situ.  The single on/off contrast above
    # tests one lambda; Proposition 1 quantifies over all of them, so the worst
    # case across the sweep is the number that actually corresponds to the
    # proposition.
    sw = d[d.arm == "sim_lambda"].sweep_rank_change.dropna().values * 100
    if len(sw):
        macro("LAMBDASWEEPMEAN", pct(sw.mean(), 2))
        macro("LAMBDASWEEPMAX", pct(sw.max(), 2))
        macro("LAMBDASWEEPZERO",
              pct(100.0 * float(np.mean(sw <= 1e-9)), 1))


# ------------------------------------------------------------- Figure: RQ2
def tradeoff_instances():
    """Instances drawn in the trade-off figure, fixed by rule.

    The members of the sensitivity subset at the smallest and the largest size:
    one instance per disruption family in each row, every layout present.
    """
    size = lambda i: int(re.sub(r"^[A-Z]+", "", i.split("-")[0]))
    fam = lambda i: next(f for f in ("INS", "WDR", "SRG") if f in i)
    sizes = sorted({size(i) for i in SUBSET_PLOT})
    keep = [i for i in SUBSET_PLOT if size(i) in (sizes[0], sizes[-1])]
    return sorted(keep, key=lambda i: (size(i), ("INS", "WDR", "SRG").index(fam(i))))


def fig_tradeoff(df, insts):
    """Distance vs deviation, median with interquartile bars, on selected instances."""
    d = df[df.study == "main"]
    ncol = 3
    nrow = int(np.ceil(len(insts) / ncol))
    # Drawn at the width it is printed (the text block is about 5.4 in).
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4, 1.9 * nrow + 0.9),
                             squeeze=False)
    for ax, inst in zip(axes.ravel(), insts):
        sub = d[d.instance == inst]
        for a in ARM_ORDER:
            s = sub[sub.arm == a]
            if s.empty:
                continue
            if s.feasible.mean() < 0.5:
                # An arm that is mostly infeasible here is not drawn: its
                # deviation is not comparable, because a plan that violates
                # capacity is not a plan.  The caption states the omission.
                continue
            x, y = s[s.feasible].f1.values / 1000, s[s.feasible].f3.values / 1000
            ax.errorbar(np.median(x), np.median(y),
                        xerr=[[np.median(x) - np.percentile(x, 25)],
                              [np.percentile(x, 75) - np.median(x)]],
                        yerr=[[np.median(y) - np.percentile(y, 25)],
                              [np.percentile(y, 75) - np.median(y)]],
                        fmt="o", ms=5, color=ARM_COLOR[a], ecolor=ARM_COLOR[a],
                        elinewidth=1.0, capsize=2,
                        mfc=ARM_COLOR[a] if a == "corridor" else "white",
                        mew=1.4, zorder=5 if a == "corridor" else 3)
        # Margins keep markers at the extremes of the data inside the axes.
        ax.margins(x=0.10, y=0.12)
        ax.set_title(inst, fontsize=8, pad=3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.tick_params(labelsize=7.5, length=2.5, pad=1.5)
    for ax in axes.ravel()[len(insts):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("travel distance (km)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("deviation $S$ (km)", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=ARM_COLOR[a],
                          mfc=ARM_COLOR[a] if a == "corridor" else "white", mew=1.4,
                          label=ARM_LABEL[a]) for a in ARM_ORDER]
    # The legend identifies arms only; the caption explains the crosses.
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.5, handletextpad=0.3,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.12, 1, 1), h_pad=1.0, w_pad=0.8)
    save(fig, f"{FIGDIR}/fig_tradeoff")


def fig_violin(df):
    """Deviation and distance per arm, normalised within instance, pooled."""
    d = df[df.study == "main"].copy()
    for col in ("f1", "f3"):
        d[col + "n"] = d.groupby("instance")[col].transform(
            lambda s: (s - s.min()) / max(s.max() - s.min(), 1e-9))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, col, lab in ((axes[0], "f3n", "trajectory deviation $S$"),
                         (axes[1], "f1n", "travel distance $f_1$")):
        data = [d[d.arm == a][col].values for a in ARM_ORDER]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for pc, a in zip(parts["bodies"], ARM_ORDER):
            pc.set_facecolor(ARM_COLOR[a]); pc.set_alpha(0.7); pc.set_edgecolor("black")
            pc.set_linewidth(0.5)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("black"); parts[key].set_linewidth(0.8)
        ax.set_xticks(range(1, len(ARM_ORDER) + 1))
        ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], rotation=32,
                           ha="right", fontsize=7.5)
        ax.set_ylabel(f"{lab}\n(normalised within instance)")
    axes[0].set_title("(a) Trajectory deviation")
    axes[1].set_title("(b) Travel distance")
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_violin")


def _maximal_cliques(sorted_ranks, cd):
    """Maximal groups of consecutive ranks spanning at most the critical difference."""
    k = len(sorted_ranks)
    spans = []
    for i in range(k):
        j = i
        while j + 1 < k and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            spans.append((i, j))
    return [(a, b) for a, b in spans
            if not any(a2 <= a and b2 >= b and (a2, b2) != (a, b) for a2, b2 in spans)]


def fig_cd(rank_tables):
    """Critical-difference diagrams (Friedman + Nemenyi post-hoc, alpha = 0.05)."""
    n = len(rank_tables)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.15 * n), squeeze=False)
    for ax, (title, (arms, avg, p, cd)) in zip(axes.ravel(), rank_tables):
        k = len(arms)
        order = list(np.argsort(avg))
        srt = np.sort(avg)
        n_left = (k + 1) // 2
        cliques = _maximal_cliques(srt, cd)

        AXIS_Y = 0.60
        ax.set_xlim(0.5, k + 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        ax.text(0.5, 0.99, title, fontsize=9, fontweight="bold",
                va="top", ha="left", transform=ax.transData)

        ax.hlines(AXIS_Y, 1, k, color="black", lw=1.0)
        for r in range(1, k + 1):
            ax.vlines(r, AXIS_Y, AXIS_Y + 0.045, color="black", lw=0.8)
            ax.text(r, AXIS_Y + 0.075, str(r), ha="center", fontsize=7.5)
        # the critical-difference ruler, drawn clear of the axis
        ax.plot([1, 1 + cd], [AXIS_Y + 0.20, AXIS_Y + 0.20], color="black", lw=1.4)
        ax.vlines([1, 1 + cd], AXIS_Y + 0.175, AXIS_Y + 0.225, color="black", lw=1.0)

        for pos, idx in enumerate(order):
            left = pos < n_left
            step = pos if left else k - 1 - pos
            y = AXIS_Y - 0.075 - 0.088 * step
            x_end = 0.72 if left else k + 0.28
            colour = ARM_COLOR.get(arms[idx], GREY)
            ax.plot([avg[idx], avg[idx], x_end], [AXIS_Y, y, y], color=colour, lw=1.1)
            ax.text(x_end + (-0.08 if left else 0.08), y,
                    f"{ARM_LABEL.get(arms[idx], arms[idx])} ({avg[idx]:.2f})",
                    va="center", ha="right" if left else "left", fontsize=7.5,
                    color="black")

        for c, (a, b) in enumerate(cliques):
            ax.hlines(AXIS_Y - 0.028 - 0.022 * c, srt[a], srt[b],
                      color="black", lw=2.2, zorder=6)
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_cd")


# ------------------------------------------------------- Figure: corridor dynamics
def fig_corridor_dynamics(rows, instance):
    """The corridor closing onto the population, and the floor taking over."""
    runs = [r for r in rows if r["study"] == "main" and r["arm"] == "corridor"
            and r["instance"] == instance]
    if not runs:
        return
    gens = [h["gen"] for h in runs[0]["history"]]

    def stack(key):
        return np.array([[h[key] for h in r["history"]] for r in runs], float)

    def stack_opt(key):
        return np.array([[(h.get(key) if h.get(key) is not None else np.nan)
                          for h in r["history"]] for r in runs], float)

    tau = np.median(stack("tau"), 0) / 1000
    q10 = np.median(stack("S_q10"), 0) / 1000
    med = np.median(stack("S_med"), 0) / 1000
    q90 = np.median(stack("S_q90"), 0) / 1000

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.fill_between(gens, q10, q90, color=BLUE, alpha=0.22, lw=0,
                    label="population, 10–90th pct")
    ax.plot(gens, med, color=BLUE, lw=1.5, label="population, median")
    ax.plot(gens, tau, color=VERMILLION, lw=2.0, label=r"corridor width $\tau(t)$")

    # The generation at which the non-vacuity floor takes over from the
    # schedule, read from the flag the solver records rather than inferred from
    # where the width happens to cross a population percentile.  Those are
    # different events: Eq. (8) hands over when the scheduled width falls below
    # the 10th percentile of the pool, not when it meets the 90th.
    # Taken per seed and summarised by the median, like every curve drawn here.
    per_seed = []
    for r in runs:
        for h in r["history"]:
            if h.get("sched_active") is False:
                per_seed.append(h["gen"])
                break
    handover = int(round(float(np.median(per_seed)))) if per_seed else None
    # ... and the generation at which the corridor starts doing work, taken as
    # the first at which it re-ranks a measurable share of the pool.  The
    # obvious alternative -- the fraction of the population outside the corridor
    # -- is measured on the survivors, who are by construction the ones
    # selection has just kept inside it, so it reads near zero throughout and
    # says nothing about when the constraint began to bite.
    binds = None
    rc = np.nanmedian(stack_opt("rank_change"), 0) if runs else None
    if rc is not None:
        for i, g in enumerate(gens):
            if np.isfinite(rc[i]) and rc[i] > 0.01:
                binds = g
                break
    if handover is not None:
        ax.axvline(handover, color="black", ls="--", lw=0.9, ymax=0.68)
    ax.set_xlabel("generation")
    ax.set_ylabel("trajectory deviation (km)")
    ax.set_title("(a) Corridor width")
    # The handover line stops below this corner and the width curve never
    # enters it, so the legend overlaps no data.
    ax.legend(frameon=False, fontsize=6.2, handlelength=1.2, borderaxespad=0.1,
              loc="upper right", bbox_to_anchor=(1.0, 1.04))

    # Panels (b) and (c): the same instance seen per arm rather than per
    # generation-statistic.  They were a separate figure until the two were
    # merged; both always ran on the instance this function is given.
    for a in ["cost_only", "sim_obj", "sim_lambda", "corridor"]:
        aruns = [r for r in rows if r["study"] == "main" and r["arm"] == a
                 and r["instance"] == instance]
        if not aruns:
            continue
        agens = [h["gen"] for h in aruns[0]["history"]]
        S = np.array([[h["S_med"] for h in r["history"]] for r in aruns]) / 1000
        D = np.array([[h["dist_med"] for h in r["history"]] for r in aruns]) / 1000
        for bx, M, lab in ((axes[1], S, "median $S$ (km)"),
                           (axes[2], D, "median $f_1$ (km)")):
            lo, hi = np.percentile(M, [25, 75], axis=0)
            bx.fill_between(agens, lo, hi, color=ARM_COLOR[a], alpha=0.18, lw=0)
            bx.plot(agens, np.median(M, 0), color=ARM_COLOR[a], lw=1.4,
                    label=ARM_LABEL[a])
            bx.set_xlabel("generation"); bx.set_ylabel(lab)
    axes[1].set_title("(b) Trajectory deviation")
    axes[2].set_title("(c) Travel distance")
    axes[1].legend(frameon=False, fontsize=7)

    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_corridor_dynamics")
    macro("CORRINSTANCE", instance.replace("_", "\\_"))
    if handover is not None:
        macro("CORRHANDOVER", handover)
    if binds is not None:
        macro("CORRBINDS", binds)
    macro("CORRTAUZERO", f"{tau[0]:.0f}")
    macro("CORRSSTART", f"{med[0]:.0f}")
    macro("CORRTAUZERORATIO", f"{tau[0]/max(med[0], 1e-9):.1f}")


# ----------------------------------------------------------- Figures: RQ4 -- RQ7
def fig_metrics(df):
    """Which trajectory metric should drive the corridor?  Common yardstick."""
    d = df[df.study.str.startswith("metricv2")]
    if d.empty:
        print("  [skip] fig_metrics: no metricv2 study in the database")
        return
    metrics = ["frechet", "dtw", "hausdorff", "lcss"]
    labels = {"frechet": "Fréchet", "dtw": "DTW", "hausdorff": "Hausdorff",
              "lcss": "LCSS"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    dat, keep = [], []
    # Feasible runs only, matching the macros in the text.  Computing the figure
    # one way and the numbers beside it another is how a figure and its caption
    # come to disagree.
    for m in metrics:
        s = d[(d.study == f"metricv2:{m}") & d.feasible]
        if s.empty:
            continue
        g = s.groupby("instance").dev_frechet.median()
        base = df[(df.study == "main") & (df.arm == "cost_only")
                  & df.feasible].groupby("instance").f3.median()
        common = g.index.intersection(base.index)
        dat.append((100 * (1 - g[common] / base[common])).values)
        keep.append(m)
    bp = ax.boxplot(dat, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, m in zip(bp["boxes"], keep):
        patch.set_facecolor(VERMILLION if m == "frechet" else GREY)
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels([labels[m] for m in keep])
    ax.set_ylabel("reduction in Fréchet deviation\nvs cost-only (%)")
    ax.set_title("(a) Deviation reduction")

    ax = axes[1]
    dat2 = []
    for m in keep:
        s = d[(d.study == f"metricv2:{m}") & d.feasible]
        g = s.groupby("instance").f1.median()
        base = df[(df.study == "main") & (df.arm == "cost_only")
                  & df.feasible].groupby("instance").f1.median()
        common = g.index.intersection(base.index)
        dat2.append((100 * (g[common] / base[common] - 1)).values)
    bp = ax.boxplot(dat2, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.2))
    for patch, m in zip(bp["boxes"], keep):
        patch.set_facecolor(VERMILLION if m == "frechet" else GREY)
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels([labels[m] for m in keep])
    ax.set_ylabel("change in travel distance\nvs cost-only (%)")
    ax.set_title("(b) Travel distance change")
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_metrics")


# The sensitivity subset is defined once, by the campaign runner, and imported
# here.  Keeping a second copy in the analysis code is how a figure ends up
# drawn over different instances than the numbers beside it.
from ..campaign.run import SUBSET as SUBSET_PLOT


def _rel_points(df, sub, col="f3"):
    """Per-instance reduction in `col` relative to cost-only, in per cent.

    Pooling raw kilometres across instances of different size would swamp the
    effect being studied with the difference between instances; the relative
    form is also what the text reports.
    """
    base = df[(df.study == "main") & (df.arm == "cost_only") & df.feasible]
    out = []
    for i in SUBSET_PLOT:
        b = base[base.instance == i]
        c = sub[(sub.instance == i) & sub.feasible]
        if b.empty or c.empty:
            continue
        out.append(100 * (1 - c[col].median() / b[col].median()))
    return out


def _ablation_test(df, sub, ref, col="f3"):
    """Seed-paired comparison of an ablation arm against the default corridor.

    Every ablation claim in the paper -- schedule, calibration, resampling step,
    initialisation, trajectory metric -- goes through this.  An earlier version
    reported those comparisons as bare medians of per-instance point estimates,
    with no test, no interval and no effect size, in a paper whose protocol
    section promised all three.
    """
    pairs = []
    for i in SUBSET_PLOT:
        a = sub[(sub.instance == i) & sub.feasible]
        b = ref[(ref.instance == i) & ref.feasible]
        common = sorted(set(a.seed) & set(b.seed))
        if len(common) >= 5:
            pairs.append((a.set_index("seed").loc[common, col].values,
                          b.set_index("seed").loc[common, col].values))
    return ind.per_instance_comparison(pairs)


def fig_ablation(df):
    """Initialisation, tightening schedule, resampling step, and calibration."""
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7))
    main = df[df.study == "main"]

    def box(ax, series, labels, colours, title, xlabel=None):
        data = [d for d in series if d]
        keep = [(l, c) for d, l, c in zip(series, labels, colours) if d]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        for patch, (_, c) in zip(bp["boxes"], keep):
            patch.set_facecolor(c); patch.set_alpha(0.78)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xticks(range(1, len(keep) + 1))
        ax.set_xticklabels([l for l, _ in keep], fontsize=7.5, rotation=25,
                           ha="right", rotation_mode="anchor")
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)

    box(axes[0],
        [_rel_points(df, main[main.arm == "corridor"]),
         _rel_points(df, df[df.study == "init:random"])],
        ["grafted", "random"], [VERMILLION, GREY], "(a) Initialisation")
    axes[0].set_ylabel("reduction in $S$ vs cost-only (%)")

    box(axes[1],
        [_rel_points(df, main[main.arm == "corridor"]),
         _rel_points(df, df[df.study == "schedule:gaussian"]),
         _rel_points(df, main[main.arm == "corridor_fixed"])],
        ["linear", "Gaussian", "fixed"], [VERMILLION, BLUE, GREY],
        "(b) Schedule")

    ax = axes[2]
    xs, med, lo, hi, rt = [], [], [], [], []
    for st in (100, 250, 500, 1000):
        sub = (main[main.arm == "corridor"] if st == 250
               else df[df.study == f"resample:{st}"])
        pts = _rel_points(df, sub)
        if not pts:
            continue
        xs.append(st); med.append(np.median(pts))
        lo.append(np.median(pts) - np.percentile(pts, 25))
        hi.append(np.percentile(pts, 75) - np.median(pts))
        rt.append(sub.runtime.median())
    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o-", color=VERMILLION, capsize=3, ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("resampling step $h$ (m)", fontsize=8)
    ax.set_title("(c) Resampling step")

    ax = axes[3]
    xs, med, lo, hi, taus = [], [], [], [], []
    for q in (25, 50, 75, 90):
        sub = (main[main.arm == "corridor"] if q == 90
               else df[df.study == f"calib:q{q}"])
        pts = _rel_points(df, sub)
        if not pts:
            continue
        xs.append(q / 100); med.append(np.median(pts))
        lo.append(np.median(pts) - np.percentile(pts, 25))
        hi.append(np.percentile(pts, 75) - np.median(pts))
        taus.append(sub[sub.instance.isin(SUBSET_PLOT)].tau0.median() / 1000)
    ax.errorbar(xs, med, yerr=[lo, hi], fmt="o-", color=VERMILLION, capsize=3, ms=4)
    ax.set_xlabel("calibration quantile $q_0$", fontsize=8)
    ax.set_title("(d) Calibration")
    ax.set_xticks([0.25, 0.50, 0.75, 0.90])
    ax.set_xticklabels(["0.25", "0.50", "0.75", "0.90"], fontsize=7)

    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_ablation")


def fig_scalability(df):
    """Cost against instance size, and whether the generation budget suffices.

    Two panels, because run time alone cannot answer a scalability question: a
    fixed population and a fixed generation budget force the per-generation cost
    to be dominated by selection, which is independent of n, so a flat time
    curve is what the experimental design produces rather than a property of the
    method.  The second panel varies the budget and asks whether 300 generations
    is enough at every size.
    """
    d = df[df.study.isin(["main", "scale"])
           & df.arm.isin(["corridor", "cost_only", "corridor_moead"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    for a in ["cost_only", "corridor", "corridor_moead"]:
        s = d[d.arm == a]
        if s.empty:
            continue
        g = s.groupby("n_customers").runtime.median()
        ax.plot(g.index, g.values, "o-", color=ARM_COLOR[a], ms=4,
                label=ARM_LABEL[a], lw=1.2)
    s = d[d.arm == "corridor"]
    g = s.groupby("n_customers").runtime.median()
    if len(g) >= 3:
        x, y = np.log(g.index.values.astype(float)), np.log(g.values)
        k, logc = np.polyfit(x, y, 1)
        # A two-parameter fit over a handful of sizes deserves an interval.
        rng = np.random.default_rng(0)
        boots = [np.polyfit(x[i], y[i], 1)[0]
                 for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))
                 if len(np.unique(i)) > 1]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        xs = np.array([g.index.min(), g.index.max()], float)
        ax.plot(xs, np.exp(logc) * xs ** k, ":", color="black", lw=1.0,
                label=f"fit: $t \\propto n^{{{k:.2f}}}$")
        macro("SCALEEXP", f"{k:.2f}")
        macro("SCALEEXPCI", f"[{lo:.2f}, {hi:.2f}]")
        macro("SCALENSIZES", int(len(g)))
        macro("SCALENMAX", int(g.index.max()))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("requests $n$"); ax.set_ylabel("run time (s)")
    ax.set_title("(a) Cost against size")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    budgets, meds, los, his = [], [], [], []
    for g_ in (150, 300, 600):
        sub = (df[(df.study == "main") & (df.arm == "corridor")] if g_ == 300
               else df[df.study == f"budget:{g_}"])
        pts = _rel_points(df, sub)
        if not pts:
            continue
        budgets.append(g_); meds.append(np.median(pts))
        los.append(np.median(pts) - np.percentile(pts, 25))
        his.append(np.percentile(pts, 75) - np.median(pts))
    if budgets:
        ax.errorbar(budgets, meds, yerr=[los, his], fmt="o-", color=VERMILLION,
                    capsize=3, ms=4)
        ax.axvline(300, color="black", ls=":", lw=0.9)
    ax.set_xlabel("generation budget $G$")
    ax.set_ylabel("reduction in $S$ vs cost-only (%)")
    ax.set_title("(b) Generation budget")
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_scalability")


def fig_maps(rows, instance):
    """Baseline versus replanned geometry, corridor against cost-only."""
    from ..rspp.instance import RSPPPair
    path = f"data/rsppbench/{instance}.json"
    if not os.path.exists(path):
        return
    pair = RSPPPair.load(path)
    picks = {}
    for a in ("cost_only", "corridor"):
        cands = [r for r in rows if r["study"] == "main" and r["arm"] == a
                 and r["instance"] == instance and r.get("routes")]
        if cands:
            picks[a] = min(cands, key=lambda r: r["seed"])
    if len(picks) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
    base_c, dis_c = pair.base.coords, pair.disrupted.coords
    new_ids = set(pair.disruption.get("new_customers", []))
    for ax, (a, r) in zip(axes, [("cost_only", picks["cost_only"]),
                                 ("corridor", picks["corridor"])]):
        for k, route in enumerate(pair.baseline_routes):
            if not route:
                continue
            pts = base_c[[0] + list(route) + [0]]
            ax.plot(pts[:, 0] / 1000, pts[:, 1] / 1000, color=GREY, lw=1.6,
                    alpha=0.55, zorder=1)
        for k, route in enumerate(r["routes"]):
            if not route:
                continue
            pts = dis_c[[0] + list(route) + [0]]
            ax.plot(pts[:, 0] / 1000, pts[:, 1] / 1000, color=ARM_COLOR[a], lw=1.0,
                    zorder=3)
        ax.scatter(dis_c[1:, 0] / 1000, dis_c[1:, 1] / 1000, s=5, color="black",
                   zorder=4, lw=0)
        if new_ids:
            idx = sorted(new_ids)
            ax.scatter(dis_c[idx, 0] / 1000, dis_c[idx, 1] / 1000, s=42,
                       facecolor="none", edgecolor=ORANGE, lw=1.4, zorder=5)
        ax.scatter([dis_c[0, 0] / 1000], [dis_c[0, 1] / 1000], marker="s", s=48,
                   color="black", zorder=6)
        ax.set_title(f"{ARM_LABEL[a]}  ($S$ = {r['objectives'][2]/1000:.1f} km, "
                     f"$f_1$ = {r['objectives'][0]/1000:.1f} km)", fontsize=8.5)
        ax.set_xlabel("east (km)"); ax.set_aspect("equal")
    axes[0].set_ylabel("north (km)")
    handles = [plt.Line2D([], [], color=GREY, lw=1.8, alpha=0.6,
                          label="plan in force (baseline)"),
               plt.Line2D([], [], marker="s", ls="", color="black", ms=6,
                          label="depot"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec=ORANGE,
                          mew=1.4, ms=7, label="requests arriving with the disruption"),
               plt.Line2D([], [], marker="o", ls="", color="black", ms=3,
                          label="requests")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_maps")
    macro("MAPINSTANCE", instance.replace("_", "\\_"))


# --------------------------------------------------------------------- tables
def write_bench_table(path="data/rsppbench/MANIFEST.json"):
    with open(path) as fh:
        suite = json.load(fh)["suite"]
    fam = {"INS": "new requests", "WDR": "vehicle withdrawal", "SRG": "demand surge"}
    lines = [r"\begin{tabular}{llrrlr}", r"\toprule",
             r"\textbf{Instance} & \textbf{Layout} & $n$ & $K$ & "
             r"\textbf{Disruption} & $k$ \\", r"\midrule"]
    lay = {"C": "clustered", "R": "uniform", "RC": "mixed"}
    for m in suite:
        n = m["n_customers"] + (m["k_new"] if m["family"] == "INS" else 0)
        mark = "$^{\\ast}$" if m.get("scale_only") else ""
        lines.append(f"\\textsf{{{m['name']}}}{mark} & {lay[m['layout']]} & {n} & "
                     f"{m['n_vehicles']} & {fam[m['family']]} & "
                     f"{m['k_new'] if m['family'] != 'WDR' else 1} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{TEXDIR}/tab_bench.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # The comparison suite, excluding the oversized pairs that exist only for
    # the scalability study.  Every "over N instances" statement in the paper
    # refers to the comparison suite, so this must not silently include them.
    macro("NINSTANCES", sum(1 for m in suite if not m.get("scale_only")))
    macro("NINSTANCESSCALE", sum(1 for m in suite if m.get("scale_only")))
    print(f"  wrote {TEXDIR}/tab_bench.tex")


def write_main_table(df, per_inst):
    """Median change in each objective per arm, aggregated over instances.

    Two rules govern every number here, and both were got wrong in an earlier
    version of this analysis.

    *Feasibility conditions everything.*  Objective statistics and the paired
    tests are computed over feasible runs only, with the feasibility rate beside
    them.  The greedy-repair arm attains its low deviation partly by returning
    plans that violate capacity -- on the demand-surge family it changes nothing
    at all, which is perfectly stable and not a plan.  A comparison that does
    not condition on feasibility rewards exactly that.

    *Comparisons happen inside an instance.*  Pooling raw deviations across
    instances spanning n = 50 to 200 makes the rank statistic a measure of the
    difference between instances rather than between methods, which drags every
    effect size towards 0.5 however consistently one arm wins.  Each instance is
    compared on seed-paired samples and only the summaries are aggregated; the
    across-instance significance statement is the Friedman control comparison.
    """
    d = df[df.study == "main"]
    insts = sorted(d.instance.unique())
    base_all = d[d.arm == "cost_only"]
    ctrl = d[(d.arm == "corridor") & d.feasible]
    rows = []
    for a in ARM_ORDER:
        s_all = d[d.arm == a]
        if s_all.empty:
            continue
        s = s_all[s_all.feasible]
        rel_S, rel_D, rel_Dmin, pairs, n_inst = [], [], [], [], 0
        for inst in insts:
            base = base_all[(base_all.instance == inst) & base_all.feasible]
            cur = s[s.instance == inst]
            if base.empty or cur.empty:
                continue
            n_inst += 1
            rel_S.append(100 * (1 - cur.f3.median() / base.f3.median()))
            rel_D.append(100 * (cur.f1.median() / base.f1.median() - 1))
            bmin = base.f1_front_min.median()
            if bmin == bmin and bmin > 0:
                rel_Dmin.append(100 * (cur.f1.median() / bmin - 1))
            # seed-paired inside the instance, on seeds feasible for both arms
            y = ctrl[ctrl.instance == inst]
            common = sorted(set(cur.seed) & set(y.seed))
            if len(common) >= 5:
                xs = cur.set_index("seed").loc[common, "f3"].values
                ys = y.set_index("seed").loc[common, "f3"].values
                pairs.append((xs, ys))
        cmp_ = ind.per_instance_comparison(pairs)
        rows.append(dict(arm=a,
                         S=np.median(rel_S) if rel_S else float("nan"),
                         Sq=(np.percentile(rel_S, [25, 75]) if len(rel_S) > 1
                             else [float("nan")] * 2),
                         D=np.median(rel_D) if rel_D else float("nan"),
                         Dmin=np.median(rel_Dmin) if rel_Dmin else float("nan"),
                         feas=100 * s_all.feasible.mean(),
                         n_inst=n_inst, rt=s_all.runtime.median(), **cmp_))

    # Across-instance significance: Holm-adjusted control comparison on the
    # Friedman ranks of median deviation, feasible runs only.
    arms = [r["arm"] for r in rows]
    ctrl_idx = arms.index("corridor")
    mat = np.array([[d[(d.instance == i) & (d.arm == a) & d.feasible].f3.median()
                     for a in arms] for i in insts], float)
    mat = np.where(np.isnan(mat), np.nanmax(mat, axis=1, keepdims=True), mat)
    _, fp, padj = ind.friedman_control(mat, ctrl_idx)

    lines = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
             r"\textbf{Arm} & $\Delta S$ (\%) & $\Delta f_1$ (\%) & "
             r"feasible (\%) & W/T/L & $\hat{A}_{12}$ & $p_{\text{Holm}}$ & "
             r"effect \\", r"\midrule"]
    for k, r_ in enumerate(rows):
        name = ARM_LABEL[r_["arm"]]
        sv = "---" if r_["S"] != r_["S"] else f"{r_['S']:+.1f}"
        dv = "---" if r_["D"] != r_["D"] else f"{r_['D']:+.1f}"
        feas = f"{r_['feas']:.1f}"
        if r_["n_inst"] < len(insts):
            feas += r"$^{\dagger}$"
        if r_["arm"] == "corridor":
            lines.append(f"\\textbf{{{name}}} & \\textbf{{{sv}}} & \\textbf{{{dv}}} & "
                         f"\\textbf{{{feas}}} & --- & --- & --- & --- \\\\")
        else:
            p_ = padj.get(k, 1.0)
            ptxt = "$<10^{-4}$" if p_ < 1e-4 else f"{p_:.3f}"
            wtl = f"{r_['win']}/{r_['tie']}/{r_['loss']}"
            lines.append(f"{name} & {sv} & {dv} & {feas} & {wtl} & "
                         f"{r_['a12']:.2f} & {ptxt} & {r_['effect']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{TEXDIR}/tab_main.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {TEXDIR}/tab_main.tex")

    cor = next(r for r in rows if r["arm"] == "corridor")
    macro("MAINSDROP", pct(cor["S"], 1))
    macro("MAINSDROPIQR", f"{cor['Sq'][0]:.1f}--{cor['Sq'][1]:.1f}\\%")
    macro("MAINDISTCOST", pct(cor["D"], 1))
    # signed and unsigned forms, so prose never has to say "reduced by -2.2%"
    macro("MAINDISTABS", pct(abs(cor["D"]), 1))
    macro("MAINDISTVERB", "reducing" if cor["D"] < 0 else "increasing")
    # Distance measured against cost-only's cheapest returned plan rather than
    # its knee: the honest reference for any claim about travel distance.
    macro("MAINDISTVSMIN", pct(cor["Dmin"], 1))
    macro("MAINFEAS", f"{cor['feas']:.1f}\\%")
    macro("FRIEDMANMAINP", "$<10^{-4}$" if fp < 1e-4 else f"{fp:.4f}")
    for k, r_ in enumerate(rows):
        key = r_["arm"].replace("_", "").upper()
        if r_["S"] == r_["S"]:
            macro("S" + key, pct(r_["S"], 1))
            macro("D" + key, pct(r_["D"], 1))
            macro("DMIN" + key, pct(r_["Dmin"], 1))
        macro("FEAS" + key, f"{r_['feas']:.1f}\\%")
        macro("A12" + key, f"{r_['a12']:.2f}")
        macro("WTL" + key, f"{r_['win']}/{r_['tie']}/{r_['loss']}")
        macro("P" + key, ("$<10^{-4}$" if padj.get(k, 1.0) < 1e-4
                          else f"{padj.get(k, 1.0):.3f}"))
        macro("EFF" + key, r_["effect"])
    macro("NINSTGREEDY", next(r["n_inst"] for r in rows if r["arm"] == "greedy"))
    macro("NINSTANCESMAIN", len(insts))


def write_family_table(df):
    """Deviation reduction of the proposed arm by disruption family."""
    d = df[(df.study == "main")]
    fams = {"INS": "new requests", "WDR": "vehicle withdrawal",
            "SRG": "demand surge"}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"\textbf{Family} & \textbf{Disruption} & $\Delta S$ (\%) & "
             r"$\Delta f_1$ (\%) & greedy feasible (\%) \\", r"\midrule"]
    for fam, label in fams.items():
        sub = d[d.family == fam]
        if sub.empty:
            continue
        rs, rd = [], []
        for inst in sorted(sub.instance.unique()):
            b = sub[(sub.instance == inst) & (sub.arm == "cost_only") & sub.feasible]
            c = sub[(sub.instance == inst) & (sub.arm == "corridor") & sub.feasible]
            if b.empty or c.empty:
                continue
            rs.append(100 * (1 - c.f3.median() / b.f3.median()))
            rd.append(100 * (c.f1.median() / b.f1.median() - 1))
        gf = 100 * sub[sub.arm == "greedy"].feasible.mean()
        lines.append(f"\\textsf{{{fam}}} & {label} & {np.median(rs):+.1f} & "
                     f"{np.median(rd):+.1f} & {gf:.0f} \\\\")
        macro("S" + fam, pct(np.median(rs), 1))
        macro("D" + fam, pct(np.median(rd), 1))
        macro("GREEDYFEAS" + fam, f"{gf:.0f}\\%")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{TEXDIR}/tab_family.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {TEXDIR}/tab_family.tex")


def write_indicator_table(per_inst):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"\textbf{Arm} & HV $\uparrow$ & IGD$^{+}$ $\downarrow$ & "
             r"$\varepsilon$ $\downarrow$ & spacing & distinct & returned \\",
             r"\midrule"]
    agg = {}
    for inst, per_arm in per_inst.items():
        for a, vals in per_arm.items():
            agg.setdefault(a, {"hv": [], "igdp": [], "eps": [], "spacing": [],
                               "n_front": [], "n_raw": []})
            for k in agg[a]:
                agg[a][k].extend([v[k] for v in vals if np.isfinite(v[k])])
    for a in ARM_ORDER:
        if a not in agg:
            continue
        v = agg[a]
        name = ARM_LABEL[a]
        if a == "corridor":
            name = r"\textbf{" + name + "}"
        lines.append(f"{name} & {np.median(v['hv']):.3f} & "
                     f"{np.median(v['igdp']):.3f} & {np.median(v['eps']):.3f} & "
                     f"{np.median(v['spacing']):.3f} & "
                     f"{np.median(v['n_front']):.0f} & "
                     f"{np.median(v['n_raw']):.0f} \\\\")
        key = a.replace("_", "").upper()
        macro("HV" + key, f"{np.median(v['hv']):.3f}")
        macro("IGD" + key, f"{np.median(v['igdp']):.3f}")
        macro("EPS" + key, f"{np.median(v['eps']):.3f}")
        macro("NDIST" + key, f"{np.median(v['n_front']):.0f}")
        # Members of the returned non-dominated set *before* duplicate objective
        # vectors are removed.  The gap between this and the previous number is
        # the precise statement of the front collapse: not a small front, but a
        # population of identical objective vectors.
        macro("NRAW" + key, f"{np.median(v['n_raw']):.0f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{TEXDIR}/tab_indicators.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {TEXDIR}/tab_indicators.tex")


# ----------------------------------------------------------------------- main
def main():
    apply()
    os.makedirs(FIGDIR, exist_ok=True)
    rows, errors = L.load()
    df = L.to_frame(rows)
    print(f"loaded {len(rows)} runs ({len(errors)} errors) "
          f"over {df.instance.nunique()} instances, {df.arm.nunique()} arms")
    if errors:
        print("  first error:", errors[0].get("error"))

    main_df = df[df.study == "main"]
    macro("NRUNS", f"{len(df):,}".replace(",", "{,}"))
    macro("NRUNSMAIN", f"{len(main_df):,}".replace(",", "{,}"))
    macro("NSEEDS", int(main_df.seed.nunique()))
    macro("NARMS", int(main_df.arm.nunique()))
    macro("NEVALS", f"{int(main_df.n_evals.median()):,}".replace(",", "{,}"))

    # The execution environment is read from the snapshot the campaign runner
    # wrote next to the results, not from whichever machine happens to be
    # drawing the figures -- those need not be the same computer.
    env_path = "experiments/results/environment.json"
    env = {}
    if os.path.exists(env_path):
        env = json.load(open(env_path))
    e = env.get("main", env.get("_last", {}))
    cpu = e.get("cpu", platform.processor())
    mem = e.get("memory_gb")
    macro("HARDWARE", f"{cpu} ({e.get('cores', os.cpu_count())} cores"
                      + (f", {mem}\\,GB RAM" if mem else "")
                      + f", macOS {e.get('mac_ver') or platform.mac_ver()[0]})")
    macro("PYVERSION", e.get("python", platform.python_version()))
    macro("NUMPYVERSION", e.get("numpy", np.__version__))
    import scipy
    macro("SCIPYVERSION", e.get("scipy", scipy.__version__))
    macro("NPROCS", e.get("procs", 7))
    macro("RUNTIMEMED", f"{main_df[main_df.arm=='corridor'].runtime.median():.1f}")

    # Uncontended timing.  Everything else in the campaign runs seven-way
    # parallel, so its wall-clock figures include contention; this study is run
    # with a single worker precisely so that a per-run cost can be quoted.
    tm = df[df.study == "timing"]
    if not tm.empty:
        macro("RUNTIMESOLO", f"{tm.runtime.median():.1f}")
        macro("RUNTIMESOLOMAX", f"{tm.runtime.max():.1f}")
        macro("CPUTIMESOLO", f"{tm.cpu_time.median():.1f}")

    # Reproducibility across seeds: the corridor converges within a run, so the
    # relevant diversity question is whether it lands in the same place twice.
    for a in ("cost_only", "sim_obj", "corridor", "corridor_fixed"):
        sub = main_df[main_df.arm == a]
        if sub.empty:
            continue
        cv = sub.groupby("instance").apply(
            lambda g: (100 * g.f3.std() / g.f3.mean()
                       if len(g) > 1 and g.f3.mean() else np.nan),
            include_groups=False).dropna()
        nd = sub.groupby("instance").apply(
            lambda g: len(np.unique(np.round(g[["f1", "f2", "f3"]].values, 3), axis=0)),
            include_groups=False)
        key = a.replace("_", "").upper()
        macro("CV" + key, pct(cv.median(), 1))
        macro("NSEEDPLANS" + key, f"{nd.median():.0f}")
    # Two maxima, because they answer different questions and conflating them
    # let an earlier draft quote a "maximum over all runs" that a sensitivity
    # study exceeded.
    macro("RUNTIMEMAXMAIN", f"{main_df[main_df.arm=='corridor'].runtime.max():.1f}")
    macro("RUNTIMEMAXALL", f"{df.runtime.max():.1f}")

    # -------- front indicators, per instance, on normalised objectives
    by_inst = L.fronts_by_instance(rows, "main")
    per_inst = {}
    for inst, runs in by_inst.items():
        fronts = [r["front"] for r in runs if r.get("front")]
        if not fronts:
            continue
        lo, hi = ind.normalise_bounds(fronts)
        ref = ind.reference_front(fronts, lo, hi)
        per_arm = {}
        for r in runs:
            per_arm.setdefault(r["arm"], []).append(
                ind.front_indicators(r.get("front"), lo, hi, ref))
        per_inst[inst] = per_arm

    # -------- Friedman / Nemenyi over instances
    insts = sorted(per_inst)
    arms = [a for a in ARM_ORDER if a in main_df.arm.unique()]
    hv_mat = np.array([[np.median([v["hv"] for v in per_inst[i].get(a, [{"hv": 0}])])
                        for a in arms] for i in insts])
    # Feasible runs only.  Ranking arms on deviation without conditioning on
    # feasibility puts the greedy repair at the top of the demand-surge
    # instances, where it scores a perfect zero by leaving an overloaded plan
    # untouched.  An infeasible plan has no deviation worth ranking.
    s_mat = np.array([[main_df[(main_df.instance == i) & (main_df.arm == a)
                               & main_df.feasible].f3.median()
                       for a in arms] for i in insts])
    # An arm with no feasible run on an instance is ranked last there rather
    # than dropped, which would silently shrink the comparison.
    s_mat = np.where(np.isnan(s_mat), np.nanmax(s_mat, axis=1, keepdims=True), s_mat)
    hv_rank = ind.friedman_nemenyi(hv_mat, higher_is_better=True)
    s_rank = ind.friedman_nemenyi(s_mat, higher_is_better=False)
    macro("FRIEDMANHVP", "$<10^{-4}$" if hv_rank[1] < 1e-4 else f"{hv_rank[1]:.4f}")
    macro("FRIEDMANSP", "$<10^{-4}$" if s_rank[1] < 1e-4 else f"{s_rank[1]:.4f}")
    macro("NEMENYICD", f"{s_rank[2]:.2f}")
    macro("CORRRANKS", f"{s_rank[0][arms.index('corridor')]:.2f}")
    macro("CORRRANKHV", f"{hv_rank[0][arms.index('corridor')]:.2f}")

    print("figures:")
    fig_mechanism(df, rows)
    fig_tradeoff(df, tradeoff_instances())
    fig_violin(df)
    fig_cd([("Trajectory deviation $S$", (arms, s_rank[0], s_rank[1], s_rank[2])),
            ("Hypervolume of the returned front",
             (arms, hv_rank[0], hv_rank[1], hv_rank[2]))])
    # Illustrative instances are chosen by a stated rule rather than by
    # position in an alphabetical listing.  Dynamics are shown on the largest
    # vehicle-withdrawal instance, the case that most invalidates the plan in
    # force and therefore gives the corridor the most to do; the map is drawn on
    # a mid-sized new-request instance, where the arriving work is visible.
    def _pick(family, by_size=max):
        cand = main_df[main_df.family == family]
        if cand.empty:
            return sorted(insts)[0]
        target = by_size(cand.n_customers.unique())
        return sorted(cand[cand.n_customers == target].instance.unique())[0]

    dyn_inst = _pick("WDR", max)
    map_inst = _pick("INS", lambda v: sorted(v)[len(v) // 2])
    fig_corridor_dynamics(rows, dyn_inst)
    fig_metrics(df)
    fig_ablation(df)
    fig_scalability(df)
    fig_maps(rows, map_inst)

    # ---- sub-study summaries, all relative to cost-only on the same instances
    SUBSET = list(SUBSET_PLOT)
    macro("NSUBSET", len(SUBSET))
    base = main_df[(main_df.arm == "cost_only") & main_df.feasible]
    corr_ref = main_df[(main_df.arm == "corridor") & main_df.feasible]
    corr_ref_all = main_df[main_df.arm == "corridor"]

    NOTEST = "\\textcolor{red}{\\textbf{[no test]}}"

    def _emit_test(name, sub, col="f3"):
        """Emit A_12 / win-tie-loss / p for one ablation against the default.

        A cell with too few paired seeds to test renders as a loud marker rather
        than as ``nan``, which would read as a number in the typeset paper.
        """
        t = _ablation_test(df, sub, corr_ref, col)
        if not t["n_cmp"]:
            for pre in ("A12", "WTL", "EFF", "P"):
                macro(pre + name, NOTEST)
            return
        macro("A12" + name, f"{t['a12']:.2f}")
        macro("WTL" + name, f"{t['win']}/{t['tie']}/{t['loss']}")
        macro("EFF" + name, t["effect"])
        macro("P" + name, "$<10^{-4}$" if t["p_min"] < 1e-4 else f"{t['p_min']:.3f}")

    def rel(sub, col="f3", ref="f3", insts=None, sign=-1, common=None):
        """Per-instance change relative to cost-only, aggregated.

        ``common`` restricts the instance set, and every ablation comparison
        passes it.  Without it an arm missing an instance is silently summarised
        over a different set than the arm it is being compared against -- and
        since the instances differ systematically in how much there is to gain,
        that is enough to reverse an ordering.
        """
        out = []
        for i in (insts or SUBSET):
            if common is not None and i not in common:
                continue
            b = base[base.instance == i]
            c = sub[(sub.instance == i) & sub.feasible]
            if b.empty or c.empty:
                continue
            ratio = c[col].median() / b[ref].median()
            out.append(100 * (1 - ratio) if sign < 0 else 100 * (ratio - 1))
        return float(np.median(out)) if out else float("nan")

    def _common(*subs):
        """Instances of the sensitivity subset on which every arm has a run."""
        sets = [set(x[x.feasible].instance.unique()) for x in subs]
        return set(SUBSET).intersection(*sets) if sets else set()

    met_subs = [df[df.study == f"metricv2:{m}"]
                for m in ("frechet", "dtw", "hausdorff", "lcss")]
    met_common = _common(*[x for x in met_subs if not x.empty]) if any(
        not x.empty for x in met_subs) else set()
    for met in ("frechet", "dtw", "hausdorff", "lcss"):
        sub = df[df.study == f"metricv2:{met}"]
        if sub.empty:
            continue
        macro("MET" + met.upper(), pct(rel(sub, "dev_frechet", common=met_common), 1))
        macro("METD" + met.upper(),
              pct(rel(sub, "f1", "f1", sign=+1, common=met_common), 1))
        if met != "frechet":
            _emit_test("MET" + met.upper(), sub, "dev_frechet")

    ir = df[df.study == "init:random"]
    if not ir.empty:
        ic = _common(ir, corr_ref)
        macro("INITGRAFT", pct(rel(corr_ref, common=ic), 1))
        macro("INITRANDOM", pct(rel(ir, common=ic), 1))
        macro("INITRANDOMTIME", f"{ir.runtime.median():.1f}")
        # Same instances as the random-initialisation runs it is set against.
        macro("INITGRAFTTIME",
              f"{corr_ref_all[corr_ref_all.instance.isin(SUBSET)].runtime.median():.1f}")
        _emit_test("INITRANDOM", ir)

    sg = df[df.study == "schedule:gaussian"]
    if not sg.empty:
        cf = main_df[main_df.arm == "corridor_fixed"]
        sc = _common(sg, corr_ref, cf)
        macro("SCHEDLINEAR", pct(rel(corr_ref, common=sc), 1))
        macro("SCHEDGAUSS", pct(rel(sg, common=sc), 1))
        macro("SCHEDFIXED", pct(rel(cf, common=sc), 1))
        _emit_test("SCHEDGAUSS", sg)
        _emit_test("SCHEDFIXED", main_df[main_df.arm == "corridor_fixed"])

    res_subs = [df[df.study == f"resample:{st}"] for st in (100, 500, 1000)]
    rc = _common(*[x for x in res_subs if not x.empty], corr_ref) if any(
        not x.empty for x in res_subs) else set(SUBSET)
    for st, rs in zip((100, 500, 1000), res_subs):
        if not rs.empty:
            macro(f"RESTIME{st}", f"{rs.runtime.median():.1f}")
            macro(f"RES{st}", pct(rel(rs, common=rc), 1))
            _emit_test(f"RES{st}", rs)
    macro("RESTIME250",
          f"{corr_ref_all[corr_ref_all.instance.isin(SUBSET)].runtime.median():.1f}")
    macro("RES250", pct(rel(corr_ref, common=rc), 1))

    cal_subs = [df[df.study == f"calib:q{q}"] for q in (25, 50, 75)]
    cc = _common(*[x for x in cal_subs if not x.empty], corr_ref) if any(
        not x.empty for x in cal_subs) else set(SUBSET)
    for q, cb in zip((25, 50, 75), cal_subs):
        if not cb.empty:
            macro(f"CALIBQ{q}", pct(rel(cb, common=cc), 1))
            macro(f"CALIBQ{q}TAU", f"{cb.tau0.median()/1000:.0f}")
            _emit_test(f"CALIBQ{q}", cb)
    macro("CALIBQNINETY", pct(rel(corr_ref, common=cc), 1))
    # Both corridor-width macros are now subset medians on the same instances,
    # so the width quoted in the results section and the width quoted in the
    # calibration study are the same quantity.
    macro("CALIBQNINETYTAU",
          f"{main_df[(main_df.arm=='corridor') & main_df.instance.isin(SUBSET)].tau0.median()/1000:.0f}")

    # Generation-budget sufficiency: is 300 generations enough?
    bud_subs = [df[df.study == f"budget:{g}"] for g in (150, 600)]
    bc = _common(*[x for x in bud_subs if not x.empty], corr_ref) if any(
        not x.empty for x in bud_subs) else set(SUBSET)
    for g, bg in zip((150, 600), bud_subs):
        if not bg.empty:
            macro(f"BUDGET{g}", pct(rel(bg, common=bc), 1))
            _emit_test(f"BUDGET{g}", bg)
    macro("BUDGET300", pct(rel(corr_ref, common=bc), 1))

    # The weight against the *unweighted* objective, head to head.  Table 5
    # compares every arm against the corridor, which does not test the claim
    # that actually matters for Proposition 1: that weighting f_3 changes
    # nothing relative to not weighting it.
    pairs = []
    for i in sorted(main_df.instance.unique()):
        a = main_df[(main_df.arm == "sim_lambda") & (main_df.instance == i)
                    & main_df.feasible]
        b = main_df[(main_df.arm == "sim_obj") & (main_df.instance == i)
                    & main_df.feasible]
        common = sorted(set(a.seed) & set(b.seed))
        if len(common) >= 5:
            pairs.append((a.set_index("seed").loc[common, "f3"].values,
                          b.set_index("seed").loc[common, "f3"].values))
    if pairs:
        t = ind.per_instance_comparison(pairs)
        macro("A12LAMBDAVSOBJ", f"{t['a12']:.2f}")
        macro("EFFLAMBDAVSOBJ", t["effect"])
        macro("WTLLAMBDAVSOBJ", f"{t['win']}/{t['tie']}/{t['loss']}")
        macro("PLAMBDAVSOBJ",
              "$<10^{-4}$" if t["p_min"] < 1e-4 else f"{t['p_min']:.3f}")

    # What the comparison would look like if each arm's knee ranged only over
    # the objectives it optimises.  Reported, not headlined: the three-objective
    # rule gives the baseline a final pick that accounts for deviation and is the
    # conservative choice, and the reader is entitled to know by how much.
    alt_rel = []
    for i in sorted(main_df.instance.unique()):
        b = main_df[(main_df.arm == "cost_only") & (main_df.instance == i)
                    & main_df.feasible]
        c = main_df[(main_df.arm == "corridor") & (main_df.instance == i)
                    & main_df.feasible]
        if b.empty or c.empty or not (b.alt_f3.median() > 0):
            continue
        alt_rel.append(100 * (1 - c.alt_f3.median() / b.alt_f3.median()))
    if alt_rel:
        macro("MAINSDROPALT", pct(float(np.median(alt_rel)), 1))
        bb = main_df[(main_df.arm == "cost_only") & main_df.feasible]
        macro("ALTKNEEBASE", f"{bb.alt_f3.median()/1000:.1f}")
        macro("KNEEBASE", f"{bb.f3.median()/1000:.1f}")

    # Equity: the corridor's real price.  Percentage and absolute minutes are
    # emitted from *one* estimator -- the median across instances of the
    # per-instance ratio, and the median across instances of the per-instance
    # median spread.  Mixing a median-of-ratios with a pooled median, as an
    # earlier version did, produces a percentage that does not reconcile with
    # the two absolute figures printed beside it.
    all_insts = sorted(main_df.instance.unique())

    def _eq_abs(sub):
        per = [sub[sub.instance == i].f2.median() / 60.0 for i in all_insts
               if not sub[sub.instance == i].empty]
        return float(np.median(per)) if per else float("nan")

    for a in ("greedy", "cost_only", "sim_obj", "eps_static", "corridor",
              "corridor_fixed", "corridor_nsga2", "corridor_moead"):
        sub = main_df[(main_df.arm == a) & main_df.feasible]
        if sub.empty:
            continue
        key = a.replace("_", "").upper()
        macro("EQ" + key, pct(rel(sub, "f2", "f2", insts=all_insts, sign=+1), 1))
        # One decimal, not zero: these are quoted next to a percentage and a
        # reader will divide them.  Rounding 22.71 and 55.54 to integers makes
        # the quotient disagree with the percentage beside it by a point.
        macro("EQ" + key + "ABS", f"{_eq_abs(sub):.1f}")

    # ------------------------------------------------- non-vacuity floor q_min
    # q_0 fixes where the corridor starts; q_min fixes how far it may close, and
    # it is the parameter Proposition 3 is stated in terms of.  Its own
    # sensitivity therefore cannot be left to the calibration study.
    QMIN_LEVELS = [("QMINQONE", 1), ("QMINQFIVE", 5), ("QMINQTWENTY", 20)]
    qsubs = [df[df.study == f"qmin:q{n}"] for _, n in QMIN_LEVELS]
    if any(not x.empty for x in qsubs):
        qc = _common(*[x for x in qsubs if not x.empty], corr_ref)
        corr_sub = corr_ref[corr_ref.instance.isin(SUBSET)]
        for (name, n), qs in zip(QMIN_LEVELS, qsubs):
            if qs.empty:
                continue
            macro(name, pct(rel(qs, common=qc), 1))
            macro(name + "FEAS", f"{100 * qs.feasible.mean():.1f}\\%")
            macro(name + "TAU", f"{qs.tau_final.median() / 1000:.1f}")
            _emit_test(name, qs)
        macro("QMINQTEN", pct(rel(corr_ref, common=qc), 1))
        macro("QMINQTENFEAS",
              f"{100 * main_df[main_df.arm == 'corridor'].feasible.mean():.1f}\\%")
        macro("QMINQTENTAU", f"{corr_sub.tau_final.median() / 1000:.1f}")
        # The span of the four levels, which is the finding: a four-fold change
        # either side of the default moves the outcome by less than a point.
        vals = [rel(q, common=qc) for q in qsubs if not q.empty]
        vals.append(rel(corr_ref, common=qc))
        macro("QMINSPAN", pct(max(vals) - min(vals), 1))

    # How often the floor rather than the schedule sets the width, and how far
    # the corridor actually closes.  Both are properties of Eq. (8) that the
    # width alone does not report.
    cs = main_df[(main_df.arm == "corridor") & main_df.feasible]
    if "sched_frac" in cs and cs.sched_frac.notna().any():
        macro("CORRSCHEDFRAC", pct(100 * cs.sched_frac.median(), 0))
        macro("CORRTAUFINAL", f"{cs.tau_final.median() / 1000:.1f}")
    # The width can reach zero, and where it does the metre guard in Eq. (7) is
    # load-bearing rather than decorative.  It happens exactly when at least
    # q_min of the pool reproduces the baseline geometry, so the floor is zero.
    ca = main_df[(main_df.arm == "corridor")].dropna(subset=["tau_min"])
    if not ca.empty:
        z = ca[ca.tau_min <= 1.0]
        macro("CORRTAUZERON", int(len(z)))
        macro("CORRTAUZEROPCT", pct(100 * len(z) / len(ca), 1))
        macro("CORRTAUZEROINST", int(z.instance.nunique()))
        macro("CORRTAUZEROINSTNAME",
              (sorted(z.instance.unique())[0].replace("_", "\\_") if len(z) else ""))
        pos = ca[ca.tau_min > 1.0]
        if not pos.empty:
            macro("CORRTAUMINPOS", f"{pos.tau_min.min() / 1000:.1f}")
        # What leaving the plan alone costs on that instance, against the arm
        # that is free to change it.
        if len(z):
            zi = sorted(z.instance.unique())[0]
            zc = main_df[(main_df.instance == zi) & (main_df.arm == "corridor")
                         & main_df.feasible]
            zb = main_df[(main_df.instance == zi) & (main_df.arm == "cost_only")
                         & main_df.feasible]
            if not zc.empty and not zb.empty:
                macro("CORRZEROINSTDIST",
                      pct(100 * (1 - zb.f1.median() / zc.f1.median()), 1))
                macro("CORRZEROINSTS", f"{zb.f3.median() / 1000:.1f}")

    # ------------------------------------------------- static epsilon-constraint
    eps = main_df[main_df.arm == "eps_static"]
    if not eps.empty:
        macro("EPSNINSTFEAS", int(eps[eps.feasible].instance.nunique()))
        macro("EPSNINSTNONE",
              int(main_df.instance.nunique() - eps[eps.feasible].instance.nunique()))
        # The static arm returns nothing on eight instances, so its column in
        # Table 5 is a median over a different instance set from every other
        # arm's.  Every comparison drawn against it in the text is therefore
        # restricted to the instances on which it returns a plan at all.
        eps_common = set(eps[eps.feasible].instance.unique())
        all_i = sorted(main_df.instance.unique())
        for a, nm in (("sim_obj", "SIMOBJ"), ("eps_static", "EPSSTATIC"),
                      ("corridor_fixed", "CORRIDORFIXED"), ("corridor", "CORRIDOR")):
            sub = main_df[main_df.arm == a]
            if sub.empty:
                continue
            macro("SCOMMON" + nm, pct(rel(sub, insts=all_i, common=eps_common), 1))
            macro("EQCOMMON" + nm,
                  pct(rel(sub, "f2", "f2", insts=all_i, sign=+1,
                          common=eps_common), 1))
        for fam in ("INS", "WDR", "SRG"):
            fs = eps[eps.family == fam]
            if fs.empty:
                continue
            macro("EPSFEAS" + fam, f"{100 * fs.feasible.mean():.1f}\\%")
            # The bound itself: the deviation of the greedy repair, recorded as
            # tau0 by the arm that uses it.
            macro("EPSBOUND" + fam, f"{fs.tau0.median() / 1000:.1f}")
            cf = main_df[(main_df.arm == "corridor") & (main_df.family == fam)]
            if not cf.empty:
                macro("TAUZERO" + fam, f"{cf.tau0.median() / 1000:.1f}")

    # --------------------------------------------------- idle-vehicle monitor
    # f_2 ranges over available vehicles carrying work, so a search could in
    # principle buy a small spread by idling a crew member.  Whether it does is
    # a question about runs, not about the definition.
    av = main_df.dropna(subset=["n_active"])
    av = av[av.feasible & (av.arm != "greedy")]
    if not av.empty:
        avail = np.where(av.family == "WDR", av.n_vehicles - 1, av.n_vehicles)
        idle = avail - av.n_active.values
        macro("IDLERUNS", int(np.sum(idle > 0)))
        macro("IDLENRUNS", f"{len(av):,}".replace(",", "{,}"))
        macro("NACTMIN", int(av.n_active.min()))

    # ------------------------------------- circumstances of the infeasible runs
    inf = main_df[(main_df.arm == "corridor") & (~main_df.feasible)]
    macro("CORRINFEASN", int(len(inf)))
    if len(inf):
        macro("CORRINFEASMAX", f"{100 * inf.violation.max():.1f}\\%")
        macro("CORRINFEASINST", int(inf.instance.nunique()))
        macro("CORRINFEASMIN", f"{100 * inf.violation.min():.1f}\\%")
        # Was the corridor the binding reason?  If a large share of the final
        # population is still inside the corridor, it was not.
        if inf.corr_feas_final.notna().any():
            macro("CORRINFEASCFEAS", pct(100 * inf.corr_feas_final.min(), 0))

    print("tables:")
    write_bench_table()
    write_main_table(df, per_inst)
    write_family_table(df)
    write_indicator_table(per_inst)

    # standalone gate experiment: the selection operator of the production
    # codebase, swept over eight decades of lambda with the random stream pinned
    gate_path = "experiments/gate_lambda/results/lambda_inertness.json"
    if os.path.exists(gate_path):
        g = json.load(open(gate_path))
        n = g["n_comparisons"]
        macro("GATENCMP", n)
        macro("GATETRIALS", g["trials"])
        macro("GATELAMLO", "10^{-4}")
        macro("GATELAMHI", "10^{6}")
        macro("GATERANK", f"{g['D_ranking_differs']}")
        macro("GATERANKPCT", pct(100 * g["D_ranking_differs"] / n, 1))
        macro("GATENORM", pct(100 * g["A_normalization_differs"] / n, 1))
        macro("GATESEL", pct(100 * g["B_selection_differs"] / n, 1))
        macro("GATENOISE", pct(100 * g["C_rng_differs"] / n, 1))
        macro("GATERATIO",
              f"{g['C_rng_differs'] / max(g['B_selection_differs'], 1):.1f}")

    # Numbers quoted by the resampling remark, emitted by the self-test so that
    # they are generated rather than transcribed.
    st_path = "experiments/results/selftest.json"
    if os.path.exists(st_path):
        st = json.load(open(st_path))
        macro("SELFH", f"{st['h_m']:.0f}")
        macro("SELFNUNDER", f"{st['n_under']}")
        macro("SELFNTRIALS", f"{st['n_trials']}")
        macro("SELFPCTUNDER", pct(st["pct_under"], 0))
        macro("SELFLEN", f"{st['length_km']:.0f}")
        macro("SELFTRUE", f"{st['true_m']:.0f}")
        macro("SELFNAIVE", f"{st['naive_m']:.0f}")
        macro("SELFKEEP", f"{st['preserving_m']:.0f}")
        _m, _e = f"{st['kernel_max_abs']:.1e}".split("e")
        macro("SELFKERNELTOL", f"{_m}\\times10^{{{int(_e)}}}")
        macro("SELFBOUNDUSE", pct(100 * st["bound_worst_ratio"], 0))

    # reference count, taken from the resolved bibliography when available
    bbl = next((f for f in (f"{TEXDIR}/RSPP_COR_revised.bbl", f"{TEXDIR}/manuscript.bbl")
                if os.path.exists(f)), None)
    bib = f"{TEXDIR}/rspp.bib"
    if bbl:
        macro("NREFS", open(bbl).read().count("\\bibitem"))
    elif os.path.exists(bib):
        macro("NREFS", open(bib).read().count("\n@"))

    with open(f"{TEXDIR}/results_macros.tex", "w") as fh:
        fh.write("% Generated by experiments/analysis/figures.py -- do not edit.\n")
        for k in sorted(MACROS):
            fh.write(f"\\newcommand{{\\{k}}}{{{MACROS[k]}}}\n")
    print(f"  wrote {TEXDIR}/results_macros.tex ({len(MACROS)} macros)")

    # Any macro a document references but the campaign has not produced renders
    # as a loud red marker rather than breaking the build.
    import re as _re
    used = set()
    for f in ("manuscript.tex", "RSPP_COR_revised.tex",
              "response_to_reviewers.tex", "cover_letter.tex",
              "highlights.tex", "tab_main.tex", "tab_family.tex",
              "tab_indicators.tex", "tab_bench.tex"):
        fp = os.path.join(TEXDIR, f)
        if os.path.exists(fp):
            text = open(fp).read()
            # Mixed case, not upper only: the generated names spell digits out
            # (\CALIBQFiveZero, \RESTIMEOneZeroZero), so an upper-only pattern
            # missed exactly the macros most likely to be mistyped and left them
            # to fail as undefined control sequences instead of loud markers.
            # The \NAME{} form also catches names the campaign never produced.
            used |= set(_re.findall(r"\\([A-Za-z]{3,})\{\}", text))
            # A generated macro may also be written bare (\GATELAMHI followed by
            # punctuation).  Those are matched against MACROS rather than taken
            # on shape, so no standard LaTeX command is ever given a
            # \providecommand that would collide with a later definition.
            used |= {m for m in _re.findall(r"\\([A-Za-z]{3,})(?![A-Za-z])", text)
                     if m in MACROS}
    used -= {"PENDING"}                     # takes an argument, not a macro
    with open(f"{TEXDIR}/macros_fallback.tex", "w") as fh:
        fh.write("% Generated. Loaded AFTER results_macros.tex; only fills gaps.\n")
        for k in sorted(used):
            fh.write("\\providecommand{\\%s}{\\textcolor{red}{\\textbf{[??%s]}}}\n"
                     % (k, k))
    missing = sorted(used - set(MACROS))
    print(f"  wrote {TEXDIR}/macros_fallback.tex"
          + (f"  -- STILL MISSING: {missing}" if missing else "  -- all resolved"))
    return df, per_inst


if __name__ == "__main__":
    main()
