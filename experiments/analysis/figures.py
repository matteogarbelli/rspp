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
from .style import (ARM_COLOR, ARM_LABEL, ARM_MARKER, ARM_ORDER, GREY, VERMILLION, BLUE,
                    ORANGE, YELLOW, apply, save)

FIGDIR = "experiments/figures"
TEXDIR = "submission_cor"
MACROS: dict[str, str] = {}

# The proposed method: the arm every comparison, test and headline number is
# taken against.  RSPP_CONTROL selects another arm, for instance to regenerate
# a database that predates the proposed arm.
CONTROL = os.environ.get("RSPP_CONTROL", "sum_lambda")


def _key(arm):
    """Macro suffix of an arm: sum_lambda -> SUMLAMBDA."""
    return arm.replace("_", "").upper()


def _uniq(seq):
    """Order-preserving de-duplication, for arm lists that may repeat CONTROL."""
    return list(dict.fromkeys(seq))


# Arms of the main comparison, in table order: the comparison arms and the
# NSGA-II and MOEA/D variants of the proposed arm.  Variants of other arms in
# the database are not reported.
ORDER = _uniq(ARM_ORDER + [a for a in (CONTROL + "_nsga2", CONTROL + "_moead")
                           if a in ARM_LABEL])


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


def pvalue(p):
    """Keep small positive p-values from rounding to a printed zero."""
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.4f}" if p < 1e-3 else f"{p:.3f}"


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
    # The same adaptive weight on a separate objective and summed into
    # distance, and the tightening bound.  An arm without in-situ measurements
    # in the database is left out.
    short = {"sim_lambda": "Weight on\nseparate objective",
             "sum_lambda": "Deviation summed\ninto distance",
             "corridor": "Tightening\nbound"}
    d = df[df.study == "main"]
    arms = [a for a in _uniq(["sim_lambda", CONTROL, "corridor"])
            if d[d.arm == a].rank_change.notna().any()]
    d = d[d.arm.isin(arms)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    pos = list(range(len(arms)))
    vals = [d[d.arm == a].rank_change.dropna().values * 100 for a in arms]
    bp = ax.boxplot(vals, positions=pos, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", lw=1.2), showfliers=False)
    for patch, a in zip(bp["boxes"], arms):
        patch.set_facecolor(ARM_COLOR[a]); patch.set_alpha(0.75)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arms):
        v = d[d.arm == a].rank_change.dropna().values * 100
        ax.scatter(np.full(len(v), i) + rng.uniform(-.13, .13, len(v)), v,
                   s=3, color="black", alpha=0.25, zorder=3)
    ax.set_xticks(pos)
    ax.set_xticklabels([short.get(a, ARM_LABEL[a]) for a in arms])
    ax.set_ylabel("individuals changing front (%)")
    ax.set_title("(a)", loc="left")
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
    ax.set_title("(b)", loc="left")
    # Headroom above the bands, so that the legend covers no data.
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(-0.6, hi_ * 1.35)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)

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
    if len(cor):
        macro("CORRRANK", pct(np.median(cor), 1))
        macro("CORRRANKIQR",
              f"{np.percentile(cor,25):.1f}--{np.percentile(cor,75):.1f}\\%")
    # The same statistic as LAMBDAMEANPCT -- the mean over all logged
    # generations -- so that the text compares the two mechanisms like for like.
    cor_gen = [h["rank_change"] for r in rows
               if r["study"] == "main" and r["arm"] == "corridor"
               for h in r.get("history", []) if h.get("rank_change") is not None]
    if cor_gen:
        macro("CORRRANKMEANPCT", pct(100.0 * float(np.mean(cor_gen)), 1))
    lt = d[d.arm == "sim_lambda"].flip_treat.dropna().values * 100
    ln_ = d[d.arm == "sim_lambda"].flip_noise.dropna().values * 100
    macro("LAMBDAFLIP", pct(np.median(lt), 1))
    macro("LAMBDANOISE", pct(np.median(ln_), 1))
    ct = d[d.arm == "corridor"].flip_treat.dropna().values * 100
    cn = d[d.arm == "corridor"].flip_noise.dropna().values * 100
    if len(ct) and len(cn):
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
    # The same sweep for both weighted placements, under one naming scheme:
    # on a separate objective it measures rounding, summed into distance the
    # dependence of the ranking on the weight.
    for a in _uniq(["sim_lambda", CONTROL]):
        v = df[(df.study == "main") & (df.arm == a)].sweep_rank_change.dropna().values * 100
        if len(v):
            macro("SWEEPMEAN" + _key(a), pct(v.mean(), 2))
            macro("SWEEPMAX" + _key(a), pct(v.max(), 2))
            macro("SWEEPZERO" + _key(a), pct(100.0 * float(np.mean(v <= 1e-9)), 1))

    # The same three statistics for every mechanism drawn, under one naming
    # scheme: median and interquartile range of the per-run means, and the mean
    # over all logged generations.
    for a in arms:
        v = d[d.arm == a].rank_change.dropna().values * 100
        g = [h["rank_change"] for r in rows if r["study"] == "main" and r["arm"] == a
             for h in r.get("history", []) if h.get("rank_change") is not None]
        if len(v):
            macro("RANKMED" + _key(a), pct(float(np.median(v)), 1))
            macro("RANKIQR" + _key(a),
                  f"{np.percentile(v, 25):.1f}--{np.percentile(v, 75):.1f}\\%")
        if g:
            macro("RANKMEAN" + _key(a), pct(100.0 * float(np.mean(g)), 3))


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
                        fmt=ARM_MARKER.get(a, "o"), ms=5, color=ARM_COLOR[a],
                        ecolor=ARM_COLOR[a],
                        elinewidth=1.0, capsize=2,
                        mfc=ARM_COLOR[a] if a == CONTROL else "white",
                        mew=1.4, zorder=5 if a == CONTROL else 3)
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
    handles = [plt.Line2D([], [], marker=ARM_MARKER.get(a, "o"), ls="", color=ARM_COLOR[a],
                          mfc=ARM_COLOR[a] if a == CONTROL else "white", mew=1.4,
                          label=ARM_LABEL[a]) for a in ARM_ORDER
               if a in d.arm.unique()]
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
        va = [a for a in ARM_ORDER if a in d.arm.unique()]
        data = [d[d.arm == a][col].values for a in va]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for pc, a in zip(parts["bodies"], va):
            pc.set_facecolor(ARM_COLOR[a]); pc.set_alpha(0.7); pc.set_edgecolor("black")
            pc.set_linewidth(0.5)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("black"); parts[key].set_linewidth(0.8)
        ax.set_xticks(range(1, len(va) + 1))
        ax.set_xticklabels([ARM_LABEL[a] for a in va], rotation=32,
                           ha="right", fontsize=7.5)
        ax.set_ylabel(f"{lab}\n(normalised within instance)")
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_violin")


def fig_summary(df, per_inst):
    """Stability and its costs: cost-only, the separate objective with and without
    the adaptive weight, the proposed arm and the two population-calibrated
    bounds.

    The statistics are those of Table 2 and the text.  Panels (a)-(c) take
    objective medians over hard-feasible runs within each instance and then the
    change relative to cost-only re-optimisation.  Distance in (b) is measured
    against the median per-run front minimum of cost-only.  Panel (d) counts
    distinct non-dominated objective vectors per run, with failed runs counting
    zero, as in Table 3.  Light points are instances in (a)-(c) and runs in (d);
    a large marker is the median and a bar the interquartile range.
    """
    d = df[df.study == "main"]
    arms = [a for a in _uniq(["cost_only", "sim_obj", "sim_lambda", CONTROL,
                              CONTROL + "_x4", CONTROL + "_x16", "corridor_fixed",
                              "corridor"]) if a in d.arm.unique()]
    vals = {k: {a: [] for a in arms} for k in ("S", "D", "W")}
    for i in sorted(d.instance.unique()):
        b = d[(d.instance == i) & (d.arm == "cost_only") & d.feasible]
        if b.empty:
            continue
        bS, bW, bmin = b.f3.median(), b.f2.median(), b.f1_front_min.median()
        for a in arms:
            c = d[(d.instance == i) & (d.arm == a) & d.feasible]
            if c.empty:
                continue
            vals["S"][a].append(100 * (1 - c.f3.median() / bS))
            vals["D"][a].append(100 * (c.f1.median() / bmin - 1))
            vals["W"][a].append(100 * (c.f2.median() / bW - 1))
    div = {a: [v["n_front"] for i in sorted(per_inst) for v in per_inst[i].get(a, [])]
           for a in arms}

    panels = [("S", "(a) Trajectory deviation", "reduction in $S$ vs cost-only (%)"),
              ("D", "(b) Travel distance",
               "increase in $f_1$ over the shortest\ncost-only plan (%)"),
              ("W", "(c) Workload spread", "increase in $f_2$ vs cost-only (%)"),
              ("N", "(d) Front diversity", "distinct objective vectors per run")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 1.4 + 0.66 * len(arms)), sharey=True)
    rng = np.random.default_rng(0)
    ypos = {a: len(arms) - 1 - j for j, a in enumerate(arms)}
    for ax, (key, title, xlabel) in zip(axes.ravel(), panels):
        for a in arms:
            v = np.asarray(div[a] if key == "N" else vals[key][a], float)
            if not len(v):
                continue
            yy = ypos[a] + rng.uniform(-0.16, 0.16, len(v))
            ax.scatter(v, yy, s=6 if key == "N" else 9, color=ARM_COLOR[a],
                       alpha=0.12 if key == "N" else 0.45, lw=0, zorder=2)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [ypos[a], ypos[a]], color="black", lw=1.6, zorder=3)
            ax.scatter([med], [ypos[a]], s=42, color=ARM_COLOR[a], edgecolor="black",
                       lw=0.8, zorder=4)
        if key != "N":
            ax.axvline(0, color="black", lw=0.8, ls=":")
        ax.set_title(title.split()[0], loc="left")
        ax.set_xlabel(xlabel)
        ax.set_yticks([ypos[a] for a in arms])
        ax.set_yticklabels([ARM_LABEL[a] for a in arms])
        ax.set_ylim(-0.6, len(arms) - 0.4)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_findings_summary")


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
    # The population drawn here is the corridor's own; it is grey because blue
    # stands for cost-only in panels (b) and (c) and in every other figure.
    ax.fill_between(gens, q10, q90, color=GREY, alpha=0.30, lw=0,
                    label="population, 10–90th pct")
    ax.plot(gens, med, color="#555555", lw=1.5, label="population, median")
    ax.plot(gens, tau, color=ARM_COLOR["corridor"], lw=2.0,
            label=r"corridor width $\tau(t)$")

    # The generation at which the non-vacuity floor takes over from the
    # schedule, read from the flag the solver records rather than inferred from
    # where the width happens to cross a population percentile.  Those are
    # different events: Eq. (12) hands over when the scheduled width falls below
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
    ax.set_title("(a)", loc="left")
    # The handover line stops below this corner and the width curve never
    # enters it, so the legend overlaps no data.
    ax.legend(frameon=False, fontsize=6.2, handlelength=1.2, borderaxespad=0.1,
              loc="upper right", bbox_to_anchor=(1.0, 1.04))

    # Panels (b) and (c): the final fronts of every seed, and the plan each
    # method returns.  Population medians per generation would compare the
    # corridor's collapsed population with spread fronts, and say nothing
    # about the plans the methods return.
    for a in ["cost_only", "sim_obj", "corridor"]:
        aruns = [r for r in rows if r["study"] == "main" and r["arm"] == a
                 and r["instance"] == instance and r.get("feasible") and r.get("front")]
        if not aruns:
            continue
        F = np.vstack([np.asarray(r["front"], float) for r in aruns])
        P = np.asarray([r["objectives"] for r in aruns], float)
        for bx, xi, scale in ((axes[1], 0, 1000.0), (axes[2], 1, 60.0)):
            bx.scatter(F[:, xi] / scale, F[:, 2] / 1000, s=5, color=ARM_COLOR[a],
                       alpha=0.20, lw=0, zorder=2)
            bx.scatter(P[:, xi] / scale, P[:, 2] / 1000, s=20, facecolor=ARM_COLOR[a],
                       edgecolor="black", lw=0.5, zorder=4, label=ARM_LABEL[a])
    axes[1].set_xlabel("travel distance $f_1$ (km)")
    axes[2].set_xlabel("workload spread $f_2$ (min)")
    for bx in axes[1:]:
        bx.set_ylabel("deviation $S$ (km)")
    axes[1].set_title("(b)", loc="left")
    axes[2].set_title("(c)", loc="left")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")

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
    # First logged generation at which the width falls below the 90th
    # percentile of the population's deviation, median over seeds.  Until then
    # at least nine tenths of the population lies inside the corridor.
    slack = []
    for r in rows:
        if r["study"] == "main" and r["arm"] == "corridor" and r["instance"] == instance:
            g = [h["gen"] for h in r.get("history", [])
                 if h.get("tau") is not None and h.get("S_q90") is not None
                 and h["tau"] < h["S_q90"]]
            if g:
                slack.append(min(g))
    if slack:
        macro("CORRSLACKGEN", int(np.median(slack)))


def fig_dynamics(rows, instance):
    """The adaptive weight over the run, and the final fronts, on one instance.

    Panel (a) gives lambda(t) for the proposed arm and, dashed, for the same
    rule applied to a separate objective; each is computed from its own
    population.  Panels (b) and (c) are the final fronts of every seed with the
    plans the methods return, for the default rate and the highest rate of the
    summed objective among others.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axes[0]
    drawn = False
    for a, ls in ((CONTROL, "-"), ("sim_lambda", "--")):
        runs = [r for r in rows if r["study"] == "main" and r["arm"] == a
                and r["instance"] == instance and r.get("history")]
        if not runs:
            continue
        gens = [h["gen"] for h in runs[0]["history"]]
        L = np.array([[h.get("lam", np.nan) for h in r["history"]] for r in runs
                      if len(r["history"]) == len(gens)], float)
        lo, hi = np.nanpercentile(L, [25, 75], axis=0)
        ax.fill_between(gens, lo, hi, color=ARM_COLOR[a], alpha=0.20, lw=0)
        ax.plot(gens, np.nanmedian(L, 0), color=ARM_COLOR[a], lw=1.6, ls=ls,
                label=ARM_LABEL[a])
        if a == CONTROL:
            macro("DYNLAMSTART", f"{np.nanmedian(L[:, 0]):.2f}")
            macro("DYNLAMEND", f"{np.nanmedian(L[:, -1]):.2f}")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("generation")
    ax.set_ylabel(r"weight $\lambda(t)$")
    ax.set_title("(a)", loc="left")

    front_arms = []
    for a in _uniq(["cost_only", "sim_obj", CONTROL, CONTROL + "_x16", "corridor"]):
        aruns = [r for r in rows if r["study"] == "main" and r["arm"] == a
                 and r["instance"] == instance and r.get("feasible") and r.get("front")]
        if not aruns:
            continue
        F = np.vstack([np.asarray(r["front"], float) for r in aruns])
        P = np.asarray([r["objectives"] for r in aruns], float)
        for bx, xi, scale in ((axes[1], 0, 1000.0), (axes[2], 1, 60.0)):
            bx.scatter(F[:, xi] / scale, F[:, 2] / 1000, s=5, color=ARM_COLOR[a],
                       alpha=0.20, lw=0, zorder=2)
            bx.scatter(P[:, xi] / scale, P[:, 2] / 1000, s=20, facecolor=ARM_COLOR[a],
                       edgecolor="black", lw=0.5, zorder=4,
                       marker=ARM_MARKER.get(a, "o"))
        front_arms.append(a)
    axes[1].set_xlabel("travel distance $f_1$ (km)")
    axes[2].set_xlabel("workload spread $f_2$ (min)")
    for bx in axes[1:]:
        bx.set_ylabel("deviation $S$ (km)")
    axes[1].set_title("(b)", loc="left")
    axes[2].set_title("(c)", loc="left")
    # One legend for the three panels, below them: a line identifies a weight
    # trajectory in (a), a marker a method's plans in (b) and (c).
    handles = []
    for a in _uniq(["cost_only", "sim_obj", "sim_lambda", CONTROL, CONTROL + "_x16",
                    "corridor"]):
        line = a in (CONTROL, "sim_lambda")
        if not line and a not in front_arms:
            continue
        handles.append(plt.Line2D(
            [], [], color=ARM_COLOR[a], lw=1.6 if line else 0,
            ls=("--" if a == "sim_lambda" else "-"),
            marker=ARM_MARKER.get(a, "o") if a in front_arms else None, ms=5,
            mfc=ARM_COLOR[a], mec="black", mew=0.5, label=ARM_LABEL[a]))
    ncol = len(handles) if len(handles) <= 4 else (len(handles) + 1) // 2
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.03), handlelength=2.0,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08 if ncol == len(handles) else 0.15, 1, 1))
    save(fig, f"{FIGDIR}/fig_dynamics")
    macro("DYNINSTANCE", instance.replace("_", "\\_"))


# ----------------------------------------------------------- Figures: RQ4 -- RQ7
def _boxes(ax, data, colours):
    """Box plot over instances, with every instance drawn as a point.

    Boxes span the interquartile range and whiskers reach the most extreme
    point within 1.5 interquartile ranges of the box.  With nine instances per
    box, no point is hidden: all are overlaid.
    """
    pos = np.arange(1, len(data) + 1)
    bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.55,
                    showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    rng = np.random.default_rng(0)
    for x, vals, c in zip(pos, data, colours):
        vals = np.asarray(vals, float)
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, s=10,
                   facecolor=c, edgecolor="black", lw=0.4, zorder=4)
    return pos


def _rel_by_instance(df, sub, col="f3"):
    """Per-instance reduction in `col` relative to cost-only, in per cent."""
    base = df[(df.study == "main") & (df.arm == "cost_only") & df.feasible]
    out = {}
    for i in SUBSET_PLOT:
        b = base[base.instance == i]
        c = sub[(sub.instance == i) & sub.feasible]
        if b.empty or c.empty:
            continue
        out[i] = 100 * (1 - c[col].median() / b[col].median())
    return out


def _paired_panel(ax, df, settings, default_sub, span, xlog=False):
    """Per-instance difference from the default setting, in percentage points.

    Each point is one instance: the reduction in S under the setting minus the
    reduction under the default, both relative to cost-only.  Positive values
    mean lower deviation than the default.  The default itself is the zero
    line.  A horizontal bar marks the median over instances.
    """
    ref = _rel_by_instance(df, default_sub)
    rng = np.random.default_rng(0)
    for x, sub in settings:
        cur = _rel_by_instance(df, sub)
        common = sorted(set(cur) & set(ref))
        diff = np.array([cur[i] - ref[i] for i in common])
        if not len(diff):
            continue
        jit = (x * 10 ** rng.uniform(-0.035, 0.035, len(diff)) if xlog
               else x + rng.uniform(-0.035, 0.035, len(diff)) * span)
        ax.scatter(jit, diff, s=12, facecolor=ARM_COLOR[CONTROL], edgecolor="black",
                   lw=0.4, alpha=0.85, zorder=4)
        ax.plot([x], [np.median(diff)], marker="_", ms=16, mew=2.2,
                color="black", zorder=5)
    ax.axhline(0, color="black", lw=0.8, ls=":")


def fig_metrics(df):
    """Which trajectory metric should drive the proposed arm?  Every metric's
    plans are scored by Fréchet deviation."""
    d = df[df.study.str.startswith("metricv2") & (df.arm == CONTROL)]
    if d.empty:
        print("  [skip] fig_metrics: no metricv2 study in the database")
        return
    metrics = ["frechet", "dtw", "hausdorff", "lcss"]
    labels = {"frechet": "Fréchet", "dtw": "DTW", "hausdorff": "Hausdorff",
              "lcss": "LCSS"}
    # Feasible runs only, matching the macros in the text.  Computing the figure
    # one way and the numbers beside it another is how a figure and its caption
    # come to disagree.
    base = df[(df.study == "main") & (df.arm == "cost_only") & df.feasible]
    b3 = base.groupby("instance").f3.median()
    b1 = base.groupby("instance").f1.median()
    dat, dat2, keep = [], [], []
    for m in metrics:
        s = d[(d.study == f"metricv2:{m}") & d.feasible]
        if s.empty:
            continue
        g3 = s.groupby("instance").dev_frechet.median()
        g1 = s.groupby("instance").f1.median()
        c3, c1 = g3.index.intersection(b3.index), g1.index.intersection(b1.index)
        dat.append((100 * (1 - g3[c3] / b3[c3])).values)
        dat2.append((100 * (g1[c1] / b1[c1] - 1)).values)
        keep.append(m)
    colours = [ARM_COLOR[CONTROL] if m == "frechet" else GREY for m in keep]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, data, ylab, title in (
            (axes[0], dat, "reduction in Fréchet deviation\nvs cost-only (%)",
             "(a) Deviation reduction"),
            (axes[1], dat2, "change in travel distance\nvs cost-only (%)",
             "(b) Travel distance change")):
        pos = _boxes(ax, data, colours)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.set_xticks(pos)
        ax.set_xticklabels([labels[m] for m in keep])
        ax.set_ylabel(ylab)
        ax.set_title(title.split()[0], loc="left")
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
    """Seed-paired comparison of a sensitivity setting against a reference,
    normally the same arm in its default configuration.

    Every sensitivity claim in the paper -- trajectory metric, initialisation,
    resampling step, generation budget, weight scale -- goes through this.
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


def _study(df, name, arm=None):
    """Rows of one sensitivity study for one arm (the proposed arm by default).

    Studies for different arms share names, so every selection is by arm too.
    """
    return df[(df.study == name) & (df.arm == (arm or CONTROL))]


def fig_ablation(df):
    """Initialisation, resampling step and generation budget of the proposed arm."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ctrl = df[(df.study == "main") & (df.arm == CONTROL)]
    ylab = "difference from default\n(percentage points)"

    ax = axes[0]
    series = [(_rel_points(df, ctrl), "grafted", ARM_COLOR[CONTROL]),
              (_rel_points(df, _study(df, "init:random")), "random", GREY)]
    keep = [(v, l, c) for v, l, c in series if v]
    if keep:
        pos = _boxes(ax, [v for v, _, _ in keep], [c for _, _, c in keep])
        ax.set_xticks(pos)
        ax.set_xticklabels([l for _, l, _ in keep])
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_ylabel("reduction in $S$ vs cost-only (%)")
    ax.set_title("(a)", loc="left")

    ax = axes[1]
    _paired_panel(ax, df, [(st, _study(df, f"resample:{st}")) for st in (100, 500, 1000)],
                  ctrl, span=None, xlog=True)
    ax.axvline(250, color="black", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([100, 250, 500, 1000])
    ax.set_xticklabels(["100", "250", "500", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("resampling step $h$ (m)")
    ax.set_ylabel(ylab)
    ax.set_title("(b)", loc="left")

    ax = axes[2]
    _paired_panel(ax, df, [(g, _study(df, f"budget:{g}")) for g in (150, 600)],
                  ctrl, span=450)
    ax.axvline(300, color="black", ls=":", lw=0.9)
    ax.set_xticks([150, 300, 600])
    ax.set_xlabel("generation budget $G$")
    ax.set_ylabel(ylab)
    ax.set_title("(c)", loc="left")

    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_ablation")


def fig_scalability(df):
    """Contended run time against instance size.

    A fixed population and generation budget make selection, which does not
    depend on n, a large share of the per-generation cost, so the fitted
    exponent describes the tested range only.
    """
    arms = _uniq(["cost_only", CONTROL, "corridor"])
    d = df[df.study.isin(["main", "scale"]) & df.arm.isin(arms)]
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 2.9))
    for a in arms:
        s = d[d.arm == a]
        if s.empty:
            continue
        g = s.groupby("n_customers").runtime.median()
        ax.plot(g.index, g.values, "o-", color=ARM_COLOR[a], ms=4,
                label=ARM_LABEL[a], lw=1.2)
    g = d[d.arm == CONTROL].groupby("n_customers").runtime.median()
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
    # Plain numbers on both logarithmic axes: the range spans less than a
    # decade, so scientific-notation minor labels would carry every tick.
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter
    t_lo, t_hi = d.groupby(["arm", "n_customers"]).runtime.median().agg(["min", "max"])
    yt = [v for v in (5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400)
          if 0.85 * t_lo <= v <= 1.15 * t_hi]
    xt = [v for v in (50, 100, 200, 400) if 0.85 * d.n_customers.min() <= v
          <= 1.15 * d.n_customers.max()]
    for axis, ticks in ((ax.yaxis, yt), (ax.xaxis, xt)):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("requests $n$"); ax.set_ylabel("run time (s)")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, f"{FIGDIR}/fig_scalability")


# Driver hues for the map: the Okabe-Ito colours other than blue (cost-only)
# and vermillion (the proposed arm), then Tol's muted palette.  fig_maps also
# drops the colour of whichever arm it draws, because panel (c) uses the arm
# colours for the methods.
DRIVER_HUE = ["#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#882255", "#332288",
              "#999933", "#44AA99", "#AA4499", "#117733", "#DDCC77", "#88CCEE"]


def fig_maps(rows, instance):
    """Plan in force against revised routes, one hue per driver.

    S compares each driver's revised route with the same driver's baseline
    route, in traversal order.  The map therefore pairs the two routes of a
    driver by hue and marks the direction of travel, and panel (c) gives each
    driver's term of S for both methods.
    """
    from matplotlib.gridspec import GridSpec
    from ..rspp.instance import RSPPPair
    from ..rspp.solution import Evaluator
    path = f"data/rsppbench/{instance}.json"
    if not os.path.exists(path):
        return
    pair = RSPPPair.load(path)
    picks = {}
    for a in ("cost_only", CONTROL):
        cands = [r for r in rows if r["study"] == "main" and r["arm"] == a
                 and r["instance"] == instance and r.get("routes")]
        if cands:
            picks[a] = min(cands, key=lambda r: r["seed"])
    if len(picks) < 2:
        return
    ev = Evaluator(pair)
    K = pair.disrupted.n_vehicles
    taken = {ARM_COLOR[a].lower() for a in picks}
    hues = [h for h in DRIVER_HUE if h.lower() not in taken]
    dev = {a: [ev.route_deviation(k, picks[a]["routes"][k]) / 1000 for k in range(K)]
           for a in picks}
    base_c, dis_c = pair.base.coords / 1000, pair.disrupted.coords / 1000
    new_ids = sorted(set(pair.disruption.get("new_customers", [])))

    fig = plt.figure(figsize=(7.2, 3.05))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 0.30, 0.78], wspace=0.06, figure=fig)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharex=axes[0], sharey=axes[0]))
    axb = fig.add_subplot(gs[0, 3])

    def draw(ax, pts, hue, baseline):
        if baseline:
            ax.plot(pts[:, 0], pts[:, 1], color=hue, lw=2.3, alpha=0.45,
                    ls=(0, (2.6, 1.4)), zorder=1)
            return
        ax.plot(pts[:, 0], pts[:, 1], color=hue, lw=1.05, zorder=3)
        p, q = pts[0], pts[1]               # one arrowhead on the first leg
        ax.annotate("", xy=p + 0.55 * (q - p), xytext=p + 0.45 * (q - p),
                    arrowprops=dict(arrowstyle="-|>", color=hue, lw=0,
                                    mutation_scale=10), zorder=4)

    for ax, a, tag in zip(axes, ("cost_only", CONTROL), ("(a)", "(b)")):
        for k in range(K):
            hue = hues[k % len(hues)]
            if pair.baseline_routes[k]:
                draw(ax, base_c[[0] + list(pair.baseline_routes[k]) + [0]], hue, True)
            if picks[a]["routes"][k]:
                draw(ax, dis_c[[0] + list(picks[a]["routes"][k]) + [0]], hue, False)
        ax.scatter(dis_c[1:, 0], dis_c[1:, 1], s=4, color="black", lw=0, zorder=5)
        if new_ids:
            ax.scatter(dis_c[new_ids, 0], dis_c[new_ids, 1], s=34, facecolor="none",
                       edgecolor="black", lw=0.9, zorder=6)
        ax.scatter([dis_c[0, 0]], [dis_c[0, 1]], marker="s", s=34, color="black",
                   zorder=7)
        ax.set_title(f"{tag} {ARM_LABEL[a]}", loc="left")
        ax.set_aspect("equal")
        ax.set_xlabel("east (km)")
    axes[0].set_ylabel("north (km)")
    plt.setp(axes[1].get_yticklabels(), visible=False)

    y = np.arange(K)[::-1]
    hgt = 0.38
    for off, a in ((+hgt / 2, "cost_only"), (-hgt / 2, CONTROL)):
        axb.barh(y + off, dev[a], height=hgt, color=ARM_COLOR[a])
        for yy, v in zip(y + off, dev[a]):
            axb.text(v + 0.08, yy, f"{v:.1f}", va="center", fontsize=7.5)
    axb.set_yticks(y)
    axb.set_yticklabels([f"driver {k + 1}" for k in range(K)])
    for tick, k in zip(axb.get_yticklabels(), range(K)):
        tick.set_color(hues[k % len(hues)])
    axb.set_xlabel(r"deviation $\delta_k$ (km)")
    axb.set_title("(c)", loc="left")
    axb.grid(axis="y", visible=False)
    axb.set_xlim(0, max(max(dev["cost_only"]), max(dev[CONTROL])) * 1.22)

    handles = [plt.Line2D([], [], color="#555555", lw=2.3, alpha=0.45,
                          ls=(0, (2.6, 1.4)), label="plan in force"),
               plt.Line2D([], [], color="#555555", lw=1.05, label="revised route"),
               plt.Line2D([], [], marker="s", ls="", color="black", ms=5, label="depot"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec="black", mew=0.9,
                          ms=6, label="new request"),
               plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR["cost_only"],
                             label=ARM_LABEL["cost_only"]),
               plt.Rectangle((0, 0), 1, 1, color=ARM_COLOR[CONTROL],
                             label=ARM_LABEL[CONTROL])]
    fig.canvas.draw()
    p0, pb = axes[0].get_position(), axb.get_position()
    axb.set_position([pb.x0, p0.y0, pb.width, p0.height])
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.07), fontsize=7.5, handlelength=1.6,
               columnspacing=1.0)
    save(fig, f"{FIGDIR}/fig_maps")
    macro("MAPINSTANCE", instance.replace("_", "\\_"))
    macro("MAPSCOSTONLY", f"{sum(dev['cost_only']):.1f}")
    macro("MAPS" + _key(CONTROL), f"{sum(dev[CONTROL]):.1f}")
    macro("MAPNDRIVERS", int(len(ev.base_curves)))
    macro("MAPNUNCHANGED", int(sum(1 for k in ev.base_curves
                                   if dev[CONTROL][k] < 1e-9)))


# ------------------------------------------------- comparison at the stable end
def stable_end(rows):
    """Compare the most stable plan in each method's front with the plan the
    proposed arm returns.

    The plan a method returns depends on its final selection rule as well as on
    its search: a compromise from a spread front, or the single plan of a
    collapsed one.  This compares, seed by seed, the most stable hard-feasible
    plan in each method's returned front with the most stable plan in the
    proposed arm's front.

    Macros, per method:
      SMIN      reduction of the most stable plan's deviation relative to the
                most stable cost-only plan (per-instance medians, then median);
      WTLMIN, A12MIN, EFFMIN
                seed-paired test of that deviation against the most stable plan
                of the proposed arm, with the orientation of Table 2 (a win is
                a method worse than the proposed arm);
      STABLEGAP median reduction of the proposed arm's deviation relative to it;
      DOMSHARE  share of runs whose front holds a plan at least as good as the
                proposed arm's on all three objectives;
      MATCHSHARE, MATCHDF, MATCHDW
                share of runs whose front holds a plan at least as stable as
                the proposed arm's, and the proposed arm's distance and
                workload spread relative to the shortest such plan.
    """
    main_rows = [r for r in rows if r["study"] == "main"]
    by = {(r["instance"], r["arm"], r["seed"]): r for r in main_rows}
    insts = sorted({r["instance"] for r in main_rows})
    seeds = sorted({r["seed"] for r in main_rows})

    def smin(r):
        if r is None or not r.get("feasible") or not r.get("front"):
            return None
        return min(p[2] for p in r["front"])

    for a in ORDER:
        key = _key(a)
        rel, pairs, dom, gap, match, d1, d2 = [], [], [], [], [], [], []
        for i in insts:
            ref = [v for v in (smin(by.get((i, "cost_only", s))) for s in seeds)
                   if v is not None]
            cur = [v for v in (smin(by.get((i, a, s))) for s in seeds) if v is not None]
            if ref and cur and np.median(ref) > 0:
                rel.append(100 * (1 - np.median(cur) / np.median(ref)))
            if a == CONTROL:
                continue
            xs, ys, dm, gp, mt, f1d, f2d = [], [], [], [], [], [], []
            for s in seeds:
                c, b = by.get((i, CONTROL, s)), by.get((i, a, s))
                if smin(c) is None or smin(b) is None:
                    continue
                co = np.asarray(c["objectives"], float)
                F = np.asarray(b["front"], float)
                xs.append(F[:, 2].min()); ys.append(smin(c))
                dm.append(bool(np.any(np.all(F <= co + 1e-9, axis=1))))
                if F[:, 2].min() > 0:
                    gp.append(100 * (1 - co[2] / F[:, 2].min()))
                cand = F[F[:, 2] <= co[2] * 1.001 + 1e-6]
                mt.append(len(cand) > 0)
                if len(cand):
                    best = cand[np.argmin(cand[:, 0])]
                    f1d.append(100 * (co[0] / best[0] - 1))
                    if best[1] > 0:
                        f2d.append(100 * (co[1] / best[1] - 1))
            if len(xs) >= 5:
                pairs.append((xs, ys))
            if dm:
                dom.append(100 * float(np.mean(dm)))
            if gp:
                gap.append(float(np.median(gp)))
            if mt:
                match.append(100 * float(np.mean(mt)))
            if f1d:
                d1.append(float(np.median(f1d)))
            if f2d:
                d2.append(float(np.median(f2d)))
        if rel:
            macro("SMIN" + key, pct(float(np.median(rel)), 1))
        if a == CONTROL:
            continue
        t = ind.per_instance_comparison(pairs)
        if t["n_cmp"]:
            macro("WTLMIN" + key, f"{t['win']}/{t['tie']}/{t['loss']}")
            macro("A12MIN" + key, f"{t['a12']:.2f}")
            macro("EFFMIN" + key, t["effect"])
        for name, vals, digits in (("DOMSHARE", dom, 0), ("STABLEGAP", gap, 1),
                                   ("MATCHSHARE", match, 0), ("MATCHDF", d1, 1),
                                   ("MATCHDW", d2, 1)):
            if vals:
                macro(name + key, pct(float(np.median(vals)), digits))


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

    Two rules govern every number here.

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
    ctrl = d[(d.arm == CONTROL) & d.feasible]
    rows = []
    for a in ORDER:
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
    ctrl_idx = arms.index(CONTROL)
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
        if r_["arm"] == CONTROL:
            lines.append(f"\\textbf{{{name}}} & \\textbf{{{sv}}} & \\textbf{{{dv}}} & "
                         f"\\textbf{{{feas}}} & --- & --- & --- & --- \\\\")
        else:
            p_ = padj.get(k, 1.0)
            ptxt = pvalue(p_)
            wtl = f"{r_['win']}/{r_['tie']}/{r_['loss']}"
            lines.append(f"{name} & {sv} & {dv} & {feas} & {wtl} & "
                         f"{r_['a12']:.2f} & {ptxt} & {r_['effect']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{TEXDIR}/tab_main.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {TEXDIR}/tab_main.tex")

    cor = next(r for r in rows if r["arm"] == CONTROL)
    macro("MAINSDROP", pct(cor["S"], 1))
    macro("MAINSDROPIQR", f"{cor['Sq'][0]:.1f}--{cor['Sq'][1]:.1f}\\%")
    macro("MAINDISTCOST", pct(cor["D"], 1))
    # signed and unsigned forms, so prose never has to say "reduced by -2.2%"
    macro("MAINDISTABS", pct(abs(cor["D"]), 1))
    macro("MAINDISTVERB", "reducing" if cor["D"] < 0 else "increasing")
    # Distance measured against cost-only's cheapest returned plan (its front
    # minimum); the compromise-to-compromise change is reported alongside.
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
        macro("P" + key, pvalue(padj.get(k, 1.0)))
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
        rs, rd, rdm = [], [], []
        for inst in sorted(sub.instance.unique()):
            b = sub[(sub.instance == inst) & (sub.arm == "cost_only") & sub.feasible]
            c = sub[(sub.instance == inst) & (sub.arm == CONTROL) & sub.feasible]
            if b.empty or c.empty:
                continue
            rs.append(100 * (1 - c.f3.median() / b.f3.median()))
            rd.append(100 * (c.f1.median() / b.f1.median() - 1))
            # against cost-only's shortest returned plan, as in Table 2
            bmin = b.f1_front_min.median()
            if bmin == bmin and bmin > 0:
                rdm.append(100 * (c.f1.median() / bmin - 1))
        if rdm:
            macro("DMIN" + fam, pct(np.median(rdm), 1))
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
    for a in ORDER:
        if a not in agg:
            continue
        v = agg[a]
        name = ARM_LABEL[a]
        if a == CONTROL:
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
    # The results database, and the environment snapshot and yardstick written
    # next to it by the campaign runner.  RSPP_DB selects a rerun directory.
    db = os.environ.get("RSPP_DB", "experiments/results/campaign.jsonl")
    print(f"database: {db}")
    rows, errors = L.load(db)
    df = L.to_frame(rows)
    print(f"loaded {len(rows)} runs ({len(errors)} errors) "
          f"over {df.instance.nunique()} instances, {df.arm.nunique()} arms")
    if errors:
        print("  first error:", errors[0].get("error"))

    # The main comparison: the reported arms only.
    main_df = df[(df.study == "main") & df.arm.isin(ORDER)]
    macro("NRUNS", f"{len(df):,}".replace(",", "{,}"))
    macro("NRUNSMAIN", f"{len(main_df):,}".replace(",", "{,}"))
    macro("NSEEDS", int(main_df.seed.nunique()))
    macro("NARMS", int(main_df.arm.nunique()))
    macro("NEVALS", f"{int(main_df.n_evals.median()):,}".replace(",", "{,}"))

    # The execution environment is read from the snapshot the campaign runner
    # wrote next to the results, not from whichever machine happens to be
    # drawing the figures -- those need not be the same computer.
    env_path = os.path.join(os.path.dirname(db), "environment.json")
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
    macro("RUNTIMEMED", f"{main_df[main_df.arm == CONTROL].runtime.median():.1f}")

    # Uncontended timing.  Everything else in the campaign runs seven-way
    # parallel, so its wall-clock figures include contention; this study is run
    # with a single worker so that a per-run cost can be quoted.
    tm = df[(df.study == "timing") & (df.arm == CONTROL)]
    if not tm.empty:
        macro("RUNTIMESOLO", f"{tm.runtime.median():.1f}")
        macro("RUNTIMESOLOMAX", f"{tm.runtime.max():.1f}")
        macro("CPUTIMESOLO", f"{tm.cpu_time.median():.1f}")

    # Reproducibility across seeds: whether a method returns the same plan
    # from different seeds.
    for a in _uniq(["cost_only", "sim_obj", CONTROL, "corridor", "corridor_fixed"]):
        sub = main_df[(main_df.arm == a) & main_df.feasible]
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
    # Two maxima, because they answer different questions: the proposed arm's
    # main-study runs, and every run including the sensitivity studies.
    macro("RUNTIMEMAXMAIN", f"{main_df[main_df.arm == CONTROL].runtime.max():.1f}")
    macro("RUNTIMEMAXALL", f"{df.runtime.max():.1f}")

    # -------- front indicators, per instance, on normalised objectives
    # Normalisation bounds and reference fronts come from the reported arms.
    by_inst = L.fronts_by_instance([r for r in rows if r.get("arm") in ORDER], "main")
    per_inst = {}
    for inst, runs in by_inst.items():
        per_arm = ind.instance_front_indicators(runs)
        if per_arm is not None:
            per_inst[inst] = per_arm

    # -------- Friedman / Nemenyi over instances
    insts = sorted(per_inst)
    arms = [a for a in ORDER if a in main_df.arm.unique()]
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
    if "corridor" in arms:
        macro("CORRRANKS", f"{s_rank[0][arms.index('corridor')]:.2f}")
        macro("CORRRANKHV", f"{hv_rank[0][arms.index('corridor')]:.2f}")
    # Mean Friedman rank of every arm, on deviation and on hypervolume.
    for j, a in enumerate(arms):
        macro("FRANKS" + _key(a), f"{s_rank[0][j]:.2f}")
        macro("FRANKHV" + _key(a), f"{hv_rank[0][j]:.2f}")

    print("figures:")
    fig_mechanism(df, rows)
    fig_tradeoff(df, tradeoff_instances())
    fig_summary(df, per_inst)
    from .driver_burden import analyse as analyse_driver_burden
    driver_summary = analyse_driver_burden(rows, FIGDIR)
    if driver_summary is not None:
        macro("DRIVERBURDENNPLANS", driver_summary["records"])
        for arm, values in driver_summary["arms"].items():
            macro("DMAXDROP" + _key(arm), pct(values["median_max_reduction_pct"], 2))
            macro("DMAXWORSE" + _key(arm), values["total_improves_max_worsens"])
    fig_violin(df)
    fig_cd([("Trajectory deviation $S$", (arms, s_rank[0], s_rank[1], s_rank[2])),
            ("Hypervolume of the returned front",
             (arms, hv_rank[0], hv_rank[1], hv_rank[2]))])
    # Illustrative instances are chosen by a stated rule rather than by
    # position in an alphabetical listing.  Dynamics are shown on the largest
    # vehicle-withdrawal instance, the case that most invalidates the plan in
    # force; the map is drawn on a mid-sized new-request instance, where the
    # arriving work is visible.
    def _pick(family, by_size=max):
        cand = main_df[main_df.family == family]
        if cand.empty:
            return sorted(insts)[0]
        target = by_size(cand.n_customers.unique())
        return sorted(cand[cand.n_customers == target].instance.unique())[0]

    dyn_inst = _pick("WDR", max)
    map_inst = _pick("INS", lambda v: sorted(v)[len(v) // 2])
    if CONTROL.startswith("corridor"):
        fig_corridor_dynamics(rows, dyn_inst)
    else:
        fig_dynamics(rows, dyn_inst)
    fig_metrics(df)
    fig_ablation(df)
    fig_scalability(df)
    fig_maps(rows, map_inst)
    from .frechet_coupling import analyse as analyse_coupling
    analyse_coupling(rows, FIGDIR)
    from .uncertainty import analyse as analyse_uncertainty
    analyse_uncertainty(rows, FIGDIR, TEXDIR)

    # ---- sub-study summaries, all relative to cost-only on the same instances
    SUBSET = list(SUBSET_PLOT)
    macro("NSUBSET", len(SUBSET))
    base = main_df[(main_df.arm == "cost_only") & main_df.feasible]
    # Sensitivity studies of the proposed arm are set against its default
    # configuration, and the studies of the tightening bound (schedule,
    # calibration, floor) against the bound's.  Studies of different arms share
    # names, so every selection goes through _study.
    ctrl_ref = main_df[(main_df.arm == CONTROL) & main_df.feasible]
    ctrl_ref_all = main_df[main_df.arm == CONTROL]
    corr_ref = main_df[(main_df.arm == "corridor") & main_df.feasible]

    NOTEST = "\\textcolor{red}{\\textbf{[no test]}}"

    def _emit_test(name, sub, col="f3", ref=None):
        """Emit A_12 / win-tie-loss / p for one setting against the default
        (the proposed arm's unless ``ref`` is given).

        A cell with too few paired seeds to test renders as a loud marker rather
        than as ``nan``, which would read as a number in the typeset paper.
        """
        t = _ablation_test(df, sub, ctrl_ref if ref is None else ref, col)
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

    met_subs = [_study(df, f"metricv2:{m}")
                for m in ("frechet", "dtw", "hausdorff", "lcss")]
    met_common = _common(*[x for x in met_subs if not x.empty]) if any(
        not x.empty for x in met_subs) else set()
    for met in ("frechet", "dtw", "hausdorff", "lcss"):
        sub = _study(df, f"metricv2:{met}")
        if sub.empty:
            continue
        macro("MET" + met.upper(), pct(rel(sub, "dev_frechet", common=met_common), 1))
        macro("METD" + met.upper(),
              pct(rel(sub, "f1", "f1", sign=+1, common=met_common), 1))
        if met != "frechet":
            _emit_test("MET" + met.upper(), sub, "dev_frechet")

    ir = _study(df, "init:random")
    if not ir.empty:
        ic = _common(ir, ctrl_ref)
        macro("INITGRAFT", pct(rel(ctrl_ref, common=ic), 1))
        macro("INITRANDOM", pct(rel(ir, common=ic), 1))
        macro("INITRANDOMTIME", f"{ir.runtime.median():.1f}")
        # Same instances as the random-initialisation runs it is set against.
        macro("INITGRAFTTIME",
              f"{ctrl_ref_all[ctrl_ref_all.instance.isin(SUBSET)].runtime.median():.1f}")
        _emit_test("INITRANDOM", ir)

    # The tightening bound's schedule, against its own linear default.
    sg = _study(df, "schedule:gaussian", "corridor")
    if not sg.empty:
        cf = main_df[main_df.arm == "corridor_fixed"]
        sc = _common(sg, corr_ref, cf)
        macro("SCHEDLINEAR", pct(rel(corr_ref, common=sc), 1))
        macro("SCHEDGAUSS", pct(rel(sg, common=sc), 1))
        macro("SCHEDFIXED", pct(rel(cf, common=sc), 1))
        _emit_test("SCHEDGAUSS", sg, ref=corr_ref)
        _emit_test("SCHEDFIXED", cf, ref=corr_ref)

    res_subs = [_study(df, f"resample:{st}") for st in (100, 500, 1000)]
    rc = _common(*[x for x in res_subs if not x.empty], ctrl_ref) if any(
        not x.empty for x in res_subs) else set(SUBSET)
    for st, rs in zip((100, 500, 1000), res_subs):
        if not rs.empty:
            macro(f"RESTIME{st}", f"{rs.runtime.median():.1f}")
            macro(f"RES{st}", pct(rel(rs, common=rc), 1))
            _emit_test(f"RES{st}", rs)
    macro("RESTIME250",
          f"{ctrl_ref_all[ctrl_ref_all.instance.isin(SUBSET)].runtime.median():.1f}")
    macro("RES250", pct(rel(ctrl_ref, common=rc), 1))

    # The tightening bound's initial width, against its default q_0 = 0.90.
    cal_subs = [_study(df, f"calib:q{q}", "corridor") for q in (25, 50, 75)]
    cc = _common(*[x for x in cal_subs if not x.empty], corr_ref) if any(
        not x.empty for x in cal_subs) else set(SUBSET)
    for q, cb in zip((25, 50, 75), cal_subs):
        if not cb.empty:
            macro(f"CALIBQ{q}", pct(rel(cb, common=cc), 1))
            macro(f"CALIBQ{q}TAU", f"{cb.tau0.median()/1000:.0f}")
            _emit_test(f"CALIBQ{q}", cb, ref=corr_ref)
    macro("CALIBQNINETY", pct(rel(corr_ref, common=cc), 1))
    # Both corridor-width macros are now subset medians on the same instances,
    # so the width quoted in the results section and the width quoted in the
    # calibration study are the same quantity.
    macro("CALIBQNINETYTAU",
          f"{main_df[(main_df.arm=='corridor') & main_df.instance.isin(SUBSET)].tau0.median()/1000:.0f}")

    # Generation-budget sufficiency: is 300 generations enough?
    bud_subs = [_study(df, f"budget:{g}") for g in (150, 600)]
    bc = _common(*[x for x in bud_subs if not x.empty], ctrl_ref) if any(
        not x.empty for x in bud_subs) else set(SUBSET)
    for g, bg in zip((150, 600), bud_subs):
        if not bg.empty:
            macro(f"BUDGET{g}", pct(rel(bg, common=bc), 1))
            _emit_test(f"BUDGET{g}", bg)
    macro("BUDGET300", pct(rel(ctrl_ref, common=bc), 1))

    # The weighted against the unweighted deviation objective, head to head.
    # Table 2 compares every arm against the proposed arm instead.
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

    # The comparison when each arm's compromise plan ranges only over the
    # objectives it optimises (Table 2, "Per-arm minimax choice").
    alt_rel = []
    for i in sorted(main_df.instance.unique()):
        b = main_df[(main_df.arm == "cost_only") & (main_df.instance == i)
                    & main_df.feasible]
        c = main_df[(main_df.arm == CONTROL) & (main_df.instance == i)
                    & main_df.feasible]
        if b.empty or c.empty or not (b.alt_f3.median() > 0):
            continue
        alt_rel.append(100 * (1 - c.alt_f3.median() / b.alt_f3.median()))
    if alt_rel:
        macro("MAINSDROPALT", pct(float(np.median(alt_rel)), 1))
        bb = main_df[(main_df.arm == "cost_only") & main_df.feasible]
        macro("ALTKNEEBASE", f"{bb.alt_f3.median()/1000:.1f}")
        macro("KNEEBASE", f"{bb.f3.median()/1000:.1f}")

    # Workload spread.  The percentage is the median across instances of the
    # per-instance ratio; the absolute minutes are the median across instances
    # of the per-instance median spread.  The ratio of two such absolute
    # medians therefore differs from the percentage.
    all_insts = sorted(main_df.instance.unique())

    def _eq_abs(sub):
        per = [sub[sub.instance == i].f2.median() / 60.0 for i in all_insts
               if not sub[sub.instance == i].empty]
        return float(np.median(per)) if per else float("nan")

    for a in ORDER:
        sub = main_df[(main_df.arm == a) & main_df.feasible]
        if sub.empty:
            continue
        key = _key(a)
        macro("EQ" + key, pct(rel(sub, "f2", "f2", insts=all_insts, sign=+1), 1))
        # One decimal, as quoted in the text.
        macro("EQ" + key + "ABS", f"{_eq_abs(sub):.1f}")

    # Greedy repair's spread is a median over the instances on which it is
    # feasible; the proposed arm's spread over the same instances makes the two
    # numbers comparable.
    g_set = sorted(main_df[(main_df.arm == "greedy") & main_df.feasible].instance.unique())
    cg = main_df[(main_df.arm == CONTROL) & main_df.feasible
                 & main_df.instance.isin(g_set)]
    per = [cg[cg.instance == i].f2.median() / 60.0 for i in g_set
           if not cg[cg.instance == i].empty]
    if per:
        macro("EQ" + _key(CONTROL) + "ABSGREEDYSET", f"{np.median(per):.1f}")

    # ------------------------------------------ relabelling output step (§6.4)
    # How much the step changes: the share of runs whose reported deviation is
    # below that of the plan the search selected, and the per-instance median
    # change.  Databases written before the step existed lack the raw column.
    if "f3_raw" in main_df and main_df.f3_raw.notna().any():
        for a in _uniq(["cost_only", CONTROL, "corridor"]):
            s = main_df[(main_df.arm == a) & main_df.feasible & main_df.f3_raw.notna()]
            if s.empty:
                continue
            key = _key(a)
            macro("RELABELSHARE" + key,
                  pct(100 * float(np.mean(s.f3 < s.f3_raw - 1e-6)), 1))
            per = [100 * (1 - g.f3.median() / g.f3_raw.median())
                   for _, g in s.groupby("instance") if g.f3_raw.median() > 0]
            if per:
                macro("RELABELDROP" + key, pct(float(np.median(per)), 1))

    # ------------------------------------------------ distance yardstick (§9.2)
    # Not an arm: the shortest plan OR-Tools finds for each disrupted instance,
    # which locates the distance end of the trade-off.
    ypath = os.environ.get("RSPP_YARDSTICK",
                           os.path.join(os.path.dirname(db), "yardstick.jsonl"))
    if os.path.exists(ypath):
        Y = {}
        with open(ypath) as fh:
            for line in fh:
                if line.strip():
                    y = json.loads(line)
                    if y.get("feasible"):
                        Y[y["instance"]] = y
        # Instances on which both cost-only and the proposed arm return a
        # hard-feasible plan; the tightening bound is reported on the same set.
        gap_c, spread = [], []
        gap = {a: [] for a in _uniq([CONTROL, CONTROL + "_x4", CONTROL + "_x16",
                                     "corridor"])}
        for i, y in sorted(Y.items()):
            b = main_df[(main_df.instance == i) & (main_df.arm == "cost_only")
                        & main_df.feasible]
            c = main_df[(main_df.instance == i) & (main_df.arm == CONTROL)
                        & main_df.feasible]
            if b.empty or c.empty:
                continue
            f1y = y["objectives"][0]
            gap_c.append(100 * (b.f1_front_min.median() / f1y - 1))
            spread.append(y["objectives"][1] / 60.0)
            for a in gap:
                s_ = main_df[(main_df.instance == i) & (main_df.arm == a)
                             & main_df.feasible]
                if not s_.empty:
                    gap[a].append(100 * (s_.f1.median() / f1y - 1))
        if gap[CONTROL]:
            macro("YARDNINST", len(gap[CONTROL]))
            macro("YARDCOSTONLY", pct(float(np.median(gap_c)), 1))
            for a, g in gap.items():
                if not g:
                    continue
                macro("YARD" + _key(a), pct(float(np.median(g)), 1))
                q1, q3 = np.percentile(g, [25, 75])
                macro("YARD" + _key(a) + "IQR", f"{q1:.1f}--{q3:.1f}\\%")
            macro("YARDSPREADABS", f"{np.median(spread):.1f}")
            first = next(iter(Y.values()))
            macro("YARDSOLUTIONS", f"{first['solution_limit']:,}".replace(",", "{,}"))
            macro("YARDVERSION", first["ortools"])
            macro("YARDCAPHIT", int(sum(1 for y in Y.values() if y.get("hit_time_cap"))))

    # ------------------------------------------------- non-vacuity floor q_min
    # q_0 fixes where the corridor starts; q_min fixes how far it may close, and
    # it is the parameter Proposition 3 is stated in terms of.  Its own
    # sensitivity therefore cannot be left to the calibration study.
    QMIN_LEVELS = [("QMINQONE", 1), ("QMINQFIVE", 5), ("QMINQTWENTY", 20)]
    qsubs = [_study(df, f"qmin:q{n}", "corridor") for _, n in QMIN_LEVELS]
    if any(not x.empty for x in qsubs):
        qc = _common(*[x for x in qsubs if not x.empty], corr_ref)
        corr_sub = corr_ref[corr_ref.instance.isin(SUBSET)]
        for (name, n), qs in zip(QMIN_LEVELS, qsubs):
            if qs.empty:
                continue
            macro(name, pct(rel(qs, common=qc), 1))
            macro(name + "FEAS", f"{100 * qs.feasible.mean():.1f}\\%")
            macro(name + "TAU", f"{qs.tau_final.median() / 1000:.1f}")
            _emit_test(name, qs, ref=corr_ref)
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
    # the corridor actually closes.  Both are properties of Eq. (12) that the
    # width alone does not report.
    cs = main_df[(main_df.arm == "corridor") & main_df.feasible]
    if "sched_frac" in cs and cs.sched_frac.notna().any():
        macro("CORRSCHEDFRAC", pct(100 * cs.sched_frac.median(), 0))
        macro("CORRTAUFINAL", f"{cs.tau_final.median() / 1000:.1f}")
    # The width can reach zero, and where it does the metre guard s_min in
    # Eq. (10) is active.  It happens exactly when at least q_min of the pool
    # reproduces the baseline geometry, so the floor is zero.
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
        # Table 2 is a median over a different instance set from every other
        # arm's.  Every comparison drawn against it in the text is therefore
        # restricted to the instances on which it returns a plan at all.
        eps_common = set(eps[eps.feasible].instance.unique())
        all_i = sorted(main_df.instance.unique())
        for a in _uniq(["sim_obj", "eps_static", "corridor_fixed", "corridor", CONTROL]):
            sub = main_df[main_df.arm == a]
            nm = _key(a)
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
    for a in _uniq([CONTROL, "corridor"]):
        inf = main_df[(main_df.arm == a) & (~main_df.feasible)]
        key = _key(a)
        macro("INFEASN" + key, int(len(inf)))
        if len(inf):
            macro("INFEASINST" + key, int(inf.instance.nunique()))
            macro("INFEASMIN" + key, f"{100 * inf.violation.min():.1f}\\%")
            macro("INFEASMAX" + key, f"{100 * inf.violation.max():.1f}\\%")
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

    stable_end(rows)

    print("tables:")
    write_bench_table()
    write_main_table(df, per_inst)
    write_family_table(df)
    write_indicator_table(per_inst)

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

    write_macro_fallback(MACROS)
    return df, per_inst


def write_macro_fallback(values):
    # Any macro a document references but the campaign has not produced renders
    # as a loud red marker rather than breaking the build.
    import re as _re
    used = set()
    for f in (("RSPP_COR_revised.tex" if os.path.exists(os.path.join(TEXDIR, "RSPP_COR_revised.tex")) else "manuscript.tex"),
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
            # A generated macro may also be written bare (followed by
            # punctuation).  Those are matched against values rather than taken
            # on shape, so no standard LaTeX command is ever given a
            # \providecommand that would collide with a later definition.
            used |= {m for m in _re.findall(r"\\([A-Za-z]{3,})(?![A-Za-z])", text)
                     if m in values}
    used -= {"PENDING"}                     # takes an argument, not a macro
    with open(f"{TEXDIR}/macros_fallback.tex", "w") as fh:
        fh.write("% Generated. Loaded AFTER results_macros.tex; only fills gaps.\n")
        for k in sorted(used):
            fh.write("\\providecommand{\\%s}{\\textcolor{red}{\\textbf{[??%s]}}}\n"
                     % (k, k))
    missing = sorted(used - set(values))
    print(f"  wrote {TEXDIR}/macros_fallback.tex"
          + (f"  -- STILL MISSING: {missing}" if missing else "  -- all resolved"))


if __name__ == "__main__":
    main()
