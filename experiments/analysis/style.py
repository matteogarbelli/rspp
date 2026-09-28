"""Shared figure style: colourblind-safe, serif, vector output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito qualitative palette (safe for all common colour-vision deficiencies)
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILLION, PURPLE = "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
GREY = "#8C8C8C"

ARM_LABEL = {
    "greedy": "Greedy repair",
    "cost_only": "Cost-only",
    "sim_obj": "Deviation objective",
    "sim_lambda": "Adaptive weight",
    "eps_static": "Static $\\varepsilon$-constraint",
    "corridor": "Corridor (proposed)",
    "corridor_fixed": "Corridor, no tightening",
    "corridor_nsga2": "Corridor + NSGA-II",
    "corridor_moead": "Corridor + MOEA/D",
}
ARM_COLOR = {
    "greedy": GREY, "cost_only": BLUE, "sim_obj": SKY, "sim_lambda": ORANGE,
    "eps_static": BLACK,          # PURPLE is taken by corridor_moead
    "corridor": VERMILLION, "corridor_fixed": YELLOW,
    "corridor_nsga2": GREEN, "corridor_moead": PURPLE,
}
ARM_ORDER = ["greedy", "cost_only", "sim_obj", "sim_lambda", "eps_static",
             "corridor", "corridor_fixed", "corridor_nsga2", "corridor_moead"]


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, path):
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"  wrote {path}.pdf")
