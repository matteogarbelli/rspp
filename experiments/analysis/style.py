"""Shared figure style: colourblind-safe, serif, vector output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito qualitative palette (safe for all common colour-vision deficiencies)
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILLION, PURPLE = "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
GREY = "#8C8C8C"

WINE = "#882255"     # Tol's muted palette, for the arm that lost vermillion

# Deviation can enter the search in three places: as a separate objective
# (plain or with the adaptive weight), summed into distance with the adaptive
# weight scaled by c (the proposed method), or as a bound (static, constant or
# tightening).
ARM_LABEL = {
    "greedy": "Greedy repair",
    "cost_only": "Cost-only",
    "sim_obj": "Separate objective",
    "sim_lambda": "Separate objective, weighted",
    "sum_lambda": "Summed deviation, $c=1$",
    "sum_lambda_x4": "Summed deviation, $c=4$",
    "sum_lambda_x16": "Summed deviation, $c=16$",
    "eps_static": "Static $\\varepsilon$-constraint",
    "corridor_fixed": "Constant bound",
    "corridor": "Tightening bound",
    "sum_lambda_nsga2": "Summed deviation + NSGA-II",
    "sum_lambda_moead": "Summed deviation + MOEA/D",
    "corridor_nsga2": "Tightening bound + NSGA-II",
    "corridor_moead": "Tightening bound + MOEA/D",
}
ARM_COLOR = {
    "greedy": GREY, "cost_only": BLUE, "sim_obj": SKY, "sim_lambda": ORANGE,
    "sum_lambda": VERMILLION, "eps_static": BLACK,
    # The rates of the summed objective darken with c.
    "sum_lambda_x4": "#9E3B00", "sum_lambda_x16": "#5C2200",
    "corridor_fixed": YELLOW, "corridor": WINE,
    "sum_lambda_nsga2": GREEN, "sum_lambda_moead": PURPLE,
    "corridor_nsga2": "#117733", "corridor_moead": "#AA4499",
}
# Arms of the main comparison, in table order.  The analysis appends the
# NSGA-II and MOEA/D variants of whichever arm it reports as proposed; the
# figures that compare arms per instance draw these arms only.
ARM_ORDER = ["greedy", "cost_only", "sim_obj", "sim_lambda", "sum_lambda",
             "sum_lambda_x4", "sum_lambda_x16", "eps_static", "corridor_fixed",
             "corridor"]
# Marker shapes separate the rates of the summed objective, whose colours share
# one hue; every other arm uses a circle.
ARM_MARKER = {"sum_lambda_x4": "s", "sum_lambda_x16": "^"}


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
