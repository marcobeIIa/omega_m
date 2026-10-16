import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, SURF = "#1a1a19", "#6b6a63", "#e5e4de", "#fcfcfb"
C1, C2, C3 = "#eb6834", "#2a78d6", "#1baf7a"          # 1 bin, 2 bins, 3 bins (fixed order)
FITCOL = {"one_bin": C1, "two_bins": C2, "three_bins": C3}
FITMK = {"one_bin": "o", "two_bins": "s", "three_bins": "^"}
FITLAB = {"one_bin": "1 bin (ΛCDM)", "two_bins": "2 bins (edge 968)", "three_bins": "3 bins (edges 871, 1230)"}
CLASSC, CPC = "#1a1a19", "#2a78d6"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "mathtext.fontset": "dejavusans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
FIG = "/tmp/claude-0/-home-claude/292e6737-93c7-5f4e-80bb-a872aa2b0bd3/scratchpad/deck/figs/"
def save(fig, name):
    fig.savefig(FIG + name + ".pdf"); fig.savefig(FIG + name + ".png", dpi=170)
    plt.close(fig)
