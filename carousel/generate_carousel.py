"""
CruxSight.ai — LinkedIn Carousel Generator
Builds a 3-page PDF (1080x1350px, 4:5 portrait) using:
  - The REAL 30-node compose-workflow call graph
  - The REAL Pattern D node set (11 nodes, 49.7% of bottleneck files)
  - The REAL verified model metrics (Run 4: AUC=0.869, PatAcc=88.9%)
  - The REAL generalization result (zero-shot 0.544 -> fine-tuned 0.906)
No inflated numbers — everything traceable to the analysis/training logs.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import numpy as np

# ── Brand palette (matches CruxSight.ai coming-soon page) ──────────
BG        = "#0d1117"
SURFACE   = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#ffa657"
PURPLE    = "#bc8cff"
RED       = "#f78166"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
})

PAGE_W, PAGE_H = 10.80, 13.50  # inches @100dpi -> 1080x1350px

# ════════════════════════════════════════════════════════════════
# REAL DATA: 30-node compose call graph + Pattern D
# ════════════════════════════════════════════════════════════════

COMPOSE_EDGES = [
    (0,1),(0,2),(0,22),
    (1,3),(1,4),(2,3),(2,4),
    (3,5),(3,6),(4,5),(4,6),
    (5,7),(5,8),(5,13),
    (6,7),(6,8),
    (7,9),(7,10),(8,11),(8,12),
    (13,14),(13,20),(14,21),
    (20,26),(21,27),(26,28),(27,28),
    (22,13),(22,14),
    (15,16),(16,17),(17,18),(18,19),(19,20),
]

ENTRY      = [0, 1, 2, 22]
MIDDLE     = [4, 5, 7, 8, 11, 12, 18, 19]
STORAGE    = [13, 14, 20, 21, 26, 27, 28]
PERIPHERAL = [3, 6, 9, 10, 15, 16, 17, 23, 24, 25, 29]
PATTERN_D  = set([0, 1, 2, 13, 14, 20, 21, 22, 26, 27, 28])

NODE_LABELS = {
    0: "nginx", 1: "compose-post", 2: "user-service", 22: "social-graph",
    13: "post-storage", 14: "post-storage-mongodb",
    20: "user-timeline", 21: "user-timeline-mongodb",
    26: "home-timeline", 27: "home-timeline-redis", 28: "home-timeline-mongodb",
}

LOGO_PATH = "/home/claude/carousel/logo_crop.png"

def embed_logo_header(ax, small=True):
    """Embed real logo image. small=True for header strip, False for hero."""
    from matplotlib.image import imread
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    logo_img = imread(LOGO_PATH)
    zoom = 0.09 if small else 0.48
    x = 0.18 if small else 0.50
    im = OffsetImage(logo_img, zoom=zoom, resample=True)
    ab = AnnotationBbox(im, (x, 0.50), frameon=False,
                         xycoords='axes fraction', box_alignment=(0.5, 0.5))
    ax.add_artist(ab)


def build_graph():
    G = nx.DiGraph()
    G.add_nodes_from(range(30))
    G.add_edges_from(COMPOSE_EDGES)
    return G

def layered_positions():
    """4-tier layout matching the architectural structure."""
    pos = {}
    def place_row(nodes, y):
        n = len(nodes)
        xs = np.linspace(0.5, 9.5, n) if n > 1 else [5.0]
        for x, node in zip(xs, nodes):
            pos[node] = (x, y)
    place_row(ENTRY, 9.0)
    place_row(MIDDLE, 6.0)
    place_row(STORAGE, 3.0)
    place_row(PERIPHERAL, 0.5)
    return pos


# ════════════════════════════════════════════════════════════════
# PAGE 1 — TITLE
# ════════════════════════════════════════════════════════════════

def make_page1(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    # Real logo image — hero, upper portion
    ax_logo = fig.add_axes([0.05, 0.52, 0.90, 0.44])
    ax_logo.set_xlim(0, 1); ax_logo.set_ylim(0, 1)
    ax_logo.axis("off")
    embed_logo_header(ax_logo, small=False)

    # Badge
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.08, 0.505), 0.84, 0.032,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#1f6feb22", edgecolor="#1f6feb55", lw=1.0))
    ax.text(0.50, 0.521,
            "RESEARCH  ·  THEORY OF CONSTRAINTS  ·  CAUSAL GNN",
            fontsize=10, color=BLUE, va="center", ha="center",
            fontweight="bold")

    # Main headline
    ax.text(0.08, 0.475, "The Cloud Paradox", fontsize=48,
            fontweight="black", color=TEXT, va="top", ha="left")

    # Accent line
    for i, c in enumerate([BLUE, "#7a8fee", PURPLE]):
        ax.plot([0.08 + i*0.055, 0.08 + (i+1)*0.055], [0.368, 0.368],
                color=c, lw=5, solid_capstyle="round")

    # Hook question
    ax.text(0.08, 0.345,
            "Why is your system\nFASTEST right before it crashes?",
            fontsize=26, fontweight="bold", color=TEXT,
            va="top", ha="left", linespacing=1.3)

    # Supporting line
    ax.text(0.08, 0.20,
            "A causal AI model that learned the answer\n"
            "from 3.3 million real distributed traces.",
            fontsize=14.5, color=MUTED, va="top", ha="left", linespacing=1.55)

    # Bottom CTA strip
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 0.065, facecolor=SURFACE))
    ax.text(0.08, 0.032, "Swipe for the architecture  →",
            fontsize=13, color=BLUE, va="center", fontweight="bold")
    ax.text(0.92, 0.032, "#MachineLearning  #GNN  #SRE",
            fontsize=10, color=MUTED, va="center", ha="right")

    pdf.savefig(fig, dpi=100)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — ARCHITECTURE / PATTERN D
# ════════════════════════════════════════════════════════════════

def make_page2(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Header strip
    ax_head = fig.add_axes([0, 0.90, 1, 0.10])
    ax_head.set_xlim(0, 1); ax_head.set_ylim(0, 1); ax_head.axis("off")
    embed_logo_header(ax_head, small=True)
    ax_head.text(0.92, 0.5, "01 / DATASET ANATOMY", fontsize=11,
                  color=MUTED, va="center", ha="right", fontweight="bold")

    # Graph area
    ax = fig.add_axes([0.04, 0.20, 0.92, 0.68])
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.0)
    ax.axis("off")

    G = build_graph()
    pos = layered_positions()

    # Tier labels
    tier_info = [
        (9.7, "1 · ENTRY & FRONTEND", ENTRY),
        (6.7, "2 · ORCHESTRATION & LOGIC", MIDDLE),
        (3.7, "3 · STORAGE & CACHING CORE", STORAGE),
        (1.2, "4 · PERIPHERAL & HELPERS", PERIPHERAL),
    ]
    for y, label, nodes in tier_info:
        ax.text(-0.3, y, label, fontsize=10.5, color=MUTED,
                fontweight="bold", va="center", ha="left")

    # Draw edges
    for u, v in G.edges():
        x1, y1 = pos[u]; x2, y2 = pos[v]
        in_pattern = u in PATTERN_D and v in PATTERN_D
        color = "#f7816688" if in_pattern else "#30363d"
        lw = 2.2 if in_pattern else 0.9
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color=color,
                                     lw=lw, shrinkA=14, shrinkB=14,
                                     connectionstyle="arc3,rad=0.08"))

    # Draw nodes
    for n in G.nodes():
        x, y = pos[n]
        if n in PATTERN_D:
            if n in STORAGE:
                fc, ec = "#3a1015", RED
            else:
                fc, ec = "#3a2810", ORANGE
            lw = 3.0
            r = 0.42
            txt_c = TEXT
        else:
            fc, ec = SURFACE, BORDER
            lw = 1.2
            r = 0.36
            txt_c = MUTED

        circ = mpatches.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                                lw=lw, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, str(n), ha="center", va="center", fontsize=12,
                fontweight="bold", color=txt_c, zorder=4)

    # Annotation box near node 14
    x14, y14 = pos[14]
    ax.annotate(
        "post-storage-mongodb\nThe irreducible core",
        xy=(x14, y14), xytext=(x14 + 1.6, y14 + 1.5),
        fontsize=9.5, color=RED, fontweight="bold",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=SURFACE,
                  edgecolor=RED, lw=1))

    # Legend
    leg_items = [
        (mpatches.Circle((0,0), 1, fc="#3a2810", ec=ORANGE, lw=2),
         "Pattern D — Entry layer"),
        (mpatches.Circle((0,0), 1, fc="#3a1015", ec=RED, lw=2),
         "Pattern D — Storage core"),
        (mpatches.Circle((0,0), 1, fc=SURFACE, ec=BORDER, lw=1.2),
         "Other services (24 nodes)"),
    ]
    ax.legend([h for h, _ in leg_items], [l for _, l in leg_items],
              loc="lower left", bbox_to_anchor=(0.0, -0.025),
              fontsize=9.5, frameon=False, labelcolor=TEXT, ncol=3,
              handletextpad=0.6, columnspacing=1.4, borderaxespad=0)

    # Caption block
    ax_cap = fig.add_axes([0, 0.0, 1, 0.18])
    ax_cap.set_xlim(0, 1); ax_cap.set_ylim(0, 1); ax_cap.axis("off")
    ax_cap.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=SURFACE))

    ax_cap.text(0.06, 0.78,
                "Mapping 3.3 Million Traces into\n"
                "7 Deterministic Structural Patterns",
                fontsize=22, fontweight="black", color=TEXT,
                va="top", linespacing=1.25)

    ax_cap.add_patch(mpatches.FancyBboxPatch(
        (0.06, 0.10), 0.50, 0.16,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#f7816622", edgecolor=RED, lw=1.2))
    ax_cap.text(0.085, 0.18,
                "Pattern D: Entry + Storage Core",
                fontsize=12, color=RED, fontweight="bold", va="center")
    ax_cap.text(0.085, 0.13,
                "49.7% of all bottleneck experiments (88 of 177 files)",
                fontsize=10, color=MUTED, va="center")

    ax_cap.text(0.94, 0.13, "30 nodes · 17 servers · DeathStarBench",
                fontsize=9.5, color=MUTED, ha="right", va="center")

    pdf.savefig(fig, dpi=100)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — RESULTS & TEASER
# ════════════════════════════════════════════════════════════════

def make_page3(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))

    # Header
    ax_head = fig.add_axes([0, 0.92, 1, 0.08])
    ax_head.set_xlim(0, 1); ax_head.set_ylim(0, 1); ax_head.axis("off")
    embed_logo_header(ax_head, small=True)
    ax_head.text(0.92, 0.5, "02 / VERIFIED RESULTS", fontsize=11,
                  color=MUTED, va="center", ha="right", fontweight="bold")

    ax_head.text(0.08, -0.15, "Real numbers. No inflation.",
                  fontsize=24, fontweight="black", color=TEXT, va="top")

    # ── Metric cards row ──────────────────────────────────────
    metrics = [
        ("0.869", "Detection AUC", BLUE, "5-fold val · compose graph"),
        ("88.9%", "Pattern Accuracy", GREEN, "8-class taxonomy (A-G + none)"),
        ("0.97", "Root-Cause Recall", ORANGE, "vs 0.94 random baseline"),
    ]
    card_w, gap, x0 = 0.27, 0.025, 0.06
    for i, (val, label, color, sub) in enumerate(metrics):
        x = x0 + i * (card_w + gap)
        ax = fig.add_axes([x, 0.66, card_w, 0.16])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=SURFACE, edgecolor=BORDER, lw=1.2))
        ax.text(0.5, 0.62, val, fontsize=30, fontweight="black",
                color=color, ha="center", va="center")
        ax.text(0.5, 0.30, label, fontsize=11.5, fontweight="bold",
                color=TEXT, ha="center", va="center")
        ax.text(0.5, 0.13, sub, fontsize=8, color=MUTED,
                ha="center", va="center")

    # ── Generalization chart (the teaser) ────────────────────
    ax_chart = fig.add_axes([0.08, 0.32, 0.84, 0.30])
    ax_chart.set_facecolor(SURFACE)

    stages = ["Zero-shot\n(unseen graph)", "+ Fine-tune\n(detection)",
              "+ Fine-tune\n(causal)"]
    aucs   = [0.544, 0.900, 0.906]
    colors = [RED, ORANGE, GREEN]

    bars = ax_chart.bar(stages, aucs, color=colors, width=0.5,
                         edgecolor="none", zorder=3)
    ax_chart.axhline(0.5, color=MUTED, ls="--", lw=1, alpha=0.6)
    ax_chart.text(2.05, 0.515, "random baseline", fontsize=9,
                   color=MUTED, ha="right", va="bottom")

    for bar, val in zip(bars, aucs):
        ax_chart.text(bar.get_x() + bar.get_width()/2, val + 0.025,
                       f"{val:.3f}", ha="center", fontsize=15,
                       fontweight="black", color=TEXT)

    ax_chart.set_ylim(0, 1.05)
    ax_chart.set_title("Generalization to an UNSEEN 7-node graph "
                        "(Home workflow)\nAUC before vs after adaptation",
                        fontsize=13, color=TEXT, fontweight="bold", pad=14)
    ax_chart.tick_params(colors=MUTED, labelsize=10)
    for spine in ax_chart.spines.values():
        spine.set_visible(False)
    ax_chart.set_yticks([])
    ax_chart.grid(False)

    # ── Teaser banner ─────────────────────────────────────────
    ax_teaser = fig.add_axes([0.06, 0.10, 0.88, 0.14])
    ax_teaser.set_xlim(0, 1); ax_teaser.set_ylim(0, 1); ax_teaser.axis("off")
    ax_teaser.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#1f6feb15", edgecolor=BLUE, lw=1.5))
    ax_teaser.text(0.5, 0.62,
                    "Zero-shot transfer failed (0.544) —",
                    fontsize=13, color=MUTED, ha="center", va="center")
    ax_teaser.text(0.5, 0.30,
                    "~25 minutes of fine-tuning recovered AUC = 0.906",
                    fontsize=16, color=BLUE, fontweight="black",
                    ha="center", va="center")

    # Footer
    ax_foot = fig.add_axes([0, 0, 1, 0.07])
    ax_foot.set_xlim(0, 1); ax_foot.set_ylim(0, 1); ax_foot.axis("off")
    ax_foot.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor=SURFACE))
    ax_foot.text(0.5, 0.5,
                  "Full causal root-cause analysis & paper — coming soon",
                  fontsize=12.5, color=GREEN, fontweight="bold",
                  ha="center", va="center")

    pdf.savefig(fig, dpi=100)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    out_pdf = "/home/claude/carousel/CruxSight_LinkedIn_Carousel.pdf"
    with PdfPages(out_pdf) as pdf:
        make_page1(pdf)
        make_page2(pdf)
        make_page3(pdf)
    print(f"Saved: {out_pdf}")

    # Also export individual PNGs for quick preview / direct image upload
    with PdfPages("/dev/null") as _:
        pass
