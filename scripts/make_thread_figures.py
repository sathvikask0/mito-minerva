"""Figures for the write-up thread. Reads only committed outputs, except the
hero figure, which folds one tRNA with the base and finetuned models.

    python scripts/make_thread_figures.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

OUT = "thread"
FT, VI, BASE, CONS, TRUTH, BAD = "#0F766E", "#6B7280", "#D97706", "#7C3AED", "#111827", "#DC2626"
plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 26, "axes.titleweight": "bold",
    "axes.labelsize": 20, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.dpi": 100, "font.family": "DejaVu Sans",
})
W, H = 16, 9


def save(fig, name, source):
    fig.text(0.99, 0.01, source, ha="right", va="bottom", fontsize=13, color="#9CA3AF")
    fig.savefig(f"{OUT}/{name}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f"{OUT}/{name}")


def bars(ax, labels, vals, lo, hi, colors, fmt="{:.0f}%", scale=100):
    x = np.arange(len(labels))
    v, l, h = np.array(vals) * scale, np.array(lo) * scale, np.array(hi) * scale
    ax.bar(x, v, color=colors, width=0.62, zorder=2)
    ax.errorbar(x, v, yerr=[v - l, h - v], fmt="none", ecolor="#111827",
                elinewidth=2, capsize=8, zorder=3)
    for xi, vi, hi_ in zip(x, v, h):
        ax.text(xi, hi_ + (2 if scale == 100 else 0.02), fmt.format(vi),
                ha="center", va="bottom", fontsize=24, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True, color="#E5E7EB", zorder=0)


# 1. hero: arcs for TRNR -------------------------------------------------------
def fig_hero():
    from step2_benchmark import Folder
    from step8_validate_structure import vienna_pairs
    from step14_grade_vs_pdb import map_pairs
    from mitominerva.loading import load_model

    gene = "TRNR"
    pdb = json.load(open("data/processed/pdb_trna_pairs.json"))["genes"][gene]
    seq, L = pdb["human_seq"], pdb["length"]
    truth = {tuple(p) for p in pdb["consensus"]}
    rows = [("Lab-measured structure", truth, TRUTH)]
    for tag, adapter, col in (("Minerva, fine-tuned", "outputs/adapters/lora-r8/adapter", FT),
                              ("Minerva, original", None, BASE)):
        tok, model = load_model(adapter=adapter)
        rows.append((tag, map_pairs(Folder(model, tok, "cpu")(seq)), col))
        del model
    rows.insert(2, ("ViennaRNA (standard tool)",
                    {tuple(sorted(p)) for p in vienna_pairs(seq) if abs(p[1] - p[0]) >= 4}, VI))

    fig, axes = plt.subplots(len(rows), 1, figsize=(W, H), sharex=True)
    for ax, (label, pairs, col) in zip(axes, rows):
        top = max((j - i) for i, j in truth) / 2 + 2
        for i, j in pairs:
            ok = (i, j) in truth or label.startswith("Lab")
            ax.add_patch(Arc(((i + j) / 2, 0), j - i, j - i, theta1=0, theta2=180,
                             color=col if ok else BAD, lw=2.6 if ok else 1.6,
                             alpha=1 if ok else 0.55))
        ax.set_xlim(0, L + 1); ax.set_ylim(0, top * 1.45); ax.axis("off")
        ax.plot([1, L], [0, 0], color="#D1D5DB", lw=3)
        if label.startswith("Lab"):
            txt = f"{label}: {len(truth)} pairs"
        else:
            hit = len(pairs & truth)
            txt = f"{label}: {hit}/{len(truth)} real pairs found, {len(pairs) - hit} wrong"
        ax.text(0, top * 1.45, txt, fontsize=20, fontweight="bold", color=col, va="top")
    fig.suptitle("Which letters of a human mitochondrial tRNA pair up? (tRNA-Arg)",
                 fontsize=26, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "each arc joins two letters that pair; red = predicted pair that isn't real",
             ha="center", fontsize=15, color="#6B7280")
    save(fig, "1_hero_arcs.png", "structure: PDB 6ZM6, 6ZM5, 7QI6, 9S7E, 9S7C")


# 2. lab structures ----------------------------------------------------------
def fig_lab():
    d = json.load(open("outputs/step14_pdb_grading.json"))
    names = [("Minerva, fine-tuned", "Minerva finetuned, tRNA alone", FT),
             ("ViennaRNA", "ViennaRNA", VI),
             ("Minerva, original", "Minerva base, tRNA alone", BASE)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [1.1, 1]})
    for ax, metric, title in ((a, "recall", "Real pairs it finds"), (b, "precision", "Its pairs that are real")):
        ps = [d["pooled"][f"all 8 | {k}"] for _, k, _ in names]
        bars(ax, [n.replace(", ", ",\n") for n, _, _ in names], [p[metric] for p in ps],
             [p[f"ci_{metric}"][0] for p in ps], [p[f"ci_{metric}"][1] for p in ps],
             [c for _, _, c in names])
        ax.set_ylim(0, 112); ax.set_title(title); ax.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("%  (bars = 95% range)")
    fig.suptitle("Graded against lab-measured 3D structures (8 tRNAs, 132 pairs)", fontsize=26,
                 fontweight="bold", y=1.02)
    save(fig, "2_lab_structures.png", "PDB X-ray / cryo-EM structures; ranges from resampling whole tRNAs")


# 3. context -----------------------------------------------------------------
def fig_context():
    d = json.load(open("outputs/step12_confidence.json"))["recall"]
    fig, ax = plt.subplots(figsize=(W, H))
    groups = [("Minerva, original", "base, tRNA alone", "base, in genome", BASE),
              ("Minerva, fine-tuned", "finetuned, tRNA alone", "finetuned, in genome", FT)]
    x = np.arange(len(groups)); w = 0.36
    for k, (label, alone, genome, col) in enumerate(groups):
        for off, key, alpha, tag in ((-w / 2, alone, 1.0, "tRNA alone"), (w / 2, genome, 0.45, "inside genome")):
            p, (lo, hi) = d[key]["point"] * 100, [c * 100 for c in d[key]["ci"]]
            ax.bar(k + off, p, w * 0.95, color=col, alpha=alpha, zorder=2)
            ax.errorbar(k + off, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=TRUTH, capsize=8, elinewidth=2, zorder=3)
            ax.text(k + off, hi + 1.5, f"{p:.0f}%", ha="center", fontsize=24, fontweight="bold")
    v = d["ViennaRNA"]["point"] * 100
    ax.axhline(v, color=VI, ls="--", lw=2.5, zorder=1)
    ax.text(-0.42, v + 1, f"ViennaRNA {v:.0f}%", color=VI, fontsize=18, ha="left", va="bottom")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#374151', label='tRNA on its own'),
                       Patch(color='#374151', alpha=0.45, label='tRNA inside its genome')],
              frameon=False, loc='upper left', fontsize=18)
    ax.set_xticks(x, [g[0] for g in groups]); ax.set_ylim(0, 80)
    ax.set_ylabel("real pairs found (%)"); ax.yaxis.grid(True, color="#E5E7EB", zorder=0)
    ax.set_title("Before: the genome around a tRNA confused the model. After: it doesn't.")
    save(fig, "3_context.png", "409 base pairs proven by co-varying letters across 8,397-15,552 species")


# 4. training curve ----------------------------------------------------------
def fig_training():
    d = json.load(open("outputs/finetune_log.json"))
    fig, ax = plt.subplots(figsize=(W, H))
    tr = [(0, None)] + [(e["step"], e["loss"]) for e in d["train"]]
    ax.plot([s for s, l in tr if l], [l for s, l in tr if l], color=FT, alpha=0.35, lw=2, label="training")
    ev = [(0, d["baseline_eval_loss"])] + [(e["step"], e["loss"]) for e in d["eval"]]
    ax.plot(*zip(*ev), color=FT, lw=4, marker="o", ms=12, label="held-out genomes")
    ax.axhline(math.log(4), color=BAD, ls="--", lw=2.5)
    ax.text(2626, math.log(4) + 0.015, "random guessing between A, C, G, T", color=BAD, ha="right", va="bottom", fontsize=18)
    ax.annotate(f"{ev[0][1]:.2f}", ev[0], xytext=(22, -8), textcoords="offset points", fontsize=22, fontweight="bold", va="top")
    ax.annotate(f"{ev[-1][1]:.2f}", ev[-1], xytext=(0, 18), textcoords="offset points", fontsize=22, fontweight="bold", ha="center")
    ax.set_xlabel("training step (1 pass over 15,589 animal mitochondrial genomes)")
    ax.set_ylabel("prediction error (lower = better)")
    ax.set_title("A bacterial DNA model starts near guessing on mitochondria, then learns")
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.55))
    ax.yaxis.grid(True, color="#E5E7EB")
    save(fig, "4_training.png", "masked-letter loss; 1 A100, 3h51m")


# 5. disease -----------------------------------------------------------------
def fig_disease():
    d = json.load(open("outputs/step6_conservation_eval.json"))["results"]
    names = [("Minerva,\nfine-tuned", "Minerva finetuned: surprise", FT),
             ("Just counting\nconserved letters", "Conservation: position conserved", CONS),
             ("ViennaRNA", "ViennaRNA: destabilisation", VI),
             ("Minerva,\noriginal", "Minerva base (pre-finetune)", BASE)]
    fig, ax = plt.subplots(figsize=(W, H))
    rs = [d[k] for _, k, _ in names]
    bars(ax, [n for n, _, _ in names], [r["auc"] for r in rs], [r["ci"][0] for r in rs],
         [r["ci"][1] for r in rs], [c for _, _, c in names], fmt="{:.2f}", scale=1)
    ax.axhline(0.5, color=BAD, ls="--", lw=2.5)
    ax.text(3.45, 0.505, "coin flip", color=BAD, ha="right", va="bottom", fontsize=18)
    ax.set_ylim(0.4, 0.97); ax.set_ylabel("AUC: disease vs harmless mutations")
    ax.set_title("Disease prediction looked good, until I built the boring baseline")
    ax.text(0.5, 0.98, "gap vs counting: +0.04 (range −0.05 to +0.12) → can't claim a win",
            ha="center", va="top", fontsize=19, color=TRUTH, transform=ax.transAxes)
    save(fig, "5_disease.png", "363 ClinVar mt-tRNA variants (52 disease-causing)")


# 6. longevity ---------------------------------------------------------------
def fig_longevity():
    from step11_longevity import group_center, pearson
    d = json.load(open("outputs/step11_longevity.json"))
    rows, summ = d["species"], d["summary"]
    colors = {"Mammalia": "#2563EB", "Aves": "#16A34A", "Teleostei": "#0891B2", "Reptilia": "#B45309"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, H))
    for cls, col in list(colors.items()) + [("other", "#9CA3AF")]:
        sel = [r for r in rows if (r["class"] == cls) or (cls == "other" and r["class"] not in colors)]
        name = {"Mammalia": "mammals", "Aves": "birds", "Teleostei": "fish", "Reptilia": "reptiles"}.get(cls, "other")
        a.scatter([r["lifespan"] for r in sel], [r["fragility"] for r in sel], s=22, color=col, alpha=0.6, label=name)
    a.set_xscale("log"); a.set_xlabel("max lifespan (years)"); a.set_ylabel("tRNA fragility")
    a.set_title("All 1,044 species"); a.legend(frameon=False, fontsize=16, markerscale=2)
    fam = [r["family"] for r in rows]
    k, cf = group_center(np.array([r["fragility"] for r in rows]), fam)
    _, cl = group_center(np.log10([r["lifespan"] for r in rows]), fam)
    b.scatter(cl, cf, s=22, color=TRUTH, alpha=0.4)
    b.axhline(0, color="#D1D5DB"); b.axvline(0, color="#D1D5DB")
    wf = summ["within_family"]
    b.set_title("Compared with close relatives")
    b.set_xlabel("lives longer than its family →"); b.set_ylabel("more fragile than its family →")
    b.text(0.03, 0.95, f"r = {wf['r']:+.2f},  p = {wf['p']:.2f}\nno link", transform=b.transAxes,
           fontsize=22, fontweight="bold", va="top")
    fig.suptitle("Do long-lived animals have sturdier tRNAs? No.", fontsize=26, fontweight="bold", y=1.02)
    save(fig, "6_longevity.png", "AnAge lifespans; fragility = share of base pairs a random mutation breaks")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_lab, fig_context, fig_training, fig_disease, fig_longevity, fig_hero):
        f()
