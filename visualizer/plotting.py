"""Visualization for iterative memory optimization — per-dialogue focus.

Each dialogue gets its own color. All 5 trajectories shown simultaneously.
"""

import os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import defaultdict

plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 12,
                     "axes.titlesize": 14, "axes.labelsize": 12,
                     "legend.fontsize": 10, "figure.dpi": 150})

DIALOGUE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
DIALOGUE_COLORS_LIGHT = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0", "#E1BEE7"]


def plot_scores_over_rounds(
    results: List[Dict],
    save_path: str = "output/score_trajectory.png",
):
    """Five per-dialogue score trajectories in distinct colors + aggregate mean."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    all_trajectories = []
    dialogue_labels = []

    for i, r in enumerate(results):
        traj = r.get("score_trajectory", [])
        label = r.get("dialogue_id", f"dialogue_{i}")
        if traj:
            all_trajectories.append(traj)
            dialogue_labels.append(label)
            color = DIALOGUE_COLORS[i % len(DIALOGUE_COLORS)]
            ax1.plot(range(len(traj)), traj, color=color, linewidth=2,
                     marker="o", markersize=5, label=label, alpha=0.85)

    # Mean line
    if all_trajectories:
        max_len = max(len(t) for t in all_trajectories)
        padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in all_trajectories])
        mean_scores = padded.mean(axis=0)
        ax1.plot(range(max_len), mean_scores, color="black", linewidth=3,
                 linestyle="--", label="Mean", alpha=0.8)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Composite Score")
    ax1.set_title("Per-Dialogue Score Trajectories")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: final delta per dialogue (bar chart)
    if all_trajectories:
        deltas = []
        colors_bar = []
        for t in all_trajectories:
            d = t[-1] - t[0] if len(t) >= 2 else 0
            deltas.append(d)
            colors_bar.append("#4CAF50" if d >= 0 else "#F44336")

        x_pos = range(len(deltas))
        ax2.bar(x_pos, deltas, color=colors_bar, edgecolor="black", alpha=0.8)
        ax2.axhline(y=0, color="black", linewidth=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(dialogue_labels, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("Score Delta (Final - Initial)")
        ax2.set_title("Improvement per Dialogue")
        # Value labels
        for j, d in enumerate(deltas):
            ax2.text(j, d + 0.005, f"{d:+.3f}", ha="center", fontsize=9,
                     fontweight="bold", color=colors_bar[j])
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Score trajectory saved to {save_path}")


def plot_component_breakdown(
    results: List[Dict],
    save_path: str = "output/component_breakdown.png",
):
    """Per-dialogue component breakdown: retention (70%) + entity_f1 (30%)."""
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, r in enumerate(results[:5]):
        ax = axes[i]
        history = r.get("history", [])
        if not history:
            ax.set_title(r.get("dialogue_id", f"dial_{i}"))
            continue

        rounds = [h["round"] for h in history]
        bs = r.get("best_score", {})

        # Extract components — try both AutoMemo and BARTScore formats
        ret = [h.get("best_score", {}).get("retention",
               h.get("best_score", {}).get("bartscore", 0)) for h in history]
        ef1 = [h.get("best_score", {}).get("entity_f1", 0) for h in history]

        ax.plot(rounds, ret, "o-", color="#2196F3", linewidth=2, label="Retention (70%)")
        ax.plot(rounds, ef1, "s-", color="#4CAF50", linewidth=2, label="Entity F1 (30%)")
        ax.set_title(r.get("dialogue_id", f"dial_{i}"))
        ax.set_xlabel("Round")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    # Hide extra subplot if fewer than 5 dialogues
    if n < 5:
        for j in range(n, 5):
            axes[j].set_visible(False)

    # 6th subplot: aggregate
    ax = axes[5]
    all_ret, all_ef1 = [], []
    for r in results:
        for h in r.get("history", []):
            bs = h.get("best_score", {})
            all_ret.append(bs.get("retention", bs.get("bartscore", 0)))
            all_ef1.append(bs.get("entity_f1", 0))
    if all_ret:
        ax.bar(["Retention (70%)", "Entity F1 (30%)"],
               [np.mean(all_ret), np.mean(all_ef1)],
               color=["#2196F3", "#4CAF50"], edgecolor="black")
        ax.set_title("Global Averages")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Per-Dialogue Component Breakdown", fontsize=15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Component breakdown saved to {save_path}")


def plot_experience_growth(
    results: List[Dict],
    save_path: str = "output/experience_growth.png",
):
    """Per-dialogue experience library growth — each dialogue starts fresh."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, r in enumerate(results):
        history = r.get("history", [])
        if not history:
            continue
        rounds = [h["round"] for h in history]
        lib_sizes = [h.get("experience_count", 0) for h in history]
        color = DIALOGUE_COLORS[i % len(DIALOGUE_COLORS)]
        label = r.get("dialogue_id", f"dial_{i}")
        ax1.plot(rounds, lib_sizes, color=color, linewidth=2, marker="o",
                 markersize=6, label=label)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Rules in Library")
    ax1.set_title("Per-Dialogue Experience Library Growth")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: final library size per dialogue
    labels = []
    final_sizes = []
    colors_bar = []
    for i, r in enumerate(results):
        history = r.get("history", [])
        labels.append(r.get("dialogue_id", f"dial_{i}"))
        final_sizes.append(history[-1].get("experience_count", 0) if history else 0)
        colors_bar.append(DIALOGUE_COLORS[i % len(DIALOGUE_COLORS)])

    ax2.bar(range(len(final_sizes)), final_sizes, color=colors_bar, edgecolor="black")
    ax2.set_xticks(range(len(final_sizes)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Final Rule Count")
    ax2.set_title("Final Library Size per Dialogue")
    for j, s in enumerate(final_sizes):
        ax2.text(j, s + 0.3, str(s), ha="center", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Experience growth saved to {save_path}")


def plot_weakness_distribution(
    results: List[Dict],
    save_path: str = "output/weakness_distribution.png",
):
    """Which scoring component was weakest most often, per dialogue."""
    fig, ax = plt.subplots(figsize=(10, 5))
    weakness_map = {"bartscore": "Retention", "retention": "Retention",
                    "entity_f1": "Entity F1", "nli_score": "Consistency",
                    "consistency": "Consistency"}

    all_labels = []
    all_rounds = []
    all_weakness = []
    color_map = {"Retention": "#2196F3", "Entity F1": "#4CAF50", "Consistency": "#FF9800"}

    for r in results:
        label = r.get("dialogue_id", "?")
        for h in r.get("history", []):
            w = weakness_map.get(h.get("weakness", "unknown"), h.get("weakness", "?"))
            all_labels.append(label)
            all_rounds.append(h.get("round", 0))
            all_weakness.append(w)

    if not all_weakness:
        return

    # Stack per dialogue
    unique_labels = list(dict.fromkeys(all_labels))
    unique_weakness_types = list(dict.fromkeys(all_weakness))
    x = np.arange(len(unique_labels))
    width = 0.6 / len(unique_weakness_types)

    for wi, wtype in enumerate(unique_weakness_types):
        counts = []
        for label in unique_labels:
            c = sum(1 for l, w in zip(all_labels, all_weakness)
                    if l == label and w == wtype)
            counts.append(c)
        ax.bar(x + wi * width - width * len(unique_weakness_types)/3,
               counts, width, label=wtype,
               color=color_map.get(wtype, "#9E9E9E"), edgecolor="black")

    ax.set_xlabel("Dialogue")
    ax.set_ylabel("Frequency (rounds)")
    ax.set_title("Weakness Distribution: Which Metric Was Lowest?")
    ax.set_xticks(x)
    ax.set_xticklabels(unique_labels, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Weakness distribution saved to {save_path}")


def plot_summary_dashboard(results: List[Dict], output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    if len(results) == 1:
        # Single dialogue — simplified plots
        plot_single_dialogue(results[0], output_dir)
    else:
        plot_scores_over_rounds(results, os.path.join(output_dir, "score_trajectory.png"))
        plot_component_breakdown(results, os.path.join(output_dir, "component_breakdown.png"))
        plot_experience_growth(results, os.path.join(output_dir, "experience_growth.png"))
        plot_weakness_distribution(results, os.path.join(output_dir, "weakness_distribution.png"))

    print(f"\n[Plot] Dashboard complete: {output_dir}/")


def plot_single_dialogue(result: Dict, output_dir: str):
    """Simplified plots for a single dialogue — 2 charts total."""
    history = result.get("history", [])
    did = result.get("dialogue_id", "user")
    if not history:
        return

    rounds = [h["round"] for h in history]
    best_scores = [h["best_score"] for h in history]
    comp = [s["composite"] for s in best_scores]
    ret = [s.get("retention", s.get("bartscore", 0)) for s in best_scores]
    ef1 = [s.get("entity_f1", 0) for s in best_scores]
    lib_sizes = [h.get("experience_count", 0) for h in history]

    # --- Figure 1: Score trajectory + component breakdown ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(rounds, comp, "ko-", linewidth=3, markersize=10, label="Composite")
    ax1.fill_between(rounds, comp, alpha=0.1, color="black")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Composite Score")
    ax1.set_title(f"Score Trajectory — {did}")
    ax1.grid(True, alpha=0.3)
    for r, c in zip(rounds, comp):
        ax1.annotate(f"{c:.3f}", (r, c), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=10, fontweight="bold")

    ax2.plot(rounds, ret, "o-", color="#2196F3", linewidth=2, label="Retention (70%)")
    ax2.plot(rounds, ef1, "s-", color="#4CAF50", linewidth=2, label="Entity F1 (30%)")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Score")
    ax2.set_title("Component Breakdown")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle(f"Single Dialogue Optimization — {did}", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Rules growth + weakness ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.fill_between(rounds, lib_sizes, alpha=0.3, color="#9C27B0")
    ax1.plot(rounds, lib_sizes, "o-", color="#9C27B0", linewidth=2, markersize=8)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Rules")
    ax1.set_title("Experience Library Growth")
    ax1.grid(True, alpha=0.3)

    # Weakness per round
    weakness_map = {"bartscore": "Retention", "retention": "Retention",
                    "entity_f1": "Entity F1", "nli_score": "Consistency",
                    "consistency": "Consistency"}
    weak_counts = {}
    for h in history:
        w = weakness_map.get(h.get("weakness", "?"), h.get("weakness", "?"))
        weak_counts[w] = weak_counts.get(w, 0) + 1
    ax2.bar(list(weak_counts.keys()), list(weak_counts.values()),
            color=["#2196F3", "#4CAF50", "#FF9800"][:len(weak_counts)], edgecolor="black")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Weakness Distribution")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Experience Library — {did}", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "library.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"[Plot] Single-dialogue charts saved to {output_dir}/")


def print_summary_table(results: List[Dict]):
    """Per-dialogue summary table — shows best score across ALL rounds, not just last."""
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY — Per Dialogue (best across all rounds)")
    print("=" * 90)
    print(f"{'Dialogue':<22} {'Round 0':>8} {'Best':>8} {'Final':>8} {'Delta':>8} {'Rules':>6} {'Best@':>6}")
    print("-" * 90)

    all_deltas = []
    for r in results:
        traj = r.get("score_trajectory", [])
        did = r.get("dialogue_id", "?")[:20]
        r0 = traj[0] if traj else 0
        best_val = max(traj) if traj else 0
        rf = traj[-1] if traj else 0
        d = best_val - r0  # improvement from start to BEST round
        all_deltas.append(d)
        rules = r.get("experience_count", 0)
        best_r = traj.index(best_val) if traj else 0
        print(f"{did:<22} {r0:>8.4f} {best_val:>8.4f} {rf:>8.4f} {d:>+8.4f} {rules:>6} {best_r:>6}")

    print("-" * 90)
    improved = sum(1 for d in all_deltas if d > 0.005)
    avg_delta = np.mean(all_deltas)
    total_rules = sum(r.get("experience_count", 0) for r in results)
    print(f"Improved: {improved}/{len(results)}  |  Avg Delta: {avg_delta:+.4f}  |  "
          f"Total Rules: {total_rules}")
    print(f"Columns: Round 0=first round score | Best=highest ever | Final=last round score | "
          f"Delta=Best-Round0 | Best@=which round")
    print("=" * 90)
