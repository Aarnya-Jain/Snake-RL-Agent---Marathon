"""
stats.py — Per-game CSV logging and live multi-agent comparison plot.
Identical to root stats.py but lives in core/.
CSV and PNG paths are relative to the project root (where marathon.py runs from).
"""

import csv
import matplotlib
matplotlib.use('Agg')   # file-only, no Qt window
import matplotlib.pyplot as plt

CSV_PATH   = 'stats.csv'
PLOT_PATH  = 'stat.png'
SAVE_EVERY = 10   # update plot every N total completed games

AGENT_COLORS = [
    '#f38ba8',   # pink   – Random
    '#89b4fa',   # blue   – BFS
    '#a6e3a1',   # green  – A*
    '#f9e2af',   # yellow – Hamiltonian
    '#cba6f7',   # mauve  – DQN
]
AGENT_NAMES = ['Random', 'BFS', 'A*', 'Hamiltonian', 'DQN']

bg       = '#1e1e2e'
grid_col = '#313244'
text_col = '#cdd6f4'
spine_col = '#45475a'


# ── CSV ───────────────────────────────────────────────────────────────────────

def init_csv():
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['agent', 'game_num', 'score', 'record', 'mean_score'])


def log_game(agent_idx: int, game_num: int, score: int, record: int, mean_score: float):
    with open(CSV_PATH, 'a', newline='') as f:
        csv.writer(f).writerow([
            AGENT_NAMES[agent_idx], game_num, score, record, f'{mean_score:.3f}'
        ])


# ── Live Plot ─────────────────────────────────────────────────────────────────

def save_plot(scores_per_agent: list, means_per_agent: list, path: str = PLOT_PATH):
    """Draw comparison plot for all 5 agents and save to disk."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9), facecolor=bg,
        constrained_layout=True
    )
    fig.patch.set_facecolor(bg)
    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        ax.tick_params(colors=text_col, labelsize=9)
        ax.grid(color=grid_col, linestyle='--', alpha=0.4)
        ax.spines['bottom'].set_color(spine_col)
        ax.spines['left'].set_color(spine_col)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax1.set_title('score per game', color=text_col, fontweight='bold', fontsize=11)
    ax1.set_ylabel('score', color=text_col, fontsize=9)
    ax1.set_xlabel('game #', color=text_col, fontsize=9)
    ax1.set_ylim(ymin=0)

    for i, scores in enumerate(scores_per_agent):
        if scores:
            ax1.plot(scores, color=AGENT_COLORS[i], label=AGENT_NAMES[i],
                     linewidth=1.3, alpha=0.85)
            ax1.text(len(scores)-1, scores[-1], f' {scores[-1]}',
                     color=AGENT_COLORS[i], fontsize=7, fontweight='bold',
                     va='center')

    ax1.legend(facecolor='#181825', edgecolor=spine_col, labelcolor=text_col,
               fontsize=8, loc='upper left', framealpha=0.85)

    ax2.set_title('mean score (running average)', color=text_col, fontweight='bold', fontsize=11)
    ax2.set_ylabel('mean score', color=text_col, fontsize=9)
    ax2.set_xlabel('game #', color=text_col, fontsize=9)
    ax2.set_ylim(ymin=0)

    for i, means in enumerate(means_per_agent):
        if means:
            ax2.plot(means, color=AGENT_COLORS[i], label=AGENT_NAMES[i],
                     linewidth=2.2)
            ax2.text(len(means)-1, means[-1], f' {means[-1]:.1f}',
                     color=AGENT_COLORS[i], fontsize=7, fontweight='bold',
                     va='center')

    ax2.legend(facecolor='#181825', edgecolor=spine_col, labelcolor=text_col,
               fontsize=8, loc='upper left', framealpha=0.85)

    plt.savefig(path, dpi=120, facecolor=bg)
    plt.close(fig)
