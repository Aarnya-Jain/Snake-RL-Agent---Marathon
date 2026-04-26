
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    plt.clf()

    plt.style.use('dark_background')
    fig = plt.gcf()
    fig.patch.set_facecolor('#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e2e')

    line1_color = '#89b4fa'
    line2_color = '#f38ba8'
    text_color = '#cdd6f4'
    grid_color = '#313244'

    ax.tick_params(colors=text_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.spines['left'].set_color(grid_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title('training model', color=text_color, pad=15, fontweight='bold')
    plt.xlabel('number of games', color=text_color)
    plt.ylabel('score', color=text_color)

    plt.plot(scores, color=line1_color, label='score', linewidth=1.5)
    plt.plot(mean_scores, color=line2_color, label='mean score', linewidth=2.5)

    plt.grid(color=grid_color, linestyle='--', alpha=0.5)
    plt.ylim(ymin=0)

    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]), color=line1_color)
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}", color=line2_color)

    plt.legend(facecolor='#181825', edgecolor=grid_color, labelcolor=text_color)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)