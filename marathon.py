"""
marathon.py — Snake Algorithm Marathon
5 snakes run simultaneously, each using a different algorithm.
UI matches the original game.py aesthetic (Catppuccin Mocha).
"""

import pygame
import sys
from core.snake_env     import SnakeGame, Direction, BLOCK_SIZE
from marathon_agents    import RandomAgent, BFSAgent, AStarAgent, HamiltonianAgent, DQNAgent
from core.stats         import init_csv, log_game, save_plot, SAVE_EVERY

pygame.init()

# ── Layout constants ──────────────────────────────────────────────────────────
PANEL_W   = 400    # 20 cols × 20 px
PANEL_H   = 300    # 15 rows × 20 px
PAD       = 10
INNER_PAD = 10
LABEL_H   = 26
STATS_H   = 110    # 5 agents × 22 px each

WIN_W = PAD + PANEL_W + INNER_PAD + PANEL_W + INNER_PAD + PANEL_W + PAD   # 1260
WIN_H = (PAD + LABEL_H + PANEL_H) * 2 + INNER_PAD + PAD + STATS_H

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
BG_WINDOW   = (24,  24,  37)
BG_PANEL    = (30,  30,  46)
GRID_COL    = (49,  50,  68)
SNAKE_HEAD  = (166, 227, 161)
SNAKE_BODY  = (148, 226, 213)
SNAKE_INNER = (116, 199, 236)
FOOD_COLOR  = (243, 139, 168)
LEAF_COLOR  = (166, 227, 161)
TEXT_COLOR  = (205, 214, 244)
EYE_COLOR   = (24,  24,  37)
SCORE_BG    = (49,  50,  68)
STATS_BG    = (36,  39,  58)

# Per-agent accent colours for border + label
ACCENT = [
    (243, 139, 168),   # pink   – Random
    (137, 180, 250),   # blue   – BFS
    (166, 227, 161),   # green  – A*
    (249, 226, 175),   # yellow – Hamiltonian
    (203, 166, 247),   # mauve  – DQN
]

SPEED = 30   # ticks per second (raise to go faster)

# ── Fonts ─────────────────────────────────────────────────────────────────────
pygame.font.init()
font_score  = pygame.font.SysFont('ubuntu, arial, sans-serif', 18, bold=True)
font_label  = pygame.font.SysFont('ubuntu, arial, sans-serif', 13, bold=True)
font_stats  = pygame.font.SysFont('ubuntu, arial, sans-serif', 15, bold=True)


# ── Panel position calculation ────────────────────────────────────────────────
def _panel_positions():
    """
    Returns list of (label_rect, panel_rect) for 5 agents.
    Row 0: agents 0,1,2 — Row 1: agents 3,4 centred.
    """
    positions = []
    row0_y = PAD
    for col in range(3):
        x = PAD + col * (PANEL_W + INNER_PAD)
        positions.append((
            pygame.Rect(x, row0_y,            PANEL_W, LABEL_H),
            pygame.Rect(x, row0_y + LABEL_H,  PANEL_W, PANEL_H),
        ))
    row1_y     = PAD + LABEL_H + PANEL_H + INNER_PAD
    row0_w     = 3 * PANEL_W + 2 * INNER_PAD
    row1_w     = 2 * PANEL_W + 1 * INNER_PAD
    row1_x0    = PAD + (row0_w - row1_w) // 2
    for col in range(2):
        x = row1_x0 + col * (PANEL_W + INNER_PAD)
        positions.append((
            pygame.Rect(x, row1_y,            PANEL_W, LABEL_H),
            pygame.Rect(x, row1_y + LABEL_H,  PANEL_W, PANEL_H),
        ))
    return positions


# ── Rendering ─────────────────────────────────────────────────────────────────
def _draw_grid(surf):
    for x in range(0, PANEL_W + 1, BLOCK_SIZE):
        pygame.draw.line(surf, GRID_COL, (x, 0), (x, PANEL_H))
    for y in range(0, PANEL_H + 1, BLOCK_SIZE):
        pygame.draw.line(surf, GRID_COL, (0, y), (PANEL_W, y))


def _draw_eyes(surf, head_pt, direction):
    B  = BLOCK_SIZE
    r  = max(2, B // 10)
    o1 = B // 3
    o2 = 2 * (B // 3)
    near, far = B // 4, 3 * (B // 4)
    x, y = head_pt.x, head_pt.y
    if direction == Direction.RIGHT:   e1, e2 = (x+far, y+o1), (x+far, y+o2)
    elif direction == Direction.LEFT:  e1, e2 = (x+near, y+o1),(x+near, y+o2)
    elif direction == Direction.UP:    e1, e2 = (x+o1, y+near), (x+o2, y+near)
    else:                              e1, e2 = (x+o1, y+far),  (x+o2, y+far)
    pygame.draw.circle(surf, EYE_COLOR, e1, r)
    pygame.draw.circle(surf, EYE_COLOR, e2, r)


def render_panel(surf, game: SnakeGame, accent):
    surf.fill(BG_PANEL)
    _draw_grid(surf)

    # Snake
    for i, pt in enumerate(game.snake):
        is_head = (i == 0)
        color   = SNAKE_HEAD if is_head else SNAKE_BODY
        rect    = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(surf, color, rect, border_radius=6)
        if not is_head:
            ir = pygame.Rect(pt.x + BLOCK_SIZE//4, pt.y + BLOCK_SIZE//4,
                             BLOCK_SIZE//2, BLOCK_SIZE//2)
            pygame.draw.rect(surf, SNAKE_INNER, ir, border_radius=4)
    _draw_eyes(surf, game.head, game.direction)

    # Food
    cx = game.food.x + BLOCK_SIZE // 2
    cy = game.food.y + BLOCK_SIZE // 2
    r  = BLOCK_SIZE // 2 - 2
    pygame.draw.circle(surf, FOOD_COLOR, (cx, cy), r)
    leaf = pygame.Rect(cx, cy - r - 2, 6, 8)
    pygame.draw.ellipse(surf, LEAF_COLOR, leaf)

    # Score badge
    score_surf = font_score.render(f" {game.score} ", True, TEXT_COLOR)
    sr         = score_surf.get_rect(topleft=(6, 5))
    bg         = sr.inflate(8, 4)
    pygame.draw.rect(surf, SCORE_BG, bg, border_radius=8)
    surf.blit(score_surf, sr)


def draw_label(display, label_rect, name, accent):
    t = font_label.render(name, True, accent)
    display.blit(t, (label_rect.x + (label_rect.w - t.get_width()) // 2,
                     label_rect.y + (label_rect.h - t.get_height()) // 2))
    pygame.draw.line(display, accent,
                     (label_rect.x, label_rect.bottom - 3),
                     (label_rect.right, label_rect.bottom - 3), 1)


def draw_stats(display, agents, games, records, total_scores, game_counts):
    sy = WIN_H - STATS_H
    pygame.draw.rect(display, STATS_BG, pygame.Rect(0, sy, WIN_W, STATS_H))
    pygame.draw.line(display, GRID_COL, (0, sy), (WIN_W, sy), 1)

    row_h = STATS_H // 5
    for i, agent in enumerate(agents):
        mean  = total_scores[i] / game_counts[i] if game_counts[i] else 0
        label = getattr(agent, 'display_name', agent.name)
        y     = sy + i * row_h + (row_h - font_stats.get_height()) // 2
        # Name badge (coloured)
        name_surf = font_stats.render(f'{label:<14}', True, ACCENT[i])
        display.blit(name_surf, (PAD, y))
        # Stats (white)
        stats_text = (
            f"score {games[i].score:>4}   "
            f"rec {records[i]:>4}   "
            f"mean {mean:>6.1f}   "
            f"games {game_counts[i]:>6}"
        )
        stats_surf = font_stats.render(stats_text, True, TEXT_COLOR)
        display.blit(stats_surf, (PAD + 160, y))
        # Thin separator line between rows (except last)
        if i < 4:
            pygame.draw.line(display, GRID_COL,
                             (0, sy + (i+1)*row_h), (WIN_W, sy + (i+1)*row_h), 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global SPEED
    display   = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("snake : algorithm marathon")
    clock     = pygame.time.Clock()
    positions = _panel_positions()

    games  = [SnakeGame(w=PANEL_W, h=PANEL_H) for _ in range(5)]
    agents = [
        RandomAgent(),
        BFSAgent(),
        AStarAgent(),
        HamiltonianAgent(w=PANEL_W, h=PANEL_H),
        DQNAgent(),
    ]

    # Pre-allocate surfaces — reused every frame (no GC churn)
    panel_surfs = [pygame.Surface((PANEL_W, PANEL_H)) for _ in range(5)]

    records      = [0] * 5
    total_scores = [0] * 5
    game_counts  = [0] * 5

    # Per-agent score history for the live plot
    all_scores  = [[] for _ in range(5)]
    all_means   = [[] for _ in range(5)]
    total_games = 0

    init_csv()   # fresh CSV every run

    while True:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_plot(all_scores, all_means)   # save stat.png before exit
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    SPEED = min(SPEED + 10, 200)
                if event.key == pygame.K_MINUS:
                    SPEED = max(SPEED - 10, 5)
                if event.key == pygame.K_d:           # D → toggle DQN mode
                    agents[4].toggle()

        # ── Step, train, reset — ALL before any rendering ─────────────────
        for i, (agent, game) in enumerate(zip(agents, games)):
            action = agent.get_action(game)
            reward, done, score = game.play_step(action)

            if hasattr(agent, 'on_step'):
                agent.on_step(reward, game, done)

            if done:
                if score > records[i]:
                    records[i] = score
                total_scores[i] += score
                game_counts[i]  += 1
                mean = total_scores[i] / game_counts[i]

                # Track history and log
                all_scores[i].append(score)
                all_means[i].append(mean)
                log_game(i, game_counts[i], score, records[i], mean)
                total_games += 1

                agent.on_game_over(score)
                game.reset()          # always reset before rendering this frame

        # ── Render — only AFTER all games are in a clean state ───────────
        display.fill(BG_WINDOW)

        for i, (game, (label_rect, panel_rect)) in enumerate(zip(games, positions)):
            surf = panel_surfs[i]         # reuse pre-allocated surface
            render_panel(surf, game, ACCENT[i])
            display.blit(surf, panel_rect.topleft)
            pygame.draw.rect(display, ACCENT[i],
                             pygame.Rect(panel_rect.x-1, panel_rect.y-1,
                                         PANEL_W+2, PANEL_H+2), 2, border_radius=3)
            label = getattr(agents[i], 'display_name', agents[i].name)
            draw_label(display, label_rect, label, ACCENT[i])

        draw_stats(display, agents, games, records, total_scores, game_counts)
        pygame.display.flip()
        clock.tick(SPEED)


if __name__ == '__main__':
    main()
