import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('ubuntu, arial, sans-serif', 24, bold=True)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('point', 'x, y')

BG_COLOR = (30, 30, 46)
GRID_COLOR = (49, 50, 68)
SNAKE_HEAD = (166, 227, 161)
SNAKE_BODY = (148, 226, 213)
SNAKE_INNER = (116, 199, 236)
FOOD_COLOR = (243, 139, 168)
TEXT_COLOR = (205, 214, 244)
EYE_COLOR  = (24, 24, 37)
SCORE_BG   = (69, 71, 90)
LEAF_COLOR = (166, 227, 161)

BLOCK_SIZE = 20
SPEED = 40

class SnakeAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('snake: learning to live')
        self.clock = pygame.time.Clock()
        self.demo_mode = False   # toggled by D key
        self.reset()


    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        global SPEED
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    SPEED = min(SPEED + 10, 300)      # speed up
                elif event.key == pygame.K_MINUS:
                    SPEED = max(SPEED - 10, 5)        # slow down
                elif event.key == pygame.K_d:
                    self.demo_mode = not self.demo_mode   # toggle demo

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BG_COLOR)

        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.w, y))

        for i, pt in enumerate(self.snake):
            is_head = (i == 0)
            color = SNAKE_HEAD if is_head else SNAKE_BODY

            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, color, rect, border_radius=6)

            if not is_head:
                inner_rect = pygame.Rect(pt.x + BLOCK_SIZE//4, pt.y + BLOCK_SIZE//4, BLOCK_SIZE//2, BLOCK_SIZE//2)
                pygame.draw.rect(self.display, SNAKE_INNER, inner_rect, border_radius=4)

            if is_head:
                eye_radius = max(2, BLOCK_SIZE // 10)
                offset1 = BLOCK_SIZE // 3
                offset2 = 2 * (BLOCK_SIZE // 3)
                near = BLOCK_SIZE // 4
                far = 3 * (BLOCK_SIZE // 4)

                if self.direction == Direction.RIGHT:
                    eye1 = (pt.x + far, pt.y + offset1)
                    eye2 = (pt.x + far, pt.y + offset2)
                elif self.direction == Direction.LEFT:
                    eye1 = (pt.x + near, pt.y + offset1)
                    eye2 = (pt.x + near, pt.y + offset2)
                elif self.direction == Direction.UP:
                    eye1 = (pt.x + offset1, pt.y + near)
                    eye2 = (pt.x + offset2, pt.y + near)
                else: # DOWN
                    eye1 = (pt.x + offset1, pt.y + far)
                    eye2 = (pt.x + offset2, pt.y + far)

                pygame.draw.circle(self.display, EYE_COLOR, eye1, eye_radius)
                pygame.draw.circle(self.display, EYE_COLOR, eye2, eye_radius)

        center_x = self.food.x + BLOCK_SIZE // 2
        center_y = self.food.y + BLOCK_SIZE // 2
        radius = BLOCK_SIZE // 2 - 2
        pygame.draw.circle(self.display, FOOD_COLOR, (center_x, center_y), radius)

        leaf_rect = pygame.Rect(center_x, center_y - radius - 2, 6, 8)
        pygame.draw.ellipse(self.display, LEAF_COLOR, leaf_rect)

        text = font.render(f" score: {self.score} ", True, TEXT_COLOR)
        text_rect = text.get_rect(topleft=(10, 10))

        bg_rect = text_rect.inflate(12, 8)
        pygame.draw.rect(self.display, SCORE_BG, bg_rect, border_radius=10)
        self.display.blit(text, text_rect)

        pygame.display.flip()


    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)