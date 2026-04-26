import csv
import os
import pygame
import torch
import random
import numpy as np
from collections import deque
from game import SnakeAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # Auto-load solo model if it exists — continues from last checkpoint
        _path = os.path.join('model', 'model_agent_solo.pth')
        if os.path.exists(_path):
            self.model.load_state_dict(
                torch.load(_path, map_location='cpu', weights_only=False))
            print(f'[agent] Loaded {_path}')

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeAI()

    # CSV logging — fresh file each run
    _csv = open('agent_training.csv', 'w', newline='')
    _writer = csv.writer(_csv)
    _writer.writerow(['game_num', 'score', 'record', 'mean_score'])
    _csv.flush()
    while True:
        state_old = agent.get_state(game)

        if game.demo_mode:
            # pure exploitation — no exploration, no training
            t = torch.tensor(state_old, dtype=torch.float)
            move = torch.argmax(agent.model(t)).item()
            final_move = [0, 0, 0]; final_move[move] = 1
        else:
            final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        # update window title to reflect mode
        pygame.display.set_caption(
            'snake: learning to live  [DEMO — D to train]'
            if game.demo_mode else
            'snake: learning to live  [TRAINING — D to demo]'
        )

        if not game.demo_mode:
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            game.reset()
            agent.n_games += 1

            if not game.demo_mode:
                agent.train_long_memory()
                if score > record:
                    record = score
                    agent.model.save('model_agent_solo.pth')

            mode_tag = '[DEMO] ' if game.demo_mode else '[TRAIN]'
            print(f'{mode_tag} game: {agent.n_games:4} | score: {score:3} | record: {record:3}')
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            _writer.writerow([agent.n_games, score, record, f'{mean_score:.3f}'])
            _csv.flush()   # write immediately so data is safe even if interrupted
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()