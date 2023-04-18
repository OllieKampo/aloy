import argparse
from itertools import count
import os
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PyQt6 import QtWidgets
from PyQt6.QtCore import QTimer

from games.snakegame import SnakeGameJinxWidget, SnakeGameLogic
from guis.gui import JinxGuiData, JinxGuiWindow
from learning.convolutional import calc_conv_output_shape_from, size_of_flat_layer
from moremath.mathutils import exp_decay_between
from moremath.vectors import vector_add, vector_distance_torus_wrapped
from learning.reinforcement.tools import ReplayMemory, Transition
from datahandling.runningstats import MovingAverage


# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class SnakeGameEnv(gym.Env[np.ndarray, tuple[int, int]]):
    metadata = {'render.modes': ['human']}

    def __init__(self, cells_width: int, cells_height: int, max_moves: int = 1000) -> None:
        cells_grid_size: tuple[int, int] = (cells_width, cells_height)
        self.game = SnakeGameLogic(cells_grid_size)
        self.max_moves: int = max_moves
        self.moves: int = 0
        self.done: bool = False
        self.observation_space = spaces.Box(low=np.array((0, 0)), high=np.array(cells_grid_size), dtype=np.int8)
        self.action_space = spaces.Discrete(4)
        self.actions_mapping: dict[int, tuple[int, int]] = {
            0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)
        }
        self.reset()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Update action and move snake
        direction = self.actions_mapping[action]
        with self.game._direction:
            self.game._direction.set_object(direction)
        self.game._move_snake()
        self.moves += 1

        observation = self.get_obs()
        distance = vector_distance_torus_wrapped(
            vector_add(
                self.game._snake[0],
                direction
            ),
            self.game._food,
            self.game._grid_size,
            manhattan=True
        )
        terminated = self.game._game_over
        if terminated:
            reward = -10.0
        else:
            reward = exp_decay_between(distance, 1.0, sum(self.game._grid_size) / 2.0) * 10.0
        truncated = self.moves >= self.max_moves
        with self.game._direction:
            info = {
                "score": self.game._score,
                "head": self.game._snake[0],
                "direction": self.game._direction.get_object(),
                "food": self.game._food,
                "distance": distance
            }
        return observation, reward, terminated, truncated, info

    def reset(
        self, *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.moves = 0
        self.done = False
        self.game._restart()
        with self.game._direction:
            info = {
                "head": self.game._snake[0],
                "direction": self.game._direction.get_object(),
                "food": self.game._food
            }
        return self.get_obs(), info

    def get_obs(self):
        return get_state(self.game)


def get_state(game: SnakeGameLogic) -> np.ndarray:
    obs = np.zeros(game._grid_size, dtype=np.int8)
    obs[game._food] = 1
    obs[tuple(segment for segment in zip(*game._snake[1:]))] = 2
    obs[game._snake[0]] = 3
    if game._obstacles:
        obs[tuple(obstacle for obstacle in zip(*game._obstacles))] = 4
    return obs


class DQN(nn.Module):

    def __init__(self, height: int, width: int, n_actions: int) -> None:
        super().__init__()
        self.n_observations = height * width
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=5,
            kernel_size=3, stride=1,
            padding=1, padding_mode="circular"
        )
        self.conv1_out_shape = calc_conv_output_shape_from(
            (height, width),
            self.conv1
        )
        self.conv1_flat_size = size_of_flat_layer(self.conv1_out_shape, 5)
        self.conv2 = nn.Conv2d(
            in_channels=5, out_channels=20,
            kernel_size=3, stride=1,
            padding=1, padding_mode="circular"
        )
        self.conv2_out_shape = calc_conv_output_shape_from(
            self.conv1_out_shape,
            self.conv2
        )
        self.conv2_flat_size = size_of_flat_layer(self.conv2_out_shape, 20)
        self.lin1 = nn.Linear(self.conv2_flat_size, self.conv2_flat_size * 2)
        self.lin2 = nn.Linear(self.conv2_flat_size * 2, self.conv2_flat_size)
        self.lin3 = nn.Linear(self.conv2_flat_size, self.conv2_flat_size // 2)
        self.lin4 = nn.Linear(self.conv2_flat_size // 2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Called with either one element to determine next action, or a batch
        during optimization.
        """
        x = x.unsqueeze(1)  # Add the "channel" dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.conv2_flat_size)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x)


def train(args: argparse.Namespace) -> None:
    plt.ion()

    snake_env = SnakeGameEnv(10, 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the AdamW optimizer
    BATCH_SIZE = 256
    GAMMA = 0.99
    EPS_START = 0.95
    EPS_END = 0.05
    EPS_DECAY = 10000
    TAU = 0.005
    LR = 1e-4

    state, info = snake_env.reset()
    n_actions = snake_env.action_space.n

    policy_net = DQN(*state.shape, n_actions).to(device)
    target_net = DQN(*state.shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    steps_done = 0

    def select_action(state):
        nonlocal steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * (steps_done / EPS_DECAY))
        steps_done += 1
        ## With random probability, choose exploitatively, the best currently known action.
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        ## Otherwise, randomly choose an action exploratorively
        else:
            return torch.tensor([[snake_env.action_space.sample()]], device=device, dtype=torch.long)

    episode_scores = []
    episode_durations = []

    def plot_output(show_result=False):
        plt.figure(1)
        scores_t = torch.tensor(episode_scores, dtype=torch.float)
        # durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.plot(scores_t.numpy())
        # Take 100 episode averages and plot them too
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    if torch.cuda.is_available():
        num_episodes = 10000
    else:
        num_episodes = 1000

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = snake_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, info = snake_env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_scores.append(info["score"])
                episode_durations.append(t + 1)
                plot_output()
                break

    print("Complete")
    plot_output(show_result=True)
    plt.ioff()
    plt.show()

    # Save the model
    if args.save:
        if not os.path.exists("models"):
            os.mkdir("models")
        torch.save(policy_net, "models/policy_net.pth")


def render(args: argparse.Namespace) -> None:
    width: int = args.width
    height: int = args.height
    debug: bool = args.debug

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = torch.load("models/policy_net.pth")
    policy_net.layer1_out_shape = calc_output_shape((10, 10), 2, 1, 0, 1)
    policy_net.layer1_flat_size = size_of_flat_layer(policy_net.layer1_out_shape, 5)
    print("policy_net.layer1_out_shape", policy_net.layer1_out_shape)
    print("policy_net.layer1_flat_size", policy_net.layer1_flat_size)

    qapp = QtWidgets.QApplication([])
    qwindow = QtWidgets.QMainWindow()
    qwindow.setWindowTitle("Snake Game")
    qwindow.resize(width, height)

    jdata = JinxGuiData("Snake GUI Data", debug=debug)
    jgui = JinxGuiWindow(qwindow, jdata, "Snake GUI Window", debug=debug)

    snake_qwidget = QtWidgets.QWidget()
    snake_game_logic = SnakeGameLogic((10, 10))
    snake_game_jwidget = SnakeGameJinxWidget(
        snake_qwidget, width, height,
        snake_game_logic=snake_game_logic, manual_update=True, debug=debug
    )
    snake_game_logic._restart()
    print(snake_game_logic._grid_size)
    # snake_options_widget = QWidget()
    # snake_game_options_jwidget = GamePerformanceDisplayJinxWidget(
    #     snake_options_widget, jdata, debug=debug
    # )

    jgui.add_view("Snake Game", snake_game_jwidget)
    # jgui.add_view("Snake Game Performance", snake_game_options_jwidget)
    jdata.desired_view_state = "Snake Game"

    def select_action(state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

    def set_action() -> None:
        if snake_game_logic._game_over:
            snake_game_logic._restart()
        
        state = get_state(snake_game_logic)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = select_action(state)
        snake_game_jwidget.manual_update_game(action.item())

    qtimer = QTimer()
    qtimer.timeout.connect(set_action)
    qtimer.setInterval(int((1.0 / args.fps) * 100))
    qtimer.start()

    qwindow.show()
    qapp.exec()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load the model")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("-wi", "--width", type=int, default=200)
    parser.add_argument("-he", "--height", type=int, default=200)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--save", action="store_true", help="Save the model")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps per episode")
    # parser.add_argument("--record", action="store_true", help="Record the environment")
    # parser.add_argument("--record_fps", type=int, default=10, help="Frames per second for recording")
    # parser.add_argument("--record_path", type=str, default="recordings", help="Path to save the recordings")
    # parser.add_argument("--record_name", type=str, default="recording", help="Name of the recording")

    args = parser.parse_args()

    if args.render:
        render(args)
    elif args.train:
        train(args)
