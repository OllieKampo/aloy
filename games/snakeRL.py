from itertools import count
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

from games.snakegame import SnakeGameLogic
from moremath.mathutils import exp_decay_between
from moremath.vectors import vector_add, vector_distance_torus_wrapped, vector_magnitude
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

    def step(self, action: tuple[int, int]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ## Update action and move snake
        with self.game._direction:
            self.game._direction.set_object(self.actions_mapping[action])
        self.game._move_snake()
        self.moves += 1

        observation = self.get_obs()
        distance = vector_distance_torus_wrapped(self.game._snake[0], self.game._food, self.game._grid_size, manhattan=True)
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
        obs = np.zeros(self.game._grid_size, dtype=np.int8)
        obs[self.game._food] = 1
        obs[tuple(segment for segment in zip(*self.game._snake))] = 2
        if self.game._obstacles:
            obs[tuple(obstacle for obstacle in zip(*self.game._obstacles))] = 3
        return obs

snake_env = SnakeGameEnv(10, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.n_observations = n_observations
        self.layer1 = nn.Linear(n_observations, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x.view(-1, self.n_observations))) # -1 is the batch size
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

state, info = snake_env.reset()
n_observations = state.size
n_actions = snake_env.action_space.n

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
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
    num_episodes = 5000
else:
    num_episodes = 500

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

print('Complete')
plot_output(show_result=True)
plt.ioff()
plt.show()
