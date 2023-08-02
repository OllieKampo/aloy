import argparse
from itertools import count
import os
from typing import Any
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
from PySide6 import QtWidgets
from PySide6.QtCore import QTimer

from aloy.games.snakegame import SnakeGameAloyWidget, SnakeGameLogic
from aloy.guis.gui import AloySystemData, AloyGuiWindow
from aloy.learning.convolutional import calc_conv_output_shape, calc_conv_output_shape_from, size_of_flat_layer
from aloy.moremath.mathutils import exp_decay_between
from aloy.moremath.vectors import vector_distance_torus_wrapped
from aloy.learning.reinforcement.tools import ReplayMemory, Transition


class SnakeGameEnv(gym.Env[np.ndarray, tuple[int, int]]):
    """Snake game environment."""

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        cells_width: int,
        cells_height: int,
        max_moves: int = 1000
    ) -> None:
        """Initialize the environment."""
        cells_grid_size: tuple[int, int] = (cells_width, cells_height)
        self.game = SnakeGameLogic(cells_grid_size)
        self.max_moves: int = max_moves
        self.moves: int = 0
        self.done: bool = False
        self.observation_space = spaces.Box(
            low=np.array((0, 0)),
            high=np.array(cells_grid_size),
            dtype=np.int8
        )
        self.action_space = spaces.Discrete(4)
        self.actions_mapping: dict[int, tuple[int, int]] = {
            0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0)
        }
        self.reset()

    def step(
        self,
        action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Perform one step in the environment."""

        current_score = self.game.score

        # Update action and move snake
        direction = self.actions_mapping[action]
        with self.game.direction:
            self.game.direction.set_obj(direction)
        self.game.move_snake()
        self.moves += 1

        observation = self.get_obs()
        distance = vector_distance_torus_wrapped(
            self.game.snake[0],
            self.game.food,
            self.game.grid_size,
            manhattan=True
        )

        terminated = self.game.game_over
        if terminated:
            reward = -10.0
        else:
            reward = exp_decay_between(
                distance,
                0.0,
                sum(self.game.grid_size)
            ) * 10.0
            if self.game.score > current_score:
                reward += 10.0

        truncated = self.moves >= self.max_moves
        info = {
            "score": self.game.score,
            "head": self.game.snake[0],
            "direction": self.game.direction.get_obj(),
            "food": self.game.food,
            "distance": distance
        }
        return observation, reward, terminated, truncated, info

    # pylint: disable=unused-argument
    def reset(
        self, *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        self.moves = 0
        self.done = False
        self.game.restart()
        with self.game.direction:
            info = {
                "head": self.game.snake[0],
                "direction": self.game.direction.get_obj(),
                "food": self.game.food
            }
        return self.get_obs(), info

    def get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return self.game.get_state()

    def render(self, mode: str = "human") -> None:
        """Render the environment."""
        if mode == "human":
            plt.imshow(self.game.get_state())
            plt.show()
        else:
            raise NotImplementedError


class DQN(nn.Module):

    def __init__(self, height: int, width: int, n_actions: int) -> None:
        super().__init__()
        self.n_observations = height * width
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16,
            kernel_size=3, stride=1,
            padding=1, padding_mode="circular"
        )
        self.conv1_out_shape = calc_conv_output_shape_from(
            (height, width),
            self.conv1
        )
        self.conv1_flat_size = size_of_flat_layer(self.conv1_out_shape, 16)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32,
            kernel_size=3, stride=1,
            padding=1, padding_mode="circular"
        )
        self.conv2_out_shape = calc_conv_output_shape_from(
            self.conv1_out_shape,
            self.conv2
        )
        self.conv2_flat_size = size_of_flat_layer(self.conv2_out_shape, 32)
        self.lin1 = nn.Linear(self.conv2_flat_size, self.conv2_flat_size * 2)
        self.lin2 = nn.Linear(self.conv2_flat_size * 2, self.conv2_flat_size)
        self.lin3 = nn.Linear(self.conv2_flat_size, self.conv2_flat_size // 2)
        self.lin4 = nn.Linear(self.conv2_flat_size // 2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Called with either one element to determine next action, or a batch
        during optimization.
        """
        # Add the "channel" dimension
        x = x.unsqueeze(1)  # B, C, H, W
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.conv2_flat_size)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x)


def select_action(
    policy_net: DQN,
    env: gym.Env,
    state: torch.Tensor,
    steps_done: int,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    device: torch.device
) -> torch.Tensor:
    """Select an action to perform."""
    sample = random.random()
    eps_threshold = (
        eps_end
        + (eps_start - eps_end)
        * math.exp(-1.0 * (steps_done / eps_decay))
    )
    # With random probability, choose exploitatively, the best currently known
    # action.
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    # Otherwise, randomly choose an action exploratorively
    else:
        return torch.tensor(
            [[env.action_space.sample()]],
            device=device,
            dtype=torch.long
        )


def plot_output(episode_scores, show_result=False):
    plt.figure(1)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
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
    # pause a bit so that plots are updated
    plt.pause(0.001)


def optimize_model(
    policy_net: DQN,
    target_net: DQN,
    memory: ReplayMemory,
    optimizer: optim.Adam,
    batch_size: int,
    gamma: float,
    device: str | torch.device
) -> None:
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None,
                  batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state
         if s is not None]
    )
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
    # This is merged based on the mask, such that we'll have either the
    # expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]
    # Compute the expected Q values.
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(
        state_action_values,
        expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def qtrain(
    env: gym.Env,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    num_episodes: int,
    batch_size: int,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    tau: float,
    save: bool,
    save_path: str
) -> None:
    """Train a DQN agent."""
    plt.ion()

    target_net.load_state_dict(policy_net.state_dict())
    memory = ReplayMemory(10000)
    steps_done = 0

    episode_scores = []
    episode_durations = []

    for _ in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        state = state.unsqueeze(0)

        for t in count():
            action = select_action(
                policy_net,
                env,
                state,
                steps_done,
                eps_start,
                eps_end,
                eps_decay,
                device
            )
            steps_done += 1
            observation, reward, terminated, truncated, info = \
                env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(
                policy_net,
                target_net,
                memory,
                optimizer,
                batch_size,
                gamma,
                device
            )

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (
                    (policy_net_state_dict[key] * tau)
                    + (target_net_state_dict[key] * (1 - tau))
                )
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_scores.append(info["score"])
                episode_durations.append(t + 1)
                plot_output(episode_scores)
                break

    print("Complete")
    plot_output(episode_scores, show_result=True)
    plt.ioff()
    plt.show()

    # Save the model
    if save:
        os.makedirs(save_path, exist_ok=True)
        torch.save(policy_net, "models/policy_net.pth")


def render(args: argparse.Namespace) -> None:
    width: int = args.width
    height: int = args.height
    debug: bool = args.debug

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = torch.load("models/policy_net.pth")
    policy_net.layer1_out_shape = calc_conv_output_shape(
        (10, 10), 2, 1, 0, 1)
    policy_net.layer1_flat_size = size_of_flat_layer(
        policy_net.layer1_out_shape, 5)
    print("policy_net.layer1_out_shape", policy_net.layer1_out_shape)
    print("policy_net.layer1_flat_size", policy_net.layer1_flat_size)

    qapp = QtWidgets.QApplication([])
    qwindow = QtWidgets.QMainWindow()
    qwindow.setWindowTitle("Snake Game")
    qwindow.resize(width, height)

    jdata = AloySystemData("Snake GUI Data", debug=debug)
    jgui = AloyGuiWindow(qwindow, jdata, "Snake GUI Window", debug=debug)

    snake_qwidget = QtWidgets.QWidget()
    snake_game_logic = SnakeGameLogic((10, 10))
    snake_game_jwidget = SnakeGameAloyWidget(
        snake_qwidget,
        size=(width, height),
        snake_game_logic=snake_game_logic,
        manual_update=True,
        debug=debug
    )
    snake_game_logic.restart()
    print(snake_game_logic.grid_size)
    # snake_options_widget = QWidget()
    # snake_game_options_jwidget = GamePerformanceDisplayAloyWidget(
    #     snake_options_widget, jdata, debug=debug
    # )

    jgui.add_view("Snake Game", snake_game_jwidget)
    # jgui.add_view("Snake Game Performance", snake_game_options_jwidget)
    jdata.desired_view_state = "Snake Game"

    def best_action(state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

    def set_action() -> None:
        if snake_game_logic.game_over:
            snake_game_logic.restart()

        state = snake_game_logic.get_state()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = best_action(state)
        snake_game_jwidget.manual_update_game(action.item())

    qtimer = QTimer()
    qtimer.timeout.connect(set_action)
    qtimer.setInterval(int((1.0 / args.fps) * 100))
    qtimer.start()

    qwindow.show()
    qapp.exec()


def main() -> None:
    parser = argparse.ArgumentParser()
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the AdamW optimizer
    parser.add_argument("--load", action="store_true", help="Load the model")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("-wi", "--width", type=int, default=200)
    parser.add_argument("-he", "--height", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--save", action="store_true", help="Save the model")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eps_start", type=float, default=0.95, help="Starting epsilon value")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Ending epsilon value")
    parser.add_argument("--eps_decay", type=int, default=10000, help="Number of episodes to decay epsilon")
    parser.add_argument("--tau", type=float, default=0.005, help="Tau value for soft update of target network")
    parser.add_argument("--save", action="store_true", help="Save the model")
    parser.add_argument("--save_path", type=str, default="models/", help="Path to save the model")
    parser.add_argument("--target_update", type=int, default=10, help="Number of episodes to update target network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--memory_size", type=int, default=10000, help="Size of the replay memory")
    parser.add_argument("--memory_min", type=int, default=1000, help="Minimum number of transitions to start training")
    parser.add_argument("--memory_batch", type=int, default=128, help="Batch size for replay memory")
    parser.add_argument("--memory_alpha", type=float, default=0.6, help="Alpha value for prioritized replay memory")
    parser.add_argument("--memory_beta", type=float, default=0.4, help="Beta value for prioritized replay memory")
    parser.add_argument("--memory_beta_end", type=float, default=1.0, help="Ending beta value for prioritized replay memory")
    parser.add_argument("--memory_beta_decay", type=int, default=200, help="Number of episodes to decay beta")
    # parser.add_argument("--record", action="store_true", help="Record the environment")
    # parser.add_argument("--record_fps", type=int, default=10, help="Frames per second for recording")
    # parser.add_argument("--record_path", type=str, default="recordings", help="Path to save the recordings")
    # parser.add_argument("--record_name", type=str, default="recording", help="Name of the recording")

    args = parser.parse_args()

    snake_env = SnakeGameEnv(10, 10)
    state, info = snake_env.reset()
    n_actions = snake_env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(*state.shape, n_actions).to(device)
    target_net = DQN(*state.shape, n_actions).to(device)

    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)

    if args.render:
        render(args)
    elif args.train:
        qtrain(
            snake_env,
            policy_net,
            target_net,
            optimizer,
            device,
            args.num_episodes,
            args.batch_size,
            args.gamma,
            args.eps_start,
            args.eps_end,
            args.eps_decay,
            args.tau,
            args.save,
            args.save_path
        )


if __name__ == "__main__":
    main()
