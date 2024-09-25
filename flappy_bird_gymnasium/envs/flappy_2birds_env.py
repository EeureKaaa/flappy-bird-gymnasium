# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal with the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==============================================================================

import gymnasium
import numpy as np
import pygame
from enum import IntEnum
from itertools import cycle
from typing import Dict, Optional, Tuple, Union, List
from flappy_bird_gymnasium.envs import utils
from flappy_bird_gymnasium.envs.constants import (
    BACKGROUND_WIDTH,
    BASE_WIDTH,
    FILL_BACKGROUND_COLOR,
    LIDAR_MAX_DISTANCE,
    PIPE_HEIGHT,
    PIPE_VEL_X,
    PIPE_WIDTH,
    PLAYER_ACC_Y,
    PLAYER_FLAP_ACC,
    PLAYER_HEIGHT,
    PLAYER_MAX_VEL_Y,
    PLAYER_PRIVATE_ZONE,
    PLAYER_ROT_THR,
    PLAYER_VEL_ROT,
    PLAYER_WIDTH,
)
from flappy_bird_gymnasium.envs.lidar import LIDAR


class Actions(IntEnum):
    """Possible actions for the player to take."""
    IDLE, FLAP = 0, 1


class Flappy2BirdsEnv(gymnasium.Env):
    """Flappy Bird Gymnasium environment that supports two birds."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        screen_size: Tuple[int, int] = (288, 512),
        audio_on: bool = False,
        normalize_obs: bool = True,
        use_lidar: bool = True,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        render_mode: Optional[str] = None,
        background: Optional[str] = "day",
        score_limit: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._debug = debug
        self._score_limit = score_limit

        self.action_space = gymnasium.spaces.MultiDiscrete([2, 2])  # Two birds
        if use_lidar:
            if normalize_obs:
                self.observation_space = gymnasium.spaces.Box(
                    0.0, 1.0, shape=(360,), dtype=np.float64  # 360 for two birds
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    0.0, np.inf, shape=(360,), dtype=np.float64
                )
        else:
            if normalize_obs:
                self.observation_space = gymnasium.spaces.Box(
                    -1.0, 1.0, shape=(24,), dtype=np.float64  # 12 for each bird
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(24,), dtype=np.float64
                )

        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap
        self._audio_on = audio_on
        self._use_lidar = use_lidar
        self._sound_cache = None
        self._player_flapped = [False, False]  # For two birds
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._bg_type = background
        self._player_idx = 0
        # 在初始化方法中设置鸟的初始位置
        # self._player_x[0] = self._screen_width * 0.2  # 左侧鸟
        # self._player_x[1] = self._screen_width * 0.4  # 右侧鸟

        self._ground = {"x": 0, "y": self._screen_height * 0.79}
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH

        self._upper_pipes = []
        self._lower_pipes = []
        self._generate_initial_pipes()  # 生成初始管道

        if use_lidar:
            self._lidar = LIDAR(LIDAR_MAX_DISTANCE)
            self._get_observation = self._get_observation_lidar
        else:
            self._get_observation = self._get_observation_features

        if render_mode is not None:
            self._fps_clock = pygame.time.Clock()
            self._display = None
            self._surface = pygame.Surface(screen_size)
            self._images = utils.load_images(
                convert=False,
                bird_color=bird_color,
                pipe_color=pipe_color,
                bg_type=background,
            )
            if audio_on:
                self._sounds = utils.load_sounds()

        self.reset()  # Initialize game state

    def _generate_initial_pipes(self):
        """生成初始管道对"""
        new_pipe = self._get_random_pipe()
        self._upper_pipes.append(new_pipe[0])
        self._lower_pipes.append(new_pipe[1])

    def step(self, actions: Tuple[Union[Actions, int], Union[Actions, int]]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        terminal = False
        truncated = False
        rewards = [0, 0]  # Initialize rewards for both birds

        for i in range(2):  # For each bird
            action = actions[i]
            if action == Actions.FLAP:
                if self._player_y[i] > -2 * PLAYER_HEIGHT:
                    self._player_vel_y[i] = PLAYER_FLAP_ACC
                    self._player_flapped[i] = True
                    self._sound_cache = "wing"

            # Check for score
            player_mid_pos = self._player_x[i] + PLAYER_WIDTH / 2
            for pipe in self._upper_pipes:
                pipe_mid_pos = pipe["x"] + PIPE_WIDTH / 2
                if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                    self._score += 1
                    rewards[i] += 1  # Reward for passing pipe
                    self._sound_cache = "point"

            # player_index base_x change
            if (self._loop_iter + 1) % 3 == 0:
                self._player_idx = next(self._player_idx_gen)

            self._loop_iter = (self._loop_iter + 1) % 30
            self._ground["x"] = -((-self._ground["x"] + 100) % self._base_shift)

            # Rotate the player
            if self._player_rot[i] > -90:
                self._player_rot[i] -= PLAYER_VEL_ROT

            # Player's movement
            if self._player_vel_y[i] < PLAYER_MAX_VEL_Y and not self._player_flapped[i]:
                self._player_vel_y[i] += PLAYER_ACC_Y

            if self._player_flapped[i]:
                self._player_flapped[i] = False
                self._player_rot[i] = 45

            self._player_y[i] += min(
                self._player_vel_y[i], self._ground["y"] - self._player_y[i] - PLAYER_HEIGHT
            )

        # Move pipes to the left
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            up_pipe["x"] += PIPE_VEL_X
            low_pipe["x"] += PIPE_VEL_X

            # If out of the screen, regenerate pipe
            if up_pipe["x"] < -PIPE_WIDTH:
                new_up_pipe, new_low_pipe = self._get_random_pipe()
                up_pipe["x"] = new_up_pipe["x"]
                up_pipe["y"] = new_up_pipe["y"]
                low_pipe["x"] = new_low_pipe["x"]
                low_pipe["y"] = new_low_pipe["y"]

        for i in range(2):
            if self._check_crash(i):
                self._sound_cache = "hit"
                rewards[i] = -2  # Penalize the bird that crashes
                terminal = True  # End the game immediately
                self._player_vel_y[i] = 0
                break  # Stop checking further once any bird crashes

        obs, reward_private_zone = self._get_observation()
        # for i in range(2):
        #     if rewards[i] == 0:
        #         if reward_private_zone is not None:
        #             rewards[i] = reward_private_zone
        #         else:
        #             rewards[i] = 0.1  # Reward for staying alive

        # Return total reward (sum of both birds' rewards), game termination status, and other info
        return obs, sum(rewards), terminal, truncated, {"score": self._score}


    def reset(self, seed=None, options=None):
        """Resets the environment (starts a new game)."""
        super().reset(seed=seed)

        # Initialize both birds' info
        self._player_x = [int(self._screen_width * 0.2), int(self._screen_width * 0.4)]
        self._player_y = [int((self._screen_height - PLAYER_HEIGHT) / 2)] * 2
        self._player_vel_y = [-9, -9]  # Players' velocity along Y
        self._player_rot = [45, 45]  # Players' rotation
        self._loop_iter = 0
        self._score = 0

        # Generate 3 new pipes to add to upper_pipes and lower_pipes lists
        new_pipe1 = self._get_random_pipe()
        new_pipe2 = self._get_random_pipe()
        new_pipe3 = self._get_random_pipe()

        # List of upper pipes:
        self._upper_pipes = [
            {"x": self._screen_width, "y": new_pipe1[0]["y"]},
            {
                "x": self._screen_width + (self._screen_width / 2),
                "y": new_pipe2[0]["y"],
            },
            {
                "x": self._screen_width + self._screen_width,
                "y": new_pipe3[0]["y"],
            },
        ]

        # List of lower pipes:
        self._lower_pipes = [
            {"x": self._screen_width, "y": new_pipe1[1]["y"]},
            {
                "x": self._screen_width + (self._screen_width / 2),
                "y": new_pipe2[1]["y"],
            },
            {
                "x": self._screen_width + self._screen_width,
                "y": new_pipe3[1]["y"],
            },
        ]

        if self.render_mode == "human":
            self.render()

        obs, _ = self._get_observation()
        info = {"score": self._score}
        return obs, info

    def render(self) -> None:
        """Renders the next frame."""
        if self.render_mode == "rgb_array":
            self._draw_surface(show_score=False, show_rays=False)
            return np.transpose(pygame.surfarray.array3d(self._surface), axes=(1, 0, 2))
        else:
            self._draw_surface(show_score=True, show_rays=self._use_lidar)
            if self._display is None:
                self._make_display()

            self._update_display()
            self._fps_clock.tick(self.metadata["render_fps"])

    def close(self):
        """Closes the environment."""
        if self.render_mode is not None:
            pygame.display.quit()
            pygame.quit()
        super().close()

    def _get_random_pipe(self) -> List[Dict[str, int]]:
        """返回随机生成的管道"""
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
        index = self.np_random.integers(0, len(gapYs))
        gap_y = gapYs[index]
        gap_y += int(self._ground["y"] * 0.2)

        pipe_x = self._screen_width + PIPE_WIDTH + (self._screen_width * 0.2)
        return [
            {"x": pipe_x, "y": gap_y - PIPE_HEIGHT},  # 上管道
            {"x": pipe_x, "y": gap_y + self._pipe_gap},  # 下管道
        ]

    def _check_crash(self, bird_index: int) -> bool:
        """Returns True if the specified bird collides with the ground (base) or a pipe."""
        if self._player_y[bird_index] <= 0:
            return True 
        if self._player_y[bird_index] + PLAYER_HEIGHT >= self._ground["y"] - 1:
            return True
        
        player_rect = pygame.Rect(
            self._player_x[bird_index], self._player_y[bird_index], PLAYER_WIDTH, PLAYER_HEIGHT
        )

        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            up_pipe_rect = pygame.Rect(
                up_pipe["x"], up_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
            )
            low_pipe_rect = pygame.Rect(
                low_pipe["x"], low_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
            )

            if player_rect.colliderect(up_pipe_rect) or player_rect.colliderect(low_pipe_rect):
                return True

        return False

    def _get_observation_features(self) -> np.ndarray:
        """Generates observations based on features for both birds."""
        pipes = []
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            if low_pipe["x"] > self._screen_width:
                pipes.append((self._screen_width, 0, self._screen_height))
            else:
                pipes.append(
                    (low_pipe["x"], (up_pipe["y"] + PIPE_HEIGHT), low_pipe["y"])
                )

        pipes = sorted(pipes, key=lambda x: x[0])
        obs = []

        for i in range(2):  # For both birds
            pos_y = self._player_y[i]
            vel_y = self._player_vel_y[i]
            rot = self._player_rot[i]

            if self._normalize_obs:
                pos_y /= self._screen_height
                vel_y /= PLAYER_MAX_VEL_Y
                rot /= 90

            obs.extend([
                pipes[0][0],  # Last pipe's horizontal position
                pipes[0][1],  # Last top pipe's vertical position
                pipes[0][2],  # Last bottom pipe's vertical position
                pipes[1][0],  # Next pipe's horizontal position
                pipes[1][1],  # Next top pipe's vertical position
                pipes[1][2],  # Next bottom pipe's vertical position
                pos_y,        # Player's vertical position
                vel_y,        # Player's vertical velocity
                rot           # Player's rotation
            ])

        return np.array(obs), None

    def _get_observation_lidar(self) -> np.ndarray:
        """Generates LIDAR observations for both birds."""
        distances = []
        for i in range(2):
            distance = self._lidar.scan(
                self._player_x[i],
                self._player_y[i],
                self._player_rot[i],
                self._upper_pipes,
                self._lower_pipes,
                self._ground,
            )
            distances.append(distance)

        distances = np.concatenate(distances)  # Combine distances for both birds

        if np.any(distances < PLAYER_PRIVATE_ZONE):
            reward = -0.5
        else:
            reward = None

        if self._normalize_obs:
            distances = distances / LIDAR_MAX_DISTANCE

        return distances, reward

    def _make_display(self) -> None:
        """Initializes the pygame's display."""
        self._display = pygame.display.set_mode(
            (self._screen_width, self._screen_height)
        )
        for name, value in self._images.items():
            if value is None:
                continue

            if type(value) in (tuple, list):
                self._images[name] = tuple([img.convert_alpha() for img in value])
            else:
                self._images[name] = (
                    value.convert() if name == "background" else value.convert_alpha()
                )

    def _draw_surface(self, show_score: bool = True, show_rays: bool = True) -> None:
        """Re-draws the renderer's surface."""
        if self._images["background"] is not None:
            self._surface.blit(self._images["background"], (0, 0))
        else:
            self._surface.fill(FILL_BACKGROUND_COLOR)

        # Draw pipes
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            self._surface.blit(self._images["pipe"][0], (up_pipe["x"], up_pipe["y"]))
            self._surface.blit(self._images["pipe"][1], (low_pipe["x"], low_pipe["y"]))

        # Draw ground
        self._surface.blit(self._images["base"], (self._ground["x"], self._ground["y"]))

        # Draw players
        for i in range(2):
            visible_rot = self._player_rot[i]
            if visible_rot <= PLAYER_ROT_THR:
                visible_rot = self._player_rot[i]

            player_surface = pygame.transform.rotate(
                self._images["player"][self._player_idx],
                visible_rot,
            )
            player_surface_rect = player_surface.get_rect(
                topleft=(self._player_x[i], self._player_y[i])
            )
            self._surface.blit(player_surface, player_surface_rect)

        # Draw score
        if show_score:
            self._draw_score()


    def _update_display(self) -> None:
        """Updates the display with the current surface of the renderer."""
        if self._display is None:
            raise RuntimeError(
                "Tried to update the display, but a display hasn't been created yet!"
            )

        pygame.event.get()
        self._display.blit(self._surface, [0, 0])
        pygame.display.update()

        # Play sounds
        if self._audio_on and self._sound_cache is not None:
            self._sounds[self._sound_cache].play()
