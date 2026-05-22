from enum import Enum
import random
from collections import deque
import pygame
import numpy as np


pygame.init()

FONT = pygame.font.SysFont('arial', 25)

BLOCK_SIZE = 20
SPEED = 40


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def copy(self):
        return Vector(self.x, self.y)

    @classmethod
    def random_within(cls, scope):
        return Vector(
            random.randint(0, scope.x - 1),
            random.randint(0, scope.y - 1)
        )


class SnakeGame:

    def __init__(self, xsize=30, ysize=30, scale=20, *, render=True, speed=SPEED):

        self.grid = Vector(xsize, ysize)
        self.scale = scale

        self.width = xsize * scale
        self.height = ysize * scale

        self.render_enabled = render
        self.speed = speed

        if self.render_enabled:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI")
        else:
            self.display = pygame.Surface((self.width, self.height))

        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):

        self.direction = Direction.RIGHT

        center = Vector(
            self.grid.x // 2,
            self.grid.y // 2
        )

        self.snake = deque([
            center,
            Vector(center.x - 1, center.y),
            Vector(center.x - 2, center.y)
        ])

        self.head = self.snake[0]

        self.score = 0
        self.frame_iteration = 0

        self._place_food()

    def _place_food(self):

        while True:

            food = Vector.random_within(self.grid)

            if food not in self.snake:
                self.food = food
                break

    def play_step(self, action):

        self.frame_iteration += 1

        prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        # quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move
        self._move(action)

        self.snake.appendleft(self.head)

        reward = -0.01
        game_over = False

        # collision
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):

            game_over = True
            reward = -10

            return reward, game_over, self.score

        # food
        if self.head == self.food:

            self.score += 1
            reward = 10

            self._place_food()

        else:
            self.snake.pop()

        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        if new_distance < prev_distance:
            reward += 0.1
        elif new_distance > prev_distance:
            reward -= 0.1

        # update UI
        self._update_ui()

        if self.speed and self.speed > 0:
            self.clock.tick(self.speed)

        return reward, game_over, self.score

    def is_collision(self, pt=None):

        if pt is None:
            pt = self.head

        # wall collision
        if (
            pt.x < 0 or
            pt.x >= self.grid.x or
            pt.y < 0 or
            pt.y >= self.grid.y
        ):
            return True

        # self collision
        if pt in list(self.snake)[1:]:
            return True

        return False

    def _draw(self):

        self.display.fill((0, 0, 0))

        # snake
        for i, pt in enumerate(self.snake):

            color = (0, max(100, 255 - i * 5), 0)

            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(
                    pt.x * self.scale,
                    pt.y * self.scale,
                    self.scale,
                    self.scale
                )
            )

        # food
        pygame.draw.rect(
            self.display,
            (255, 0, 0),
            pygame.Rect(
                self.food.x * self.scale,
                self.food.y * self.scale,
                self.scale,
                self.scale
            )
        )

        # score
        text = FONT.render(f"Score: {self.score}", True, (255,255,255))
        self.display.blit(text, [0,0])

    def _update_ui(self):

        if not self.render_enabled:
            return

        self._draw()
        pygame.display.flip()

    def render_frame(self):

        self._draw()
        frame = pygame.surfarray.array3d(self.display)
        return np.transpose(frame, (1, 0, 2))

    def _move(self, action):

        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP
        ]

        idx = clock_wise.index(self.direction)

        # straight
        if np.array_equal(action, [1,0,0]):

            new_dir = clock_wise[idx]

        # right turn
        elif np.array_equal(action, [0,1,0]):

            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]

        # left turn
        else:

            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += 1

        elif self.direction == Direction.LEFT:
            x -= 1

        elif self.direction == Direction.DOWN:
            y += 1

        elif self.direction == Direction.UP:
            y -= 1

        self.head = Vector(x, y)


if __name__ == '__main__':

    game = SnakeGame()

    while True:

        # temporary random AI
        action = random.choice([
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])

        reward, game_over, score = game.play_step(action)

        if game_over:

            print("Final Score:", score)

            break