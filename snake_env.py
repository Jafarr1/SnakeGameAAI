import numpy as np

from snakeai import Direction, SnakeGame, Vector


ACTION_STRAIGHT = 0
ACTION_RIGHT = 1
ACTION_LEFT = 2


def _danger(game: SnakeGame, direction: Direction) -> bool:
    head = game.head
    if direction == Direction.RIGHT:
        point = Vector(head.x + 1, head.y)
    elif direction == Direction.LEFT:
        point = Vector(head.x - 1, head.y)
    elif direction == Direction.UP:
        point = Vector(head.x, head.y - 1)
    else:
        point = Vector(head.x, head.y + 1)
    return game.is_collision(point)


def get_state(game: SnakeGame) -> np.ndarray:
    head = game.head

    dir_left = game.direction == Direction.LEFT
    dir_right = game.direction == Direction.RIGHT
    dir_up = game.direction == Direction.UP
    dir_down = game.direction == Direction.DOWN

    danger_straight = (
        (dir_right and _danger(game, Direction.RIGHT))
        or (dir_left and _danger(game, Direction.LEFT))
        or (dir_up and _danger(game, Direction.UP))
        or (dir_down and _danger(game, Direction.DOWN))
    )

    danger_right = (
        (dir_up and _danger(game, Direction.RIGHT))
        or (dir_down and _danger(game, Direction.LEFT))
        or (dir_left and _danger(game, Direction.UP))
        or (dir_right and _danger(game, Direction.DOWN))
    )

    danger_left = (
        (dir_down and _danger(game, Direction.RIGHT))
        or (dir_up and _danger(game, Direction.LEFT))
        or (dir_right and _danger(game, Direction.UP))
        or (dir_left and _danger(game, Direction.DOWN))
    )

    food_left = game.food.x < head.x
    food_right = game.food.x > head.x
    food_up = game.food.y < head.y
    food_down = game.food.y > head.y

    state = np.array(
        [
            danger_straight,
            danger_right,
            danger_left,
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            food_left,
            food_right,
            food_up,
            food_down,
        ],
        dtype=np.int32,
    )

    return state


class SnakeEnv:
    def __init__(self, *, render: bool = False, speed: int = 0, xsize: int = 30, ysize: int = 30, scale: int = 20):
        self.game = SnakeGame(xsize=xsize, ysize=ysize, scale=scale, render=render, speed=speed)

    def reset(self):
        self.game.reset()
        return get_state(self.game)

    def step(self, action: int):
        if action == ACTION_RIGHT:
            move = [0, 1, 0]
        elif action == ACTION_LEFT:
            move = [0, 0, 1]
        else:
            move = [1, 0, 0]

        reward, done, score = self.game.play_step(move)
        next_state = get_state(self.game)
        info = {"score": score}
        return next_state, reward, done, False, info

    def render(self):
        return self.game.render_frame()
