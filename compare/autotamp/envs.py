"""AutoTAMP's single-agent environments (env_and_optimize_single_agent.py).

Every region is an axis-aligned box stored the way the release stores it:
`[-x_min, x_max, -y_min, y_max, name]`, so that with
A = [[-1, 0], [1, 0], [0, -1], [0, 1]] the box is {p : A p <= b}. The numbers
are the release's, including the ones it shows the LLM as "position and size".

    HouseWorld   the house of Finucane et al. (LTLMoP): eight coloured rooms
                 inside a 3 x 3 square, the paper's HouseWorld1 and HouseWorld2
    chip         Chip's Challenge: keys, doors, goals and walls in an 8 x 8
                 maze, three layouts (env1, env2, env3)

`regions(...)` returns the environment as the release's `tag_and_shape`
(colour -> boxes) and `prop2block` (prop_i -> colour, box, function, (A, b)),
numbered in the order the release numbers them, which is the order the rooms
are named to the LLM (room1, room2, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


def _thicken(segments: list[list[float]], half_width: float) -> list[list[float]]:
    """Axis-aligned wall or door segments [x1, x2, y1, y2] as boxes of the given half width."""
    boxes = []
    for segment in segments:
        wall = np.array(segment, dtype=np.float64)
        if wall[0] == wall[1]:
            wall[0] -= half_width
            wall[1] += half_width
        elif wall[2] == wall[3]:
            wall[2] -= half_width
            wall[3] += half_width
        else:
            raise ValueError("wrong shape for axis-aligned wall")
        wall *= np.array([-1, 1, -1, 1])
        boxes.append(wall.tolist())
    return boxes


def houseworld() -> dict[str, list[list]]:
    half_len, half_width = 1.5, 0.05
    walls = _thicken(
        [
            [-half_len, -half_len, -half_len, half_len],
            [half_len, half_len, -half_len, half_len],
            [-half_len, half_len, -half_len, -half_len],
            [-half_len, half_len, half_len, half_len],
        ],
        half_width,
    )
    return {
        "cyan": [[1.2, -0.8, 1.4, -0.7, "Kitchen"]],
        "red": [[0, 0.9, 1.4, -0.7, "RestRoom2"]],
        "green": [[-0.2, 1.43, -0.8, 1.43, "MasterBedroom"]],
        "pink": [[-0.6, 1.43, -0.2, 0.8, "Bedroom"]],
        "blue": [[0.5, 0.5, 0.5, 0.5, "LivingRoom"], [1.2, -0.7, 0.05, 0.5, "ExerciseRoom"]],
        "yellow": [[1.4, -0.2, -0.8, 1.43, "RestRoom"]],
        "purple": [[0.7, -0.3, 1.4, -0.8, "DiningRoom"]],
        "black": [wall + ["wall"] for wall in walls],
    }


_CHIP_WALLS_23 = [
    [0, 0, 0, 8], [8, 8, 0, 8], [0, 8, 0, 0], [0, 8, 8, 8], [3.5, 7, 6, 6], [0, 2.5, 6, 6], [2, 4, 1, 1],
    [5, 6, 1, 1], [2, 2, 1, 4], [2, 4, 4, 4], [4, 4, 4, 6], [2, 2, 4, 6], [6, 6, 0, 5],
]


def chip(environment: str = "env2") -> dict[str, list[list]]:
    half_width = 0.1
    if environment == "env1":
        doors = _thicken([[2, 2, 0, 1], [6, 6, 5, 6], [6, 8, 2, 2], [0, 2, 4, 4], [7, 8, 6, 6]], half_width)
        key_half_width = 0.55
        keys = [
            [-(x - key_half_width), x + key_half_width, -(y - key_half_width), y + key_half_width]
            for x, y in [(5, 1), (3, 3), (1, 1), (7, 1), (3, 5)]
        ]
        walls = _thicken(
            [[0, 0, 0, 8], [8, 8, 0, 8], [0, 8, 0, 0], [0, 8, 8, 8], [0, 7, 6, 6], [2, 2, 1, 4], [2, 4, 4, 4],
             [4, 4, 4, 6], [6, 6, 0, 5]],
            half_width,
        )
        return {
            "green": [key + [f"key{i + 1}"] for i, key in enumerate(keys)],
            "blue": [[-0.5, 1.5, -6.5, 7.5, "goal"], [-6.7, 7.5, -3.7, 4.5, "goal"]],
            "red": [door + [f"door{i + 1}"] for i, door in enumerate(doors)],
            "black": [wall + ["wall"] for wall in walls],
        }
    walls = [wall + ["wall"] for wall in _thicken(_CHIP_WALLS_23, half_width)]
    if environment == "env2":
        return {
            "red": [[-2.5, 3.5, -5.9, 6.1, "door1"], [-5.9, 6.1, -5, 6, "door2"], [-4, 5, -0.9, 1.1, "door3"]],
            "green": [[-2.5, 2.9, -4.5, 4.9, "key1"], [-3.4, 3.8, -5.4, 5.8, "key2"], [-6.5, 7.5, -1.5, 2.5, "key3"]],
            "black": walls,
            "blue": [[-0.5, 1.5, -6.5, 7.5, "goal"], [-2.5, 3.5, -1.7, 2.7, "goal"], [-4.5, 5.5, -2, 3, "goal"],
                     [-0.5, 1.5, -3.5, 4.5, "goal"]],
        }
    if environment == "env3":
        return {
            "red": [[-2.5, 3.5, -5.9, 6.1, "door1"], [-7, 8, -5.9, 6.1, "door2"], [-5.9, 6.1, -5, 6, "door3"],
                    [-4, 5, -0.9, 1.1, "door4"], [-6, 8, -0.9, 1.1, "door5"], [0, 2, -2.9, 3.1, "door6"],
                    [0, 2, -1.9, 2.1, "door7"]],
            "green": [[-2.5, 2.9, -4.5, 4.9, "key1"], [-3.2, 3.8, -7.2, 7.8, "key2"], [-3.4, 3.8, -5.4, 5.8, "key3"],
                      [-6.5, 7.5, -1.5, 2.5, "key4"], [-4.5, 5.5, -6.5, 7.5, "key5"], [-2.2, 2.9, -5.2, 5.9, "key6"],
                      [-7.3, 7.8, -0.2, 0.7, "key7"]],
            "black": walls,
            "blue": [[-0.5, 1.5, -6.5, 7.5, "goal"], [-2.5, 3.5, -1.7, 2.7, "goal"], [-4.5, 5.5, -2, 3, "goal"],
                     [-0.5, 1.5, -3.5, 4.5, "goal"], [-6.5, 7.1, -0.2, 0.8, "goal"]],
        }
    raise ValueError(f"unknown chip environment {environment!r}")


@dataclass
class Environment:
    domain: str
    name: str
    tag_and_shape: dict[str, list[list]]
    #: 'prop_<i>' -> [colour, box, function, (A, b)], i counting from 1 in the release's order.
    prop2block: dict[str, list]

    @property
    def action_list(self) -> str:
        """The scene as the release describes it to the LLM."""
        text = "available scene objects: "
        for i in range(len(self.prop2block)):
            colour, block, function, _ = self.prop2block["prop_" + str(i + 1)]
            text += (
                "[name : room" + str(i + 1) + ", color: " + colour + ", function: " + function
                + ", position and size: " + "(" + str(block)[1:-1] + ")" + "], "
            )
        return text[:-2] + "\n" + "available actions: [enter(), not_enter()]"

    def rooms(self) -> dict[str, tuple[str, str, tuple[float, float, float, float]]]:
        """room<i> -> (colour, function, (x_min, x_max, y_min, y_max))."""
        out = {}
        for i in range(len(self.prop2block)):
            colour, block, function, _ = self.prop2block["prop_" + str(i + 1)]
            out[f"room{i + 1}"] = (colour, function, (-block[0], block[1], -block[2], block[3]))
        return out


def regions(domain: str, environment: str = "env2") -> Environment:
    if domain == "HouseWorld":
        tag_and_shape, name = houseworld(), "HouseWorld"
    elif domain == "chip":
        tag_and_shape, name = chip(environment), environment
    else:
        raise ValueError("unknown domain")
    prop2block: dict[str, list] = {}
    index = 0
    for colour, block_array in tag_and_shape.items():
        for block_item_list in block_array:
            block_item = block_item_list[:-1]
            index += 1
            prop2block["prop_" + str(index)] = [
                colour, block_item, block_item_list[-1], (A, np.array(block_item, dtype=np.float64))
            ]
    return Environment(domain, name, tag_and_shape, prop2block)
