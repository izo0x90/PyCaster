from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any

import pygame


class GameEventType(Enum):
    TOGGLE_DISTANCE_CORRECTION = auto()
    TOGGLE_POLAR_TO_CARTESIAN_CORRECTION = auto()
    TOGGLE_TEXTURE_TYPE = auto()
    MOUSE_ACTION = auto()
    SET_HELP_TEXT = auto()


@dataclass
class GameEvent:
    type_: GameEventType
    meta: dict[str, Any] = field(default_factory=dict)


GameEventList = list[GameEvent | pygame.event.Event]
