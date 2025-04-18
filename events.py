from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any

class GameEventType(Enum):
    TOGGLE_DISTANCE_CORRECTION = auto()
    TOGGLE_POLAR_TO_CARTESIAN_CORRECTION = auto()
    TOGGLE_TEXTURE_TYPE = auto()
    MOUSE_CLICK = auto()

@dataclass
class GameEvent:
    type_: GameEventType
    meta: dict[str, Any] = field(default_factory=dict)

