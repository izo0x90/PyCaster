from enum import Enum, auto
from functools import partial
from dataclasses import dataclass, field
from typing import Protocol

import pygame

from events import GameEvent, GameEventType, GameEventList
from raycast import Map, Player


class View(Protocol):
    pos: pygame.Vector2
    surface: pygame.Surface

    def clear(self): ...

    def render(self, tick: int): ...

    def handle_events(self, events: GameEventList): ...


class GameState(Enum):
    RUNNING = auto()
    PAUSED = auto()
    QUITING = auto()


@dataclass
class Game:
    focused_player: Player
    map: Map
    main_view: pygame.Surface
    views: list[View]
    state: GameState = GameState.RUNNING
    key_map: dict = field(default_factory=dict)
    key_map_repeat: dict = field(default_factory=dict)
    game_events_for_tick: GameEventList = field(default_factory=list)
    tick_count: int = 0

    def __post_init__(self):
        self.key_map = {
            pygame.K_q: (self.action_quit, "Quit"),
            pygame.K_6: (
                self.action_toggle_distance_correction,
                "Toggle distance to projection plane correction",
            ),
            pygame.K_7: (
                self.action_toggle_porlar_to_cartesian_correction,
                "Toggle polar to cartesion correction",
            ),
            pygame.K_8: (self.action_toggle_wall_texture_type, "Rotate texture type"),
        }

        self.key_map_repeat = {
            pygame.K_w: (self.action_player_move, "Move player forward"),
            pygame.K_s: (
                partial(self.action_player_move, False),
                "Move player backward",
            ),
            pygame.K_d: (self.action_player_rotate, "Rotate player clockwise"),
            pygame.K_a: (
                partial(self.action_player_rotate, False),
                "Rotate player counter clockwise",
            ),
        }

        help_text = list(self.key_map.items())
        help_text += list(self.key_map_repeat.items())

        self.game_events_for_tick.append(
            GameEvent(GameEventType.SET_HELP_TEXT, {"text": help_text})
        )

    def util_clear_events(self):
        self.game_events_for_tick.clear()

    def action_toggle_distance_correction(self):
        self.game_events_for_tick.append(
            GameEvent(GameEventType.TOGGLE_DISTANCE_CORRECTION)
        )

    def action_toggle_wall_texture_type(self):
        self.game_events_for_tick.append(GameEvent(GameEventType.TOGGLE_TEXTURE_TYPE))

    def action_toggle_porlar_to_cartesian_correction(self):
        self.game_events_for_tick.append(
            GameEvent(GameEventType.TOGGLE_POLAR_TO_CARTESIAN_CORRECTION)
        )

    def action_player_move(self, forward=True):
        self.focused_player.calc_next_pos(forward=forward)
        next_bound_box = self.focused_player.get_bound_box(next_pos=True)
        has_collision, suggested_position = self.map.will_collide(
            next_bound_box,
            self.focused_player.pos,
            self.focused_player.radius,
            self.focused_player.next_pos,
        )

        if not has_collision:
            self.focused_player.update_pos()
        elif suggested_position is not None:
            self.focused_player.update_pos(suggested_position)

    def action_player_rotate(self, clock_wise=True):
        self.focused_player.update_dir(clock_wise)

    @property
    def is_quitting(self):
        return self.state == GameState.QUITING

    def action_quit(self):
        self.state = GameState.QUITING

    def handle_repeat_input(self, keys):
        for key_code, action_hint in self.key_map_repeat.items():
            if keys[key_code]:
                action, _ = action_hint
                action()

    def handle_input(self, key_code: int):
        if action_help := self.key_map.get(key_code, None):
            action, _ = action_help
            action()

    def handle_mouse_input(self, event: pygame.event.Event):
        self.game_events_for_tick.append(event)

    def dispatch_events(self):
        # Dispatch to views
        for view in self.views:
            view.handle_events(self.game_events_for_tick)

        self.focused_player.handle_events(self.game_events_for_tick)

        self.util_clear_events()

    def render(self):
        for view in self.views:
            view.render(tick=self.tick_count)
            self.main_view.blit(view.surface, view.pos)

    def tick(self):
        self.dispatch_events()
        self.render()
        self.tick_count += 1
