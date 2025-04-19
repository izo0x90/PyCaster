from enum import Enum, auto
from functools import partial
from dataclasses import dataclass, field
import math
import logging
from typing import Callable

import pygame

from events import GameEvent, GameEventType
from raycast import Map, Player, Ray, Intersect, VERY_NEAR_ZERO

logger = logging.getLogger(__name__)

TEST_TEXTURE = [
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
    0,
    255,
]


class BaseView:
    pos: pygame.Vector2
    surface: pygame.Surface

    def localize_mouse(self, x: int, y: int) -> tuple[int | None, int | None]:
        local_region_x_start = self.pos.x
        local_region_x_end = self.pos.x + self.surface.get_width()
        local_region_y_start = self.pos.y
        local_region_y_end = self.pos.y + self.surface.get_height()

        if (
            x >= local_region_x_start
            and x <= local_region_x_end
            and y >= local_region_y_start
            and y <= local_region_y_end
        ):
            local_x = int(x - self.pos.x)
            local_y = int(y - self.pos.y)

            return (local_x, local_y)

        return None, None


@dataclass
class HelpView(BaseView):
    pos: pygame.Vector2
    surface: pygame.Surface
    help_text: list[tuple[int, tuple[Callable | None, str]]] = field(
        default_factory=list
    )

    def __post_init__(self):
        self.update_text()

    def update_text(self):
        font = pygame.font.Font(None, 32)
        self.text = []
        self.help_text.append((103, (None, "TEST")))
        for key, action_hint in self.help_text:
            _, hint = action_hint
            text = f"{pygame.key.name(key)}: {hint}"
            text_obj = font.render(text, True, "green")
            self.text.append(text_obj)

    def clear(self):
        self.surface.fill("chocolate")

    def draw_help_text(self):
        x_offset = 0
        y_offset = 0
        longest_line = 0
        for text_obj in self.text:
            text_rect = text_obj.get_rect()
            longest_line = max(longest_line, text_rect.width)
            text_rect.y = y_offset
            y_offset += text_rect.height
            if y_offset > self.surface.get_height():
                y_offset = 0
                x_offset = longest_line
                longest_line = 0
                text_rect.x = x_offset
                text_rect.y = y_offset

            self.surface.blit(text_obj, text_rect)

    def render(self, tick):
        self.tick_count = tick
        self.clear()
        self.draw_help_text()

    def handle_events(self, events: list[GameEvent]):
        for event in events:
            if isinstance(event, pygame.event.Event):
                pass
            else:
                if event.type_ == GameEventType.SET_HELP_TEXT:
                    self.help_text = event.meta.get("text", "")
                    self.update_text()


class WallShadingType(Enum):
    TEST = auto()
    SOLID = auto()
    TEXTURE = auto()


@dataclass
class FPSView:
    pos: pygame.Vector2
    surface: pygame.Surface
    map: Map
    player: Player
    true_distance = False
    max_brightness_decrese: float = 0.85
    light_loss_multiplier: float = 10
    col_max_scale_h: int = 10000
    wall_shading: WallShadingType = WallShadingType.TEXTURE
    test_texture: list[int] = field(default_factory=partial(list, TEST_TEXTURE))
    tick_count: int = 0

    def __post_init__(self):
        self.wall_height = self.surface.get_height() / self.map.cell_count
        self.surface_col_width = int(
            math.ceil(self.surface.get_width() / self.player.fov_ray_count)
        )
        self.sample_wall_texture = self.map.sample_wall_texture or pygame.Surface(
            (0, 0)
        )
        self.wall_tex_size = pygame.Vector2(
            self.sample_wall_texture.get_width(), self.sample_wall_texture.get_height()
        )
        self.texture_rect = pygame.Rect((0, 0), (0, 0))
        self.col_surface = pygame.Surface(
            (self.surface_col_width, self.col_max_scale_h)
        ).convert(self.sample_wall_texture)
        self.dark = pygame.Surface(
            (self.surface_col_width, self.col_max_scale_h), flags=pygame.SRCALPHA
        )
        self.dark.fill((1, 1, 1))
        self.floor_rect = pygame.Rect(
            (0, self.surface.get_height() / 2),
            (self.surface.get_width(), self.surface.get_height() / 2),
        )
        # self.floor_view = Mode7SubView(self.surface, self.player)

    def draw_col(
        self,
        wall_tex_scalar: float,
        width_scalar: float,
        height_scalar: float,
        color: pygame.Color,
        wall_texture: pygame.Surface | None = None,
    ):
        x_surface = self.surface.get_width() * width_scalar
        height_surface = min(
            self.wall_height / (height_scalar + VERY_NEAR_ZERO), self.col_max_scale_h
        )
        y_surface = (self.surface.get_height() - height_surface) / 2
        distance_squared_light_loss = height_scalar**2 * self.light_loss_multiplier
        if self.wall_shading == WallShadingType.TEXTURE and wall_texture:
            # TODO: Wall texture direction needs to be based on player direction
            texture_w = self.wall_tex_size.x - 1
            tex_x = math.ceil(texture_w * wall_tex_scalar)
            texture_rect = pygame.Rect(
                (tex_x, 0),
                (1, self.wall_tex_size.y),
            )
            self.texture_rect.x = tex_x
            self.texture_rect.y = 0
            self.texture_rect.width = 1
            self.texture_rect.height = int(self.wall_tex_size.y)

            col_texture = self.col_surface.subsurface(
                (0, 0), (self.surface_col_width, height_surface)
            )
            capped_light_loss = int(
                255 * min(self.max_brightness_decrese, distance_squared_light_loss)
            )
            self.dark.set_alpha(capped_light_loss, pygame.RLEACCEL)
            wall_col_tex_source = wall_texture.subsurface(texture_rect)
            pygame.transform.scale(
                wall_col_tex_source,
                (self.surface_col_width, height_surface),
                col_texture,
            )
            self.surface.blit(col_texture, (x_surface, y_surface))
            self.surface.blit(self.dark, (x_surface, y_surface))
        else:
            if self.wall_shading == WallShadingType.TEST:
                tex_x = math.ceil((len(self.test_texture) - 1) * wall_tex_scalar)
                color = pygame.Color(
                    self.test_texture[tex_x],
                    self.test_texture[tex_x],
                    self.test_texture[tex_x],
                )

            self.texture_rect.x = int(x_surface)
            self.texture_rect.y = int(y_surface)
            self.texture_rect.width = int(self.surface_col_width)
            self.texture_rect.height = int(height_surface)

            pygame.draw.rect(
                self.surface,
                color.lerp(
                    0, min(self.max_brightness_decrese, distance_squared_light_loss)
                ),
                self.texture_rect,
            )

    def draw_collision_bound(self, rays=None):
        rays = self.player.fov_rays()
        ray_count = self.player.fov_ray_count
        for ray_idx, player_ray in enumerate(rays):
            intersects, count = self.map.intersect(
                player_ray, self.player.dir, self.tick_count
            )
            for intersect_idx in range(count):
                intersect = intersects[intersect_idx]
                if intersect.collision:
                    if self.true_distance:
                        distance = intersect.distance
                    else:
                        distance = intersect.cos_distance
                    width_scalar = (ray_count - ray_idx) / ray_count
                    wall_tex_scalar = (
                        getattr(intersect.intersect, intersect.intersect_type)
                        % self.map.grid_step
                    ) / self.map.grid_step

                    self.draw_col(
                        wall_tex_scalar,
                        width_scalar,
                        distance,
                        intersect.color,
                        intersect.wall_texture,
                    )
                    break

    def clear(self):
        self.surface.fill("pink")
        pygame.draw.rect(self.surface, "cyan4", self.floor_rect)

    def render(self, tick):
        self.tick_count = tick
        self.clear()
        # self.floor_view.draw_floor_top_down()
        # self.floor_view.draw_floor_fps()
        self.draw_collision_bound()

    def handle_events(self, events: list[GameEvent]):
        for event in events:
            if isinstance(event, pygame.event.Event):
                pass
            else:
                if event.type_ == GameEventType.TOGGLE_DISTANCE_CORRECTION:
                    self.true_distance = not self.true_distance
                elif event.type_ == GameEventType.TOGGLE_TEXTURE_TYPE:
                    self.wall_shading = WallShadingType(
                        (self.wall_shading.value + 1) % (len(WallShadingType)) + 1
                    )


@dataclass
class TopDownDebugView(BaseView):
    pos: pygame.Vector2
    surface: pygame.Surface
    map: Map
    player: Player
    tick_count: int = 0
    selected_cell_meta: int = 1
    wall_pen_down: bool = False

    def handle_mouse_input(self, event: pygame.event.Event):
        x, y = event.pos
        action = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.wall_pen_down = True
                action = 1
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.wall_pen_down = False
                return

            elif event.button == 3:
                action = 3

        elif event.type == pygame.MOUSEMOTION:
            if self.wall_pen_down:
                action = 1

        if action is None:
            return

        local_x, local_y = self.localize_mouse(x, y)
        if local_x is None or local_y is None:
            return

        cell = self.map.get_cell(self.upmap_point(pygame.Vector2(local_x, local_y)))
        if (cell_meta := self.map.map_data.get_cell_meta(cell)) is not None:
            if action == 3:
                logger.info(f"{cell}, {cell_meta}")
            elif action == 1:
                self.map.map_data.set_wall_texture(cell, self.selected_cell_meta)

    def upmap_point(self, cord: pygame.Vector2):
        x = 2 * (cord.x / self.surface.get_width()) - 1
        y = 1 - 2 * (cord.y / self.surface.get_width())
        return pygame.Vector2(x, y)

    def map_point(self, cord: pygame.Vector2):
        half_w = self.surface.get_width() / 2
        half_h = self.surface.get_height() / 2
        x = half_w + half_w * cord.x
        y = half_h + half_h * cord.y * -1
        return pygame.Vector2(int(x), int(y))

    def map_scalar(self, size_x: float = 0, size_y: float = 0):
        # size_y / 2 == mapped_y / surface.get_height
        # (size_y * surface.get_height) / 2 == mapped_y
        return pygame.Vector2(
            (size_x * self.surface.get_width()) / 2,
            (size_y * self.surface.get_height()) / 2,
        )

    def draw_ray(self, ray: Ray, color="black"):
        screen_cord = self.map_point(ray.pos)
        end_cord = ray.pos + ray.mag.rotate(ray.dir)
        end_screen_cord = self.map_point(end_cord)
        pygame.draw.line(self.surface, color, screen_cord, end_screen_cord)
        pygame.draw.circle(self.surface, "blue", screen_cord, self.map_scalar(0.01).x)
        pygame.draw.circle(
            self.surface, "red", end_screen_cord, self.map_scalar(0.01).x
        )

    def map_cell(self, cell_cord: pygame.Vector2):
        screen_grid_step_x = self.surface.get_width() / self.map.cell_count
        screen_grid_step_y = self.surface.get_height() / self.map.cell_count
        x = int(screen_grid_step_x * cell_cord.x)
        y = int(screen_grid_step_y * cell_cord.y)
        return pygame.Rect((int(x), int(y)), (screen_grid_step_x, screen_grid_step_y))

    def draw_intersect(
        self, ray: Ray, intersects: list[Intersect], count: int, marked_only=False
    ):
        for intersect_idx in range(count):
            intersect = intersects[intersect_idx]
            color = "black"
            if not intersect.marked:
                if marked_only:
                    continue
                color = "red"

            # Draw step x
            pygame.draw.line(
                self.surface,
                "blue",
                self.map_point(intersect.origin),
                self.map_point(
                    pygame.Vector2(intersect.vert_intersect.x, intersect.origin.y)
                ),
                math.ceil(self.map_scalar(0.008).x),
            )

            # Draw step y
            pygame.draw.line(
                self.surface,
                "red",
                self.map_point(intersect.origin),
                self.map_point(
                    pygame.Vector2(intersect.origin.x, intersect.horiz_intersect.y)
                ),
                math.ceil(self.map_scalar(0.008).x),
            )

            # Draw target cell
            pygame.draw.rect(
                self.surface,
                "yellow",
                self.map_cell(self.map.get_cell(intersect.origin)),
                math.ceil(self.map_scalar(0.008).x),
            )

            self.draw_ray(ray, color)

            # Vert intersect
            pygame.draw.circle(
                self.surface,
                "red",
                self.map_point(intersect.vert_intersect),
                math.ceil(self.map_scalar(0.008).x),
            )

            # Horiz intersect
            pygame.draw.circle(
                self.surface,
                "blue",
                self.map_point(intersect.horiz_intersect),
                math.ceil(self.map_scalar(0.008).x),
            )

            # Intersect
            pygame.draw.circle(
                self.surface,
                "black",
                self.map_point(intersect.intersect),
                math.ceil(self.map_scalar(0.012).x),
            )

            # Draw target point
            pygame.draw.circle(
                self.surface,
                "green",
                self.map_point(intersect.origin),
                math.ceil(self.map_scalar(0.008).x),
            )

    def draw_grid(self):
        cur_pos = -1

        # Draw axis origin
        pygame.draw.circle(
            self.surface,
            "yellow",
            self.map_point(pygame.Vector2(0, 0)),
            self.map_scalar(0.0125).x,
        )

        # Draw map cell grid
        for _ in range(self.map.cell_count):
            pygame.draw.line(
                self.surface,
                "blue",
                self.map_point(pygame.Vector2(-1, cur_pos)),
                self.map_point(pygame.Vector2(1, cur_pos)),
            )
            pygame.draw.line(
                self.surface,
                "red",
                self.map_point(pygame.Vector2(cur_pos, -1)),
                self.map_point(pygame.Vector2(cur_pos, 1)),
            )
            cur_pos += self.map.grid_step

    def draw_single_ray_cast(self):
        intersects, count = self.map.intersect(
            self.player.get_entity_ray(), self.player.dir, self.tick_count
        )
        self.draw_intersect(self.player.get_entity_ray(), intersects, count)

    def draw_ray_cast_fov(self, marked_only=False):
        for player_ray in self.player.fov_rays():
            intersects, count = self.map.intersect(
                player_ray, self.player.dir, self.tick_count
            )
            self.draw_intersect(player_ray, intersects, count, marked_only)

    def draw_collision_bound(self):
        first_collision_per_ray = []
        for player_ray in self.player.fov_rays():
            intersects, count = self.map.intersect(
                player_ray, self.player.dir, self.tick_count
            )
            for intersect_idx in range(count):
                intersect = intersects[intersect_idx]
                if intersect.collision:
                    first_collision_per_ray.append(intersect)

                    pygame.draw.circle(
                        self.surface,
                        "cyan",
                        self.map_point(intersect.intersect),
                        math.ceil(self.map_scalar(0.005).x),
                    )
                    break

    def draw_player(self):
        self.draw_ray(self.player.get_entity_ray())
        self.draw_ray(self.player.get_entity_speed_ray())

    def draw_cells(self):
        for row_idx, row in enumerate(self.map.map_data.cell_data):
            for col_idx, col in enumerate(row):
                cel_pos = pygame.Vector2(col_idx, row_idx)
                if self.map.map_data.cell_metas[col].solid:
                    pygame.draw.rect(
                        self.surface,
                        self.map.map_data.cell_metas[col].color,
                        self.map_cell(cel_pos),
                    )

    def draw_next_step(self):
        self.player.calc_next_pos()
        bound_box = self.player.get_bound_box()
        bound_box_next_step = self.player.get_bound_box(next_pos=True)
        bound_box_rect = pygame.Rect(
            self.map_point(bound_box.top_left),
            self.map_scalar(
                abs(bound_box.bottom_rigth.x - bound_box.top_left.x),
                abs(bound_box.bottom_rigth.y - bound_box.top_left.y),
            ),
        )

        pygame.draw.rect(
            self.surface, "green", bound_box_rect, math.ceil(self.map_scalar(0.005).x)
        )

        next_step_rect = pygame.Rect(
            self.map_point(bound_box_next_step.top_left),
            self.map_scalar(
                abs(
                    bound_box_next_step.bottom_rigth.x - bound_box_next_step.top_left.x
                ),
                abs(
                    bound_box_next_step.bottom_rigth.y - bound_box_next_step.top_left.y
                ),
            ),
        )

        pygame.draw.rect(
            self.surface, "green", next_step_rect, math.ceil(self.map_scalar(0.005).x)
        )

    def clear(self):
        self.surface.fill("purple")

    def render(self, tick):
        self.tick_count = tick
        self.clear()
        self.draw_grid()
        self.draw_cells()
        # self.draw_single_ray_cast()
        self.draw_ray_cast_fov(marked_only=True)
        # self.draw_ray_cast_fov()
        self.draw_collision_bound()
        self.draw_player()
        self.draw_next_step()

    def handle_events(self, events: list[GameEvent]):
        for event in events:
            if isinstance(event, pygame.event.Event):
                if event.type in [
                    pygame.MOUSEMOTION,
                    pygame.MOUSEBUTTONUP,
                    pygame.MOUSEBUTTONDOWN,
                ]:
                    self.handle_mouse_input(event)
