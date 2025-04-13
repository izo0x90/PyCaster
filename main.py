"""
This code it is not structured in a way an actual game that is concerned with performance should be structured.

Instead the aim to separate things conceptually where maybe they can be understood individually when reading through
the code and also visualized on screen with the different "views".

If it accomplished this or not, **shrugs**, but certainly don't structure actual games like this even in Python.
"""

from enum import Enum, auto
from functools import partial
from dataclasses import dataclass, field
import json
import math
from typing import Any, Protocol

import pygame

MAX_STEPS_PER_CAST = 100
FULL_CIRCLE_DEGREES = 360
VERY_NEAR_ZERO = (
    0.0000000000000000000000000000000000000000000000000000000000000000000001
)


@dataclass
class Ray:
    pos: pygame.Vector2
    mag: pygame.Vector2
    dir: float


@dataclass
class Intersect:
    origin: pygame.Vector2
    vert_intersect: pygame.Vector2
    horiz_intersect: pygame.Vector2
    intersect: pygame.Vector2
    distance: float
    cos_distance: float
    collision: bool
    color: pygame.Color
    marked: bool


@dataclass
class BoundBox:
    top_left: pygame.Vector2
    bottom_rigth: pygame.Vector2


class GameEventType(Enum):
    TOGGLE_DISTANCE_CORRECTION = auto()
    TOGGLE_POLAR_TO_CARTESIAN_CORRECTION = auto()


@dataclass
class GameEvent:
    type_: GameEventType
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DirectionalEntity:
    pos: pygame.Vector2
    ray_cast_depth: pygame.Vector2
    dir: float
    speed: pygame.Vector2
    rotation_speed: float
    radius: float
    next_pos: pygame.Vector2 | None = None

    def get_entity_ray(self):
        return Ray(self.pos, self.ray_cast_depth, self.dir)

    def get_entity_speed_ray(self):
        return Ray(self.pos, self.speed, self.dir)

    def get_bound_box(self, next_pos=False):
        if next_pos and self.next_pos:
            pos = self.next_pos
        else:
            pos = self.pos

        return BoundBox(
            pygame.Vector2(pos.x - self.radius, pos.y + self.radius),
            pygame.Vector2(pos.x + self.radius, pos.y - self.radius),
        )

    def calc_next_pos(self, forward=True) -> pygame.Vector2:
        if forward:
            self.next_pos = self.pos + self.speed.rotate(self.dir)
        else:
            self.next_pos = self.pos - self.speed.rotate(self.dir)

        return self.next_pos

    def update_pos(self, suggested_position: pygame.Vector2 | None = None):
        if suggested_position is not None:
            self.pos = suggested_position
            self.next_pos = None
        elif self.next_pos:
            self.pos = self.next_pos
            self.next_pos = None

    def constrain_angle(self, angle: float) -> float:
        return (FULL_CIRCLE_DEGREES + angle) % FULL_CIRCLE_DEGREES

    def update_dir(self, clock_wise=True):
        # Module operator "%" keeps angle in range 0, 360
        if clock_wise:
            # Negative angles will be converted to their positive equivalent by adding to 360
            self.dir = self.constrain_angle((self.dir - self.rotation_speed))
        else:
            self.dir = self.constrain_angle(self.dir + self.rotation_speed)


@dataclass
class Player(DirectionalEntity):
    fov: int = 60
    fov_ray_count: int = 320
    half_fov: float = 0
    half_ray_count: float = 0
    ang_step: float = 0
    rays: list[Ray] = field(default_factory=list)
    focal_len: float = 0.0
    correct_polar_to_cart: bool = True

    def __post_init__(self):
        self.half_fov = self.fov / 2
        self.ang_step = self.fov / self.fov_ray_count
        for _ in range(self.fov_ray_count):
            self.rays.append(Ray(self.pos, self.ray_cast_depth, self.dir))

        self.half_ray_count = self.fov_ray_count / 2
        self.focal_len = self.half_ray_count / math.tan(math.radians(self.half_fov))

    def convert_to_projected(self, ray_index):
        screen_max = self.fov_ray_count - 1  # Avoid "fence post" error

        # Where does ray index fall on plane ranged from -half_ray_count to +half_ray_count
        projected_x = (
            ((ray_index * 2) - screen_max) / screen_max
        ) * self.half_ray_count

        # Get "correct" angle of ray as it would pass through middle of cell in projection plane
        return math.degrees(math.atan2(projected_x, self.focal_len))

    def fov_rays(self) -> list[Ray]:
        cur_ang = self.dir - self.half_fov
        for idx, ray in enumerate(self.rays):
            ray.dir = self.constrain_angle(cur_ang)
            if self.correct_polar_to_cart:
                ray.dir = self.constrain_angle(
                    self.convert_to_projected(idx) + self.dir
                )
            ray.pos = self.pos
            cur_ang += self.ang_step

        return self.rays

    def handle_events(self, events: list[GameEvent]):
        for event in events:
            if event.type_ == GameEventType.TOGGLE_POLAR_TO_CARTESIAN_CORRECTION:
                self.correct_polar_to_cart = not self.correct_polar_to_cart


@dataclass
class CellMeta:
    solid: bool = True
    color: str = "orange"


@dataclass
class MapData:
    cell_data: list[list[int]]
    cell_metas: dict[int, CellMeta]
    default_color: pygame.Color = field(default_factory=partial(pygame.Color, "black"))

    def get_color(self, cell_cord: pygame.Vector2) -> pygame.Color:
        cell_y = int(cell_cord.y)
        cell_x = int(cell_cord.x)

        if (cell_y < 0 or cell_y >= len(self.cell_data)) or (
            cell_x < 0 or cell_x >= len(self.cell_data[cell_y])
        ):
            return self.default_color

        return pygame.Color(self.cell_metas[self.cell_data[cell_y][cell_x]].color)

    def is_solid_cell(self, cell_cord: pygame.Vector2) -> bool:
        cell_y = int(cell_cord.y)
        cell_x = int(cell_cord.x)

        if (cell_y < 0 or cell_y >= len(self.cell_data)) or (
            cell_x < 0 or cell_x >= len(self.cell_data[cell_y])
        ):
            return False

        return self.cell_metas[self.cell_data[cell_y][cell_x]].solid


@dataclass
class Map:
    map_data: MapData
    cell_count: int = 0
    intersect_padding = 0.001

    def __post_init__(self) -> None:
        self.cell_count = len(self.map_data.cell_data[0])

    @property
    def grid_step(self):
        return 2 / self.cell_count

    def get_cell(self, cord: pygame.Vector2):
        norm_x = (1 + cord.x) / 2
        norm_y = (1 - cord.y) / 2
        cell_x = math.floor(self.cell_count * norm_x)
        cell_y = math.floor(self.cell_count * norm_y)
        return pygame.Vector2(cell_x, cell_y)

    def cell_origin(self, cell_cord: pygame.Vector2):
        x = 2 * (cell_cord.x / self.cell_count) - 1
        y = 1 - 2 * (cell_cord.y / self.cell_count)
        return pygame.Vector2(x, y)

    def will_collide(
        self,
        entity_bound_box: BoundBox,
        entity_pos: pygame.Vector2,
        entity_radius: float,
        entity_next_pos: pygame.Vector2,
    ) -> tuple[bool, pygame.Vector2 | None]:
        start = self.get_cell(entity_bound_box.top_left)
        end = self.get_cell(entity_bound_box.bottom_rigth)

        has_collision = False
        entity_cell = self.get_cell(entity_pos)
        limit_cell_y = None
        dir_up = None
        limit_cell_x = None
        dir_right = None

        for row in range(
            int(start.y), int(min(end.y + 1, len(self.map_data.cell_data)))
        ):
            for col in range(
                int(start.x), int(min(end.x + 1, len(self.map_data.cell_data[0])))
            ):
                if self.map_data.is_solid_cell(pygame.Vector2(col, row)):
                    has_collision = True

                    if entity_cell.x == col:
                        if entity_cell.y < row:
                            if not limit_cell_y or limit_cell_y > row:
                                dir_up = False
                                limit_cell_y = row
                        elif entity_cell.y > row:
                            if not limit_cell_y or limit_cell_y < row:
                                dir_up = True
                                limit_cell_y = row

                    if entity_cell.y == row:
                        if entity_cell.x < col:
                            if not limit_cell_x or limit_cell_x > col:
                                dir_right = True
                                limit_cell_x = col
                        elif entity_cell.x > col:
                            if not limit_cell_x or limit_cell_x < col:
                                dir_right = False
                                limit_cell_x = col

        limit_x = limit_y = None
        if limit_cell_y is not None:
            direction_cell_offset = 0
            direction_sign = 1
            if dir_up:
                direction_cell_offset = 1
                direction_sign = -1

            limit_y = self.cell_origin(
                pygame.Vector2(0, limit_cell_y + direction_cell_offset)
            ).y
            limit_y = limit_y + (direction_sign * entity_radius)

        if limit_cell_x is not None:
            direction_cell_offset = 0
            direction_sign = 1
            if not dir_right:
                direction_cell_offset = 1
                direction_sign = -1

            limit_x = self.cell_origin(
                pygame.Vector2(limit_cell_x + direction_cell_offset, 0)
            ).x
            limit_x = limit_x - (direction_sign * entity_radius)

        suggested_position = None
        if has_collision:
            if limit_y is None:
                limit_y = entity_next_pos.y

            if limit_x is None:
                limit_x = entity_next_pos.x

            suggested_position = pygame.Vector2(limit_x, limit_y)

        return has_collision, suggested_position

    def intersect(self, ray: Ray, reference_angle: float):
        # This not an optimal implementation of a ray-caster, many objects are created "on the fly"
        # to just be disposed a short time later, redundant calculations exist, this function is
        # invoked for each ray etc. etc.
        intersects = []
        next_cast_origin = ray.pos
        end_point = ray.pos + ray.mag.rotate(ray.dir)
        max_cast_distance = math.sqrt(
            (ray.pos.x - end_point.x) ** 2 + (ray.pos.y - end_point.y) ** 2
        )
        angle = ray.dir if ray.dir != 0 else VERY_NEAR_ZERO
        angle_true_distance_to_adj_distance = abs(reference_angle - angle)

        traversed_cast_distance = 0
        # Travel along the cast ray and find where it intersects with horizontal or vertical gird lines
        for _ in range(MAX_STEPS_PER_CAST):
            cell = self.get_cell(next_cast_origin)
            cell_origin = self.cell_origin(cell)

            # Vert intersect x component
            # The players direction in respect to the gird changes if addition or
            # subtraction is needed to calculate intersect position
            if (angle >= 0 and angle <= 90) or (
                angle >= 270 and angle <= FULL_CIRCLE_DEGREES
            ):
                step_x = self.grid_step - (next_cast_origin.x - cell_origin.x)
                vert_intersect_x_comp = next_cast_origin.x + step_x
                next_cell_x = cell_origin.x + self.grid_step + self.intersect_padding
            else:
                step_x = next_cast_origin.x - cell_origin.x
                vert_intersect_x_comp = next_cast_origin.x - step_x
                next_cell_x = cell_origin.x - self.intersect_padding

            # Vert intersect
            # adj = step_x
            # tan(angle) = opposite / adj
            # opposite = tan(angle) * adj
            # delta_y = opposite
            delta_y = abs(math.tan(math.radians(angle)) * step_x)
            if angle >= 0 and angle <= 180:
                vert_intersect_y_comp = next_cast_origin.y + delta_y
            else:
                vert_intersect_y_comp = next_cast_origin.y - delta_y

            vert_intersect = pygame.Vector2(
                vert_intersect_x_comp, vert_intersect_y_comp
            )

            # Horiz intersect y component
            # Same as for the vertical intersect, players direction in respect to the gird changes if addition or
            # subtraction is needed to calculate horizontal intersect position
            if angle >= 0 and angle <= 180:
                step_y = cell_origin.y - next_cast_origin.y
                horiz_intersect_y_comp = next_cast_origin.y + step_y
                next_cell_y = cell_origin.y + self.intersect_padding
            else:
                step_y = self.grid_step - (cell_origin.y - next_cast_origin.y)
                horiz_intersect_y_comp = next_cast_origin.y - step_y
                next_cell_y = cell_origin.y - self.grid_step - self.intersect_padding

            # Horiz intersect
            # opposite = step_y
            # tan(angle) = opposite / adj
            # adj = opposite / tan(angle)
            # delta_x = adj
            delta_x = abs(step_y / math.tan(math.radians(angle)))
            if (angle >= 0 and angle <= 90) or (
                angle >= 270 and angle <= FULL_CIRCLE_DEGREES
            ):
                horiz_intersect_x_comp = next_cast_origin.x + delta_x
            else:
                horiz_intersect_x_comp = next_cast_origin.x - delta_x

            horiz_intersect = pygame.Vector2(
                horiz_intersect_x_comp, horiz_intersect_y_comp
            )

            # Calculate distance (distance squared, more on why below) from the previous intersect to both the
            # horizontal and vertical intersects in this step.
            vertical_intersect_dist_squared = (
                next_cast_origin.x - vert_intersect.x
            ) ** 2 + (next_cast_origin.y - vert_intersect.y) ** 2

            horiz_intersect_dist_squared = (
                next_cast_origin.x - horiz_intersect.x
            ) ** 2 + (next_cast_origin.y - horiz_intersect.y) ** 2

            # The closer intersect is the intersect that the ray would encounter first as it projects out
            # so it is the one we want for the current step
            # Is the vertical or horizontal intersect at this step closer in distance squared
            # because we don't need the actual distance just which one is relative to the other closer
            # we can avoid having to take the square root here and save on computation
            if horiz_intersect_dist_squared < vertical_intersect_dist_squared:
                closest_intersect = pygame.Vector2(horiz_intersect.x, next_cell_y)
                intersect_distance_squared = horiz_intersect_dist_squared
            else:
                closest_intersect = pygame.Vector2(next_cell_x, vert_intersect.y)
                intersect_distance_squared = vertical_intersect_dist_squared

            # Capturing all intersects along the ray so that we can visualize them later
            # however it we could optimize and only capture the first intersect, only having to take
            # one square root per projected ray and only taking as many steps along the way as it takes
            # it to encounter an obstacle in "best cases"
            # Capturing more than the first intersect also allows for effects like transparency etc. in
            # "fancier" ray-casters
            traversed_cast_distance += math.sqrt(intersect_distance_squared)

            # TODO: Remove distance correctness verification
            # inter_d = math.sqrt((ray.pos.x - closest_intersect.x) ** 2 + (ray.pos.y - closest_intersect.y) ** 2)

            marked = False
            # Stop traversing down the ray once its length has exceeded the viewing distance
            # that has been defined for the player/ entity "casting the rays"
            if traversed_cast_distance <= max_cast_distance:
                # If the traversed distance to an intersect is used to "calculate the height" of parts of
                # obstacle that the rays intersect, parts of the object to the left and right of the players
                # view center line would show up as being further away from the player
                # Because those rays are "more diagonal" to the players view, in real life things like eyes
                # and camera lenses are curved and they account for that
                # Here some trig. is used to calculate the straight line distance from the player to the intersect
                #
                # hypotenuse = traversed_cast_distance
                # angle = abs(angle_of_player - angle_of_ray)
                # cos(angle) = adj / hypotenuse
                # adj = cos(angle) * hypotenuse
                # adj = cos_distance = corrected_distance_to_intersect
                cos_distance = (
                    math.cos(math.radians(angle_true_distance_to_adj_distance))
                    * traversed_cast_distance
                )

                # The Map has all the "data" for the cells that build the grid "world"
                # so it is extracted here and sent down along with the rest of the intersect data
                cell = self.get_cell(closest_intersect)
                has_collided = self.map_data.is_solid_cell(cell)
                color = self.map_data.get_color(cell)

                if has_collided:
                    marked = True

                intersects.append(
                    Intersect(
                        next_cast_origin,
                        vert_intersect,
                        horiz_intersect,
                        closest_intersect,
                        traversed_cast_distance,
                        cos_distance,
                        has_collided,
                        color,
                        marked,
                    )
                )
            else:
                break

            next_cast_origin = closest_intersect

        return intersects


class MapLoader:
    DEFAULT_LEVEL = [
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 2, 2, 0, 0, 0],
        [0, 0, 1, 0, 0, 2, 2, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 3, 3, 3, 3, 3, 3, 0, 0],
    ]

    DEFAULT_LEVEL_CELL_METAS = {
        0: CellMeta(solid=False),
        1: CellMeta(),
        2: CellMeta(color="blue"),
        3: CellMeta(color="cyan"),
    }

    @classmethod
    def generate_test_level(cls, size_in_cells: int) -> Map:
        data = []
        col_fill = [0] * (size_in_cells - len(cls.DEFAULT_LEVEL[0]))
        for row in cls.DEFAULT_LEVEL:
            data.append(row + col_fill)

        data.extend([[0] * size_in_cells] * (size_in_cells - len(cls.DEFAULT_LEVEL)))

        return Map(MapData(data, cls.DEFAULT_LEVEL_CELL_METAS))

    @classmethod
    def from_file(cls, file_name: str) -> Map:
        with open(file_name, "r") as f:
            serialized_data = f.read()

        map_data = json.loads(serialized_data)

        return Map(MapData(map_data, cls.DEFAULT_LEVEL_CELL_METAS))


class View(Protocol):
    pos: pygame.Vector2
    surface: pygame.Surface

    def clear(self): ...

    def render(self): ...

    def handle_events(self, events: list[GameEvent]): ...


@dataclass
class FPSView:
    pos: pygame.Vector2
    surface: pygame.Surface
    map: Map
    player: Player
    true_distance = False
    max_brightness_decrese: float = 0.85
    light_loss_multiplier: float = 4

    def draw_col(
        self, ray_index: int, ray_count: int, height_scalar: float, color: pygame.Color
    ):
        x_surface = self.surface.get_width() * (ray_index / ray_count)
        height_surface = (
            self.surface.get_height() / self.map.cell_count
        ) / height_scalar
        width_surface = self.surface.get_width() / ray_count
        y_surface = (self.surface.get_height() - height_surface) / 2
        distance_squared_light_loss = height_scalar**2 * self.light_loss_multiplier
        pygame.draw.rect(
            self.surface,
            color.lerp(
                0, min(self.max_brightness_decrese, distance_squared_light_loss)
            ),
            pygame.Rect(
                (x_surface, y_surface),
                (width_surface, height_surface),
            ),
        )

    def draw_collision_bound(self, rays=None):
        rays = self.player.fov_rays()
        ray_count = len(rays)
        for ray_idx, player_ray in enumerate(rays):
            intersects = self.map.intersect(player_ray, self.player.dir)
            for intersect in intersects:
                if intersect.collision:
                    if self.true_distance:
                        distance = intersect.distance
                    else:
                        distance = intersect.cos_distance

                    self.draw_col(
                        (ray_count - ray_idx), ray_count, distance, intersect.color
                    )
                    break

    def clear(self):
        self.surface.fill("pink")

    def render(self):
        self.clear()
        self.draw_collision_bound()

    def handle_events(self, events: list[GameEvent]):
        for event in events:
            if event.type_ == GameEventType.TOGGLE_DISTANCE_CORRECTION:
                self.true_distance = not self.true_distance


@dataclass
class TopDownDebugView:
    pos: pygame.Vector2
    surface: pygame.Surface
    map: Map
    player: Player

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
        screen_grid_step_x = int(self.surface.get_width() / self.map.cell_count)
        screen_grid_step_y = int(self.surface.get_height() / self.map.cell_count)
        x = screen_grid_step_x * cell_cord.x
        y = screen_grid_step_y * cell_cord.y
        return pygame.Rect((int(x), int(y)), (screen_grid_step_x, screen_grid_step_y))

    def draw_intersect(self, ray: Ray, intersects: list[Intersect], marked_only=False):
        for intersect in intersects:
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
        intersects = self.map.intersect(self.player.get_entity_ray(), self.player.dir)
        self.draw_intersect(self.player.get_entity_ray(), intersects)

    def draw_ray_cast_fov(self, marked_only=False):
        for player_ray in self.player.fov_rays():
            intersects = self.map.intersect(player_ray, self.player.dir)
            self.draw_intersect(player_ray, intersects, marked_only)

    def draw_collision_bound(self):
        first_collision_per_ray = []
        for player_ray in self.player.fov_rays():
            intersects = self.map.intersect(player_ray, self.player.dir)
            for intersect in intersects:
                if intersect.collision:
                    first_collision_per_ray.append(intersect)
                    break

        mapped_points = []
        for collision_intersect in first_collision_per_ray:
            mapped_points.append(self.map_point(collision_intersect.intersect))
            pygame.draw.circle(
                self.surface,
                "cyan",
                mapped_points[-1],
                math.ceil(self.map_scalar(0.005).x),
            )

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

    def render(self):
        self.clear()
        self.draw_grid()
        self.draw_cells()
        # self.draw_single_ray_cast()
        # self.draw_ray_cast_fov(marked_only=True)
        self.draw_collision_bound()
        self.draw_player()
        self.draw_next_step()

    def handle_events(self, events: list[GameEvent]):
        pass


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
    game_events_for_tick: list[GameEvent] = field(default_factory=list)

    def __post_init__(self):
        self.key_map = {
            pygame.K_q: self.action_quit,
            pygame.K_6: self.action_toggle_distance_correction,
            pygame.K_7: self.action_toggle_porlar_to_cartesian_correction,
        }

        self.key_map_repeat = {
            pygame.K_w: self.action_player_move,
            pygame.K_s: partial(self.action_player_move, False),
            pygame.K_d: self.action_player_rotate,
            pygame.K_a: partial(self.action_player_rotate, False),
        }

    def util_clear_events(self):
        self.game_events_for_tick.clear()

    def action_toggle_distance_correction(self):
        self.game_events_for_tick.append(
            GameEvent(GameEventType.TOGGLE_DISTANCE_CORRECTION)
        )

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
        for key_code, action in self.key_map_repeat.items():
            if keys[key_code]:
                action()

    def handle_input(self, key_code: int):
        if action := self.key_map.get(key_code, None):
            action()

    def dispatch_events(self):
        # Dispatch to views
        for view in self.views:
            view.handle_events(self.game_events_for_tick)

        self.focused_player.handle_events(self.game_events_for_tick)

        self.util_clear_events()

    def render(self):
        for view in self.views:
            view.render()
            self.main_view.blit(view.surface, view.pos)


TOP_DOWN_VIEW_SIZE = 800
FPS_VIEW_SIZE = 640
MARGIN = 10
RESOLUTION = 1


def main():
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode(
        (
            TOP_DOWN_VIEW_SIZE + FPS_VIEW_SIZE + MARGIN,
            max(TOP_DOWN_VIEW_SIZE, FPS_VIEW_SIZE),
        )
    )
    clock = pygame.time.Clock()
    # map = MapLoader.generate_test_level(25)
    map = MapLoader.from_file("level_1.json")
    player_size_from_cell_size = map.grid_step / 8
    speed = player_size_from_cell_size / 5
    ray_cast_depth = map.grid_step * 8
    player = Player(
        pos=pygame.Vector2(-0.37, -0.245),
        ray_cast_depth=pygame.Vector2(ray_cast_depth, 0),
        dir=145,
        speed=pygame.Vector2(speed, 0),
        rotation_speed=2,
        fov=90,
        fov_ray_count=math.ceil(FPS_VIEW_SIZE * min(1, RESOLUTION)),
        radius=player_size_from_cell_size,
    )
    top_down_view = TopDownDebugView(
        pygame.Vector2(0, 0),
        pygame.Surface((TOP_DOWN_VIEW_SIZE, TOP_DOWN_VIEW_SIZE)),
        map,
        player,
    )

    fps_view = FPSView(
        pygame.Vector2(TOP_DOWN_VIEW_SIZE + MARGIN, 0),
        pygame.Surface((FPS_VIEW_SIZE, FPS_VIEW_SIZE)),
        map,
        player,
    )

    game = Game(player, map, screen, [top_down_view, fps_view])
    while not game.is_quitting:
        # TODO: Move remaining input in to Game
        keys = pygame.key.get_pressed()
        if keys[pygame.K_0]:
            player.dir = 0

        if keys[pygame.K_1]:
            player.dir = 90

        if keys[pygame.K_2]:
            player.dir = 180

        if keys[pygame.K_3]:
            player.dir = 270

        if keys[pygame.K_4]:
            player.dir = FULL_CIRCLE_DEGREES

        if keys[pygame.K_5]:
            player.dir = VERY_NEAR_ZERO

        game.handle_repeat_input(keys)

        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.action_quit()
            elif event.type == pygame.KEYDOWN:
                game.handle_input(event.key)

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("green")

        # RENDER YOUR GAME HERE
        game.dispatch_events()
        game.render()

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    pygame.quit()


if __name__ == "__main__":
    main()
