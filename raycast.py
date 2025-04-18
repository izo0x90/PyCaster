from functools import partial
from dataclasses import dataclass, field
import json
import math

import pygame

from events import GameEvent, GameEventType

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
    intersect_type: str
    wall_texture: pygame.Surface | None


@dataclass
class BoundBox:
    top_left: pygame.Vector2
    bottom_rigth: pygame.Vector2


@dataclass
class DirectionalEntity:
    pos: pygame.Vector2
    ray_cast_depth: pygame.Vector2
    dir: float
    speed: pygame.Vector2
    rotation_speed: float
    radius: float
    next_pos: pygame.Vector2 | None = None

    def __post_init__(self):
        self.entity_ray = Ray(self.pos, self.ray_cast_depth, self.dir)
        self.entity_speed_ray =  Ray(self.pos, self.speed, self.dir)
        self.bound_box = BoundBox(
            pygame.Vector2(self.pos.x - self.radius, self.pos.y + self.radius),
            pygame.Vector2(self.pos.x + self.radius, self.pos.y - self.radius),
        )

    def get_entity_ray(self):
        self.entity_ray.pos = self.pos
        self.entity_ray.mag = self.ray_cast_depth
        self.entity_ray.dir = self.dir
        return self.entity_ray

    def get_entity_speed_ray(self):
        self.entity_speed_ray.pos = self.pos
        self.entity_speed_ray.mag = self.ray_cast_depth
        self.entity_speed_ray.dir = self.dir
        return self.entity_speed_ray

    def get_bound_box(self, next_pos=False):
        if next_pos and self.next_pos:
            pos = self.next_pos
        else:
            pos = self.pos

        self.bound_box.top_left.x = pos.x - self.radius
        self.bound_box.top_left.y = pos.y + self.radius
        self.bound_box.bottom_rigth.x = pos.x + self.radius
        self.bound_box.bottom_rigth.y = pos.y - self.radius

        return  self.bound_box

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
        super().__post_init__()
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
    wall_texture: pygame.Surface | None = None


@dataclass
class MapData:
    cell_data: list[list[int]]
    cell_metas: dict[int, CellMeta]
    default_color: pygame.Color = field(default_factory=partial(pygame.Color, "black"))

    def get_cell_meta(self, cell_cord: pygame.Vector2) -> CellMeta | None:
        cell_y = int(cell_cord.y)
        cell_x = int(cell_cord.x)

        if (cell_y < 0 or cell_y >= len(self.cell_data)) or (
            cell_x < 0 or cell_x >= len(self.cell_data[cell_y])
        ):
            return None

        return self.cell_metas[self.cell_data[cell_y][cell_x]]

    def get_color(self, cell_cord: pygame.Vector2) -> pygame.Color:
        if not (cell_meta := self.get_cell_meta(cell_cord)):
            return self.default_color

        return pygame.Color(cell_meta.color)

    def get_wall_texture(self, cell_cord: pygame.Vector2) -> pygame.Surface | None:
        cell_meta = self.get_cell_meta(cell_cord)
        return cell_meta.wall_texture if cell_meta else None

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
    sample_wall_texture: pygame.Surface | None
    cell_count: int = 0
    intersect_padding = 0.001
    tick_id: tuple = tuple()

    def __post_init__(self) -> None:
        self.cell_count = len(self.map_data.cell_data[0])
        self.intersect_buffer = [
            Intersect(
                origin = pygame.Vector2(),
                vert_intersect = pygame.Vector2(),
                horiz_intersect = pygame.Vector2(),
                intersect = pygame.Vector2(),
                distance = 0,
                cos_distance = 0,
                collision = False,
                color = pygame.Color("red"),
                marked = False,
                intersect_type = "x",
                wall_texture = None
            ) for _ in range(MAX_STEPS_PER_CAST)
        ]
        self.last_intersect_count = 0

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

    def intersect(self, ray: Ray, reference_angle: float, tick_count):
        # This not an optimal implementation of a ray-caster, many objects are created "on the fly"
        # to just be disposed a short time later, redundant calculations exist, this function is
        # invoked for each ray etc. etc.

        if self.tick_id == (ray.pos.x, ray.pos.y, ray.dir, reference_angle, tick_count):
            return self.intersect_buffer, self.last_intersect_count
        else:
            self.tick_id = (ray.pos.x, ray.pos.y, ray.dir, reference_angle, tick_count)

        next_cast_origin = ray.pos
        end_point = ray.pos + ray.mag.rotate(ray.dir)
        max_cast_distance = math.sqrt(
            (ray.pos.x - end_point.x) ** 2 + (ray.pos.y - end_point.y) ** 2
        )
        angle = ray.dir if ray.dir != 0 else VERY_NEAR_ZERO
        angle_true_distance_to_adj_distance = abs(reference_angle - angle)

        traversed_cast_distance = 0
        intersect_count = 0
        # Travel along the cast ray and find where it intersects with horizontal or vertical gird lines
        for idx in range(MAX_STEPS_PER_CAST):
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

            # Vert intersect_count
            # adj = step_x
            # tan(angle) = opposite / adj
            # opposite = tan(angle) * adj
            # delta_y = opposite
            delta_y = abs(math.tan(math.radians(angle)) * step_x)
            if angle >= 0 and angle <= 180:
                vert_intersect_y_comp = next_cast_origin.y + delta_y
            else:
                vert_intersect_y_comp = next_cast_origin.y - delta_y

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

            # Calculate distance (distance squared, more on why below) from the previous intersect to both the
            # horizontal and vertical intersects in this step.
            vertical_intersect_dist_squared = (
                next_cast_origin.x - vert_intersect_x_comp
            ) ** 2 + (next_cast_origin.y - vert_intersect_y_comp) ** 2

            horiz_intersect_dist_squared = (
                next_cast_origin.x - horiz_intersect_x_comp
            ) ** 2 + (next_cast_origin.y - horiz_intersect_y_comp) ** 2

            # The closer intersect is the intersect that the ray would encounter first as it projects out
            # so it is the one we want for the current step
            # Is the vertical or horizontal intersect at this step closer in distance squared
            # because we don't need the actual distance just which one is relative to the other closer
            # we can avoid having to take the square root here and save on computation
            if horiz_intersect_dist_squared < vertical_intersect_dist_squared:
                closest_intersect_tuple = (horiz_intersect_x_comp, next_cell_y)
                intersect_distance_squared = horiz_intersect_dist_squared
                intersect_type = "x"
            else:
                closest_intersect_tuple = (next_cell_x, vert_intersect_y_comp)
                intersect_distance_squared = vertical_intersect_dist_squared
                intersect_type = "y"

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
                current_intersect = self.intersect_buffer[idx]
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

                current_intersect.intersect.x = closest_intersect_tuple[0]
                current_intersect.intersect.y = closest_intersect_tuple[1]

                # The Map has all the "data" for the cells that build the grid "world"
                # so it is extracted here and sent down along with the rest of the intersect data
                cell = self.get_cell(current_intersect.intersect)
                has_collided = self.map_data.is_solid_cell(cell)
                color = self.map_data.get_color(cell)
                wall_texture = self.map_data.get_wall_texture(cell)

                if has_collided:
                    marked = True

                current_intersect.origin = next_cast_origin
                current_intersect.vert_intersect.x = vert_intersect_x_comp
                current_intersect.vert_intersect.x = vert_intersect_x_comp
                current_intersect.horiz_intersect.x = horiz_intersect_x_comp
                current_intersect.horiz_intersect.y = horiz_intersect_y_comp
                current_intersect.distance = traversed_cast_distance
                current_intersect.cos_distance = cos_distance
                current_intersect.collision = has_collided
                current_intersect.color = color
                current_intersect.marked = marked
                current_intersect.intersect_type = intersect_type
                current_intersect.wall_texture = wall_texture
                intersect_count = idx
            else:
                break

            next_cast_origin = current_intersect.intersect
            self.last_intersect_count = intersect_count + 1

        return self.intersect_buffer, self.last_intersect_count


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

        return Map(MapData(data, cls.DEFAULT_LEVEL_CELL_METAS), None)

    @classmethod
    def from_file(cls, file_name: str) -> Map:
        with open(file_name, "r") as f:
            serialized_data = f.read()

        level_data = json.loads(serialized_data)
        cell_meta = {}
        sample_wall_texture = None
        for k, v in level_data.get("cell_meta", {}).items():
            wall_texture_path = v.pop("wall_texture", None)
            wall_texture = (
                pygame.image.load(wall_texture_path) if wall_texture_path else None
            )
            if wall_texture:
                sample_wall_texture = wall_texture
            cell_meta[int(k)] = CellMeta(**v, wall_texture=wall_texture)

        return Map(
            MapData(
                level_data.get("layout"), (cell_meta or cls.DEFAULT_LEVEL_CELL_METAS)
            ),
            sample_wall_texture,
        )


