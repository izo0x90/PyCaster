from dataclasses import dataclass
import math

import pygame

from raycast import Player, VERY_NEAR_ZERO


@dataclass
class Mode7SubView:
    surface: pygame.Surface
    player: Player
    tex_name: str = "./dirt.jpg"

    def __post_init__(self):
        self.floor_tex = pygame.image.load(self.tex_name)
        diagonal = math.sqrt((self.floor_tex.get_width() ** 2) * 2)
        self.big_tex = pygame.Surface((diagonal, diagonal)).convert(self.floor_tex)
        self.top_down_surface = pygame.Surface((diagonal, diagonal)).convert(
            self.floor_tex
        )
        self.last_sample_angle = float("inf")

    def map_point(self, cord: pygame.Vector2, tex):
        half_w = tex.get_width() / 2
        half_h = tex.get_height() / 2
        x = half_w + half_w * cord.x
        y = half_h + half_h * cord.y * -1
        return pygame.Vector2(int(x), int(y))

    def orient_tex_top_down_dir(self):
        sampling_angle = -1 * (self.player.dir - 90)
        if self.last_sample_angle == sampling_angle:
            return

        self.last_sample_angle = sampling_angle

        tex = self.floor_tex
        rotated_tex = pygame.transform.rotate(tex, sampling_angle)
        rot_w = rotated_tex.get_width()
        center_offset = round(abs(self.big_tex.get_width() - rot_w) / 2)
        self.big_tex.blit(rotated_tex, (center_offset, center_offset))

    def orient_player_top_down_dir(self):
        sampling_angle = -1 * (self.player.dir - 90)
        player_pos = self.player.pos.rotate(sampling_angle)
        player_pos_tex = self.map_point(player_pos, self.top_down_surface)

        return player_pos, player_pos_tex

    def draw_floor_top_down(self):
        self.orient_tex_top_down_dir()
        self.top_down_surface.blit(self.big_tex, (0, 0))

        player_pos, player_pos_tex = self.orient_player_top_down_dir()

        pygame.draw.circle(
            self.top_down_surface,
            "yellow",
            self.map_point(pygame.Vector2(0, 0), self.top_down_surface),
            20,
        )
        pygame.draw.circle(self.top_down_surface, "purple", player_pos_tex, 20)

        half_screen_height = math.floor(self.surface.get_height() / 2)
        sample_tex = self.top_down_surface
        for row in range(half_screen_height, self.surface.get_height()):
            scaler = (abs(row - half_screen_height) / half_screen_height) * 4
            inv_scalar = 1 / (scaler + VERY_NEAR_ZERO)
            distance = 0.05 * inv_scalar

            sample_angle_start = 90 + self.player.half_fov
            sample_angle_end = 90 - self.player.half_fov

            start_sample = pygame.Vector2(distance, 0)
            start_sample.rotate_ip(sample_angle_start)
            start_sample += player_pos
            start_sample_tex = self.map_point(start_sample, sample_tex)
            start_sample_tex.x = max(
                0, min(start_sample_tex.x, sample_tex.get_width() - 1)
            )
            start_sample_tex.y = max(
                0, min(start_sample_tex.y, sample_tex.get_width() - 1)
            )
            pygame.draw.circle(sample_tex, "blue", start_sample_tex, 50)

            end_sample = pygame.Vector2(distance, 0)
            end_sample.rotate_ip(sample_angle_end)
            end_sample += player_pos
            end_sample_tex = self.map_point(end_sample, sample_tex)
            end_sample_tex.x = max(0, min(end_sample_tex.x, sample_tex.get_width() - 1))
            end_sample_tex.y = max(
                0, min(end_sample_tex.y, sample_tex.get_height() - 1)
            )
            pygame.draw.circle(sample_tex, "red", end_sample_tex, 50)

            row_width = end_sample_tex - start_sample_tex
            row_width.x = min(row_width.x, sample_tex.get_width() - 1)
            row_width.y = 1

        self.surface.blit(
            pygame.transform.scale(self.top_down_surface, (640, 640)), (0, 0)
        )

    def draw_floor_fps(self):
        half_screen_height = math.floor(self.surface.get_height() / 2)
        sample_tex = self.big_tex

        player_pos, _ = self.orient_player_top_down_dir()
        self.orient_tex_top_down_dir()

        for row in range(half_screen_height, self.surface.get_height()):
            scaler = (abs(row - half_screen_height) / half_screen_height) * 4
            inv_scalar = 1 / (scaler + VERY_NEAR_ZERO)
            distance = 0.05 * inv_scalar

            sample_angle_start = 90 + self.player.half_fov
            sample_angle_end = 90 - self.player.half_fov

            start_sample = pygame.Vector2(distance, 0)
            start_sample.rotate_ip(sample_angle_start)
            start_sample += player_pos
            start_sample_tex = self.map_point(start_sample, sample_tex)
            start_sample_tex.x = max(
                0, min(start_sample_tex.x, sample_tex.get_width() - 1)
            )
            start_sample_tex.y = max(
                0, min(start_sample_tex.y, sample_tex.get_width() - 1)
            )

            end_sample = pygame.Vector2(distance, 0)
            end_sample.rotate_ip(sample_angle_end)
            end_sample += player_pos
            end_sample_tex = self.map_point(end_sample, sample_tex)
            end_sample_tex.x = max(0, min(end_sample_tex.x, sample_tex.get_width() - 1))
            end_sample_tex.y = max(
                0, min(end_sample_tex.y, sample_tex.get_height() - 1)
            )

            row_width = end_sample_tex - start_sample_tex
            row_width.x = min(row_width.x, sample_tex.get_width() - 1)
            row_width.y = 1

            pygame.draw.line(
                self.surface, "red", (0, row), (self.surface.get_width(), row)
            )
            row_tex = sample_tex.subsurface(start_sample_tex, row_width)
            scaled_row = pygame.transform.scale(row_tex, (self.surface.get_width(), 1))
            self.surface.blit(scaled_row, (0, row))
