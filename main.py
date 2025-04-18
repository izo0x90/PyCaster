"""
This code it is not structured in a way an actual game that is concerned with performance should be structured.

Instead the aim to separate things conceptually where maybe they can be understood individually when reading through
the code and also visualized on screen with the different "views".

If it accomplished this or not, **shrugs**, but certainly don't structure actual games like this even in Python.
"""
import math
import logging

import pygame

from views import TopDownDebugView, FPSView
from game import Game
from raycast import MapLoader, Player, FULL_CIRCLE_DEGREES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



TOP_DOWN_VIEW_SIZE = 640
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
    # map = MapLoader.from_file("level_2.json")
    player_size_from_cell_size = map.grid_step / 8
    speed = player_size_from_cell_size / 5
    ray_cast_depth = map.grid_step * 8
    player = Player(
        # pos=pygame.Vector2(-0.37, -0.245),
        pos=pygame.Vector2(0, 0),
        ray_cast_depth=pygame.Vector2(ray_cast_depth, 0),
        # dir=145,
        dir=90,
        speed=pygame.Vector2(speed, 0),
        rotation_speed=2,
        fov=90,
        fov_ray_count=math.ceil(FPS_VIEW_SIZE * min(1, RESOLUTION)),
        radius=player_size_from_cell_size,
    )
    top_down_debug_view = TopDownDebugView(
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

    game = Game(player, map, screen, [top_down_debug_view, fps_view])
    # game = Game(player, map, screen, [fps_view])
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
            player.pos.x = 0
            player.pos.y = 0

        game.handle_repeat_input(keys)

        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.action_quit()
            elif event.type == pygame.KEYDOWN:
                game.handle_input(event.key)
            elif event.type == pygame.MOUSEBUTTONUP:
                print(event)
                game.handle_mouse_input(event.pos[0], event.pos[1], event.button)

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
