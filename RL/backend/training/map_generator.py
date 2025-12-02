"""
Procedural Map Generator for Side-Scrolling Platformer
Python translation of launcher/frontend/src/components/rl/MapGenerator.js
"""

import random
import time


class MapGenerator:
    """
    Generates random but playable platformer levels.
    Direct translation from JavaScript version.
    """

    def __init__(self, width=1920, height=1080, tile_size=32, difficulty=1.0):
        """
        Initialize the map generator.

        Args:
            width: Width of the map in pixels
            height: Height of the map in pixels
            tile_size: Size of each tile (32x32 pixels)
            difficulty: Difficulty level (not currently used, kept for future)
        """
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.difficulty = difficulty
        self.seed = int(time.time() * 1000)  # Milliseconds like Date.now()

    def generate_map(self, seed=None):
        """
        Generate a complete level.
        Exact translation of JavaScript MapGenerator.generateMap()

        Args:
            seed: Random seed for reproducibility

        Returns:
            dict: Map data containing:
                - platforms: List of platform dictionaries
                - coins: List of coin dictionaries
                - enemies: List of enemy dictionaries
                - goal: Goal dictionary
                - spawn: Spawn position dictionary
                - width: Total map width
                - height: Total map height
        """
        if seed is not None:
            self.seed = seed
            random.seed(seed)

        length = 200  # tiles (not currently used)
        platforms = []
        coins = []
        enemies = []

        # Starting platform - small and elevated
        start_height = self.height - 500  # Even higher up (moved up 100px)
        platforms.append({
            'x': 100,  # Moved right 100px
            'y': start_height,
            'width': 200,  # Small starting platform
            'height': 40,
            'type': 'ground'
        })

        # Generate floating platforms
        current_x = 300  # Start at the edge of the starting platform
        current_y = start_height - 50  # Start at similar height

        for i in range(50):
            # Platform dimensions
            platform_width = self.random_int(3, 8) * self.tile_size
            platform_height = self.tile_size

            # Position - smaller gaps, more reasonable
            # First platform is guaranteed close, others are random
            if i == 0:
                gap_x = self.random_int(100, 140)
                gap_y = self.random_int(-30, 20)  # First platform has less height variation
            else:
                gap_x = self.random_int(120, 250)
                gap_y = self.random_int(-100, 80)

            current_x += gap_x
            current_y = max(200, min(self.height - 200, current_y + gap_y))

            # Check for overlap with existing platforms
            platform_overlaps = False
            attempts = 0
            max_attempts = 10

            while True:
                platform_overlaps = False

                for existing_platform in platforms:
                    if self.check_overlap(
                        {'x': current_x, 'y': current_y, 'width': platform_width, 'height': platform_height},
                        existing_platform
                    ):
                        platform_overlaps = True
                        # Try adjusting position
                        current_y += self.random_int(-50, 50)
                        current_y = max(200, min(self.height - 200, current_y))
                        break

                attempts += 1
                if not (platform_overlaps and attempts < max_attempts):
                    break

            # Only add platform if no overlap found
            if not platform_overlaps:
                new_platform = {
                    'x': current_x,
                    'y': current_y,
                    'width': platform_width,
                    'height': platform_height,
                    'type': 'platform'
                }
                platforms.append(new_platform)

                # Add coins on some platforms
                if random.random() < 0.4:
                    coin_x = current_x + platform_width / 2
                    coin_y = current_y - 40

                    coins.append({
                        'x': coin_x,
                        'y': coin_y,
                        'radius': 15,
                        'collected': False
                    })

                # Add enemies on some platforms (but not on same spot as coins)
                if random.random() < 0.2 and i > 5:
                    enemy_x = current_x + self.random_int(20, platform_width - 52)  # Random position on platform
                    enemy_y = current_y - 40

                    # Check if coin exists at similar position
                    coin_too_close = any(
                        abs(coin['x'] - enemy_x) < 50 and abs(coin['y'] - enemy_y) < 50
                        for coin in coins
                    )

                    if not coin_too_close:
                        enemies.append({
                            'x': enemy_x,
                            'y': enemy_y,
                            'width': 32,
                            'height': 32,
                            'type': 'walker',
                            'direction': 1,
                            'speed': 2
                        })

        # Goal at the end
        goal = {
            'x': current_x + 200,
            'y': current_y - 100,
            'width': 50,
            'height': 100
        }

        # Spawn point - on the starting platform
        spawn = {
            'x': 180,  # Adjusted for new platform position
            'y': start_height - 60  # Above the starting platform
        }

        return {
            'platforms': platforms,
            'coins': coins,
            'enemies': enemies,
            'goal': goal,
            'spawn': spawn,
            'width': current_x + 500,
            'height': self.height
        }

    def random_int(self, min_val, max_val):
        """
        Generate random integer (inclusive).
        Matches JavaScript's Math.floor(Math.random() * (max - min + 1)) + min

        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            int: Random integer in range [min_val, max_val]
        """
        return random.randint(min_val, max_val)

    def check_overlap(self, rect1, rect2):
        """
        Check if two rectangles overlap (AABB collision).

        Args:
            rect1: Dictionary with x, y, width, height
            rect2: Dictionary with x, y, width, height

        Returns:
            bool: True if rectangles overlap
        """
        return (
            rect1['x'] < rect2['x'] + rect2['width'] and
            rect1['x'] + rect1['width'] > rect2['x'] and
            rect1['y'] < rect2['y'] + rect2['height'] and
            rect1['y'] + rect1['height'] > rect2['y']
        )
