"""
Configuration constants for the platformer game
"""

# Game dimensions
GAME_WIDTH = 1920
GAME_HEIGHT = 1080
TILE_SIZE = 32

# AI observation dimensions (smaller for performance)
OBS_WIDTH = 84
OBS_HEIGHT = 84

# Player physics
PLAYER_WIDTH = 32
PLAYER_HEIGHT = 32
PLAYER_SPEED = 5
JUMP_VELOCITY = 15
GRAVITY = 0.8
MAX_FALL_SPEED = 20

# Action space
# 0: Left
# 1: Right
# 2: Jump
# 3: Sprint (run faster)
# 4: Duck (slide under obstacles)
# 5: Idle
NUM_ACTIONS = 6

# Map generation
MAP_LENGTH_TILES = 200  # Length of each level
MIN_PLATFORM_WIDTH = 3  # Minimum platform width in tiles
MAX_PLATFORM_WIDTH = 10
MIN_PLATFORM_GAP = 2  # Minimum gap between platforms
MAX_PLATFORM_GAP = 5
PLATFORM_HEIGHT_VARIANCE = 5  # How much platforms can vary in height

# Enemies
ENEMY_TYPES = ['walker', 'jumper', 'flyer']
ENEMY_SPAWN_RATE = 0.1  # Probability of enemy on each platform

# Collectibles
COIN_SPAWN_RATE = 0.3
POWERUP_SPAWN_RATE = 0.05

# Rewards
REWARD_MOVE_RIGHT = 1
REWARD_COIN = 10
REWARD_DEFEAT_ENEMY = 50
REWARD_REACH_GOAL = 1000
REWARD_DEATH = -100
REWARD_IDLE = -1

# Training
TRAINING_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 10_000
EVAL_FREQ = 50_000
