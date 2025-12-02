/**
 * Game constants (must match backend config.py)
 */

// Game dimensions
export const GAME_WIDTH = 1920;
export const GAME_HEIGHT = 1080;
export const TILE_SIZE = 32;

// AI observation dimensions
export const OBS_WIDTH = 84;
export const OBS_HEIGHT = 84;

// Player physics
export const PLAYER_WIDTH = 32;
export const PLAYER_HEIGHT = 32;
export const PLAYER_SPEED = 5;
export const SPRINT_MULTIPLIER = 1.5;
export const JUMP_VELOCITY = 15;
export const GRAVITY = 0.8;
export const MAX_FALL_SPEED = 20;

// Actions
export enum Action {
  LEFT = 0,
  RIGHT = 1,
  JUMP = 2,
  SPRINT = 3,
  DUCK = 4,
  IDLE = 5
}

export const ACTION_NAMES = [
  'Left',
  'Right',
  'Jump',
  'Sprint',
  'Duck',
  'Idle'
];

// Keyboard mappings
export const KEY_BINDINGS = {
  LEFT: ['ArrowLeft', 'a', 'A'],
  RIGHT: ['ArrowRight', 'd', 'D'],
  JUMP: [' ', 'ArrowUp', 'w', 'W'],
  SPRINT: ['Shift'],
  DUCK: ['ArrowDown', 's', 'S']
};

// Map generation
export const MAP_LENGTH_TILES = 200;
export const MIN_PLATFORM_WIDTH = 3;
export const MAX_PLATFORM_WIDTH = 10;
export const MIN_PLATFORM_GAP = 2;
export const MAX_PLATFORM_GAP = 5;

// Colors (simple placeholders)
export const COLORS = {
  SKY: '#87CEEB',
  GROUND: '#8B4513',
  PLATFORM: '#2E8B57',
  PLAYER_HUMAN: '#FFFFFF',
  PLAYER_AI: '#FF4444',
  ENEMY: '#FF00FF',
  COIN: '#FFD700',
  GOAL: '#00FF00'
};

// Game settings
export const FPS = 60;
export const FRAME_TIME = 1000 / FPS;

// Model path
export const MODEL_PATH = '/models/tfjs_model/model.json';
