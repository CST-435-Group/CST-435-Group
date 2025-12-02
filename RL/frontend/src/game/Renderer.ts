/**
 * Game renderer
 * Handles all canvas drawing operations
 */

import { MapData, Platform, Enemy, Coin } from './MapGenerator';
import { Player } from './Player';

export class Renderer {
  private ctx: CanvasRenderingContext2D;
  private width: number;
  private height: number;
  private cameraX: number;

  constructor(ctx: CanvasRenderingContext2D, width: number, height: number) {
    // TODO: Initialize renderer
  }

  /**
   * Clear canvas
   */
  clear(): void {
    // TODO: Clear entire canvas
  }

  /**
   * Render background (sky, clouds, etc.)
   */
  renderBackground(): void {
    // TODO: Draw sky gradient, clouds
  }

  /**
   * Render the entire map
   */
  renderMap(mapData: MapData): void {
    // TODO: Render all platforms, enemies, coins
  }

  /**
   * Render a single platform
   */
  renderPlatform(platform: Platform): void {
    // TODO: Draw platform (green rectangle for now)
    // Apply camera offset
  }

  /**
   * Render a player
   */
  renderPlayer(player: Player): void {
    // TODO: Draw player (white rectangle for now)
    // Different color for AI (red) vs human (white)
    // Apply camera offset
  }

  /**
   * Render an enemy
   */
  renderEnemy(enemy: Enemy): void {
    // TODO: Draw enemy (different colors for different types)
    // Apply camera offset
  }

  /**
   * Render a coin
   */
  renderCoin(coin: Coin): void {
    // TODO: Draw coin (yellow circle)
    // Skip if collected
    // Apply camera offset
  }

  /**
   * Render goal flag
   */
  renderGoal(x: number, y: number): void {
    // TODO: Draw goal (checkered flag or star)
  }

  /**
   * Render UI elements (score, distance, etc.)
   */
  renderUI(humanPlayer: Player, aiPlayer: Player): void {
    // TODO: Draw scores, distances, timer
    // Fixed position (not affected by camera)
  }

  /**
   * Update camera to follow player
   */
  updateCamera(player: Player): void {
    // TODO: Smooth camera following
    // Keep player in view while showing level ahead
  }

  /**
   * Convert world coordinates to screen coordinates
   */
  worldToScreen(x: number, y: number): { x: number; y: number } {
    // TODO: Apply camera offset
    return { x: 0, y: 0 };
  }

  /**
   * Draw text with shadow
   */
  private drawText(text: string, x: number, y: number, size = 20, color = 'white'): void {
    // TODO: Draw text with outline/shadow for visibility
  }

  /**
   * Draw a rectangle with optional border
   */
  private drawRect(
    x: number,
    y: number,
    width: number,
    height: number,
    fillColor: string,
    strokeColor?: string
  ): void {
    // TODO: Draw rectangle
  }
}
