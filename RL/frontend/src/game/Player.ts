/**
 * Player class for both human and AI players
 */

export class Player {
  public x: number;
  public y: number;
  public width: number;
  public height: number;
  public velocityX: number;
  public velocityY: number;
  public isOnGround: boolean;
  public isAlive: boolean;
  public distance: number;
  public score: number;
  public isAI: boolean;

  constructor(x: number, y: number, isAI = false) {
    // TODO: Initialize player properties
  }

  /**
   * Update player physics and position
   */
  update(platforms: any[], deltaTime: number): void {
    // TODO: Apply physics, check collisions
  }

  /**
   * Move player left
   */
  moveLeft(speed = 5): void {
    // TODO: Set velocity for left movement
  }

  /**
   * Move player right
   */
  moveRight(speed = 5): void {
    // TODO: Set velocity for right movement
  }

  /**
   * Make player jump
   */
  jump(velocity = 15): void {
    // TODO: Apply upward velocity if on ground
  }

  /**
   * Sprint (move faster)
   */
  sprint(): void {
    // TODO: Increase movement speed
  }

  /**
   * Duck/slide
   */
  duck(): void {
    // TODO: Reduce height, increase speed temporarily
  }

  /**
   * Stop horizontal movement
   */
  stopMovement(): void {
    // TODO: Set horizontal velocity to 0
  }

  /**
   * Check collision with a rectangle
   */
  collidesWith(rect: { x: number; y: number; width: number; height: number }): boolean {
    // TODO: AABB collision detection
    return false;
  }

  /**
   * Check if player is on a platform
   */
  checkGroundCollision(platforms: any[]): void {
    // TODO: Check if standing on any platform
  }

  /**
   * Apply gravity
   */
  private applyGravity(gravity = 0.8): void {
    // TODO: Increase downward velocity
  }

  /**
   * Kill player (fell in pit, hit enemy, etc.)
   */
  die(): void {
    // TODO: Set isAlive to false
  }

  /**
   * Reset player to spawn position
   */
  reset(x: number, y: number): void {
    // TODO: Reset all properties
  }

  /**
   * Get current state for AI observation
   */
  getState(): number[] {
    // TODO: Return state vector for AI
    return [];
  }
}
