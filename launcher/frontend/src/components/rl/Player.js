/**
 * Player class with physics and controls
 */

export class Player {
  constructor(x, y, width = 32, height = 32) {
    this.x = x
    this.y = y
    this.width = width
    this.height = height
    this.velocityX = 0
    this.velocityY = 0
    this.speed = 5
    this.jumpPower = 15
    this.gravity = 0.8
    this.maxFallSpeed = 20
    this.isOnGround = false
    this.isAlive = true
    this.distance = 0
    this.score = 0
    this.isSprinting = false
    this.isDucking = false
    this.jumpCount = 0  // Track total jumps for metrics
  }

  update(platforms, deltaTime = 1) {
    if (!this.isAlive) return

    // Store previous position for collision detection
    const prevX = this.x
    const prevY = this.y

    // Apply gravity with deltaTime for frame-rate independence
    this.velocityY += this.gravity * deltaTime
    if (this.velocityY > this.maxFallSpeed) {
      this.velocityY = this.maxFallSpeed
    }

    // Apply vertical velocity first (scaled by deltaTime)
    this.y += this.velocityY * deltaTime

    // Check vertical collision FIRST (landing on platforms has priority)
    this.isOnGround = false
    for (const platform of platforms) {
      if (this.checkPlatformCollision(platform, prevY)) {
        this.isOnGround = true
        this.velocityY = 0
        this.y = platform.y - this.height
        break
      }
    }

    // Apply horizontal velocity with collision detection (scaled by deltaTime)
    this.x += this.velocityX * deltaTime

    // Check horizontal collision with platforms (only if not landing on top)
    for (const platform of platforms) {
      if (this.checkHorizontalCollision(platform, prevX)) {
        // Stop horizontal movement and push player out
        if (this.velocityX > 0) {
          // Moving right, hit left side of platform
          this.x = platform.x - this.width
        } else if (this.velocityX < 0) {
          // Moving left, hit right side of platform
          this.x = platform.x + platform.width
        }
        this.velocityX = 0
        break
      }
    }

    // Track distance
    this.distance = Math.max(this.distance, this.x)

    // Die if fall off map
    if (this.y > 1200) {
      this.die()
    }
  }

  checkHorizontalCollision(platform, prevX) {
    // Only check horizontal collision when moving horizontally
    if (this.velocityX === 0) return false

    // Don't apply horizontal collision when jumping upward through platforms
    if (this.velocityY < 0) return false

    // Check if player overlaps with platform horizontally and vertically
    const horizontalOverlap =
      this.x < platform.x + platform.width &&
      this.x + this.width > platform.x

    const verticalOverlap =
      this.y < platform.y + platform.height &&
      this.y + this.height > platform.y

    const playerBottom = this.y + this.height
    const platformTop = platform.y
    const platformBottom = platform.y + platform.height

    // Check if we're actually standing on top of THIS platform
    // (not just at the same height as another platform)
    const feetAtSameLevel = Math.abs(playerBottom - platformTop) < 5
    const playerCenterX = this.x + this.width / 2
    const horizontallyOnPlatform = playerCenterX > platform.x && playerCenterX < platform.x + platform.width
    const isStandingOnThisPlatform = feetAtSameLevel && horizontallyOnPlatform

    // Don't collide horizontally if we're actually standing on top of THIS platform
    // (allows walking along the platform without hitting its own edges)
    if (isStandingOnThisPlatform) return false

    // Don't collide if player is below the platform (coming from underneath)
    // This prevents teleporting when jumping through bottom of platforms
    const playerTop = this.y
    const playerComingFromBelow = playerTop < platformBottom && playerBottom > platformTop

    // Only collide if we're actually moving into the side of the platform
    // (not if we're above it or jumping through from below)
    const playerAbovePlatform = this.y + this.height < platform.y + 5

    return horizontalOverlap && verticalOverlap && !playerAbovePlatform && !playerComingFromBelow
  }

  checkPlatformCollision(platform, prevY) {
    // Only check landing collision when falling or on the platform
    if (this.velocityY < 0) return false // Moving up, don't land

    const playerBottom = this.y + this.height
    const prevPlayerBottom = prevY + this.height
    const platformTop = platform.y

    // Check horizontal overlap
    const horizontalOverlap =
      this.x + this.width > platform.x &&
      this.x < platform.x + platform.width

    if (!horizontalOverlap) return false

    // Check if we crossed the platform top this frame (prevents tunneling)
    const crossedPlatform = prevPlayerBottom <= platformTop && playerBottom >= platformTop

    // Or if we're very close to the top (already on it)
    const onPlatform = playerBottom >= platformTop && playerBottom <= platformTop + Math.abs(this.velocityY) + 10

    return crossedPlatform || onPlatform
  }

  moveLeft() {
    const speed = this.isSprinting ? this.speed * 1.5 : this.speed
    this.velocityX = -speed
  }

  moveRight() {
    const speed = this.isSprinting ? this.speed * 1.5 : this.speed
    this.velocityX = speed
  }

  jump() {
    // Only jump if on ground and not already jumping
    if (this.isOnGround && this.velocityY >= 0) {
      this.velocityY = -this.jumpPower
      this.isOnGround = false
      this.jumpCount++  // Increment jump counter for metrics
    }
  }

  sprint(active) {
    this.isSprinting = active
  }

  duck(active) {
    this.isDucking = active
    // Could reduce height when ducking
  }

  stopMovement() {
    this.velocityX = 0
  }

  die() {
    this.isAlive = false
    this.velocityX = 0
    this.velocityY = 0
  }

  reset(x, y) {
    this.x = x
    this.y = y
    this.velocityX = 0
    this.velocityY = 0
    this.isAlive = true
    this.isOnGround = false
    this.distance = 0
    this.score = 0
    this.jumpCount = 0  // Reset jump counter for new game
  }

  collectCoin(coin) {
    if (!coin.collected) {
      const dx = this.x + this.width / 2 - coin.x
      const dy = this.y + this.height / 2 - coin.y
      const distance = Math.sqrt(dx * dx + dy * dy)

      if (distance < this.width / 2 + coin.radius) {
        coin.collected = true
        this.score += 10
        return true
      }
    }
    return false
  }

  checkEnemyCollision(enemy) {
    return (
      this.x < enemy.x + enemy.width &&
      this.x + this.width > enemy.x &&
      this.y < enemy.y + enemy.height &&
      this.y + this.height > enemy.y
    )
  }

  checkGoalReached(goal) {
    return (
      this.x < goal.x + goal.width &&
      this.x + this.width > goal.x &&
      this.y < goal.y + goal.height &&
      this.y + this.height > goal.y
    )
  }
}
