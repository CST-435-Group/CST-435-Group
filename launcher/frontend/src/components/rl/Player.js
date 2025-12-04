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
  }

  update(platforms, deltaTime = 1) {
    if (!this.isAlive) return

    // Apply gravity with deltaTime for frame-rate independence
    this.velocityY += this.gravity * deltaTime
    if (this.velocityY > this.maxFallSpeed) {
      this.velocityY = this.maxFallSpeed
    }

    // Apply horizontal velocity with collision detection (scaled by deltaTime)
    this.x += this.velocityX * deltaTime

    // Check horizontal collision with platforms
    for (const platform of platforms) {
      if (this.checkHorizontalCollision(platform)) {
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

    // Apply vertical velocity (scaled by deltaTime)
    this.y += this.velocityY * deltaTime

    // Check ground collision (vertical - can jump up through platforms)
    this.isOnGround = false
    for (const platform of platforms) {
      if (this.checkPlatformCollision(platform)) {
        this.isOnGround = true
        this.velocityY = 0
        this.y = platform.y - this.height
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

  checkHorizontalCollision(platform) {
    // Only check horizontal collision when moving horizontally
    if (this.velocityX === 0) return false

    // Check if player overlaps with platform horizontally and vertically
    const horizontalOverlap =
      this.x < platform.x + platform.width &&
      this.x + this.width > platform.x

    const verticalOverlap =
      this.y < platform.y + platform.height &&
      this.y + this.height > platform.y

    // Only collide if we're not landing on top (allow jumping through from below)
    // We're moving horizontally into the side of the platform
    const notLandingOnTop = this.y + this.height <= platform.y + 10

    return horizontalOverlap && verticalOverlap && !notLandingOnTop
  }

  checkPlatformCollision(platform) {
    // Check if player is falling and landing on top of platform
    if (this.velocityY >= 0) {
      const playerBottom = this.y + this.height
      const platformTop = platform.y

      // Check if player's feet are near platform top
      if (
        playerBottom >= platformTop &&
        playerBottom <= platformTop + this.velocityY + this.gravity + 5 &&
        this.x + this.width > platform.x &&
        this.x < platform.x + platform.width
      ) {
        return true
      }
    }
    return false
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
