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
    this.onIcePlatform = false  // Track if on ice (slippery)
  }

  update(platforms, deltaTime = 1) {
    if (!this.isAlive) return

    // Apply gravity with deltaTime for frame-rate independence
    this.velocityY += this.gravity * deltaTime
    if (this.velocityY > this.maxFallSpeed) {
      this.velocityY = this.maxFallSpeed
    }

    // Move horizontally first
    this.x += this.velocityX * deltaTime

    // Check horizontal collisions - platforms are SOLID
    for (const platform of platforms) {
      if (this.isOverlapping(platform)) {
        // Resolve collision by pushing player out
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

    // Move vertically
    this.y += this.velocityY * deltaTime

    // Check vertical collisions - platforms are SOLID
    this.isOnGround = false
    this.onIcePlatform = false
    for (const platform of platforms) {
      if (this.isOverlapping(platform)) {
        // Resolve collision by pushing player out
        if (this.velocityY > 0) {
          // Moving down, hit top of platform (landing)
          this.y = platform.y - this.height
          this.isOnGround = true
          // Check if platform is ice (slippery)
          if (platform.type === 'ice') {
            this.onIcePlatform = true
          }
        } else if (this.velocityY < 0) {
          // Moving up, hit bottom of platform
          this.y = platform.y + platform.height
        }
        this.velocityY = 0
        break
      }
    }

    // Apply friction (reduced on ice)
    if (this.isOnGround) {
      if (this.onIcePlatform) {
        // Ice: very low friction (0.95 = keeps 95% of velocity)
        this.velocityX *= 0.95
        // Cap velocity on ice to prevent infinite acceleration
        const maxIceVelocity = 15
        if (Math.abs(this.velocityX) > maxIceVelocity) {
          this.velocityX = Math.sign(this.velocityX) * maxIceVelocity
        }
      } else {
        // Normal: high friction (0.7 = keeps 70% of velocity)
        this.velocityX *= 0.7
      }
    }

    // Track distance
    this.distance = Math.max(this.distance, this.x)

    // Die if fall off map
    if (this.y > 1200) {
      this.die()
    }
  }

  // Simple AABB collision detection - checks if player overlaps with platform
  isOverlapping(platform) {
    return (
      this.x < platform.x + platform.width &&
      this.x + this.width > platform.x &&
      this.y < platform.y + platform.height &&
      this.y + this.height > platform.y
    )
  }

  moveLeft() {
    const speed = this.isSprinting ? this.speed * 1.5 : this.speed
    if (this.onIcePlatform) {
      // On ice: add to velocity (acceleration-based for sliding)
      this.velocityX -= speed * 0.3  // 30% acceleration on ice
    } else {
      // Normal: direct control
      this.velocityX = -speed
    }
  }

  moveRight() {
    const speed = this.isSprinting ? this.speed * 1.5 : this.speed
    if (this.onIcePlatform) {
      // On ice: add to velocity (acceleration-based for sliding)
      this.velocityX += speed * 0.3  // 30% acceleration on ice
    } else {
      // Normal: direct control
      this.velocityX = speed
    }
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
