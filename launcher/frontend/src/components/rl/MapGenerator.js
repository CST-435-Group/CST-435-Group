/**
 * Procedural Map Generator for Platformer
 * Generates random but playable levels
 */

export class MapGenerator {
  constructor(width = 1920, height = 1080, tileSize = 32) {
    this.width = width
    this.height = height
    this.tileSize = tileSize
    this.seed = Date.now()
  }

  generateMap(seed, difficulty = 'easy') {
    if (seed) this.seed = seed

    // Determine number of platforms based on difficulty
    let numPlatforms = 50 // Easy (default)
    if (difficulty === 'medium') {
      numPlatforms = 75 // 1.5x
    } else if (difficulty === 'hard') {
      numPlatforms = 100 // 2x
    }

    const length = 200 // tiles
    const platforms = []
    const coins = []
    const enemies = []

    // Starting platform - small and elevated
    const startHeight = this.height - 500 // Even higher up (moved up 100px)
    platforms.push({
      x: 100, // Moved right 100px
      y: startHeight,
      width: 200, // Small starting platform
      height: 40,
      type: 'ground'
    })

    // Generate floating platforms
    let currentX = 300 // Start at the edge of the starting platform
    let currentY = startHeight - 50 // Start at similar height

    for (let i = 0; i < numPlatforms; i++) {
      // Platform dimensions
      const platformWidth = this.randomInt(3, 8) * this.tileSize
      const platformHeight = this.tileSize

      // Position - smaller gaps, more reasonable
      // First platform is guaranteed close, others are random
      const gapX = i === 0 ? this.randomInt(100, 140) : this.randomInt(120, 250)
      const gapY = i === 0 ? this.randomInt(-30, 20) : this.randomInt(-100, 80) // First platform has less height variation

      currentX += gapX
      currentY = Math.max(200, Math.min(this.height - 200, currentY + gapY))

      // Check for overlap with existing platforms
      let platformOverlaps = false
      let attempts = 0
      const maxAttempts = 10

      do {
        platformOverlaps = false

        for (const existingPlatform of platforms) {
          if (this.checkOverlap(
            { x: currentX, y: currentY, width: platformWidth, height: platformHeight },
            existingPlatform
          )) {
            platformOverlaps = true
            // Try adjusting position
            currentY += this.randomInt(-50, 50)
            currentY = Math.max(200, Math.min(this.height - 200, currentY))
            break
          }
        }

        attempts++
      } while (platformOverlaps && attempts < maxAttempts)

      // Only add platform if no overlap found
      if (!platformOverlaps) {
        // 10% chance for ice platform (slippery)
        const platformType = Math.random() < 0.1 ? 'ice' : 'platform'

        const newPlatform = {
          x: currentX,
          y: currentY,
          width: platformWidth,
          height: platformHeight,
          type: platformType
        }
        platforms.push(newPlatform)

        // Add coins on some platforms
        if (Math.random() < 0.4) {
          const coinX = currentX + platformWidth / 2
          const coinY = currentY - 40

          coins.push({
            x: coinX,
            y: coinY,
            radius: 15,
            collected: false
          })
        }

        // Add enemies on some platforms (but not on same spot as coins)
        if (Math.random() < 0.2 && i > 5) {
          const enemyX = currentX + this.randomInt(20, platformWidth - 52) // Random position on platform
          const enemyY = currentY - 40

          // Check if coin exists at similar position
          const coinTooClose = coins.some(coin =>
            Math.abs(coin.x - enemyX) < 50 && Math.abs(coin.y - enemyY) < 50
          )

          if (!coinTooClose) {
            enemies.push({
              x: enemyX,
              y: enemyY,
              width: 32,
              height: 32,
              type: 'walker',
              direction: 1,
              speed: 2
            })
          }
        }
      }
    }

    // Goal at the end
    const goal = {
      x: currentX + 200,
      y: currentY - 100,
      width: 50,
      height: 100
    }

    // Spawn point - on the starting platform
    const spawn = {
      x: 180, // Adjusted for new platform position
      y: startHeight - 60 // Above the starting platform
    }

    return {
      platforms,
      coins,
      enemies,
      goal,
      spawn,
      width: currentX + 500,
      height: this.height
    }
  }

  randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min
  }

  checkOverlap(rect1, rect2) {
    // Check if two rectangles overlap
    return (
      rect1.x < rect2.x + rect2.width &&
      rect1.x + rect1.width > rect2.x &&
      rect1.y < rect2.y + rect2.height &&
      rect1.y + rect1.height > rect2.y
    )
  }
}
