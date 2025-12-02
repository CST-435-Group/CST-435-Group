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

  generateMap(seed) {
    if (seed) this.seed = seed

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
    let currentX = 350 // Start close to the starting platform (adjusted for new start position)
    let currentY = startHeight - 50 // Start at similar height

    for (let i = 0; i < 50; i++) {
      // Platform dimensions
      const platformWidth = this.randomInt(3, 8) * this.tileSize
      const platformHeight = this.tileSize

      // Position - smaller gaps, more reasonable
      const gapX = this.randomInt(120, 250) // Jumpable distances
      const gapY = this.randomInt(-100, 80) // Height variation

      currentX += gapX
      currentY = Math.max(200, Math.min(this.height - 200, currentY + gapY))

      platforms.push({
        x: currentX,
        y: currentY,
        width: platformWidth,
        height: platformHeight,
        type: 'platform'
      })

      // Add coins on some platforms
      if (Math.random() < 0.4) {
        coins.push({
          x: currentX + platformWidth / 2,
          y: currentY - 40,
          radius: 15,
          collected: false
        })
      }

      // Add enemies on some platforms
      if (Math.random() < 0.2 && i > 5) {
        enemies.push({
          x: currentX + platformWidth / 2,
          y: currentY - 40,
          width: 32,
          height: 32,
          type: 'walker',
          direction: 1,
          speed: 2
        })
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
}
