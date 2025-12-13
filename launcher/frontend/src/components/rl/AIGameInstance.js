/**
 * AIGameInstance - Separate headless game instance for AI
 *
 * This runs the AI in an isolated environment where:
 * - AI gets the same visual input it was trained on
 * - Camera is centered on AI (not affected by human player position)
 * - AI can't see the human player sprite
 *
 * The main game just reads AI's world position and renders the sprite
 */

import { Player } from './Player'
import AIPlayer from './AIPlayer'

export class AIGameInstance {
  constructor(map, modelPath, difficulty, onAIReady, onAIError) {
    this.map = map
    this.modelPath = modelPath
    this.difficulty = difficulty
    this.onAIReady = onAIReady
    this.onAIError = onAIError

    // Create offscreen canvas for AI's view (not visible to player)
    this.canvas = document.createElement('canvas')
    this.canvas.width = 1920
    this.canvas.height = 1080
    this.ctx = this.canvas.getContext('2d')

    // AI player object
    this.aiPlayer = new Player(map.spawn.x, map.spawn.y - 50)

    // AI agent (model)
    this.aiAgent = new AIPlayer()

    // AI camera (follows AI, not human)
    this.cameraX = 0

    // AI state
    this.aiActionCooldown = 0
    this.aiLastAction = 0
    this.aiPredicting = false
    this.isReady = false
    this.isLoading = true

    // For accessing in predictAction
    this.gameContext = {
      difficulty: difficulty,
      enemies: map.enemies,
      goal: map.goal,
      map: map
    }

    // Load AI model
    this.loadAI()
  }

  async loadAI() {
    console.log('[AI-Instance] Loading AI model from:', this.modelPath)

    const onProgress = (progress, message) => {
      console.log(`[AI-Instance] ${progress}% - ${message}`)
    }

    try {
      const success = await this.aiAgent.loadModel(this.modelPath, onProgress)

      if (success) {
        console.log('[AI-Instance] AI model loaded successfully')
        this.isReady = true
        this.isLoading = false
        if (this.onAIReady) this.onAIReady()
      } else {
        console.error('[AI-Instance] Failed to load AI model')
        this.isLoading = false
        if (this.onAIError) this.onAIError('Failed to load model')
      }
    } catch (error) {
      console.error('[AI-Instance] Error loading AI:', error)
      this.isLoading = false
      if (this.onAIError) this.onAIError(error.message)
    }
  }

  /**
   * Update AI player state and render to offscreen canvas
   * Call this every frame from main game loop
   */
  update(deltaTime = 1) {
    if (!this.aiPlayer.isAlive || !this.isReady) {
      return
    }

    // Update AI action (prediction)
    this.aiActionCooldown--
    if (this.aiActionCooldown <= 0) {
      this.aiActionCooldown = 3 // Predict every 3 frames

      // Start async prediction
      if (!this.aiPredicting) {
        this.aiPredicting = true
        this.aiAgent.predictAction(this.canvas, this.aiPlayer.x, this.cameraX, this.gameContext, this.aiPlayer)
          .then(action => {
            this.aiLastAction = action
            this.aiPredicting = false
          })
          .catch(error => {
            console.error('[AI-Instance] Prediction error:', error)
            this.aiPredicting = false
          })
      }
    }

    // Apply cached action
    const action = this.aiLastAction
    this.aiPlayer.stopMovement()

    switch (action) {
      case 1: // Left
        this.aiPlayer.moveLeft()
        break
      case 2: // Right
        this.aiPlayer.moveRight()
        break
      case 3: // Jump (straight up)
        this.aiPlayer.jump()
        break
      case 4: // Sprint + Right
        this.aiPlayer.sprint(true)
        this.aiPlayer.moveRight()
        break
      case 5: // Duck
        this.aiPlayer.duck(true)
        break
      case 6: // Jump + Left
        this.aiPlayer.jump()
        this.aiPlayer.moveLeft()
        break
      case 7: // Jump + Right
        this.aiPlayer.jump()
        this.aiPlayer.moveRight()
        break
      case 8: // Sprint + Jump + Right
        this.aiPlayer.sprint(true)
        this.aiPlayer.jump()
        this.aiPlayer.moveRight()
        break
      default: // Idle
        break
    }

    // Update AI physics
    this.aiPlayer.update(this.map.platforms, deltaTime)

    // Check coin collection
    this.map.coins.forEach(coin => {
      this.aiPlayer.collectCoin(coin)
    })

    // Check enemy collisions
    this.map.enemies.forEach(enemy => {
      if (this.aiPlayer.checkEnemyCollision(enemy)) {
        this.aiPlayer.die()
      }
    })

    // Update camera - CRITICAL: Camera follows AI (like training)
    const targetCameraX = this.aiPlayer.x - 400
    this.cameraX += (targetCameraX - this.cameraX) * 0.1 * deltaTime

    // Render to offscreen canvas (AI sees this view)
    this.render()
  }

  /**
   * Render AI's view to offscreen canvas
   * This is what the CNN model sees - AI at consistent left position
   */
  render() {
    const canvas = this.canvas
    const ctx = this.ctx

    // Clear canvas
    ctx.fillStyle = '#87CEEB' // Sky blue
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Save context
    ctx.save()
    ctx.translate(-this.cameraX, 0)

    // Draw platforms
    this.map.platforms.forEach(platform => {
      if (platform.type === 'ground') {
        ctx.fillStyle = '#8B4513'
      } else if (platform.type === 'ice') {
        ctx.fillStyle = '#4DA6FF'
      } else {
        ctx.fillStyle = '#2E8B57'
      }
      ctx.fillRect(platform.x, platform.y, platform.width, platform.height)

      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.strokeRect(platform.x, platform.y, platform.width, platform.height)
    })

    // Draw coins
    this.map.coins.forEach(coin => {
      if (!coin.collected) {
        ctx.fillStyle = '#FFD700'
        ctx.beginPath()
        ctx.arc(coin.x, coin.y, coin.radius, 0, Math.PI * 2)
        ctx.fill()
        ctx.strokeStyle = '#FFA500'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })

    // Draw enemies
    this.map.enemies.forEach(enemy => {
      ctx.fillStyle = '#FF00FF'
      ctx.fillRect(enemy.x, enemy.y, enemy.width, enemy.height)
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.strokeRect(enemy.x, enemy.y, enemy.width, enemy.height)
    })

    // Draw goal
    ctx.fillStyle = '#00FF00'
    ctx.fillRect(this.map.goal.x, this.map.goal.y, this.map.goal.width, this.map.goal.height)
    ctx.strokeStyle = '#006400'
    ctx.lineWidth = 3
    ctx.strokeRect(this.map.goal.x, this.map.goal.y, this.map.goal.width, this.map.goal.height)

    // Goal flag
    ctx.fillStyle = '#FFD700'
    ctx.font = 'bold 60px Arial'
    ctx.fillText('🏁', this.map.goal.x + 5, this.map.goal.y - 10)

    // Draw AI player (ONLY AI, no human player sprite)
    ctx.fillStyle = '#FF6B6B'
    ctx.fillRect(this.aiPlayer.x, this.aiPlayer.y, this.aiPlayer.width, this.aiPlayer.height)
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 2
    ctx.strokeRect(this.aiPlayer.x, this.aiPlayer.y, this.aiPlayer.width, this.aiPlayer.height)

    // Restore context
    ctx.restore()
  }

  /**
   * Get AI player's current world position (for rendering on main canvas)
   */
  getAIPosition() {
    return {
      x: this.aiPlayer.x,
      y: this.aiPlayer.y,
      width: this.aiPlayer.width,
      height: this.aiPlayer.height,
      isAlive: this.aiPlayer.isAlive,
      score: this.aiPlayer.score,
      distance: this.aiPlayer.distance
    }
  }

  /**
   * Check if AI reached goal
   */
  checkGoalReached() {
    return this.aiPlayer.checkGoalReached(this.map.goal)
  }

  /**
   * Dispose resources
   */
  dispose() {
    if (this.aiAgent) {
      this.aiAgent.dispose()
    }
    this.canvas = null
    this.ctx = null
  }
}
