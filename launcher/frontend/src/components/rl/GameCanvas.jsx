import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { MapGenerator } from './MapGenerator'
import { Player } from './Player'
import AIPlayer from './AIPlayer'
import './GameCanvas.css'

/**
 * Main game canvas component
 * Handles rendering, input, and game loop
 * @param {boolean} enableAI - Enable AI opponent
 * @param {string} modelPath - Path to AI model
 */
export default function GameCanvas({ onGameEnd, enableAI = false, modelPath = '/models/rl/tfjs_model/model.json' }) {
  const canvasRef = useRef(null)
  const navigate = useNavigate()
  const [gameState, setGameState] = useState('playing') // playing, won, lost
  const [stats, setStats] = useState({ score: 0, distance: 0, aiScore: 0, aiDistance: 0 })
  const [aiStatus, setAiStatus] = useState('loading') // loading, ready, error, disabled

  // Game objects
  const gameRef = useRef({
    player: null,
    aiPlayer: null,
    aiAgent: null,
    map: null,
    cameraX: 0,
    keys: {},
    lastTime: 0,
    animationFrame: null,
    aiActionCooldown: 0
  })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    const game = gameRef.current

    // Initialize game
    const mapGen = new MapGenerator(1920, 1080, 32)
    game.map = mapGen.generateMap()
    game.player = new Player(game.map.spawn.x, game.map.spawn.y)
    game.cameraX = 0
    game.jumpKeyWasPressed = false // Track if jump key was already pressed

    // Initialize AI if enabled
    if (enableAI) {
      game.aiPlayer = new Player(game.map.spawn.x, game.map.spawn.y - 50) // Start slightly above
      game.aiAgent = new AIPlayer()
      game.aiActionCooldown = 0

      // Load AI model
      setAiStatus('loading')
      game.aiAgent.loadModel(modelPath)
        .then(success => {
          if (success) {
            setAiStatus('ready')
            console.log('[GAME] AI opponent loaded and ready')
          } else {
            setAiStatus('error')
            console.error('[GAME] Failed to load AI model')
          }
        })
        .catch(err => {
          setAiStatus('error')
          console.error('[GAME] AI loading error:', err)
        })
    } else {
      setAiStatus('disabled')
    }

    // Clear all keys helper
    const clearAllKeys = () => {
      game.keys = {}
      game.jumpKeyWasPressed = false
    }

    // Keyboard controls - attach to canvas for better focus control
    const handleKeyDown = (e) => {
      // Prevent default for game keys first
      if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' ', 'Shift', 'a', 'A', 'd', 'D', 'w', 'W', 's', 'S'].includes(e.key)) {
        e.preventDefault()
      }

      // Only set to true if not already true (prevents key repeat)
      if (!game.keys[e.key]) {
        game.keys[e.key] = true
      }
    }

    const handleKeyUp = (e) => {
      // Completely remove the key from the object
      delete game.keys[e.key]

      // Reset jump flag when jump keys are released
      if ([' ', 'ArrowUp', 'w', 'W'].includes(e.key)) {
        game.jumpKeyWasPressed = false
      }
    }

    // Clear keys when window loses focus
    const handleBlur = () => {
      clearAllKeys()
    }

    // Ensure canvas is focusable
    canvas.setAttribute('tabindex', '0')
    canvas.focus()

    window.addEventListener('keydown', handleKeyDown)
    window.addEventListener('keyup', handleKeyUp)
    window.addEventListener('blur', handleBlur)
    canvas.addEventListener('blur', handleBlur)

    // Game loop
    const gameLoop = (timestamp) => {
      const deltaTime = timestamp - game.lastTime
      game.lastTime = timestamp

      // Update
      updateGame(game)

      // Render
      renderGame(ctx, game)

      // Check game state
      const humanWon = game.player.checkGoalReached(game.map.goal)
      const humanLost = !game.player.isAlive
      const aiWon = enableAI && game.aiPlayer && game.aiPlayer.checkGoalReached(game.map.goal)
      const aiLost = enableAI && game.aiPlayer && !game.aiPlayer.isAlive

      if (humanWon || (enableAI && aiLost)) {
        setGameState('won')
        setStats({
          score: game.player.score,
          distance: Math.floor(game.player.distance),
          aiScore: game.aiPlayer ? game.aiPlayer.score : 0,
          aiDistance: game.aiPlayer ? Math.floor(game.aiPlayer.distance) : 0
        })
      } else if (humanLost || (enableAI && aiWon)) {
        setGameState('lost')
        setStats({
          score: game.player.score,
          distance: Math.floor(game.player.distance),
          aiScore: game.aiPlayer ? game.aiPlayer.score : 0,
          aiDistance: game.aiPlayer ? Math.floor(game.aiPlayer.distance) : 0
        })
      } else {
        game.animationFrame = requestAnimationFrame(gameLoop)
      }
    }

    // Start game loop
    game.lastTime = performance.now()
    game.animationFrame = requestAnimationFrame(gameLoop)

    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleBlur)
      canvas.removeEventListener('blur', handleBlur)
      if (game.animationFrame) {
        cancelAnimationFrame(game.animationFrame)
      }
    }
  }, [])

  const updateGame = (game) => {
    const { player, keys, map } = game

    if (!player.isAlive) return

    // Handle input - always reset movement first
    player.stopMovement()

    // Check which keys are currently pressed (must exist in keys object)
    const leftPressed = ('ArrowLeft' in keys) || ('a' in keys) || ('A' in keys)
    const rightPressed = ('ArrowRight' in keys) || ('d' in keys) || ('D' in keys)
    const jumpPressed = (' ' in keys) || ('ArrowUp' in keys) || ('w' in keys) || ('W' in keys)
    const sprintPressed = ('Shift' in keys)
    const duckPressed = ('ArrowDown' in keys) || ('s' in keys) || ('S' in keys)

    // Movement - only move if one direction is pressed
    if (leftPressed && !rightPressed) {
      player.moveLeft()
    } else if (rightPressed && !leftPressed) {
      player.moveRight()
    }

    // Jump - only jump once per key press (not continuous while held)
    if (jumpPressed && !game.jumpKeyWasPressed) {
      player.jump()
      game.jumpKeyWasPressed = true
    }

    player.sprint(sprintPressed)
    player.duck(duckPressed)

    // Update player physics
    player.update(map.platforms, 1)

    // Update camera (smooth follow)
    const targetCameraX = player.x - 400
    game.cameraX += (targetCameraX - game.cameraX) * 0.1

    // Check coin collection
    map.coins.forEach(coin => {
      player.collectCoin(coin)
    })

    // Update enemies
    map.enemies.forEach(enemy => {
      enemy.x += enemy.direction * enemy.speed

      // Simple AI: turn around at platform edges
      const onPlatform = map.platforms.find(p =>
        enemy.x >= p.x && enemy.x <= p.x + p.width &&
        Math.abs(enemy.y + enemy.height - p.y) < 5
      )

      if (!onPlatform || enemy.x < 0 || enemy.x > map.width) {
        enemy.direction *= -1
      }

      // Check collision with player
      if (player.checkEnemyCollision(enemy)) {
        player.die()
      }
    })

    // Update AI player
    if (enableAI && game.aiPlayer && game.aiAgent && game.aiAgent.isReady()) {
      updateAIPlayer(game, canvas)
    }

    // Update stats display
    setStats({
      score: player.score,
      distance: Math.floor(player.distance),
      aiScore: game.aiPlayer ? game.aiPlayer.score : 0,
      aiDistance: game.aiPlayer ? Math.floor(game.aiPlayer.distance) : 0
    })
  }

  const updateAIPlayer = async (game, canvas) => {
    const { aiPlayer, aiAgent, map } = game

    if (!aiPlayer.isAlive) return

    // AI action cooldown (predict every N frames to reduce computational load)
    game.aiActionCooldown--
    if (game.aiActionCooldown <= 0) {
      game.aiActionCooldown = 3 // Predict every 3 frames

      try {
        // Get AI action prediction
        const action = await aiAgent.predictAction(canvas, aiPlayer.x, game.cameraX)

        // Apply action
        aiPlayer.stopMovement()

        switch (action) {
          case 1: // Left
            aiPlayer.moveLeft()
            break
          case 2: // Right
            aiPlayer.moveRight()
            break
          case 3: // Jump
            aiPlayer.jump()
            break
          case 4: // Sprint + Right
            aiPlayer.sprint(true)
            aiPlayer.moveRight()
            break
          case 5: // Duck
            aiPlayer.duck(true)
            break
          default: // Idle
            break
        }
      } catch (error) {
        console.error('[GAME] AI prediction error:', error)
      }
    }

    // Update AI player physics
    aiPlayer.update(map.platforms, 1)

    // Check coin collection for AI
    map.coins.forEach(coin => {
      aiPlayer.collectCoin(coin)
    })

    // Check enemy collisions for AI
    map.enemies.forEach(enemy => {
      if (aiPlayer.checkEnemyCollision(enemy)) {
        aiPlayer.die()
      }
    })
  }

  const renderGame = (ctx, game) => {
    const { player, map, cameraX } = game
    const canvas = ctx.canvas

    // Clear canvas
    ctx.fillStyle = '#87CEEB' // Sky blue
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Save context
    ctx.save()
    ctx.translate(-cameraX, 0)

    // Draw platforms
    map.platforms.forEach(platform => {
      ctx.fillStyle = platform.type === 'ground' ? '#8B4513' : '#2E8B57'
      ctx.fillRect(platform.x, platform.y, platform.width, platform.height)

      // Platform outline
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.strokeRect(platform.x, platform.y, platform.width, platform.height)
    })

    // Draw coins
    map.coins.forEach(coin => {
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
    map.enemies.forEach(enemy => {
      ctx.fillStyle = '#FF00FF'
      ctx.fillRect(enemy.x, enemy.y, enemy.width, enemy.height)
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.strokeRect(enemy.x, enemy.y, enemy.width, enemy.height)
    })

    // Draw goal
    ctx.fillStyle = '#00FF00'
    ctx.fillRect(map.goal.x, map.goal.y, map.goal.width, map.goal.height)
    ctx.strokeStyle = '#006400'
    ctx.lineWidth = 3
    ctx.strokeRect(map.goal.x, map.goal.y, map.goal.width, map.goal.height)

    // Goal flag
    ctx.fillStyle = '#FFD700'
    ctx.font = 'bold 60px Arial'
    ctx.fillText('🏁', map.goal.x + 5, map.goal.y - 10)

    // Draw player (human)
    ctx.fillStyle = '#4CAF50' // Green for human
    ctx.fillRect(player.x, player.y, player.width, player.height)
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 2
    ctx.strokeRect(player.x, player.y, player.width, player.height)

    // Draw player label
    ctx.fillStyle = '#FFFFFF'
    ctx.font = 'bold 14px Arial'
    ctx.fillText('YOU', player.x + 2, player.y - 5)

    // Draw AI player if enabled
    if (enableAI && game.aiPlayer) {
      ctx.fillStyle = '#FF6B6B' // Red for AI
      ctx.fillRect(game.aiPlayer.x, game.aiPlayer.y, game.aiPlayer.width, game.aiPlayer.height)
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.strokeRect(game.aiPlayer.x, game.aiPlayer.y, game.aiPlayer.width, game.aiPlayer.height)

      // Draw AI label
      ctx.fillStyle = '#FFFFFF'
      ctx.font = 'bold 14px Arial'
      ctx.fillText('AI', game.aiPlayer.x + 6, game.aiPlayer.y - 5)
    }

    // Restore context
    ctx.restore()

    // Draw UI (fixed position)
    if (enableAI && game.aiPlayer) {
      // Split view: Human on left, AI on right
      ctx.fillStyle = 'rgba(76, 175, 80, 0.9)' // Green for human
      ctx.fillRect(10, 10, 300, 100)

      ctx.fillStyle = '#FFFFFF'
      ctx.font = 'bold 18px Arial'
      ctx.fillText('YOU (Human)', 20, 35)
      ctx.font = 'bold 16px Arial'
      ctx.fillText(`Score: ${player.score}`, 20, 60)
      ctx.fillText(`Distance: ${Math.floor(player.distance)}m`, 20, 85)

      ctx.fillStyle = 'rgba(255, 107, 107, 0.9)' // Red for AI
      ctx.fillRect(canvas.width - 310, 10, 300, 100)

      ctx.fillStyle = '#FFFFFF'
      ctx.font = 'bold 18px Arial'
      ctx.fillText('AI Opponent', canvas.width - 290, 35)
      ctx.font = 'bold 16px Arial'
      ctx.fillText(`Score: ${game.aiPlayer.score}`, canvas.width - 290, 60)
      ctx.fillText(`Distance: ${Math.floor(game.aiPlayer.distance)}m`, canvas.width - 290, 85)

      // AI status indicator
      if (aiStatus === 'loading') {
        ctx.fillStyle = 'rgba(255, 193, 7, 0.9)'
        ctx.fillRect(canvas.width - 310, 120, 300, 40)
        ctx.fillStyle = '#000'
        ctx.font = '14px Arial'
        ctx.fillText('AI Loading...', canvas.width - 290, 145)
      } else if (aiStatus === 'error') {
        ctx.fillStyle = 'rgba(244, 67, 54, 0.9)'
        ctx.fillRect(canvas.width - 310, 120, 300, 40)
        ctx.fillStyle = '#FFF'
        ctx.font = '14px Arial'
        ctx.fillText('AI Error - Playing Solo', canvas.width - 290, 145)
      }
    } else {
      // Solo play
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(10, 10, 300, 80)

      ctx.fillStyle = '#FFD700'
      ctx.font = 'bold 20px Arial'
      ctx.fillText(`Score: ${player.score}`, 20, 40)
      ctx.fillText(`Distance: ${Math.floor(player.distance)}m`, 20, 70)
    }

    // Controls hint
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.fillRect(canvas.width - 310, 10, 300, 80)
    ctx.fillStyle = '#FFFFFF'
    ctx.font = '16px Arial'
    ctx.fillText('← → : Move', canvas.width - 290, 35)
    ctx.fillText('Space: Jump', canvas.width - 290, 55)
    ctx.fillText('Shift: Sprint', canvas.width - 290, 75)
  }

  const handlePlayAgain = () => {
    // Go back to intro screen to start a new game
    onGameEnd()
  }

  const handleBackToHome = () => {
    // Navigate to home page
    navigate('/')
  }

  return (
    <div className="game-canvas-wrapper">
      <canvas
        ref={canvasRef}
        width={1920}
        height={1080}
        className="game-canvas"
      />

      {gameState !== 'playing' && (
        <div className="game-overlay">
          <div className="game-result">
            <h2>{gameState === 'won' ? '🎉 You Win!' : (enableAI ? '🤖 AI Wins!' : '💀 Game Over')}</h2>

            {enableAI ? (
              <div className="final-stats-race">
                <div className="player-stats human-stats">
                  <h3>🧑 You</h3>
                  <p>Score: <strong>{stats.score}</strong></p>
                  <p>Distance: <strong>{stats.distance}m</strong></p>
                  <p className={stats.distance > stats.aiDistance ? 'winner' : ''}>
                    {stats.distance > stats.aiDistance ? '👑 Winner' : ''}
                  </p>
                </div>
                <div className="vs-divider">VS</div>
                <div className="player-stats ai-stats">
                  <h3>🤖 AI</h3>
                  <p>Score: <strong>{stats.aiScore}</strong></p>
                  <p>Distance: <strong>{stats.aiDistance}m</strong></p>
                  <p className={stats.aiDistance > stats.distance ? 'winner' : ''}>
                    {stats.aiDistance > stats.distance ? '👑 Winner' : ''}
                  </p>
                </div>
              </div>
            ) : (
              <div className="final-stats">
                <p>Score: <strong>{stats.score}</strong></p>
                <p>Distance: <strong>{stats.distance}m</strong></p>
              </div>
            )}

            <div className="game-buttons">
              <button onClick={handlePlayAgain} className="restart-btn">
                🔄 Race Again
              </button>
              <button onClick={handleBackToHome} className="menu-btn">
                🏠 Back to Menu
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
