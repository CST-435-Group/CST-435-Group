import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { MapGenerator } from './MapGenerator'
import { Player } from './Player'
import './GameCanvas.css'

/**
 * Main game canvas component
 * Handles rendering, input, and game loop
 */
export default function GameCanvas({ onGameEnd }) {
  const canvasRef = useRef(null)
  const navigate = useNavigate()
  const [gameState, setGameState] = useState('playing') // playing, won, lost
  const [stats, setStats] = useState({ score: 0, distance: 0 })

  // Game objects
  const gameRef = useRef({
    player: null,
    map: null,
    cameraX: 0,
    keys: {},
    lastTime: 0,
    animationFrame: null
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
      if (game.player.checkGoalReached(game.map.goal)) {
        setGameState('won')
        setStats({ score: game.player.score, distance: Math.floor(game.player.distance) })
      } else if (!game.player.isAlive) {
        setGameState('lost')
        setStats({ score: game.player.score, distance: Math.floor(game.player.distance) })
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

    // Update stats display
    setStats({ score: player.score, distance: Math.floor(player.distance) })
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

    // Draw player
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(player.x, player.y, player.width, player.height)
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 2
    ctx.strokeRect(player.x, player.y, player.width, player.height)

    // Restore context
    ctx.restore()

    // Draw UI (fixed position)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    ctx.fillRect(10, 10, 300, 80)

    ctx.fillStyle = '#FFD700'
    ctx.font = 'bold 20px Arial'
    ctx.fillText(`Score: ${player.score}`, 20, 40)
    ctx.fillText(`Distance: ${Math.floor(player.distance)}m`, 20, 70)

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
            <h2>{gameState === 'won' ? '🎉 You Win!' : '💀 Game Over'}</h2>
            <div className="final-stats">
              <p>Score: <strong>{stats.score}</strong></p>
              <p>Distance: <strong>{stats.distance}m</strong></p>
            </div>
            <div className="game-buttons">
              <button onClick={handlePlayAgain} className="restart-btn">
                🔄 Play Again
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
