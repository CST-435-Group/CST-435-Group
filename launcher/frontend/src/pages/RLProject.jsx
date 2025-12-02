import { useState, useEffect } from 'react'
import GameCanvas from '../components/rl/GameCanvas'
import './RLProject.css'

/**
 * RL Platformer Project Page
 * Human vs AI racing game with procedurally generated levels
 */
function RLProject() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [gameStarted, setGameStarted] = useState(false)

  // Check backend status on mount
  useEffect(() => {
    checkStatus()
  }, [])

  const checkStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/rl/status')
      const data = await response.json()
      setStatus(data)
    } catch (error) {
      console.error('Failed to fetch RL status:', error)
      setStatus({ error: 'Backend not available' })
    } finally {
      setLoading(false)
    }
  }

  const startGame = () => {
    setGameStarted(true)
  }

  const resetGame = () => {
    setGameStarted(false)
    // TODO: Reset game state
  }

  if (loading) {
    return (
      <div className="rl-project">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading RL Platformer...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="rl-project">
      <div className="rl-header">
        <h1>🎮 RL Platformer</h1>
        <p>Race against a trained AI agent on procedurally generated levels!</p>
      </div>

      {/* Status Banner */}
      {status && (
        <div className={`status-banner ${status.ready_for_gameplay ? 'ready' : 'not-ready'}`}>
          <div className="status-item">
            <span className="label">Backend:</span>
            <span className={`value ${status.backend_available ? 'success' : 'error'}`}>
              {status.backend_available ? '✓ Available' : '✗ Unavailable'}
            </span>
          </div>
          <div className="status-item">
            <span className="label">GPU:</span>
            <span className={`value ${status.gpu_available ? 'success' : 'warning'}`}>
              {status.gpu_available ? '✓ Available' : 'CPU Only'}
            </span>
          </div>
          <div className="status-item">
            <span className="label">Model:</span>
            <span className={`value ${status.tfjs_model_exists ? 'success' : 'warning'}`}>
              {status.tfjs_model_exists ? '✓ Loaded' : 'Not Trained'}
            </span>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="rl-content">
        {/* Disabled model check for single player - always show game */}
        {false ? (
          <div className="setup-instructions">
            <h2>🤖 Setup Required</h2>
            <p>The AI agent hasn't been trained yet. Follow these steps:</p>

            <div className="steps">
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h3>Install Dependencies</h3>
                  <code>cd RL/backend</code>
                  <code>pip install -r requirements.txt</code>
                </div>
              </div>

              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h3>Train the Agent</h3>
                  <code>python training/train_agent.py</code>
                  <p className="note">⏱️ Training takes 4-8 hours on GPU, 12-24 hours on CPU</p>
                  {status?.gpu_available && (
                    <p className="note success">✓ GPU detected - training will be faster!</p>
                  )}
                </div>
              </div>

              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h3>Export for Web</h3>
                  <code>python training/export_model.py</code>
                </div>
              </div>

              <div className="step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h3>Copy Model to Frontend</h3>
                  <code>cp -r RL/backend/models/tfjs_model/ launcher/frontend/public/models/rl/</code>
                </div>
              </div>

              <div className="step">
                <div className="step-number">5</div>
                <div className="step-content">
                  <h3>Refresh This Page</h3>
                  <p>The game will be ready to play!</p>
                </div>
              </div>
            </div>

            <div className="info-box">
              <h3>📖 About the RL Agent</h3>
              <ul>
                <li><strong>Algorithm:</strong> PPO (Proximal Policy Optimization)</li>
                <li><strong>Input:</strong> Visual observation (84x84 pixels)</li>
                <li><strong>Actions:</strong> Left, Right, Jump, Sprint, Duck, Idle</li>
                <li><strong>Training:</strong> 1 million timesteps on random levels</li>
                <li><strong>Framework:</strong> PyTorch + Stable-Baselines3</li>
              </ul>
            </div>
          </div>
        ) : (
          // Show game (model not required for single player)
          <div className="game-section">
            {!gameStarted ? (
              <div className="game-intro">
                <h2>🎮 Ready to Play!</h2>
                <p className="intro-text">
                  Navigate to the goal flag! The map is randomly generated each time.
                  After testing, we'll train an AI to compete against you!
                </p>

                <div className="controls-info">
                  <h3>🎮 Controls</h3>
                  <div className="control-grid">
                    <div className="control">
                      <kbd>←</kbd> <kbd>→</kbd>
                      <span>Move Left/Right</span>
                    </div>
                    <div className="control">
                      <kbd>Space</kbd> or <kbd>↑</kbd>
                      <span>Jump</span>
                    </div>
                    <div className="control">
                      <kbd>Shift</kbd>
                      <span>Sprint</span>
                    </div>
                    <div className="control">
                      <kbd>↓</kbd>
                      <span>Duck/Slide</span>
                    </div>
                  </div>
                </div>

                <button className="start-button" onClick={startGame}>
                  🎮 Start Race
                </button>

                <div className="features">
                  <div className="feature">
                    <span className="icon">🗺️</span>
                    <div>
                      <strong>Procedural Generation</strong>
                      <p>Different map every race</p>
                    </div>
                  </div>
                  <div className="feature">
                    <span className="icon">🧠</span>
                    <div>
                      <strong>Visual AI</strong>
                      <p>AI "sees" the game like you do</p>
                    </div>
                  </div>
                  <div className="feature">
                    <span className="icon">⚡</span>
                    <div>
                      <strong>Real-time</strong>
                      <p>AI runs in browser, no lag</p>
                    </div>
                  </div>
                  <div className="feature">
                    <span className="icon">🏆</span>
                    <div>
                      <strong>Fair Competition</strong>
                      <p>AI has same abilities as you</p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="game-container">
                <GameCanvas onGameEnd={resetGame} />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Project Info */}
      <div className="project-info">
        <h3>🤔 How It Works</h3>
        <div className="info-grid">
          <div className="info-card">
            <h4>Training Phase</h4>
            <p>
              The AI learns by playing thousands of randomly generated levels.
              It receives rewards for progress, collecting coins, and reaching the goal.
              Over time, it learns optimal strategies for jumping, timing, and navigation.
            </p>
          </div>
          <div className="info-card">
            <h4>Visual Input</h4>
            <p>
              Unlike traditional game AI with direct access to game state, this agent
              sees the game as pixels (84x84 grayscale image). It must learn to recognize
              platforms, enemies, and coins purely from visual observation.
            </p>
          </div>
          <div className="info-card">
            <h4>Browser Inference</h4>
            <p>
              The trained PyTorch model is converted to TensorFlow.js and runs entirely
              in your browser. This means zero latency - the AI makes decisions at 60+ FPS
              without any network calls to a server.
            </p>
          </div>
          <div className="info-card">
            <h4>Procedural Maps</h4>
            <p>
              Levels are generated using Perlin noise for smooth terrain and smart algorithms
              that ensure all platforms are reachable. The AI never sees the same level twice
              during training, so it learns general platforming skills, not memorization.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RLProject
