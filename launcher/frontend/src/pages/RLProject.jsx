import { useState, useEffect, useRef } from 'react'
import GameCanvas from '../components/rl/GameCanvas'
import TrainingDashboard from '../components/rl/TrainingDashboard'
import EpisodeCheckpoints from '../components/rl/EpisodeCheckpoints'
import Scoreboard from '../components/rl/Scoreboard'
import { rlAPI } from '../services/api'
import './RLProject.css'

/**
 * RL Platformer Project Page
 * Human vs AI racing game with procedurally generated levels + AI Training UI
 */
function RLProject() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [gameStarted, setGameStarted] = useState(false)
  const [enableAI, setEnableAI] = useState(false)
  const [activeTab, setActiveTab] = useState('game') // 'game' or 'training'
  const [isTrainingAuthenticated, setIsTrainingAuthenticated] = useState(false)
  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false)
  const [passwordInput, setPasswordInput] = useState('')
  const [passwordError, setPasswordError] = useState('')
  const [episodeModelPath, setEpisodeModelPath] = useState(null) // Model path for selected episode
  const [playingEpisode, setPlayingEpisode] = useState(null) // Episode number being played
  const [trainingStatus, setTrainingStatus] = useState(null) // Training status for checkpoints
  const [availableModels, setAvailableModels] = useState([]) // List of available models
  const [selectedModel, setSelectedModel] = useState(null) // Currently selected model for gameplay

  // Scoreboard state
  const [showNamePrompt, setShowNamePrompt] = useState(false)
  const [playerName, setPlayerName] = useState('')
  const [nameInput, setNameInput] = useState('')
  const [difficulty, setDifficulty] = useState('easy') // easy, medium, hard
  const [selectedDifficulty, setSelectedDifficulty] = useState('easy')
  const scoreboardRef = useRef(null)

  const TRAINING_PASSWORD = 'John117@home'

  // Check authentication on mount
  useEffect(() => {
    const authenticated = sessionStorage.getItem('rl_training_auth') === 'true'
    setIsTrainingAuthenticated(authenticated)
  }, [])

  // Check backend status on mount
  useEffect(() => {
    checkStatus()
    fetchAvailableModels()
  }, [])

  const fetchAvailableModels = async () => {
    try {
      console.log('[RLProject] Fetching available models...')
      const response = await rlAPI.getAvailableModels()
      console.log('[RLProject] Available models:', response.data.models)
      setAvailableModels(response.data.models || [])
      // Auto-select first model if available
      if (response.data.models && response.data.models.length > 0) {
        const firstModel = response.data.models[0]
        console.log('[RLProject] Auto-selecting first model:', firstModel)
        setSelectedModel(firstModel)
      } else {
        console.log('[RLProject] No models available')
      }
    } catch (error) {
      console.error('[RLProject] Failed to fetch available models:', error)
    }
  }

  // Fetch training status when training tab is active
  useEffect(() => {
    const fetchTrainingStatus = async () => {
      try {
        const response = await rlAPI.getTrainingStatus()
        setTrainingStatus(response.data)
      } catch (error) {
        console.error('Failed to fetch training status:', error)
      }
    }

    if (activeTab === 'training' && isTrainingAuthenticated) {
      fetchTrainingStatus()
      // Poll every 5 seconds while training tab is active
      const interval = setInterval(fetchTrainingStatus, 5000)
      return () => clearInterval(interval)
    }
  }, [activeTab, isTrainingAuthenticated])

  const checkStatus = async () => {
    try {
      const response = await rlAPI.getStatus()
      setStatus(response.data)
    } catch (error) {
      console.error('Failed to fetch RL status:', error)
      setStatus({ error: 'Backend not available' })
    } finally {
      setLoading(false)
    }
  }

  const startGame = () => {
    // Check if name is already locked
    const lockedName = localStorage.getItem('rl_platformer_player_name_locked')

    if (lockedName) {
      // Name is locked, use it directly and start game
      setPlayerName(lockedName)
      setDifficulty('easy')
      setSelectedDifficulty('easy')
      console.log('[RLProject] Using locked name:', lockedName)
      setGameStarted(true)
    } else {
      // Show name prompt for first-time players
      setShowNamePrompt(true)
      const savedName = localStorage.getItem('rl_platformer_player_name')
      setNameInput(savedName || '')
      // Reset difficulty to 'easy' when opening prompt
      setSelectedDifficulty('easy')
    }
  }

  const handleNameSubmit = (e) => {
    e.preventDefault()
    const name = nameInput.trim() || 'Anonymous'
    setPlayerName(name)
    setDifficulty(selectedDifficulty)

    // Lock the name permanently - can't change it after first submission
    localStorage.setItem('rl_platformer_player_name_locked', name)
    localStorage.setItem('rl_platformer_player_name', name)
    setShowNamePrompt(false)

    console.log('[RLProject] Starting game with locked name:', {
      enableAI,
      selectedModel,
      episodeModelPath,
      playingEpisode,
      playerName: name,
      difficulty: selectedDifficulty
    })
    setGameStarted(true)
  }

  const handleNameCancel = () => {
    setShowNamePrompt(false)
    setNameInput('')
  }

  const handleGameComplete = (gameData) => {
    console.log('[RLProject] Game completed:', gameData)

    // Save score if player won
    if (gameData.won && scoreboardRef.current) {
      scoreboardRef.current(playerName, gameData.time, gameData.score, gameData.distance, gameData.won, difficulty)
    }
  }

  const resetGame = () => {
    setGameStarted(false)
    // Clear episode model selection
    setEpisodeModelPath(null)
    setPlayingEpisode(null)
  }

  const handleTrainingTabClick = () => {
    if (!isTrainingAuthenticated) {
      setShowPasswordPrompt(true)
      setPasswordInput('')
      setPasswordError('')
    } else {
      setActiveTab('training')
    }
  }

  const handlePasswordSubmit = (e) => {
    e.preventDefault()
    if (passwordInput === TRAINING_PASSWORD) {
      setIsTrainingAuthenticated(true)
      sessionStorage.setItem('rl_training_auth', 'true')
      setShowPasswordPrompt(false)
      setActiveTab('training')
      setPasswordError('')
    } else {
      setPasswordError('Incorrect password. Access denied.')
      setPasswordInput('')
    }
  }

  const handlePasswordCancel = () => {
    setShowPasswordPrompt(false)
    setPasswordInput('')
    setPasswordError('')
  }

  const handlePlayAgainst = (episode, modelPath) => {
    setEpisodeModelPath(modelPath)
    setPlayingEpisode(episode)
    setGameStarted(true)
    setEnableAI(true)
    setActiveTab('game')
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
            <span className={`value ${status.any_model_exists ? 'success' : 'warning'}`}>
              {status.any_model_exists ? '✓ Loaded' : 'Not Trained'}
            </span>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'game' ? 'active' : ''}`}
          onClick={() => setActiveTab('game')}
        >
          🎮 Play Game
        </button>
        <button
          className={`tab-button ${activeTab === 'training' ? 'active' : ''}`}
          onClick={handleTrainingTabClick}
        >
          🧠 Train AI {!isTrainingAuthenticated && '🔒'}
        </button>
      </div>

      {/* Password Prompt Modal */}
      {showPasswordPrompt && (
        <div className="password-modal-overlay" onClick={handlePasswordCancel}>
          <div className="password-modal" onClick={(e) => e.stopPropagation()}>
            <h3>🔒 Training Access Required</h3>
            <p>Enter password to access AI training controls</p>
            <form onSubmit={handlePasswordSubmit}>
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="Enter password"
                autoFocus
                className="password-input"
              />
              {passwordError && <div className="password-error">{passwordError}</div>}
              <div className="password-actions">
                <button type="submit" className="password-submit-btn">
                  Unlock
                </button>
                <button type="button" onClick={handlePasswordCancel} className="password-cancel-btn">
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Name Prompt Modal */}
      {showNamePrompt && (
        <div className="password-modal-overlay" onClick={handleNameCancel}>
          <div className="password-modal" onClick={(e) => e.stopPropagation()}>
            <h3>🎮 Choose Your Player Name</h3>
            <p style={{ marginBottom: '10px' }}>Enter your name to track your score on the leaderboard!</p>
            <p style={{ color: '#f44336', fontWeight: 'bold', fontSize: '0.9rem', marginBottom: '15px', background: '#ffebee', padding: '10px', borderRadius: '6px' }}>
              ⚠️ WARNING: Your name will be LOCKED and cannot be changed!
            </p>
            <form onSubmit={handleNameSubmit}>
              <input
                type="text"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                placeholder="Your name (or leave blank for Anonymous)"
                autoFocus
                className="password-input"
                maxLength={20}
              />

              <div className="difficulty-selector">
                <label className="difficulty-label">Select Difficulty:</label>
                <div className="difficulty-options">
                  <label className={`difficulty-option ${selectedDifficulty === 'easy' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="difficulty"
                      value="easy"
                      checked={selectedDifficulty === 'easy'}
                      onChange={(e) => setSelectedDifficulty(e.target.value)}
                    />
                    <div className="difficulty-content">
                      <span className="difficulty-name">🟢 Easy</span>
                      <span className="difficulty-desc">Standard length</span>
                    </div>
                  </label>
                  <label className={`difficulty-option ${selectedDifficulty === 'medium' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="difficulty"
                      value="medium"
                      checked={selectedDifficulty === 'medium'}
                      onChange={(e) => setSelectedDifficulty(e.target.value)}
                    />
                    <div className="difficulty-content">
                      <span className="difficulty-name">🟡 Medium</span>
                      <span className="difficulty-desc">1.5x length</span>
                    </div>
                  </label>
                  <label className={`difficulty-option ${selectedDifficulty === 'hard' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="difficulty"
                      value="hard"
                      checked={selectedDifficulty === 'hard'}
                      onChange={(e) => setSelectedDifficulty(e.target.value)}
                    />
                    <div className="difficulty-content">
                      <span className="difficulty-name">🔴 Hard</span>
                      <span className="difficulty-desc">2x length</span>
                    </div>
                  </label>
                </div>
              </div>

              <div className="password-actions">
                <button type="submit" className="password-submit-btn">
                  Start Game
                </button>
                <button type="button" onClick={handleNameCancel} className="password-cancel-btn">
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="rl-content">
        {/* Game Tab */}
        {activeTab === 'game' && (
          <>
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

                {/* AI Mode Toggle */}
                {status?.any_model_exists && (
                  <div className="ai-mode-toggle">
                    <label className="toggle-container">
                      <input
                        type="checkbox"
                        checked={enableAI}
                        onChange={(e) => setEnableAI(e.target.checked)}
                      />
                      <span className="toggle-label">
                        🤖 Race against AI {enableAI ? '(Enabled)' : '(Disabled)'}
                      </span>
                    </label>
                    <p className="toggle-hint">
                      {enableAI
                        ? 'You will race against a trained AI opponent!'
                        : 'Play solo or enable AI for a competitive race'}
                    </p>
                  </div>
                )}

                {/* Model Selector */}
                {enableAI && availableModels.length > 0 && (
                  <div className="model-selector">
                    <label htmlFor="model-select" className="model-selector-label">
                      🎯 Select AI Model:
                    </label>
                    <select
                      id="model-select"
                      className="model-select"
                      value={selectedModel?.id || ''}
                      onChange={(e) => {
                        const model = availableModels.find(m => m.id === e.target.value)
                        setSelectedModel(model)
                      }}
                    >
                      {availableModels.map((model) => (
                        <option key={model.id} value={model.id}>
                          {model.name} - {model.description}
                        </option>
                      ))}
                    </select>
                    {selectedModel && (
                      <p className="model-hint">
                        Playing against: <strong>{selectedModel.name}</strong>
                      </p>
                    )}
                  </div>
                )}

                <button className="start-button" onClick={startGame}>
                  {enableAI ? '🏁 Start Race vs AI' : '🎮 Start Solo Game'}
                </button>

                {/* Scoreboard - below the play button */}
                <Scoreboard onNewScore={scoreboardRef} difficulty={difficulty} />

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
                <GameCanvas
                  onGameEnd={resetGame}
                  enableAI={enableAI}
                  episodeModelPath={episodeModelPath || selectedModel?.path}
                  playingEpisode={playingEpisode}
                  onGameComplete={handleGameComplete}
                  difficulty={difficulty}
                />
              </div>
            )}
          </div>
        )}

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
          </>
        )}

        {/* Training Tab */}
        {activeTab === 'training' && (
          <>
            <TrainingDashboard status={status} />
            <EpisodeCheckpoints
              isTraining={trainingStatus?.is_training && trainingStatus?.process_alive}
              onPlayAgainst={handlePlayAgainst}
              onModelExported={fetchAvailableModels}
            />
          </>
        )}
      </div>
    </div>
  )
}

export default RLProject
