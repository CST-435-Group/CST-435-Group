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

  // User authentication state
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [authMode, setAuthMode] = useState('login') // 'login' or 'register'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [authError, setAuthError] = useState('')
  const [authToken, setAuthToken] = useState(null)
  const [currentUser, setCurrentUser] = useState(null)
  const [difficulty, setDifficulty] = useState('easy') // easy, medium, hard
  const [selectedDifficulty, setSelectedDifficulty] = useState('easy')
  const scoreboardRef = useRef(null)

  // Profile settings modal state
  const [showProfileModal, setShowProfileModal] = useState(false)
  const [profileFormData, setProfileFormData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
    newUsername: '',
    playerColor: ''
  })
  const [profileError, setProfileError] = useState('')
  const [profileSuccess, setProfileSuccess] = useState('')

  const TRAINING_PASSWORD = 'John117@home'

  // Check authentication on mount
  useEffect(() => {
    const authenticated = sessionStorage.getItem('rl_training_auth') === 'true'
    setIsTrainingAuthenticated(authenticated)

    // Check for saved auth token
    const savedToken = localStorage.getItem('rl_auth_token')
    if (savedToken) {
      verifyAndLoadUser(savedToken)
    }
  }, [])

  const verifyAndLoadUser = async (token) => {
    try {
      const response = await rlAPI.verifyToken(token)
      setAuthToken(token)
      setCurrentUser(response.data.user)
    } catch (error) {
      console.error('Token verification failed:', error)
      localStorage.removeItem('rl_auth_token')
      setAuthToken(null)
      setCurrentUser(null)
    }
  }

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
    // Check if user is logged in
    if (!currentUser) {
      // Show login/register modal
      setShowAuthModal(true)
      setAuthMode('login')
      setAuthError('')
      return
    }

    // User is authenticated, start game
    setDifficulty(selectedDifficulty)

    const modelPathToPass = episodeModelPath || selectedModel?.path

    console.log('\n🎮 ═══════════ GAME START ═══════════')
    console.log('[RLProject] Starting game with configuration:')
    console.log('  enableAI:', enableAI)
    console.log('  selectedModel:', selectedModel)
    console.log('  selectedModel.path:', selectedModel?.path)
    console.log('  episodeModelPath:', episodeModelPath)
    console.log('  Final model path to pass:', modelPathToPass)
    console.log('  playingEpisode:', playingEpisode)
    console.log('  username:', currentUser.username)
    console.log('  difficulty:', selectedDifficulty)
    console.log('🎮 ════════════════════════════════════\n')

    setGameStarted(true)
  }

  const handleAuthSubmit = async (e) => {
    e.preventDefault()
    setAuthError('')

    try {
      if (authMode === 'register') {
        // Register new user
        const response = await rlAPI.register(username, password)
        const { token, user } = response.data

        // Save token and user info
        localStorage.setItem('rl_auth_token', token)
        setAuthToken(token)
        setCurrentUser(user)
        setShowAuthModal(false)

        // Start game automatically after registration
        setDifficulty(selectedDifficulty)
        setGameStarted(true)
      } else {
        // Login existing user
        const response = await rlAPI.login(username, password)
        const { token, user } = response.data

        // Save token and user info
        localStorage.setItem('rl_auth_token', token)
        setAuthToken(token)
        setCurrentUser(user)
        setShowAuthModal(false)

        // Start game automatically after login
        setDifficulty(selectedDifficulty)
        setGameStarted(true)
      }
    } catch (error) {
      console.error('Auth error:', error)
      setAuthError(error.response?.data?.detail || 'Authentication failed. Please try again.')
      setPassword('')
    }
  }

  const handleAuthCancel = () => {
    setShowAuthModal(false)
    setUsername('')
    setPassword('')
    setAuthError('')
  }

  const handleLogout = () => {
    localStorage.removeItem('rl_auth_token')
    setAuthToken(null)
    setCurrentUser(null)
    setGameStarted(false)
  }

  const handleOpenProfile = () => {
    setProfileFormData({
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
      newUsername: currentUser?.username || '',
      playerColor: currentUser?.player_color || '#4287f5'
    })
    setProfileError('')
    setProfileSuccess('')
    setShowProfileModal(true)
  }

  const handleProfileSubmit = async (e) => {
    e.preventDefault()
    setProfileError('')
    setProfileSuccess('')

    try {
      const updateData = {}

      // Only include fields that were changed
      if (profileFormData.newPassword) {
        if (!profileFormData.currentPassword) {
          setProfileError('Current password required to change password')
          return
        }
        if (profileFormData.newPassword !== profileFormData.confirmPassword) {
          setProfileError('New passwords do not match')
          return
        }
        updateData.current_password = profileFormData.currentPassword
        updateData.new_password = profileFormData.newPassword
      }

      if (profileFormData.newUsername && profileFormData.newUsername !== currentUser.username) {
        updateData.new_username = profileFormData.newUsername
      }

      if (profileFormData.playerColor && profileFormData.playerColor !== currentUser.player_color) {
        updateData.player_color = profileFormData.playerColor
      }

      if (Object.keys(updateData).length === 0) {
        setProfileError('No changes to save')
        return
      }

      const response = await rlAPI.updateProfile(updateData, authToken)

      // Update current user with new data
      setCurrentUser({
        ...currentUser,
        username: response.data.user.username,
        player_color: response.data.user.player_color
      })

      setProfileSuccess(response.data.message)

      // Clear password fields
      setProfileFormData({
        ...profileFormData,
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      })

      // Close modal after 2 seconds
      setTimeout(() => {
        setShowProfileModal(false)
      }, 2000)
    } catch (error) {
      console.error('Profile update error:', error)
      setProfileError(error.response?.data?.detail || 'Failed to update profile. Please try again.')
    }
  }

  const handleGameComplete = async (gameData) => {
    console.log('[RLProject] Game completed:', gameData)

    // Submit metrics if authenticated
    if (currentUser && authToken) {
      try {
        await rlAPI.submitMetrics({
          jumps: gameData.jumps || 0,
          points: gameData.score || 0,
          distance: gameData.distance || 0,
          time_played: gameData.time || 0
        }, authToken)
        console.log('[RLProject] Metrics submitted successfully')
      } catch (error) {
        console.error('[RLProject] Failed to submit metrics:', error)
      }
    }

    // Save score if player won
    if (gameData.won && scoreboardRef.current && currentUser && authToken) {
      try {
        // Request completion token from backend (proof that player reached goal)
        console.log('[RLProject] Requesting completion token...')
        const tokenResponse = await rlAPI.requestCompletionToken(authToken)
        const completionToken = tokenResponse.data.completion_token
        console.log('[RLProject] Got completion token, submitting score...')

        // Submit score with completion token
        scoreboardRef.current(
          currentUser.username,
          gameData.time,
          gameData.score,
          gameData.distance,
          gameData.won,
          difficulty,
          authToken,
          completionToken  // NEW: Pass completion token
        )
      } catch (error) {
        console.error('[RLProject] Failed to get completion token or submit score:', error)
        setError('Failed to submit score. Please try again.')
      }
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

      {/* Login/Register Modal */}
      {showAuthModal && (
        <div className="password-modal-overlay" onClick={handleAuthCancel}>
          <div className="password-modal" onClick={(e) => e.stopPropagation()}>
            <h3>{authMode === 'login' ? '🔐 Login' : '📝 Create Account'}</h3>
            <p style={{ marginBottom: '20px', color: '#666' }}>
              {authMode === 'login'
                ? 'Login to track your stats and compete on the leaderboard!'
                : 'Create an account to save your progress and stats!'}
            </p>
            <form onSubmit={handleAuthSubmit}>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username (3-20 characters)"
                autoFocus
                className="password-input"
                minLength={3}
                maxLength={20}
                required
              />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password (minimum 6 characters)"
                className="password-input"
                minLength={6}
                required
                style={{ marginTop: '10px' }}
              />

              {authError && (
                <div className="password-error" style={{ marginTop: '10px' }}>
                  {authError}
                </div>
              )}

              <div className="password-actions" style={{ marginTop: '20px' }}>
                <button type="submit" className="password-submit-btn">
                  {authMode === 'login' ? 'Login' : 'Create Account'}
                </button>
                <button type="button" onClick={handleAuthCancel} className="password-cancel-btn">
                  Cancel
                </button>
              </div>
            </form>

            <div style={{ marginTop: '15px', textAlign: 'center', borderTop: '1px solid #eee', paddingTop: '15px' }}>
              {authMode === 'login' ? (
                <p style={{ margin: 0, color: '#666' }}>
                  Don't have an account?{' '}
                  <button
                    onClick={() => {
                      setAuthMode('register')
                      setAuthError('')
                      setUsername('')
                      setPassword('')
                    }}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#667eea',
                      cursor: 'pointer',
                      fontWeight: 'bold',
                      textDecoration: 'underline'
                    }}
                  >
                    Register here
                  </button>
                </p>
              ) : (
                <p style={{ margin: 0, color: '#666' }}>
                  Already have an account?{' '}
                  <button
                    onClick={() => {
                      setAuthMode('login')
                      setAuthError('')
                      setUsername('')
                      setPassword('')
                    }}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#667eea',
                      cursor: 'pointer',
                      fontWeight: 'bold',
                      textDecoration: 'underline'
                    }}
                  >
                    Login here
                  </button>
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Profile Settings Modal */}
      {showProfileModal && (
        <div className="password-modal-overlay" onClick={() => setShowProfileModal(false)}>
          <div className="password-modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '500px' }}>
            <h3>⚙️ Profile Settings</h3>
            <p style={{ marginBottom: '20px', color: '#666' }}>
              Update your username, password, or player color
            </p>
            <form onSubmit={handleProfileSubmit}>
              {/* Username Change */}
              <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#333' }}>
                  Username
                </label>
                <input
                  type="text"
                  value={profileFormData.newUsername}
                  onChange={(e) => setProfileFormData({ ...profileFormData, newUsername: e.target.value })}
                  placeholder="Enter new username"
                  className="password-input"
                  minLength={3}
                  maxLength={20}
                />
                <p style={{ fontSize: '0.85rem', color: '#999', margin: '3px 0 0 0' }}>
                  This will update your name on the leaderboard
                </p>
              </div>

              {/* Player Color */}
              <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#333' }}>
                  Player Color
                </label>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <input
                    type="color"
                    value={profileFormData.playerColor}
                    onChange={(e) => setProfileFormData({ ...profileFormData, playerColor: e.target.value })}
                    style={{
                      width: '60px',
                      height: '40px',
                      border: '2px solid #ddd',
                      borderRadius: '6px',
                      cursor: 'pointer'
                    }}
                  />
                  <div style={{
                    width: '40px',
                    height: '40px',
                    backgroundColor: profileFormData.playerColor,
                    border: '2px solid #333',
                    borderRadius: '4px'
                  }}></div>
                  <span style={{ color: '#666', fontSize: '0.9rem' }}>{profileFormData.playerColor}</span>
                </div>
                <p style={{ fontSize: '0.85rem', color: '#999', margin: '3px 0 0 0' }}>
                  Your player square will be this color in-game
                </p>
              </div>

              {/* Password Change Section */}
              <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '2px solid #eee' }}>
                <h4 style={{ margin: '0 0 10px 0', color: '#333' }}>Change Password (Optional)</h4>

                <div style={{ marginBottom: '10px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#333' }}>
                    Current Password
                  </label>
                  <input
                    type="password"
                    value={profileFormData.currentPassword}
                    onChange={(e) => setProfileFormData({ ...profileFormData, currentPassword: e.target.value })}
                    placeholder="Enter current password"
                    className="password-input"
                  />
                </div>

                <div style={{ marginBottom: '10px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#333' }}>
                    New Password
                  </label>
                  <input
                    type="password"
                    value={profileFormData.newPassword}
                    onChange={(e) => setProfileFormData({ ...profileFormData, newPassword: e.target.value })}
                    placeholder="Enter new password (min 6 chars)"
                    className="password-input"
                    minLength={6}
                  />
                </div>

                <div style={{ marginBottom: '10px' }}>
                  <label style={{ display: 'block', marginBottom: '5px', fontWeight: 'bold', color: '#333' }}>
                    Confirm New Password
                  </label>
                  <input
                    type="password"
                    value={profileFormData.confirmPassword}
                    onChange={(e) => setProfileFormData({ ...profileFormData, confirmPassword: e.target.value })}
                    placeholder="Confirm new password"
                    className="password-input"
                    minLength={6}
                  />
                </div>
              </div>

              {profileError && (
                <div className="password-error" style={{ marginTop: '15px' }}>
                  {profileError}
                </div>
              )}

              {profileSuccess && (
                <div style={{
                  marginTop: '15px',
                  padding: '10px',
                  background: '#e8f5e9',
                  border: '1px solid #4caf50',
                  borderRadius: '6px',
                  color: '#2e7d32',
                  fontWeight: 'bold'
                }}>
                  ✓ {profileSuccess}
                </div>
              )}

              <div className="password-actions" style={{ marginTop: '20px' }}>
                <button type="submit" className="password-submit-btn">
                  Save Changes
                </button>
                <button type="button" onClick={() => setShowProfileModal(false)} className="password-cancel-btn">
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
                        console.log('🎯 [MODEL-SELECT] User selected model:', model)
                        console.log('[MODEL-SELECT] Model ID:', model?.id)
                        console.log('[MODEL-SELECT] Model name:', model?.name)
                        console.log('[MODEL-SELECT] Model path:', model?.path)
                        console.log('[MODEL-SELECT] Model format:', model?.format)
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

                {/* Difficulty Selector - always visible */}
                <div className="difficulty-selector-main">
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

                {/* User Info Display */}
                {currentUser && (
                  <div style={{
                    background: 'white',
                    borderRadius: '10px',
                    padding: '15px 20px',
                    margin: '20px 0',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <div>
                      <p style={{ margin: 0, fontSize: '0.9rem', color: '#666' }}>Logged in as</p>
                      <p
                        onClick={handleOpenProfile}
                        style={{
                          margin: '3px 0 0 0',
                          fontSize: '1.2rem',
                          fontWeight: 'bold',
                          color: '#667eea',
                          cursor: 'pointer',
                          textDecoration: 'underline',
                          transition: 'color 0.2s'
                        }}
                        onMouseEnter={(e) => e.target.style.color = '#4a5fcc'}
                        onMouseLeave={(e) => e.target.style.color = '#667eea'}
                        title="Click to edit profile"
                      >
                        {currentUser.username}
                      </p>
                    </div>
                    <button
                      onClick={handleLogout}
                      style={{
                        background: '#f44336',
                        color: 'white',
                        border: 'none',
                        padding: '8px 16px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: 'bold',
                        fontSize: '0.9rem',
                        transition: 'all 0.2s'
                      }}
                      onMouseOver={(e) => e.target.style.background = '#da190b'}
                      onMouseOut={(e) => e.target.style.background = '#f44336'}
                    >
                      Logout
                    </button>
                  </div>
                )}

                <button className="start-button" onClick={startGame}>
                  {currentUser
                    ? (enableAI ? '🏁 Start Race vs AI' : '🎮 Start Solo Game')
                    : '🔐 Login to Play'
                  }
                </button>

                {/* Scoreboard - below the play button */}
                <Scoreboard
                  onNewScore={scoreboardRef}
                  difficulty={difficulty}
                  authToken={authToken}
                  currentUser={currentUser}
                />

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
                  playerColor={currentUser?.player_color}
                  username={currentUser?.username || 'Anonymous'}
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
