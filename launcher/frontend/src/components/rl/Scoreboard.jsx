import { useState, useEffect } from 'react'
import { rlAPI } from '../../services/api'
import './Scoreboard.css'

/**
 * Scoreboard component for displaying and managing player scores
 * Uses backend API with shared database for global leaderboard
 */
export default function Scoreboard({ onNewScore, difficulty = 'easy' }) {
  const [scores, setScores] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedDifficulty, setSelectedDifficulty] = useState(difficulty)
  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false)
  const [passwordInput, setPasswordInput] = useState('')
  const [passwordError, setPasswordError] = useState('')

  const ADMIN_PASSWORD = 'John117@home'

  useEffect(() => {
    loadScores()
  }, [selectedDifficulty])

  useEffect(() => {
    setSelectedDifficulty(difficulty)
  }, [difficulty])

  // Notify parent of new score
  useEffect(() => {
    if (onNewScore) {
      // Expose function to add score from parent
      onNewScore.current = addScore
    }
  }, [onNewScore])

  const loadScores = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await rlAPI.getScores(10, selectedDifficulty)
      setScores(response.data.scores || [])
    } catch (err) {
      console.error('Failed to load scores:', err)
      setError('Failed to load leaderboard')
      setScores([])
    } finally {
      setLoading(false)
    }
  }

  const addScore = async (playerName, time, score, distance, won, gameDifficulty) => {
    // Only save winning scores
    if (!won) return

    try {
      const scoreData = {
        name: playerName.trim() || 'Anonymous',
        time: Math.round(time * 10) / 10,
        score,
        distance,
        difficulty: gameDifficulty || 'easy'
      }

      console.log('[Scoreboard] Submitting score:', scoreData)
      const response = await rlAPI.submitScore(scoreData)
      console.log('[Scoreboard] Server response:', response.data)

      // Reload scores to get updated leaderboard
      await loadScores()
    } catch (err) {
      console.error('Failed to submit score:', err)
      // Don't show error to user, just log it
    }
  }

  const handleClearClick = () => {
    setShowPasswordPrompt(true)
    setPasswordInput('')
    setPasswordError('')
  }

  const handlePasswordSubmit = async (e) => {
    e.preventDefault()

    if (passwordInput !== ADMIN_PASSWORD) {
      setPasswordError('Incorrect password. Access denied.')
      setPasswordInput('')
      return
    }

    // Password correct, proceed with clearing
    setShowPasswordPrompt(false)
    setPasswordInput('')
    setPasswordError('')

    if (!window.confirm('Are you sure you want to clear ALL scores from the global leaderboard? This will affect all players!')) {
      return
    }

    try {
      await rlAPI.clearScores()
      await loadScores() // Reload to show empty leaderboard
    } catch (err) {
      console.error('Failed to clear scores:', err)
      alert('Failed to clear scores. Please try again.')
    }
  }

  const handlePasswordCancel = () => {
    setShowPasswordPrompt(false)
    setPasswordInput('')
    setPasswordError('')
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(1)
    return mins > 0 ? `${mins}:${secs.padStart(4, '0')}` : `${secs}s`
  }

  const getDifficultyLabel = (diff) => {
    const labels = {
      easy: '🟢 Easy',
      medium: '🟡 Medium',
      hard: '🔴 Hard'
    }
    return labels[diff] || '🟢 Easy'
  }

  return (
    <div className="scoreboard-container">
      <div className="scoreboard-header">
        <h3>🏆 Global Leaderboard - Best Times</h3>
        <button onClick={handleClearClick} className="clear-scores-btn" title="Clear all scores (requires password)">
          🔒 Clear All
        </button>
      </div>

      {/* Password Prompt Modal */}
      {showPasswordPrompt && (
        <div className="password-modal-overlay" onClick={handlePasswordCancel}>
          <div className="password-modal" onClick={(e) => e.stopPropagation()}>
            <h3>🔒 Admin Access Required</h3>
            <p>Enter password to clear the leaderboard</p>
            <form onSubmit={handlePasswordSubmit}>
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="Enter admin password"
                autoFocus
                className="password-input"
              />
              {passwordError && <div className="password-error">{passwordError}</div>}
              <div className="password-actions">
                <button type="submit" className="password-submit-btn">
                  Clear Scores
                </button>
                <button type="button" onClick={handlePasswordCancel} className="password-cancel-btn">
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Difficulty Tabs */}
      <div className="difficulty-tabs">
        <button
          className={`difficulty-tab ${selectedDifficulty === 'easy' ? 'active' : ''}`}
          onClick={() => setSelectedDifficulty('easy')}
        >
          🟢 Easy
        </button>
        <button
          className={`difficulty-tab ${selectedDifficulty === 'medium' ? 'active' : ''}`}
          onClick={() => setSelectedDifficulty('medium')}
        >
          🟡 Medium
        </button>
        <button
          className={`difficulty-tab ${selectedDifficulty === 'hard' ? 'active' : ''}`}
          onClick={() => setSelectedDifficulty('hard')}
        >
          🔴 Hard
        </button>
      </div>

      {loading ? (
        <div className="no-scores">
          <p>Loading leaderboard...</p>
        </div>
      ) : error ? (
        <div className="no-scores">
          <p style={{ color: '#f44336' }}>{error}</p>
          <button onClick={loadScores} style={{ marginTop: '10px', padding: '8px 16px', cursor: 'pointer' }}>
            Retry
          </button>
        </div>
      ) : scores.length === 0 ? (
        <div className="no-scores">
          <p>No scores yet! Be the first to complete the level.</p>
        </div>
      ) : (
        <div className="scores-table">
          <div className="scores-header">
            <div className="col-rank">Rank</div>
            <div className="col-name">Player</div>
            <div className="col-time">Time</div>
            <div className="col-score">Score</div>
            <div className="col-distance">Distance</div>
            <div className="col-date">Date</div>
          </div>

          {scores.map((score, index) => (
            <div key={index} className={`score-row ${index === 0 ? 'first-place' : ''}`}>
              <div className="col-rank">
                {index === 0 && '🥇'}
                {index === 1 && '🥈'}
                {index === 2 && '🥉'}
                {index > 2 && `#${index + 1}`}
              </div>
              <div className="col-name">{score.name}</div>
              <div className="col-time">{formatTime(score.time)}</div>
              <div className="col-score">{score.score}</div>
              <div className="col-distance">{score.distance}m</div>
              <div className="col-date">{score.date}</div>
            </div>
          ))}
        </div>
      )}

      <div className="scoreboard-footer">
        <p>⏱️ Complete the level as fast as you can to claim your spot!</p>
        <p style={{ fontSize: '0.85rem', color: '#999', marginTop: '5px' }}>
          🌐 Global leaderboard - compete with players worldwide!
        </p>
      </div>
    </div>
  )
}
