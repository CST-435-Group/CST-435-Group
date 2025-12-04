import { useState, useEffect } from 'react'
import './Scoreboard.css'

/**
 * Scoreboard component for displaying and managing player scores
 * Stores scores in localStorage with player names and completion times
 */
export default function Scoreboard({ onNewScore }) {
  const [scores, setScores] = useState([])

  useEffect(() => {
    loadScores()
  }, [])

  // Notify parent of new score
  useEffect(() => {
    if (onNewScore) {
      // Expose function to add score from parent
      onNewScore.current = addScore
    }
  }, [onNewScore])

  const loadScores = () => {
    const storedScores = localStorage.getItem('rl_platformer_scores')
    if (storedScores) {
      try {
        const parsed = JSON.parse(storedScores)
        setScores(parsed.sort((a, b) => a.time - b.time)) // Sort by time (fastest first)
      } catch (error) {
        console.error('Failed to load scores:', error)
        setScores([])
      }
    }
  }

  const addScore = (playerName, time, score, distance, won) => {
    // Only save winning scores
    if (!won) return

    const newScore = {
      name: playerName.trim() || 'Anonymous',
      time: Math.round(time * 10) / 10, // Round to 1 decimal
      score,
      distance,
      timestamp: new Date().toISOString(),
      date: new Date().toLocaleDateString()
    }

    let updatedScores = [...scores]

    // Check if player already exists
    const existingIndex = updatedScores.findIndex(s =>
      s.name.toLowerCase() === newScore.name.toLowerCase()
    )

    if (existingIndex !== -1) {
      // Only update if new time is better (faster)
      if (newScore.time < updatedScores[existingIndex].time) {
        updatedScores[existingIndex] = newScore
      }
    } else {
      // Add new player
      updatedScores.push(newScore)
    }

    // Sort by time (fastest first) and keep top 10
    updatedScores.sort((a, b) => a.time - b.time)
    updatedScores = updatedScores.slice(0, 10)

    // Save to localStorage
    localStorage.setItem('rl_platformer_scores', JSON.stringify(updatedScores))
    setScores(updatedScores)
  }

  const clearScores = () => {
    if (window.confirm('Are you sure you want to clear all scores?')) {
      localStorage.removeItem('rl_platformer_scores')
      setScores([])
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(1)
    return mins > 0 ? `${mins}:${secs.padStart(4, '0')}` : `${secs}s`
  }

  return (
    <div className="scoreboard-container">
      <div className="scoreboard-header">
        <h3>🏆 Leaderboard - Best Times</h3>
        {scores.length > 0 && (
          <button onClick={clearScores} className="clear-scores-btn" title="Clear all scores">
            🗑️ Clear
          </button>
        )}
      </div>

      {scores.length === 0 ? (
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
      </div>
    </div>
  )
}
