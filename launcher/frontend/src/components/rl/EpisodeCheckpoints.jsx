import { useState, useEffect } from 'react'
import { rlAPI } from '../../services/api'
import './EpisodeCheckpoints.css'

/**
 * Episode Checkpoints Component
 * Displays recent episode checkpoints and allows playing against them
 */
function EpisodeCheckpoints({ isTraining, onPlayAgainst, onModelExported }) {
  const [bestModel, setBestModel] = useState(null)
  const [recentCheckpoints, setRecentCheckpoints] = useState([])
  const [loading, setLoading] = useState(false)
  const [exporting, setExporting] = useState(null) // Episode number being exported
  const [error, setError] = useState(null)

  // Fetch checkpoints
  const fetchCheckpoints = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await rlAPI.getEpisodeCheckpoints()
      setBestModel(response.data.best_model || null)
      setRecentCheckpoints(response.data.recent_checkpoints || [])
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err)
      setError('Failed to load checkpoints')
    } finally {
      setLoading(false)
    }
  }

  // Auto-refresh checkpoints during training
  useEffect(() => {
    fetchCheckpoints()

    let interval
    if (isTraining) {
      interval = setInterval(fetchCheckpoints, 10000) // Refresh every 10 seconds
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isTraining])

  // Export episode checkpoint to ONNX (Recommended)
  const handleExportONNX = async (episode) => {
    setExporting(episode)
    setError(null)

    try {
      // Export the checkpoint to ONNX format
      const response = await rlAPI.exportEpisodeCheckpointONNX(episode)

      // Show success message with more details
      const usageGuide = response.data.usage_guide || 'See documentation'
      alert(
        `✅ Episode ${episode} exported to ONNX successfully!\n\n` +
        `Format: ${response.data.format}\n` +
        `Runtime: ${response.data.runtime}\n` +
        `Model Path: ${response.data.model_path}\n\n` +
        `Check ${usageGuide} for integration instructions.`
      )

      // Refresh checkpoints to show updated status
      await fetchCheckpoints()

      // Notify parent to refresh available models list
      if (onModelExported) {
        onModelExported()
      }

    } catch (err) {
      console.error('Failed to export checkpoint to ONNX:', err)
      const errorMsg = err.response?.data?.detail || err.message || 'Unknown error'
      setError(`Failed to export episode ${episode} to ONNX: ${errorMsg}`)
    } finally {
      setExporting(null)
    }
  }

  // Export episode checkpoint to TensorFlow.js (Legacy - has conversion issues)
  const handleExport = async (episode) => {
    setExporting(episode)
    setError(null)

    try {
      // Export the checkpoint to TensorFlow.js
      const response = await rlAPI.exportEpisodeCheckpoint(episode)

      // Show success message
      alert(`✅ Episode ${episode} exported successfully!\n\nYou can now select it from the model dropdown in the Play Game tab.`)

      // Refresh checkpoints to show updated status
      await fetchCheckpoints()

    } catch (err) {
      console.error('Failed to export checkpoint:', err)
      const errorMsg = err.response?.data?.detail || err.message || 'Unknown error'
      setError(`Failed to export episode ${episode}: ${errorMsg}`)
    } finally {
      setExporting(null)
    }
  }

  if (loading && !bestModel && recentCheckpoints.length === 0) {
    return (
      <div className="episode-checkpoints">
        <h3>Episode Checkpoints</h3>
        <div className="loading">Loading checkpoints...</div>
      </div>
    )
  }

  const renderCheckpointRow = (checkpoint, isBest = false) => (
    <tr key={`${checkpoint.episode}-${isBest ? 'best' : 'recent'}`} className={isBest ? 'best-model-row' : ''}>
      <td className="episode-num">
        #{checkpoint.episode}
        {isBest && <span className="best-badge">🏆 BEST</span>}
      </td>
      <td className={`reward ${checkpoint.reward >= 0 ? 'positive' : 'negative'}`}>
        {checkpoint.reward.toFixed(1)}
      </td>
      <td>{checkpoint.length} steps</td>
      <td>{checkpoint.timestep.toLocaleString()}</td>
      <td className="export-actions">
        <button
          className="play-btn export-onnx-btn"
          onClick={() => handleExportONNX(checkpoint.episode)}
          disabled={exporting !== null}
          title="Export to ONNX format (Recommended - uses ONNX Runtime Web)"
        >
          {exporting === checkpoint.episode ? (
            <span>⏳ Exporting...</span>
          ) : (
            <span>📦 Export (ONNX) ⭐</span>
          )}
        </button>
      </td>
    </tr>
  )

  return (
    <div className="episode-checkpoints">
      <div className="checkpoints-header">
        <h3>🎮 Episode Checkpoints</h3>
        <button onClick={fetchCheckpoints} className="refresh-btn" disabled={loading}>
          ↻ Refresh
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {!bestModel && recentCheckpoints.length === 0 ? (
        <div className="no-checkpoints">
          <p>No episode checkpoints yet.</p>
          <p className="hint">Checkpoints are saved automatically during training.</p>
        </div>
      ) : (
        <div className="checkpoints-list">
          <div className="checkpoints-info">
            {isTraining && <span className="training-badge">⚡ Training Active</span>}
          </div>

          {/* Best Model Section */}
          {bestModel && (
            <>
              <h4 className="section-title">🏆 Best Model (Highest Reward)</h4>
              <table className="checkpoints-table best-model-table">
                <thead>
                  <tr>
                    <th>Episode</th>
                    <th>Reward</th>
                    <th>Length</th>
                    <th>Timestep</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {renderCheckpointRow(bestModel, true)}
                </tbody>
              </table>
            </>
          )}

          {/* Recent Episodes Section */}
          {recentCheckpoints.length > 0 && (
            <>
              <h4 className="section-title">📊 Recent 10 Episodes</h4>
              <table className="checkpoints-table">
                <thead>
                  <tr>
                    <th>Episode</th>
                    <th>Reward</th>
                    <th>Length</th>
                    <th>Timestep</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {recentCheckpoints.map((checkpoint) => renderCheckpointRow(checkpoint, false))}
                </tbody>
              </table>
            </>
          )}

          <div className="checkpoints-legend">
            <p><strong>⭐ ONNX Export (Recommended):</strong> Exports to ONNX format for use with ONNX Runtime Web in the browser.</p>
            <p><strong>Benefits:</strong> ✅ No TensorFlow conversion issues ✅ GPU acceleration via WebGL ✅ Smaller bundle size</p>
            <p>🏆 The best model has the highest reward score across all training episodes.</p>
            <p><strong>Next steps:</strong> After exporting, integrate the model.onnx file with ONNX Runtime Web (see USAGE.md for code examples)</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default EpisodeCheckpoints
