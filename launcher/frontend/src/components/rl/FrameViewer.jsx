import { useState, useEffect } from 'react'
import { rlAPI } from '../../services/api'
import './FrameViewer.css'

/**
 * Frame Viewer Component
 * Displays captured training frames showing the AI playing
 */
function FrameViewer({ isTraining }) {
  const [frames, setFrames] = useState([])
  const [selectedFrame, setSelectedFrame] = useState(null)
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Fetch frames
  const fetchFrames = async () => {
    setLoading(true)
    try {
      const response = await rlAPI.getTrainingFrames(20)
      const data = response.data
      if (data.frames) {
        setFrames(data.frames)
        // Auto-select the latest frame if none selected
        if (!selectedFrame && data.frames.length > 0) {
          setSelectedFrame(data.frames[0])
        }
      }
    } catch (error) {
      console.error('Failed to fetch frames:', error)
    } finally {
      setLoading(false)
    }
  }

  // Auto-refresh frames during training
  useEffect(() => {
    fetchFrames()

    let interval
    if (isTraining && autoRefresh) {
      interval = setInterval(fetchFrames, 5000) // Refresh every 5 seconds
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isTraining, autoRefresh])

  const handleFrameSelect = (frame) => {
    setSelectedFrame(frame)
  }

  const getFrameUrl = (frame) => {
    const filename = frame.url.split('/').pop()
    return `${rlAPI.getTrainingFrame(filename)}?t=${Date.now()}`
  }

  return (
    <div className="frame-viewer">
      <div className="viewer-header">
        <h3>🎬 Training Visualization</h3>
        <div className="viewer-controls">
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            <span>Auto-refresh</span>
          </label>
          <button onClick={fetchFrames} disabled={loading} className="refresh-btn">
            {loading ? '⟳ Loading...' : '↻ Refresh'}
          </button>
        </div>
      </div>

      {frames.length > 0 ? (
        <div className="viewer-content">
          {/* Main frame display */}
          <div className="main-frame">
            {selectedFrame ? (
              <>
                <img
                  src={getFrameUrl(selectedFrame)}
                  alt="Training frame"
                  className="frame-image"
                />
                <div className="frame-info">
                  <span className="frame-filename">{selectedFrame.filename}</span>
                  <span className="frame-timestamp">
                    {new Date(selectedFrame.timestamp * 1000).toLocaleString()}
                  </span>
                </div>
              </>
            ) : (
              <div className="frame-placeholder">
                Select a frame to view
              </div>
            )}
          </div>

          {/* Frame thumbnails */}
          <div className="frame-thumbnails">
            <div className="thumbnails-label">Recent Frames</div>
            <div className="thumbnails-grid">
              {frames.map((frame, index) => (
                <div
                  key={frame.filename}
                  className={`thumbnail ${selectedFrame?.filename === frame.filename ? 'selected' : ''}`}
                  onClick={() => handleFrameSelect(frame)}
                >
                  <img
                    src={getFrameUrl(frame)}
                    alt={`Frame ${index}`}
                    className="thumbnail-image"
                  />
                  <div className="thumbnail-label">
                    Frame {frames.length - index}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="viewer-placeholder">
          <div className="placeholder-icon">🎬</div>
          <p>No training frames captured yet</p>
          <p className="placeholder-detail">
            {isTraining
              ? 'Frames will appear as training progresses...'
              : 'Start training to capture gameplay frames'}
          </p>
        </div>
      )}

      {/* Info box */}
      <div className="viewer-info">
        <span className="info-icon">ℹ️</span>
        <span className="info-text">
          The AI sees the game as 84×84 pixel images. Frames are captured at key moments during training.
        </span>
      </div>
    </div>
  )
}

export default FrameViewer
