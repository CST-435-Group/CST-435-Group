import { useState, useEffect, useRef } from 'react'
import './LogViewer.css'

/**
 * Log Viewer Component
 * Displays training logs and TensorBoard info
 */
function LogViewer({ isTraining }) {
  const [logs, setLogs] = useState('')
  const [logInfo, setLogInfo] = useState(null)
  const [loading, setLoading] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const logContainerRef = useRef(null)

  // Fetch logs
  const fetchLogs = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/rl/training/logs?lines=100')
      const data = await response.json()

      if (data.logs) {
        setLogs(data.logs)
        setLogInfo({
          log_file: data.log_file,
          total_lines: data.total_lines,
          returned_lines: data.returned_lines
        })
      } else if (data.message) {
        setLogs(`[INFO] ${data.message}`)
        if (data.tensorboard_command) {
          setLogInfo({
            tensorboard_command: data.tensorboard_command,
            event_files: data.event_files
          })
        }
      }
    } catch (error) {
      console.error('Failed to fetch logs:', error)
      setLogs(`[ERROR] Failed to fetch logs: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Auto-refresh logs during training
  useEffect(() => {
    fetchLogs()

    let interval
    if (isTraining && autoRefresh) {
      interval = setInterval(fetchLogs, 3000) // Refresh every 3 seconds
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isTraining, autoRefresh])

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleClearLogs = () => {
    setLogs('')
  }

  return (
    <div className="log-viewer">
      <div className="log-header">
        <h3>📋 Training Logs</h3>
        <div className="log-controls">
          <label className="log-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            <span>Auto-refresh</span>
          </label>
          <label className="log-toggle">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>
          <button onClick={fetchLogs} disabled={loading} className="refresh-btn">
            {loading ? '⟳ Loading...' : '↻ Refresh'}
          </button>
          <button onClick={handleClearLogs} className="clear-btn">
            🗑️ Clear
          </button>
        </div>
      </div>

      {/* Log info */}
      {logInfo && (
        <div className="log-info">
          {logInfo.log_file && (
            <div className="log-info-item">
              <span className="info-label">File:</span>
              <span className="info-value">{logInfo.log_file}</span>
            </div>
          )}
          {logInfo.total_lines && (
            <div className="log-info-item">
              <span className="info-label">Lines:</span>
              <span className="info-value">
                Showing {logInfo.returned_lines} of {logInfo.total_lines}
              </span>
            </div>
          )}
          {logInfo.tensorboard_command && (
            <div className="log-info-tensorboard">
              <span className="info-icon">📊</span>
              <div>
                <div className="info-label">TensorBoard Available</div>
                <code className="tensorboard-cmd">{logInfo.tensorboard_command}</code>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Log content */}
      <div className="log-content" ref={logContainerRef}>
        {logs ? (
          <pre className="log-text">{logs}</pre>
        ) : (
          <div className="log-placeholder">
            <p>📋 No logs available yet</p>
            <p className="log-placeholder-detail">
              {isTraining
                ? 'Logs will appear as training progresses...'
                : 'Start training to see logs'}
            </p>
          </div>
        )}
      </div>

      {/* Help text */}
      <div className="log-help">
        <span className="help-icon">💡</span>
        <span className="help-text">
          Logs show training progress, rewards, and episode statistics. For detailed metrics, use TensorBoard.
        </span>
      </div>
    </div>
  )
}

export default LogViewer
