import { useState, useEffect, useRef } from 'react'
import './MetricsChart.css'

/**
 * Metrics Chart Component
 * Displays real-time training metrics (rewards, episodes, etc.)
 */
function MetricsChart({ trainingStatus }) {
  const [metricsHistory, setMetricsHistory] = useState([])
  const [selectedMetric, setSelectedMetric] = useState('avg_reward')
  const canvasRef = useRef(null)

  // Store metrics history
  useEffect(() => {
    if (trainingStatus && trainingStatus.current_step > 0) {
      setMetricsHistory((prev) => {
        const newPoint = {
          step: trainingStatus.current_step,
          timestamp: trainingStatus.timestamp || Date.now(),
          avg_reward: trainingStatus.avg_reward || 0,
          best_reward: trainingStatus.best_reward || 0,
          episodes: trainingStatus.episodes || 0,
          progress: trainingStatus.progress || 0
        }

        // Keep last 100 points
        const updated = [...prev, newPoint].slice(-100)
        return updated
      })
    }
  }, [trainingStatus])

  // Draw chart
  useEffect(() => {
    if (!canvasRef.current || metricsHistory.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background grid
    ctx.strokeStyle = '#2a2a2a'
    ctx.lineWidth = 1

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = (height - 60) * (i / 5) + 30
      ctx.beginPath()
      ctx.moveTo(50, y)
      ctx.lineTo(width - 20, y)
      ctx.stroke()
    }

    // Get data points
    const dataPoints = metricsHistory.map((m) => m[selectedMetric])
    const maxValue = Math.max(...dataPoints, 1)
    const minValue = Math.min(...dataPoints, 0)
    const range = maxValue - minValue || 1

    // Draw line
    ctx.strokeStyle = '#4CAF50'
    ctx.lineWidth = 2
    ctx.beginPath()

    metricsHistory.forEach((point, i) => {
      const x = 50 + ((width - 70) * i) / (metricsHistory.length - 1 || 1)
      const normalizedValue = (point[selectedMetric] - minValue) / range
      const y = height - 30 - normalizedValue * (height - 60)

      if (i === 0) {
        ctx.moveTo(x, y)
      } else {
        ctx.lineTo(x, y)
      }
    })
    ctx.stroke()

    // Draw points
    ctx.fillStyle = '#4CAF50'
    metricsHistory.forEach((point, i) => {
      const x = 50 + ((width - 70) * i) / (metricsHistory.length - 1 || 1)
      const normalizedValue = (point[selectedMetric] - minValue) / range
      const y = height - 30 - normalizedValue * (height - 60)

      ctx.beginPath()
      ctx.arc(x, y, 3, 0, 2 * Math.PI)
      ctx.fill()
    })

    // Draw axis labels
    ctx.fillStyle = '#999'
    ctx.font = '12px monospace'

    // Y-axis labels
    for (let i = 0; i <= 5; i++) {
      const y = (height - 60) * (i / 5) + 30
      const value = maxValue - (range * i) / 5
      ctx.fillText(value.toFixed(2), 5, y + 4)
    }

    // X-axis label
    ctx.fillText('Steps', width / 2 - 20, height - 5)

    // Min/max labels on chart
    if (metricsHistory.length > 0) {
      const firstStep = metricsHistory[0].step
      const lastStep = metricsHistory[metricsHistory.length - 1].step

      ctx.fillText(firstStep.toLocaleString(), 50, height - 10)
      ctx.fillText(lastStep.toLocaleString(), width - 80, height - 10)
    }
  }, [metricsHistory, selectedMetric])

  return (
    <div className="metrics-chart">
      <div className="chart-header">
        <h3>📊 Training Metrics</h3>
        <div className="metric-selector">
          <button
            className={selectedMetric === 'avg_reward' ? 'active' : ''}
            onClick={() => setSelectedMetric('avg_reward')}
          >
            Avg Reward
          </button>
          <button
            className={selectedMetric === 'best_reward' ? 'active' : ''}
            onClick={() => setSelectedMetric('best_reward')}
          >
            Best Reward
          </button>
          <button
            className={selectedMetric === 'episodes' ? 'active' : ''}
            onClick={() => setSelectedMetric('episodes')}
          >
            Episodes
          </button>
        </div>
      </div>

      {metricsHistory.length > 0 ? (
        <canvas
          ref={canvasRef}
          width={800}
          height={300}
          className="chart-canvas"
        />
      ) : (
        <div className="chart-placeholder">
          <p>📈 Chart will appear when training starts</p>
          <p className="chart-placeholder-detail">
            Metrics are updated in real-time during training
          </p>
        </div>
      )}

      {/* Current Values */}
      {metricsHistory.length > 0 && (
        <div className="chart-summary">
          <div className="summary-item">
            <span className="label">Latest:</span>
            <span className="value">
              {metricsHistory[metricsHistory.length - 1][selectedMetric].toFixed(2)}
            </span>
          </div>
          <div className="summary-item">
            <span className="label">Peak:</span>
            <span className="value">
              {Math.max(...metricsHistory.map((m) => m[selectedMetric])).toFixed(2)}
            </span>
          </div>
          <div className="summary-item">
            <span className="label">Data Points:</span>
            <span className="value">{metricsHistory.length}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default MetricsChart
