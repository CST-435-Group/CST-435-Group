import { useState, useEffect } from 'react'
import TrainingControls from './TrainingControls'
import MetricsChart from './MetricsChart'
import FrameViewer from './FrameViewer'
import LogViewer from './LogViewer'
import { rlAPI } from '../../services/api'
import './TrainingDashboard.css'

/**
 * Training Dashboard Component
 * Main container for AI training UI
 */
function TrainingDashboard({ status }) {
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [isTraining, setIsTraining] = useState(false)
  const [useSSE, setUseSSE] = useState(true)

  // Fetch training status
  useEffect(() => {
    let eventSource = null
    let pollInterval = null

    const fetchStatus = async () => {
      try {
        const response = await rlAPI.getTrainingStatus()
        const data = response.data
        setTrainingStatus(data)
        setIsTraining(data.is_training && data.process_alive)
      } catch (error) {
        console.error('Failed to fetch training status:', error)
      }
    }

    if (useSSE && isTraining) {
      // Use Server-Sent Events for real-time updates
      eventSource = new EventSource(rlAPI.streamTrainingStatus())

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setTrainingStatus(data)
          setIsTraining(data.is_training && data.process_alive)
        } catch (error) {
          console.error('Failed to parse SSE data:', error)
        }
      }

      eventSource.onerror = () => {
        console.log('SSE connection lost, falling back to polling')
        eventSource.close()
        setUseSSE(false)
      }
    } else {
      // Polling fallback
      fetchStatus()
      pollInterval = setInterval(fetchStatus, 2000) // Poll every 2 seconds
    }

    return () => {
      if (eventSource) {
        eventSource.close()
      }
      if (pollInterval) {
        clearInterval(pollInterval)
      }
    }
  }, [isTraining, useSSE])

  const handleTrainingStart = async (params) => {
    try {
      const response = await rlAPI.startTraining(params)
      const data = response.data
      setIsTraining(true)
      setUseSSE(true)
      console.log('Training started:', data)
    } catch (error) {
      console.error('Failed to start training:', error)
      const errorMsg = error.response?.data?.detail || error.message
      alert(`Failed to start training: ${errorMsg}`)
    }
  }

  const handleTrainingStop = async () => {
    try {
      const response = await rlAPI.stopTraining()
      const data = response.data
      setIsTraining(false)
      console.log('Training stopped:', data)
    } catch (error) {
      console.error('Failed to stop training:', error)
      const errorMsg = error.response?.data?.detail || error.message
      alert(`Failed to stop training: ${errorMsg}`)
    }
  }

  return (
    <div className="training-dashboard">
      <div className="dashboard-header">
        <h2>🧠 AI Training Dashboard</h2>
        <p>Configure and monitor reinforcement learning agent training</p>
      </div>

      {/* Training Controls */}
      <TrainingControls
        isTraining={isTraining}
        onStart={handleTrainingStart}
        onStop={handleTrainingStop}
        status={status}
      />

      {/* Training Status & Metrics */}
      {trainingStatus && (
        <div className="training-status-section">
          <div className="status-cards">
            <div className="status-card">
              <div className="card-label">Status</div>
              <div className={`card-value ${isTraining ? 'training' : 'idle'}`}>
                {isTraining ? '🟢 Training' : '⚪ Idle'}
              </div>
              {trainingStatus.training_mode && (
                <div className="card-detail">
                  {trainingStatus.training_mode === 'behavioral_cloning' ? 'Behavioral Cloning' : 'Reinforcement Learning'}
                </div>
              )}
            </div>

            <div className="status-card">
              <div className="card-label">Progress</div>
              <div className="card-value">
                {trainingStatus.progress !== undefined
                  ? `${(trainingStatus.progress * 100).toFixed(1)}%`
                  : '0%'}
              </div>
              {trainingStatus.training_mode === 'behavioral_cloning' ? (
                trainingStatus.current_epoch !== undefined && (
                  <div className="card-detail">
                    Epoch {trainingStatus.current_epoch} / {trainingStatus.total_epochs || 0}
                  </div>
                )
              ) : (
                trainingStatus.current_step !== undefined && (
                  <div className="card-detail">
                    {trainingStatus.current_step.toLocaleString()} / {trainingStatus.total_steps?.toLocaleString() || 0} steps
                  </div>
                )
              )}
            </div>

            {trainingStatus.training_mode === 'behavioral_cloning' ? (
              // Behavioral Cloning metrics
              <>
                <div className="status-card">
                  <div className="card-label">Train Loss</div>
                  <div className="card-value">
                    {trainingStatus.train_loss !== undefined
                      ? trainingStatus.train_loss.toFixed(4)
                      : '0.0000'}
                  </div>
                </div>

                <div className="status-card">
                  <div className="card-label">Val Loss</div>
                  <div className="card-value">
                    {trainingStatus.val_loss !== undefined
                      ? trainingStatus.val_loss.toFixed(4)
                      : '0.0000'}
                  </div>
                  {trainingStatus.best_val_loss !== undefined && (
                    <div className="card-detail">
                      Best: {trainingStatus.best_val_loss.toFixed(4)}
                    </div>
                  )}
                </div>

                <div className="status-card">
                  <div className="card-label">Accuracy</div>
                  <div className="card-value">
                    {trainingStatus.val_accuracy !== undefined
                      ? `${(trainingStatus.val_accuracy * 100).toFixed(1)}%`
                      : '0%'}
                  </div>
                </div>
              </>
            ) : (
              // Reinforcement Learning metrics
              <>
                <div className="status-card">
                  <div className="card-label">Episodes</div>
                  <div className="card-value">
                    {trainingStatus.episodes?.toLocaleString() || 0}
                  </div>
                </div>

                <div className="status-card">
                  <div className="card-label">Avg Reward</div>
                  <div className="card-value">
                    {trainingStatus.avg_reward !== undefined
                      ? trainingStatus.avg_reward.toFixed(2)
                      : '0.00'}
                  </div>
                </div>

                <div className="status-card">
                  <div className="card-label">Best Reward</div>
                  <div className="card-value">
                    {trainingStatus.best_reward !== undefined
                      ? trainingStatus.best_reward.toFixed(2)
                      : '0.00'}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Progress Bar */}
          {trainingStatus.progress !== undefined && (
            <div className="progress-bar-container">
              <div
                className="progress-bar-fill"
                style={{ width: `${trainingStatus.progress * 100}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* Metrics Chart */}
      <MetricsChart trainingStatus={trainingStatus} />

      {/* Frame Viewer */}
      <FrameViewer isTraining={isTraining} />

      {/* Log Viewer */}
      <LogViewer isTraining={isTraining} />
    </div>
  )
}

export default TrainingDashboard
