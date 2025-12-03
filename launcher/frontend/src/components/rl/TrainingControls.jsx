import { useState } from 'react'
import './TrainingControls.css'

/**
 * Training Controls Component
 * UI for starting/stopping training and configuring parameters
 */
function TrainingControls({ isTraining, onStart, onStop, status }) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Basic parameters
  const [timesteps, setTimesteps] = useState(1000000)

  // Learning parameters
  const [learningRate, setLearningRate] = useState(0.0003)
  const [batchSize, setBatchSize] = useState(64)
  const [nEpochs, setNEpochs] = useState(10)

  // Environment parameters
  const [mapLength, setMapLength] = useState(50)
  const [difficulty, setDifficulty] = useState('medium')

  // Reward weights
  const [progressWeight, setProgressWeight] = useState(1.0)
  const [coinWeight, setCoinWeight] = useState(0.5)
  const [goalWeight, setGoalWeight] = useState(100.0)
  const [deathPenalty, setDeathPenalty] = useState(-10.0)

  // Visualization
  const [captureFrames, setCaptureFrames] = useState(true)

  const handleStart = () => {
    const params = {
      timesteps,
      learning_rate: learningRate,
      batch_size: batchSize,
      n_epochs: nEpochs,
      map_length: mapLength,
      difficulty,
      reward_weights: {
        progress: progressWeight,
        coin: coinWeight,
        goal: goalWeight,
        death: deathPenalty
      },
      capture_frames: captureFrames
    }
    onStart(params)
  }

  return (
    <div className="training-controls">
      <div className="controls-header">
        <h3>⚙️ Training Configuration</h3>
        <button
          className="toggle-advanced-btn"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? '▼ Hide' : '▶ Show'} Advanced Settings
        </button>
      </div>

      {/* Basic Controls */}
      <div className="control-section">
        <div className="control-group">
          <label htmlFor="timesteps">
            Training Timesteps
            <span className="control-info">Total number of environment steps</span>
          </label>
          <div className="input-with-presets">
            <input
              type="number"
              id="timesteps"
              value={timesteps}
              onChange={(e) => setTimesteps(Number(e.target.value))}
              min="10000"
              max="10000000"
              step="10000"
              disabled={isTraining}
            />
            <div className="preset-buttons">
              <button onClick={() => setTimesteps(100000)} disabled={isTraining}>100K</button>
              <button onClick={() => setTimesteps(500000)} disabled={isTraining}>500K</button>
              <button onClick={() => setTimesteps(1000000)} disabled={isTraining}>1M</button>
              <button onClick={() => setTimesteps(2000000)} disabled={isTraining}>2M</button>
            </div>
          </div>
          <div className="control-estimate">
            Estimated time: {(timesteps / 50000).toFixed(1)} hours on GPU
          </div>
        </div>

        {showAdvanced && (
          <>
            {/* Learning Parameters */}
            <div className="control-subsection">
              <h4>Learning Parameters</h4>

              <div className="control-group">
                <label htmlFor="learningRate">
                  Learning Rate
                  <span className="control-info">Step size for policy updates</span>
                </label>
                <input
                  type="number"
                  id="learningRate"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  min="0.00001"
                  max="0.01"
                  step="0.0001"
                  disabled={isTraining}
                />
              </div>

              <div className="control-group">
                <label htmlFor="batchSize">
                  Batch Size
                  <span className="control-info">Number of samples per update</span>
                </label>
                <input
                  type="number"
                  id="batchSize"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  min="16"
                  max="512"
                  step="16"
                  disabled={isTraining}
                />
              </div>

              <div className="control-group">
                <label htmlFor="nEpochs">
                  Epochs per Update
                  <span className="control-info">Number of passes through batch</span>
                </label>
                <input
                  type="number"
                  id="nEpochs"
                  value={nEpochs}
                  onChange={(e) => setNEpochs(Number(e.target.value))}
                  min="1"
                  max="20"
                  disabled={isTraining}
                />
              </div>
            </div>

            {/* Environment Parameters */}
            <div className="control-subsection">
              <h4>Environment Parameters</h4>

              <div className="control-group">
                <label htmlFor="mapLength">
                  Map Length (platforms)
                  <span className="control-info">Number of platforms to generate</span>
                </label>
                <input
                  type="number"
                  id="mapLength"
                  value={mapLength}
                  onChange={(e) => setMapLength(Number(e.target.value))}
                  min="20"
                  max="100"
                  disabled={isTraining}
                />
              </div>

              <div className="control-group">
                <label htmlFor="difficulty">
                  Difficulty
                  <span className="control-info">Gap sizes and platform spacing</span>
                </label>
                <select
                  id="difficulty"
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                  disabled={isTraining}
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </div>
            </div>

            {/* Reward Weights */}
            <div className="control-subsection">
              <h4>Reward Shaping</h4>

              <div className="control-group">
                <label htmlFor="progressWeight">
                  Progress Reward
                  <span className="control-info">Reward per pixel moved forward</span>
                </label>
                <input
                  type="number"
                  id="progressWeight"
                  value={progressWeight}
                  onChange={(e) => setProgressWeight(Number(e.target.value))}
                  min="0"
                  max="10"
                  step="0.1"
                  disabled={isTraining}
                />
              </div>

              <div className="control-group">
                <label htmlFor="coinWeight">
                  Coin Reward
                  <span className="control-info">Reward per coin collected</span>
                </label>
                <input
                  type="number"
                  id="coinWeight"
                  value={coinWeight}
                  onChange={(e) => setCoinWeight(Number(e.target.value))}
                  min="0"
                  max="10"
                  step="0.1"
                  disabled={isTraining}
                />
              </div>

              <div className="control-group">
                <label htmlFor="goalWeight">
                  Goal Reward
                  <span className="control-info">Reward for reaching goal flag</span>
                </label>
                <input
                  type="number"
                  id="goalWeight"
                  value={goalWeight}
                  onChange={(e) => setGoalWeight(Number(e.target.value))}
                  min="0"
                  max="1000"
                  step="10"
                  disabled={isTraining}
                />
              </div>

              <div className="control-group">
                <label htmlFor="deathPenalty">
                  Death Penalty
                  <span className="control-info">Penalty for falling or hitting enemies</span>
                </label>
                <input
                  type="number"
                  id="deathPenalty"
                  value={deathPenalty}
                  onChange={(e) => setDeathPenalty(Number(e.target.value))}
                  min="-100"
                  max="0"
                  step="1"
                  disabled={isTraining}
                />
              </div>
            </div>

            {/* Visualization Options */}
            <div className="control-subsection">
              <h4>Visualization Options</h4>

              <div className="control-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    checked={captureFrames}
                    onChange={(e) => setCaptureFrames(e.target.checked)}
                    disabled={isTraining}
                  />
                  <span>Capture training frames (for visualization)</span>
                </label>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Action Buttons */}
      <div className="control-actions">
        {!isTraining ? (
          <button
            className="start-training-btn"
            onClick={handleStart}
            disabled={!status?.backend_available}
          >
            🚀 Start Training
          </button>
        ) : (
          <button className="stop-training-btn" onClick={onStop}>
            ⏹️ Stop Training
          </button>
        )}

        {!status?.backend_available && (
          <div className="error-message">
            ⚠️ Backend not available. Start the backend server first.
          </div>
        )}
      </div>
    </div>
  )
}

export default TrainingControls
