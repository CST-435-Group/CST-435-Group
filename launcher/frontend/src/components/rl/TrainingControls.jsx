import { useState } from 'react'
import './TrainingControls.css'

/**
 * Training Controls Component
 * UI for starting/stopping training and configuring parameters
 */
function TrainingControls({ isTraining, onStart, onStop, status }) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Training mode selection
  const [trainingMode, setTrainingMode] = useState('behavioral_cloning')

  // RL parameters
  const [timesteps, setTimesteps] = useState(1000000)

  // BC parameters
  const [epochs, setEpochs] = useState(100)
  const [learningRate, setLearningRate] = useState(0.001)
  const [batchSize, setBatchSize] = useState(256)
  const [valSplit, setValSplit] = useState(0.2)

  // Legacy RL parameters (kept for compatibility)
  const [nEpochs, setNEpochs] = useState(10)
  const [mapLength, setMapLength] = useState(50)
  const [difficulty, setDifficulty] = useState('medium')
  const [progressWeight, setProgressWeight] = useState(1.0)
  const [coinWeight, setCoinWeight] = useState(0.5)
  const [goalWeight, setGoalWeight] = useState(100.0)
  const [deathPenalty, setDeathPenalty] = useState(-10.0)
  const [captureFrames, setCaptureFrames] = useState(true)

  const handleStart = () => {
    if (trainingMode === 'behavioral_cloning') {
      // Behavioral cloning parameters
      const params = {
        training_mode: 'behavioral_cloning',
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        val_split: valSplit
      }
      onStart(params)
    } else {
      // Reinforcement learning parameters
      const params = {
        training_mode: 'reinforcement_learning',
        timesteps,
        learning_rate: 0.0003,
        batch_size: 64,
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

      {/* Training Mode Selector */}
      <div className="control-section">
        <div className="control-group">
          <label htmlFor="trainingMode">
            Training Mode
            <span className="control-info">Choose how to train the AI</span>
          </label>
          <select
            id="trainingMode"
            value={trainingMode}
            onChange={(e) => setTrainingMode(e.target.value)}
            disabled={isTraining}
            className="training-mode-select"
          >
            <option value="behavioral_cloning">
              Behavioral Cloning (Learn from Human Gameplay)
            </option>
            <option value="reinforcement_learning">
              Reinforcement Learning (Self-Play with PPO)
            </option>
          </select>
          <div className="mode-description">
            {trainingMode === 'behavioral_cloning' ? (
              <p>🎮 Trains AI to mimic human players using collected gameplay data. Fast and sample-efficient.</p>
            ) : (
              <p>🤖 Trains AI through trial and error in the environment. Requires long training times.</p>
            )}
          </div>
        </div>

        {/* Behavioral Cloning Controls */}
        {trainingMode === 'behavioral_cloning' && (
          <>
            <div className="control-group">
              <label htmlFor="epochs">
                Training Epochs
                <span className="control-info">Number of passes through the dataset</span>
              </label>
              <div className="input-with-presets">
                <input
                  type="number"
                  id="epochs"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  min="10"
                  max="1000"
                  step="10"
                  disabled={isTraining}
                />
                <div className="preset-buttons">
                  <button onClick={() => setEpochs(50)} disabled={isTraining}>50</button>
                  <button onClick={() => setEpochs(100)} disabled={isTraining}>100</button>
                  <button onClick={() => setEpochs(200)} disabled={isTraining}>200</button>
                  <button onClick={() => setEpochs(500)} disabled={isTraining}>500</button>
                </div>
              </div>
              <div className="control-estimate">
                Estimated time: {(epochs / 10).toFixed(1)} minutes
              </div>
            </div>

            <div className="control-group">
              <label htmlFor="bcBatchSize">
                Batch Size
                <span className="control-info">Number of samples per training batch</span>
              </label>
              <input
                type="number"
                id="bcBatchSize"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                min="32"
                max="1024"
                step="32"
                disabled={isTraining}
              />
            </div>

            <div className="control-group">
              <label htmlFor="bcLearningRate">
                Learning Rate
                <span className="control-info">Step size for model updates</span>
              </label>
              <input
                type="number"
                id="bcLearningRate"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                min="0.00001"
                max="0.01"
                step="0.0001"
                disabled={isTraining}
              />
            </div>

            <div className="control-group">
              <label htmlFor="valSplit">
                Validation Split
                <span className="control-info">Fraction of data used for validation</span>
              </label>
              <input
                type="number"
                id="valSplit"
                value={valSplit}
                onChange={(e) => setValSplit(Number(e.target.value))}
                min="0.1"
                max="0.5"
                step="0.05"
                disabled={isTraining}
              />
            </div>
          </>
        )}

        {/* Reinforcement Learning Controls */}
        {trainingMode === 'reinforcement_learning' && (
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
        )}

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
