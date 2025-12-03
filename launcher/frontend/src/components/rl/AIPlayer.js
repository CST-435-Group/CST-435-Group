import * as tf from '@tensorflow/tfjs'

/**
 * AI Player Class
 * Handles loading and running the trained RL model for AI gameplay
 */
class AIPlayer {
  constructor() {
    this.model = null
    this.isLoaded = false
    this.inputShape = [84, 84, 3] // Height, Width, Channels
    this.numActions = 6 // idle, left, right, jump, sprint+right, duck
  }

  /**
   * Load the TensorFlow.js model
   * @param {string} modelPath - Path to model.json
   * @returns {Promise<boolean>} Success status
   */
  async loadModel(modelPath = '/models/rl/tfjs_model/model.json') {
    try {
      console.log('[AI] Loading model from:', modelPath)

      this.model = await tf.loadGraphModel(modelPath)
      this.isLoaded = true

      console.log('[AI] Model loaded successfully')
      console.log('[AI] Model inputs:', this.model.inputs.map(i => `${i.name}: ${i.shape}`))
      console.log('[AI] Model outputs:', this.model.outputs.map(o => `${o.name}: ${o.shape}`))

      // Warm up model with dummy input
      await this.warmUp()

      return true
    } catch (error) {
      console.error('[AI] Failed to load model:', error)
      this.isLoaded = false
      return false
    }
  }

  /**
   * Warm up the model with a dummy prediction
   * This helps improve performance for the first real prediction
   */
  async warmUp() {
    if (!this.isLoaded) return

    console.log('[AI] Warming up model...')
    const dummyInput = tf.zeros([1, ...this.inputShape])
    const prediction = await this.model.predict(dummyInput)

    // Clean up
    dummyInput.dispose()
    if (Array.isArray(prediction)) {
      prediction.forEach(t => t.dispose())
    } else {
      prediction.dispose()
    }

    console.log('[AI] Model warm-up complete')
  }

  /**
   * Preprocess canvas frame for model input
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position for camera
   * @param {number} cameraX - Camera X offset
   * @returns {tf.Tensor} Preprocessed tensor [1, 84, 84, 3]
   */
  preprocessFrame(canvas, playerX, cameraX) {
    return tf.tidy(() => {
      // Get image data from canvas
      const ctx = canvas.getContext('2d')

      // Create a view centered on the player
      const viewWidth = 400
      const viewHeight = 300
      const viewX = Math.max(0, playerX - cameraX - viewWidth / 2)
      const viewY = 0

      // Get image data
      const imageData = ctx.getImageData(
        viewX,
        viewY,
        Math.min(viewWidth, canvas.width - viewX),
        Math.min(viewHeight, canvas.height - viewY)
      )

      // Convert to tensor and resize to 84x84
      let tensor = tf.browser.fromPixels(imageData)

      // Resize to model input size (84x84)
      tensor = tf.image.resizeBilinear(tensor, [84, 84])

      // Normalize to [0, 1]
      tensor = tensor.div(255.0)

      // Add batch dimension [1, 84, 84, 3]
      tensor = tensor.expandDims(0)

      return tensor
    })
  }

  /**
   * Predict action from current game state
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @returns {Promise<number>} Action index (0-5)
   */
  async predictAction(canvas, playerX, cameraX) {
    if (!this.isLoaded) {
      console.warn('[AI] Model not loaded')
      return 0 // Default to idle
    }

    try {
      // Preprocess frame
      const inputTensor = this.preprocessFrame(canvas, playerX, cameraX)

      // Run inference
      const prediction = await this.model.predict(inputTensor)

      // Get action probabilities
      let actionProbs
      if (Array.isArray(prediction)) {
        actionProbs = prediction[0]
      } else {
        actionProbs = prediction
      }

      // Apply softmax if not already applied
      const probs = tf.softmax(actionProbs)

      // Get action with highest probability
      const action = await probs.argMax(-1).data()

      // Clean up tensors
      inputTensor.dispose()
      if (Array.isArray(prediction)) {
        prediction.forEach(t => t.dispose())
      } else {
        prediction.dispose()
      }
      probs.dispose()

      return action[0]
    } catch (error) {
      console.error('[AI] Prediction error:', error)
      return 0 // Default to idle on error
    }
  }

  /**
   * Predict action with exploration (stochastic policy)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @param {number} temperature - Sampling temperature (default 1.0)
   * @returns {Promise<number>} Sampled action index
   */
  async predictActionStochastic(canvas, playerX, cameraX, temperature = 1.0) {
    if (!this.isLoaded) {
      return 0
    }

    try {
      const inputTensor = this.preprocessFrame(canvas, playerX, cameraX)
      const prediction = await this.model.predict(inputTensor)

      let actionProbs
      if (Array.isArray(prediction)) {
        actionProbs = prediction[0]
      } else {
        actionProbs = prediction
      }

      // Apply temperature scaling
      const scaledLogits = actionProbs.div(temperature)
      const probs = tf.softmax(scaledLogits)

      // Sample from distribution
      const probsArray = await probs.data()
      const action = this.sampleFromDistribution(probsArray)

      // Clean up
      inputTensor.dispose()
      if (Array.isArray(prediction)) {
        prediction.forEach(t => t.dispose())
      } else {
        prediction.dispose()
      }
      scaledLogits.dispose()
      probs.dispose()

      return action
    } catch (error) {
      console.error('[AI] Stochastic prediction error:', error)
      return 0
    }
  }

  /**
   * Sample action from probability distribution
   * @param {Float32Array} probs - Action probabilities
   * @returns {number} Sampled action index
   */
  sampleFromDistribution(probs) {
    const rand = Math.random()
    let cumSum = 0

    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i]
      if (rand < cumSum) {
        return i
      }
    }

    return probs.length - 1
  }

  /**
   * Convert action index to action name
   * @param {number} actionIndex - Action index (0-5)
   * @returns {string} Action name
   */
  getActionName(actionIndex) {
    const actions = ['idle', 'left', 'right', 'jump', 'sprint_right', 'duck']
    return actions[actionIndex] || 'unknown'
  }

  /**
   * Unload model and free resources
   */
  dispose() {
    if (this.model) {
      this.model.dispose()
      this.model = null
      this.isLoaded = false
      console.log('[AI] Model disposed')
    }
  }

  /**
   * Check if model is loaded and ready
   * @returns {boolean}
   */
  isReady() {
    return this.isLoaded && this.model !== null
  }

  /**
   * Get model info
   * @returns {object} Model information
   */
  getModelInfo() {
    if (!this.isLoaded) {
      return {
        loaded: false,
        message: 'Model not loaded'
      }
    }

    return {
      loaded: true,
      inputShape: this.inputShape,
      numActions: this.numActions,
      backend: tf.getBackend(),
      memory: tf.memory()
    }
  }
}

export default AIPlayer
