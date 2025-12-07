import * as tf from '@tensorflow/tfjs'
import ort from './onnxInit'  // Use pre-configured ONNX Runtime

/**
 * AI Player Class
 * Handles loading and running the trained RL model for AI gameplay
 * Supports both TensorFlow.js and ONNX models
 */
class AIPlayer {
  constructor() {
    this.model = null
    this.session = null // For ONNX Runtime
    this.isLoaded = false
    this.isLoading = false // Prevent concurrent loads
    this.modelFormat = null // 'tfjs' or 'onnx'
    this.inputShape = [84, 84, 3] // Height, Width, Channels (for TensorFlow.js)
    this.numActions = 6 // Default for TensorFlow.js models
  }

  /**
   * Load the model (automatically detects TensorFlow.js vs ONNX)
   * @param {string} modelPath - Path to model.json or model.onnx
   * @param {Function} onProgress - Progress callback (progress, message)
   * @returns {Promise<boolean>} Success status
   */
  async loadModel(modelPath = '/models/rl/tfjs_model/model.json', onProgress = null) {
    // Prevent concurrent loads
    if (this.isLoading) {
      console.log('[AI] Model is already loading, skipping duplicate request')
      return this.isLoaded
    }

    this.isLoading = true

    try {
      console.log('[AI] Loading model from:', modelPath)
      if (onProgress) onProgress(0, 'Starting model load...')

      // Detect model format from file extension
      if (modelPath.endsWith('.onnx')) {
        return await this.loadONNXModel(modelPath, onProgress)
      } else {
        return await this.loadTFJSModel(modelPath, onProgress)
      }
    } catch (error) {
      console.error('[AI] Failed to load model:', error)
      if (onProgress) onProgress(100, `Error: ${error.message}`)
      this.isLoaded = false
      return false
    } finally {
      this.isLoading = false
    }
  }

  /**
   * Load TensorFlow.js model
   * @param {string} modelPath - Path to model.json
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<boolean>} Success status
   */
  async loadTFJSModel(modelPath, onProgress = null) {
    try {
      console.log('[AI] Loading TensorFlow.js model...')
      if (onProgress) onProgress(10, 'Fetching TensorFlow.js model...')

      this.model = await tf.loadGraphModel(modelPath)
      if (onProgress) onProgress(60, 'Model loaded, initializing...')

      this.modelFormat = 'tfjs'
      this.inputShape = [84, 84, 3] // HWC format
      this.numActions = 6 // TensorFlow.js models have 6 actions
      this.isLoaded = true

      console.log('[AI] TensorFlow.js model loaded successfully')
      console.log('[AI] Model inputs:', this.model.inputs.map(i => `${i.name}: ${i.shape}`))
      console.log('[AI] Model outputs:', this.model.outputs.map(o => `${o.name}: ${o.shape}`))

      // Warm up model
      if (onProgress) onProgress(80, 'Warming up model...')
      await this.warmUp()

      if (onProgress) onProgress(100, 'Ready!')
      return true
    } catch (error) {
      console.error('[AI] Failed to load TensorFlow.js model:', error)
      this.isLoaded = false
      return false
    }
  }

  /**
   * Load ONNX model
   * @param {string} modelPath - Path to model.onnx
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<boolean>} Success status
   */
  async loadONNXModel(modelPath, onProgress = null) {
    try {
      console.log('[AI] Loading ONNX model...')
      console.log('[AI] Model path:', modelPath)
      if (onProgress) onProgress(10, 'Fetching ONNX model file...')
      console.log('[AI] Fetching model from:', modelPath)
      console.log('[AI] ONNX Runtime already configured (via onnxInit.js)')
      console.log('[AI] WASM path:', ort.env.wasm.wasmPaths)
      console.log('[AI] Proxy disabled:', !ort.env.wasm.proxy)

      // Fetch the .onnx file as ArrayBuffer
      console.log('[AI] Fetching .onnx file...')
      const modelResponse = await fetch(modelPath)
      if (!modelResponse.ok) {
        throw new Error(`Failed to fetch model: ${modelResponse.status} ${modelResponse.statusText}`)
      }
      const modelBuffer = await modelResponse.arrayBuffer()
      console.log('[AI] Model file fetched:', modelBuffer.byteLength, 'bytes')

      if (onProgress) onProgress(20, 'Fetching model weights...')

      // Also fetch the .onnx.data file (external weights)
      const dataPath = modelPath + '.data'
      console.log('[AI] Fetching .onnx.data file from:', dataPath)
      const dataResponse = await fetch(dataPath)

      let sessionOptions = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      }

      if (dataResponse.ok) {
        const dataBuffer = await dataResponse.arrayBuffer()
        console.log('[AI] External data file fetched:', dataBuffer.byteLength, 'bytes')

        // Provide external data to ONNX Runtime
        sessionOptions.externalData = [
          {
            data: new Uint8Array(dataBuffer),
            path: 'model.onnx.data'
          }
        ]
        if (onProgress) onProgress(30, 'Loading model with external weights...')
      } else {
        console.log('[AI] No external data file found, model may be self-contained')
        if (onProgress) onProgress(30, 'Loading model...')
      }

      // Create session with the model buffer
      console.log('[AI] Creating InferenceSession with WASM-only backend...')
      this.session = await ort.InferenceSession.create(modelBuffer, sessionOptions)

      console.log('[AI] InferenceSession created successfully')
      if (onProgress) onProgress(70, 'Model loaded, initializing...')

      this.modelFormat = 'onnx'
      this.inputShape = [3, 84, 84] // CHW format for ONNX
      this.numActions = 9 // ONNX models from new training have 9 actions
      this.isLoaded = true

      console.log('[AI] ONNX model loaded successfully')
      console.log('[AI] Input names:', this.session.inputNames)
      console.log('[AI] Output names:', this.session.outputNames)
      console.log('[AI] Execution provider: WebAssembly (WASM)')

      // Warm up model (with error handling for concurrent session issues)
      if (onProgress) onProgress(85, 'Warming up model with test inference...')
      console.log('[AI] Starting model warm-up...')
      try {
        await this.warmUp()
        console.log('[AI] Model warm-up complete')
      } catch (warmupError) {
        // If warmup fails (e.g., concurrent session calls in React StrictMode),
        // log it but don't fail the entire load - model is still usable
        console.warn('[AI] Warm-up failed (non-critical):', warmupError.message)
        console.log('[AI] Model loaded successfully despite warm-up issue')
      }

      if (onProgress) onProgress(100, 'AI Ready!')
      return true
    } catch (error) {
      console.error('[AI] Failed to load ONNX model:', error)
      console.error('[AI] Error details:', error.message)
      console.error('[AI] Stack trace:', error.stack)
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

    if (this.modelFormat === 'onnx') {
      // ONNX warm-up
      const dummyInput = new Float32Array(1 * 3 * 84 * 84).fill(0)
      const tensor = new ort.Tensor('float32', dummyInput, [1, 3, 84, 84])
      const feeds = { [this.session.inputNames[0]]: tensor }
      await this.session.run(feeds)
    } else {
      // TensorFlow.js warm-up
      const dummyInput = tf.zeros([1, ...this.inputShape])
      const prediction = await this.model.predict(dummyInput)

      // Clean up
      dummyInput.dispose()
      if (Array.isArray(prediction)) {
        prediction.forEach(t => t.dispose())
      } else {
        prediction.dispose()
      }
    }

    console.log('[AI] Model warm-up complete')
  }

  /**
   * Preprocess canvas frame for TensorFlow.js model (HWC format)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position for camera
   * @param {number} cameraX - Camera X offset
   * @returns {tf.Tensor} Preprocessed tensor [1, 84, 84, 3]
   */
  preprocessFrameTFJS(canvas, playerX, cameraX) {
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
   * Preprocess canvas frame for ONNX model (CHW format)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position for camera
   * @param {number} cameraX - Camera X offset
   * @returns {ort.Tensor} Preprocessed ONNX tensor [1, 3, 84, 84]
   */
  preprocessFrameONNX(canvas, playerX, cameraX) {
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

    // Create temporary canvas for resizing
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = 84
    tempCanvas.height = 84
    const tempCtx = tempCanvas.getContext('2d')

    // Draw and resize to 84x84
    const srcCanvas = document.createElement('canvas')
    srcCanvas.width = imageData.width
    srcCanvas.height = imageData.height
    srcCanvas.getContext('2d').putImageData(imageData, 0, 0)
    tempCtx.drawImage(srcCanvas, 0, 0, imageData.width, imageData.height, 0, 0, 84, 84)

    // Get resized image data
    const resizedData = tempCtx.getImageData(0, 0, 84, 84)
    const pixels = resizedData.data

    // Convert to CHW format: [1, 3, 84, 84]
    const input = new Float32Array(1 * 3 * 84 * 84)

    for (let i = 0; i < 84; i++) {
      for (let j = 0; j < 84; j++) {
        const idx = (i * 84 + j) * 4
        const r = pixels[idx] / 255.0
        const g = pixels[idx + 1] / 255.0
        const b = pixels[idx + 2] / 255.0

        // HWC → CHW conversion
        input[0 * 84 * 84 + i * 84 + j] = r  // Red channel
        input[1 * 84 * 84 + i * 84 + j] = g  // Green channel
        input[2 * 84 * 84 + i * 84 + j] = b  // Blue channel
      }
    }

    return new ort.Tensor('float32', input, [1, 3, 84, 84])
  }

  /**
   * Preprocess canvas frame (automatically uses correct format)
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position for camera
   * @param {number} cameraX - Camera X offset
   * @returns {tf.Tensor|ort.Tensor} Preprocessed tensor
   */
  preprocessFrame(canvas, playerX, cameraX) {
    if (this.modelFormat === 'onnx') {
      return this.preprocessFrameONNX(canvas, playerX, cameraX)
    } else {
      return this.preprocessFrameTFJS(canvas, playerX, cameraX)
    }
  }

  /**
   * Predict action from current game state
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @returns {Promise<number>} Action index
   */
  async predictAction(canvas, playerX, cameraX) {
    if (!this.isLoaded) {
      console.warn('[AI] Model not loaded')
      return 0 // Default to idle
    }

    try {
      if (this.modelFormat === 'onnx') {
        return await this.predictActionONNX(canvas, playerX, cameraX)
      } else {
        return await this.predictActionTFJS(canvas, playerX, cameraX)
      }
    } catch (error) {
      console.error('[AI] Prediction error:', error)
      return 0 // Default to idle on error
    }
  }

  /**
   * Predict action using TensorFlow.js model
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @returns {Promise<number>} Action index (0-5)
   */
  async predictActionTFJS(canvas, playerX, cameraX) {
    try {
      // Preprocess frame
      const inputTensor = this.preprocessFrameTFJS(canvas, playerX, cameraX)

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
      console.error('[AI] TensorFlow.js prediction error:', error)
      return 0
    }
  }

  /**
   * Predict action using ONNX model
   * @param {HTMLCanvasElement} canvas - Game canvas
   * @param {number} playerX - Player X position
   * @param {number} cameraX - Camera X offset
   * @returns {Promise<number>} Action index (0-8)
   */
  async predictActionONNX(canvas, playerX, cameraX) {
    try {
      // Preprocess frame
      const inputTensor = this.preprocessFrameONNX(canvas, playerX, cameraX)

      // Run inference
      const feeds = { [this.session.inputNames[0]]: inputTensor }
      const results = await this.session.run(feeds)

      // Get logits from output
      const outputName = this.session.outputNames[0]
      const logits = results[outputName].data

      // Apply softmax
      const maxLogit = Math.max(...logits)
      const expValues = Array.from(logits).map(x => Math.exp(x - maxLogit))
      const sumExp = expValues.reduce((a, b) => a + b, 0)
      const probs = expValues.map(x => x / sumExp)

      // Get action with highest probability (greedy)
      const action = probs.indexOf(Math.max(...probs))

      // Debug: Log probabilities occasionally
      if (Math.random() < 0.05) { // 5% chance to log
        console.log('[AI-PREDICT] Logits:', Array.from(logits).map(x => x.toFixed(2)))
        console.log('[AI-PREDICT] Probs:', probs.map(x => (x * 100).toFixed(1) + '%'))
        console.log('[AI-PREDICT] Selected action:', action, '| Top 3:',
          probs.map((p, i) => ({action: i, prob: p}))
            .sort((a, b) => b.prob - a.prob)
            .slice(0, 3)
            .map(x => `${x.action}:${(x.prob * 100).toFixed(1)}%`).join(', ')
        )
      }

      return action
    } catch (error) {
      console.error('[AI] ONNX prediction error:', error)
      return 0
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
    }
    if (this.session) {
      this.session = null
    }
    this.isLoaded = false
    this.modelFormat = null
    console.log('[AI] Model disposed')
  }

  /**
   * Check if model is loaded and ready
   * @returns {boolean}
   */
  isReady() {
    return this.isLoaded && (this.model !== null || this.session !== null)
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

    const info = {
      loaded: true,
      format: this.modelFormat,
      inputShape: this.inputShape,
      numActions: this.numActions
    }

    if (this.modelFormat === 'tfjs') {
      info.backend = tf.getBackend()
      info.memory = tf.memory()
    } else if (this.modelFormat === 'onnx') {
      info.runtime = 'ONNX Runtime Web'
      info.executionProviders = ['webgl', 'wasm']
    }

    return info
  }
}

export default AIPlayer
