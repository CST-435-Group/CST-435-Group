import axios from 'axios'

// Get API URL from environment or default to proxy
const API_URL = import.meta.env.VITE_API_URL || '/api'

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  timeout: 300000, // 5 minutes for long-running operations
  headers: {
    'Content-Type': 'application/json',
  },
})

// ANN API calls
export const annAPI = {
  getInfo: () => api.get('/ann/'),
  getHealth: () => api.get('/ann/health'),
  getDataInfo: () => api.get('/ann/data-info'),
  selectTeam: (data) => api.post('/ann/select-team', data),
  getPlayers: (limit = 20) => api.get(`/ann/players?limit=${limit}`),
  preload: () => api.post('/ann/preload'),
  unload: () => api.post('/ann/unload'),
}

// CNN API calls
export const cnnAPI = {
  getInfo: () => api.get('/cnn/'),
  getHealth: () => api.get('/cnn/health'),
  getModelInfo: () => api.get('/cnn/info'),
  getFruitList: () => api.get('/cnn/fruit-list'),
  predictImage: (file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post('/cnn/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  predictBase64: (imageBase64) => api.post('/cnn/predict-base64', { image_base64: imageBase64 }),
  getSampleImages: (perFruit = 5) => api.get(`/cnn/sample-images?per_fruit=${perFruit}`),
  getSampleImageUrl: (fruit, filename) => `${API_URL}/cnn/sample-image/${fruit}/${filename}`,
  predictSampleImage: (fruit, filename) => api.post(`/cnn/predict-sample/${fruit}/${filename}`),
  preload: () => api.post('/cnn/preload'),
  unload: () => api.post('/cnn/unload'),
}

// NLP API calls
export const nlpAPI = {
  getInfo: () => api.get('/nlp/'),
  getHealth: () => api.get('/nlp/health'),
  analyzeSentiment: (text) => api.post('/nlp/analyze', { text }),
  analyzeBatch: (texts) => api.post('/nlp/analyze/batch', { texts }),
  getExamples: () => api.get('/nlp/examples'),
  getSentimentScale: () => api.get('/nlp/sentiment-scale'),
  preload: () => api.post('/nlp/preload'),
  unload: () => api.post('/nlp/unload'),
}

// RNN API calls
export const rnnAPI = {
  getInfo: () => api.get('/rnn/'),
  getHealth: () => api.get('/rnn/health'),
  getModelInfo: () => api.get('/rnn/model/info'),
  generateText: (data) => api.post('/rnn/generate', data),
  generateTextStream: async (data, onToken, onPunctuation, onComplete, onError) => {
    const url = `${API_URL}/rnn/generate/stream`
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonData = JSON.parse(line.slice(6))

            if (jsonData.type === 'token') {
              onToken(jsonData.word, jsonData.index, jsonData.grammar_score)
            } else if (jsonData.type === 'punctuation') {
              onPunctuation(jsonData.formatted_text)
            } else if (jsonData.type === 'done') {
              onComplete(jsonData.full_text)
            } else if (jsonData.type === 'error') {
              onError(jsonData.message)
            }
          }
        }
      }
    } catch (error) {
      onError(error.message)
    }
  },
  testModel: (useBeamSearch = true, beamWidth = 5) =>
    api.get(`/rnn/model/test?use_beam_search=${useBeamSearch}&beam_width=${beamWidth}`),
  getAvailableModels: () => api.get('/rnn/models/available'),
  switchModel: (data) => api.post('/rnn/models/switch', data),
  getTechnicalReport: () => api.get('/rnn/technical-report'),
  preload: () => api.post('/rnn/preload'),
  unload: () => api.post('/rnn/unload'),
}

// GAN API calls
export const ganAPI = {
  getInfo: () => api.get('/gan/'),
  getHealth: () => api.get('/gan/health'),
  getModelInfo: () => api.get('/gan/info'),
  getAvailableModels: () => api.get('/gan/models'),
  switchModel: (modelName) => api.post('/gan/models/switch', null, { params: { model_name: modelName } }),
  getTanks: () => api.get('/gan/tanks'),
  getViews: () => api.get('/gan/views'),
  generate: (data) => api.post('/gan/generate', data),
  preload: () => api.post('/gan/preload'),
  unload: () => api.post('/gan/unload'),
}

// RL API calls
export const rlAPI = {
  getInfo: () => api.get('/rl/'),
  getStatus: () => api.get('/rl/status'),
  getGpuInfo: () => api.get('/rl/gpu/info'),
  getModelInfo: () => api.get('/rl/model/info'),
  getAvailableModels: () => api.get('/rl/models/available'),
  exportModel: () => api.post('/rl/model/export'),
  startTraining: (params) => api.post('/rl/training/start', null, { params }),
  stopTraining: () => api.post('/rl/training/stop'),
  getTrainingStatus: () => api.get('/rl/training/status'),
  getTrainingLogs: (lines = 100) => api.get(`/rl/training/logs?lines=${lines}`),
  getTrainingFrames: (limit = 10) => api.get(`/rl/training/frames?limit=${limit}`),
  getTrainingFrame: (filename) => `${API_URL}/rl/training/frame/${filename}`,
  // SSE stream endpoint - use raw fetch since it's not JSON
  streamTrainingStatus: () => `${API_URL}/rl/training/stream`,
  // Episode checkpoints
  getEpisodeCheckpoints: () => api.get('/rl/checkpoints/episodes'),
  exportEpisodeCheckpoint: (episode) => api.post(`/rl/checkpoints/export/${episode}`),
  exportEpisodeCheckpointONNX: (episode) => api.post(`/rl/checkpoints/export-onnx/${episode}`),
  // Scoreboard
  getScores: (limit = 10, difficulty = 'easy') => api.get(`/rl/scores?limit=${limit}&difficulty=${difficulty}`),
  submitScore: (scoreData) => api.post('/rl/scores', scoreData),
  clearScores: () => api.delete('/rl/scores'),
  deleteScore: (playerName, difficulty) => api.delete(`/rl/scores/${encodeURIComponent(playerName)}/${difficulty}`),
}

// General API calls
export const generalAPI = {
  getRoot: () => api.get('/'),
  getHealth: () => api.get('/health'),
}

export default api

// Docs API
export const docsAPI = {
  getTechnical: (projectId) => api.get(`/docs/${projectId}/technical`),
  getCost: (projectId) => api.get(`/docs/${projectId}/cost`),
  getCostJson: (projectId) => api.get(`/docs/${projectId}/cost/json`),
}
