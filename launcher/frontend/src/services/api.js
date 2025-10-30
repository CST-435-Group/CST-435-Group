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

// General API calls
export const generalAPI = {
  getRoot: () => api.get('/'),
  getHealth: () => api.get('/health'),
}

export default api
