import axios from 'axios'

// Get API URL from environment or default to proxy
const API_URL = import.meta.env.VITE_API_URL || '/api'

// Create axios instance
const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
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

// General API calls
export const generalAPI = {
  getRoot: () => api.get('/'),
  getHealth: () => api.get('/health'),
}

export default api
