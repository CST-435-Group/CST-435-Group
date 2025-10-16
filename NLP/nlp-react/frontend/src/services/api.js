/**
 * API Service for Sentiment Analyzer
 * Handles all communication with the FastAPI backend
 */

import axios from 'axios';

// Configure base URL - uses proxy in development, environment variable in production
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      console.error('API Error:', error.response.status, error.response.data);
    } else if (error.request) {
      console.error('Network Error:', error.message);
    } else {
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

/**
 * Check API health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('Failed to connect to API');
  }
};

/**
 * Analyze sentiment of a single text
 * @param {string} text - Text to analyze
 * @returns {Promise<Object>} Sentiment analysis result
 */
export const analyzeSentiment = async (text) => {
  try {
    const response = await api.post('/analyze', { text });
    return response.data;
  } catch (error) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to analyze sentiment');
  }
};

/**
 * Analyze sentiment of multiple texts in batch
 * @param {string[]} texts - Array of texts to analyze
 * @returns {Promise<Array>} Array of sentiment analysis results
 */
export const analyzeBatch = async (texts) => {
  try {
    const response = await api.post('/analyze/batch', { texts });
    return response.data;
  } catch (error) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error('Failed to analyze batch');
  }
};

/**
 * Get example reviews
 * @returns {Promise<Object>} Example reviews for each sentiment level
 */
export const getExamples = async () => {
  try {
    const response = await api.get('/examples');
    return response.data;
  } catch (error) {
    throw new Error('Failed to load examples');
  }
};

/**
 * Get sentiment scale information
 * @returns {Promise<Object>} Sentiment scale with labels and emojis
 */
export const getSentimentScale = async () => {
  try {
    const response = await api.get('/sentiment-scale');
    return response.data;
  } catch (error) {
    throw new Error('Failed to load sentiment scale');
  }
};

export default api;
