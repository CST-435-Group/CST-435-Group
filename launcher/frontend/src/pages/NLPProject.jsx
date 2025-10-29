import { useState, useEffect } from 'react'
import { nlpAPI, annAPI, cnnAPI } from '../services/api'
import { MessageSquare, Send, AlertCircle, Smile } from 'lucide-react'
import { useModelManager } from '../hooks/useModelManager'

export default function NLPProject() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [examples, setExamples] = useState({})
  const [sentimentScale, setSentimentScale] = useState({})

  // Preload NLP model and unload others when this page loads
  useModelManager(nlpAPI, [annAPI, cnnAPI])

  useEffect(() => {
    loadExamples()
    loadSentimentScale()
  }, [])

  const loadExamples = async () => {
    try {
      const response = await nlpAPI.getExamples()
      setExamples(response.data)
    } catch (err) {
      console.error('Error loading examples:', err)
    }
  }

  const loadSentimentScale = async () => {
    try {
      const response = await nlpAPI.getSentimentScale()
      setSentimentScale(response.data)
    } catch (err) {
      console.error('Error loading sentiment scale:', err)
    }
  }

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await nlpAPI.analyzeSentiment(text)
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error analyzing sentiment')
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (exampleText) => {
    setText(exampleText)
    setResult(null)
    setError(null)
  }

  const getSentimentColor = (score) => {
    if (score === 3) return 'text-green-600 bg-green-100 border-green-400'
    if (score === 2) return 'text-yellow-600 bg-yellow-100 border-yellow-400'
    if (score === 1) return 'text-red-600 bg-red-100 border-red-400'
    return 'text-gray-600 bg-gray-100 border-gray-400'
  }

  // Get top 3 probabilities
  const getTopProbabilities = () => {
    if (!result) return []

    return Object.entries(result.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <div className="flex items-center mb-4">
          <MessageSquare size={48} className="text-purple-600 mr-4" />
          <div>
            <h1 className="text-4xl font-bold text-gray-800">Sentiment Analysis</h1>
            <p className="text-gray-600 text-lg">3-Point Scale NLP Model for Text Classification</p>
          </div>
        </div>
      </div>

      {/* Sentiment Scale */}
      {Object.keys(sentimentScale).length > 0 && (
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
            <Smile className="mr-3 text-purple-600" />
            Sentiment Scale
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {Object.entries(sentimentScale).map(([score, info]) => (
              <div
                key={score}
                className={`rounded-lg p-4 text-center ${getSentimentColor(parseInt(score))}`}
              >
                <div className="text-3xl mb-2">{info.emoji}</div>
                <div className="font-bold text-lg mb-1">{score}</div>
                <div className="text-sm">{info.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
            <Send className="mr-3 text-purple-600" />
            Analyze Text
          </h2>

          <div className="mb-6">
            <label className="block text-gray-700 font-semibold mb-2">Enter Text</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter a hospital review, patient feedback, or healthcare service comment to analyze..."
              rows={6}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
            />
            <p className="text-sm text-gray-500 mt-2">{text.length} / 5000 characters</p>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {loading ? 'Analyzing...' : 'Analyze Sentiment'}
          </button>

          {error && (
            <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg flex items-center">
              <AlertCircle className="mr-2" size={20} />
              {error}
            </div>
          )}

          {/* Example Texts */}
          {Object.keys(examples).length > 0 && (
            <div className="mt-8">
              <h3 className="text-lg font-bold text-gray-800 mb-3">Try Examples:</h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {Object.entries(examples).map(([score, texts]) => (
                  <div key={score}>
                    {texts.map((exampleText, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleExampleClick(exampleText)}
                        className="w-full text-left bg-gray-50 hover:bg-gray-100 px-4 py-2 rounded-lg text-sm text-gray-700 transition-colors mb-2"
                      >
                        <span className="font-semibold mr-2">({score}):</span>
                        {exampleText}
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Analysis Results</h2>

          {result ? (
            <div>
              {/* Main Result */}
              <div className={`border-2 rounded-lg p-6 mb-6 ${getSentimentColor(result.sentiment_score)}`}>
                <div className="text-center mb-4">
                  <div className="text-6xl mb-2">{result.emoji}</div>
                  <p className="text-3xl font-bold mb-2">{result.sentiment_label}</p>
                  <p className="text-xl">Score: {result.sentiment_score}</p>
                  <p className="text-sm">Confidence: {(result.confidence * 100).toFixed(2)}%</p>
                </div>
              </div>

              {/* Analyzed Text */}
              <div className="bg-gray-50 rounded-lg p-4 mb-6">
                <p className="text-gray-700 text-sm font-semibold mb-2">Analyzed Text:</p>
                <p className="text-gray-600 italic">"{result.text}"</p>
              </div>

              {/* Top 3 Probabilities */}
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">Probability Distribution</h3>
                <div className="space-y-3">
                  {getTopProbabilities().map(([label, prob], idx) => (
                    <div key={idx} className="bg-gray-50 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-semibold text-gray-800">{label}</span>
                        <span className="text-purple-600 font-bold">{(prob * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-12">
              <MessageSquare size={64} className="mx-auto mb-4 opacity-50" />
              <p>Enter text and analyze to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
