import { useState, useEffect } from 'react'
import { rnnAPI, annAPI, cnnAPI, nlpAPI } from '../services/api'
import { MessageCircle, Send, AlertCircle, Sparkles, BookOpen, FileText } from 'lucide-react'
import { useModelManager } from '../hooks/useModelManager'
import ReactMarkdown from 'react-markdown'

export default function RNNProject() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [seedText, setSeedText] = useState('In the beginning')
  const [numWords, setNumWords] = useState(50)
  const [temperature, setTemperature] = useState(1.0)
  const [useBeamSearch, setUseBeamSearch] = useState(false)
  const [beamWidth, setBeamWidth] = useState(5)
  const [lengthPenalty, setLengthPenalty] = useState(1.0)
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.2)
  const [addPunctuation, setAddPunctuation] = useState(false)
  const [validateGrammar, setValidateGrammar] = useState(false)
  const [generatedText, setGeneratedText] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [activeTab, setActiveTab] = useState('generate') // 'generate' or 'report'
  const [technicalReport, setTechnicalReport] = useState('')
  const [loadingReport, setLoadingReport] = useState(false)
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [switchingModel, setSwitchingModel] = useState(false)
  const [modelSwitchMessage, setModelSwitchMessage] = useState(null)

  // Preload RNN model and unload others when this page loads
  useModelManager(rnnAPI, [annAPI, cnnAPI, nlpAPI])

  useEffect(() => {
    loadModelInfo()
    loadAvailableModels()
  }, [])

  const loadModelInfo = async () => {
    try {
      const response = await rnnAPI.getModelInfo()
      setModelInfo(response.data)
      if (response.data.current_model) {
        setSelectedModel(response.data.current_model)
      }
    } catch (err) {
      console.error('Error loading model info:', err)
    }
  }

  const loadAvailableModels = async () => {
    try {
      const response = await rnnAPI.getAvailableModels()
      setAvailableModels(response.data.models || [])
      if (response.data.current_model) {
        setSelectedModel(response.data.current_model)
      }
    } catch (err) {
      console.error('Error loading available models:', err)
    }
  }

  const handleModelSwitch = async (modelName) => {
    if (modelName === selectedModel) return

    setSwitchingModel(true)
    setModelSwitchMessage(null)
    setError(null)

    try {
      const response = await rnnAPI.switchModel({ model_name: modelName })
      setSelectedModel(modelName)
      setModelSwitchMessage({ type: 'success', text: response.data.message })

      // Reload model info after switching
      await loadModelInfo()
    } catch (err) {
      const errorMsg = err.response?.data?.detail || 'Error switching model'
      setError(errorMsg)
      setModelSwitchMessage({ type: 'error', text: errorMsg })
    } finally {
      setSwitchingModel(false)
    }
  }

  const loadTechnicalReport = async () => {
    if (technicalReport) return // Already loaded

    setLoadingReport(true)
    try {
      const response = await rnnAPI.getTechnicalReport()
      setTechnicalReport(response.data.content)
    } catch (err) {
      console.error('Error loading technical report:', err)
      setError('Failed to load technical report')
    } finally {
      setLoadingReport(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'report') {
      loadTechnicalReport()
    }
  }, [activeTab])

  const handleGenerate = async () => {
    if (!seedText.trim()) {
      setError('Please enter seed text')
      return
    }

    setLoading(true)
    setError(null)
    setGeneratedText(seedText) // Start with seed text

    const requestData = {
      seed_text: seedText,
      num_words: numWords,
      temperature: temperature,
      use_beam_search: useBeamSearch,
      beam_width: beamWidth,
      length_penalty: lengthPenalty,
      repetition_penalty: repetitionPenalty,
      beam_temperature: 0.0,
      add_punctuation: addPunctuation,
      validate_grammar: validateGrammar
    }

    // Use regular API for beam search, streaming for sampling
    if (useBeamSearch) {
      try {
        const response = await rnnAPI.generateText(requestData)
        setGeneratedText(response.data.generated_text)
      } catch (err) {
        setError(err.response?.data?.detail || 'Error generating text')
      } finally {
        setLoading(false)
      }
    } else {
      // Use streaming for sampling (supports grammar & punctuation)
      await rnnAPI.generateTextStream(
        requestData,
        // onToken: called for each word
        (word, index, grammarScore) => {
          setGeneratedText(prev => prev + word)
        },
        // onPunctuation: called when punctuation is applied
        (formattedText) => {
          setGeneratedText(formattedText)
        },
        // onComplete: called when done
        (fullText) => {
          setGeneratedText(fullText)
          setLoading(false)
        },
        // onError: called on error
        (errorMsg) => {
          setError(errorMsg)
          setLoading(false)
        }
      )
    }
  }

  const exampleSeeds = [
    'In the beginning',
    'To be or not to be',
    'It was the best of times',
    'Call me Ishmael',
    'Once upon a time',
    'All happy families are alike'
  ]

  const handleExampleClick = (example) => {
    setSeedText(example)
    setGeneratedText(null)
    setError(null)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <div className="flex items-center mb-4">
          <MessageCircle size={48} className="text-blue-600 mr-4" />
          <div>
            <h1 className="text-4xl font-bold text-gray-800">RNN Text Generation</h1>
            <p className="text-gray-600 text-lg">LSTM-Based Neural Network for Next-Word Prediction</p>
          </div>
        </div>

        {/* Model Selection */}
        {availableModels.length > 0 && (
          <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-3">
                <label className="text-sm font-semibold text-gray-700">Select Model:</label>
                <select
                  value={selectedModel || ''}
                  onChange={(e) => handleModelSwitch(e.target.value)}
                  disabled={switchingModel}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white disabled:bg-gray-100 disabled:cursor-not-allowed"
                >
                  {availableModels.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.display_name}
                    </option>
                  ))}
                </select>
                {switchingModel && (
                  <div className="flex items-center gap-2 text-blue-600">
                    <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                    <span className="text-sm">Switching...</span>
                  </div>
                )}
              </div>
              {modelSwitchMessage && (
                <div className={`text-sm font-medium ${modelSwitchMessage.type === 'success' ? 'text-green-600' : 'text-red-600'}`}>
                  {modelSwitchMessage.text}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Model Info */}
        {modelInfo && (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mt-6">
            <div className="bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-600 mb-1">Vocabulary</p>
              <p className="text-lg font-bold text-blue-600">{modelInfo.vocab_size.toLocaleString()}</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-600 mb-1">Context</p>
              <p className="text-lg font-bold text-blue-600">{modelInfo.sequence_length} words</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-600 mb-1">Embedding</p>
              <p className="text-lg font-bold text-blue-600">{modelInfo.embedding_dim}D</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-600 mb-1">LSTM Units</p>
              <p className="text-lg font-bold text-blue-600">{modelInfo.lstm_units}</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-600 mb-1">Layers</p>
              <p className="text-lg font-bold text-blue-600">{modelInfo.num_layers}</p>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 text-center">
              <p className="text-xs text-gray-600 mb-1">Parameters</p>
              <p className="text-lg font-bold text-blue-600">{(modelInfo.total_neurons / 1000000).toFixed(1)}M</p>
            </div>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-2xl shadow-xl mb-8">
        <div className="flex border-b border-gray-200">
          <button
            onClick={() => setActiveTab('generate')}
            className={`flex-1 py-4 px-6 text-center font-semibold transition-colors ${
              activeTab === 'generate'
                ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
            }`}
          >
            <Sparkles className="inline-block mr-2" size={20} />
            Generate Text
          </button>
          <button
            onClick={() => setActiveTab('report')}
            className={`flex-1 py-4 px-6 text-center font-semibold transition-colors ${
              activeTab === 'report'
                ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
            }`}
          >
            <FileText className="inline-block mr-2" size={20} />
            Technical Report
          </button>
        </div>
      </div>

      {/* Generate Tab */}
      {activeTab === 'generate' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
              <Send className="mr-3 text-blue-600" />
              Configuration
            </h2>

            {/* Seed Text */}
            <div className="mb-6">
              <label className="block text-gray-700 font-semibold mb-2">Seed Text</label>
              <input
                type="text"
                value={seedText}
                onChange={(e) => setSeedText(e.target.value)}
                placeholder="Enter seed text..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Number of Words */}
            <div className="mb-6">
              <label className="block text-gray-700 font-semibold mb-2">
                Number of Words: <span className="text-blue-600">{numWords}</span>
              </label>
              <input
                type="range"
                min="10"
                max="200"
                value={numWords}
                onChange={(e) => setNumWords(parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            {/* Generation Method */}
            <div className="mb-6">
              <label className="block text-gray-700 font-semibold mb-2">Generation Method</label>
              <div className="flex gap-4">
                <button
                  onClick={() => setUseBeamSearch(false)}
                  className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors ${
                    !useBeamSearch
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Sampling
                </button>
                <button
                  onClick={() => setUseBeamSearch(true)}
                  className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors ${
                    useBeamSearch
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Beam Search
                </button>
              </div>
            </div>

            {/* Temperature (Sampling) */}
            {!useBeamSearch && (
              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-2">
                  Temperature: <span className="text-blue-600">{temperature.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Very Conservative (0.1)</span>
                  <span>Extremely Creative (5.0)</span>
                </div>
              </div>
            )}

            {/* Beam Search Parameters */}
            {useBeamSearch && (
              <>
                <div className="mb-6">
                  <label className="block text-gray-700 font-semibold mb-2">
                    Beam Width: <span className="text-blue-600">{beamWidth}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={beamWidth}
                    onChange={(e) => setBeamWidth(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div className="mb-6">
                  <label className="block text-gray-700 font-semibold mb-2">
                    Repetition Penalty: <span className="text-blue-600">{repetitionPenalty.toFixed(2)}</span>
                  </label>
                  <input
                    type="range"
                    min="1.0"
                    max="2.0"
                    step="0.1"
                    value={repetitionPenalty}
                    onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </>
            )}

            {/* Post-Processing Options */}
            <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-teal-50 rounded-lg border border-green-200">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Post-Processing Options</h3>

              <div className="space-y-2">
                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={addPunctuation}
                    onChange={(e) => setAddPunctuation(e.target.checked)}
                    className="mr-3 h-4 w-4 text-green-600 rounded focus:ring-green-500"
                  />
                  <span className="text-sm text-gray-700">
                    <span className="font-semibold">Add Punctuation & Capitalization</span>
                    <span className="block text-xs text-gray-500 mt-1">
                      Automatically add periods, capitalize sentences, and format contractions
                    </span>
                  </span>
                </label>

                <label className="flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={validateGrammar}
                    onChange={(e) => setValidateGrammar(e.target.checked)}
                    className="mr-3 h-4 w-4 text-green-600 rounded focus:ring-green-500"
                  />
                  <span className="text-sm text-gray-700">
                    <span className="font-semibold">Validate Grammar</span>
                    <span className="block text-xs text-gray-500 mt-1">
                      Check for basic grammar rules during generation (slower but more grammatical)
                    </span>
                  </span>
                </label>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={loading || !seedText.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed mb-4"
            >
              {loading ? 'Generating...' : 'Generate Text'}
            </button>

            {error && (
              <div className="mb-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg flex items-center">
                <AlertCircle className="mr-2" size={20} />
                {error}
              </div>
            )}

            {/* Example Seeds */}
            <div className="mt-6">
              <h3 className="text-lg font-bold text-gray-800 mb-3">Try Examples:</h3>
              <div className="grid grid-cols-2 gap-2">
                {exampleSeeds.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleExampleClick(example)}
                    className="text-left bg-gray-50 hover:bg-gray-100 px-3 py-2 rounded-lg text-sm text-gray-700 transition-colors"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Generated Text</h2>

            {generatedText ? (
              <div>
                <div className="bg-gray-50 rounded-lg p-6 mb-4">
                  <p className="text-gray-800 text-lg leading-relaxed">{generatedText}</p>
                </div>

                <div className="bg-blue-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700">
                    <span className="font-semibold">Word Count:</span> {generatedText.split(' ').length}
                  </p>
                  <p className="text-sm text-gray-700">
                    <span className="font-semibold">Method:</span> {useBeamSearch ? 'Beam Search' : 'Sampling'}
                  </p>
                  {!useBeamSearch && (
                    <p className="text-sm text-gray-700">
                      <span className="font-semibold">Temperature:</span> {temperature.toFixed(2)}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-12">
                <BookOpen size={64} className="mx-auto mb-4 opacity-50" />
                <p>Configure parameters and generate text to see results</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Technical Report Tab */}
      {activeTab === 'report' && (
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {loadingReport ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-gray-600 mt-4">Loading technical report...</p>
            </div>
          ) : technicalReport ? (
            <div className="prose prose-lg max-w-none">
              <ReactMarkdown>{technicalReport}</ReactMarkdown>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-12">
              <FileText size={64} className="mx-auto mb-4 opacity-50" />
              <p>Technical report not available</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
