import { useState, useEffect } from 'react'
import { cnnAPI, annAPI, nlpAPI, docsAPI } from '../services/api'
import ReactMarkdown from 'react-markdown'
import { Camera, Upload, AlertCircle, CheckCircle, Image as ImageIcon } from 'lucide-react'
import { useModelManager } from '../hooks/useModelManager'
import OptimizationReport from '../components/OptimizationReport.jsx'


export default function CNNProject() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [fruits, setFruits] = useState([])
  const [prediction, setPrediction] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [uploadMode, setUploadMode] = useState('dataset') // 'upload' or 'dataset'
  const [sampleImages, setSampleImages] = useState([])
  const [selectedSampleImage, setSelectedSampleImage] = useState(null)

  // Preload CNN model and unload others when this page loads
  useModelManager(cnnAPI, [annAPI, nlpAPI])

  useEffect(() => {
    loadModelInfo()
    loadFruits()
    loadSampleImages()
  }, [])

  const loadModelInfo = async () => {
    try {
      const response = await cnnAPI.getModelInfo()
      setModelInfo(response.data)
    } catch (err) {
      console.error('Error loading model info:', err)
    }
  }

  const loadFruits = async () => {
    try {
      const response = await cnnAPI.getFruitList()
      setFruits(response.data.fruits)
    } catch (err) {
      console.error('Error loading fruits:', err)
    }
  }

  const loadSampleImages = async () => {
    try {
      const response = await cnnAPI.getSampleImages(5)
      setSampleImages(response.data)
    } catch (err) {
      console.error('Error loading sample images:', err)
    }
  }

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPrediction(null)
      setError(null)

      // Create preview URL
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreviewUrl(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await cnnAPI.predictImage(selectedFile)
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error classifying image')
    } finally {
      setLoading(false)
    }
  }

  const handleSampleImageSelect = (sampleImage) => {
    setSelectedSampleImage(sampleImage)
    setPrediction(null)
    setError(null)
  }

  const handlePredictSample = async () => {
    if (!selectedSampleImage) {
      setError('Please select an image from the dataset first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const [fruit, filename] = selectedSampleImage.id.split('/')
      const response = await cnnAPI.predictSampleImage(fruit, filename)
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error classifying image')
    } finally {
      setLoading(false)
    }
  }

  // Get top 3 probabilities
  const getTopProbabilities = () => {
    if (!prediction) return []

    return Object.entries(prediction.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <div className="flex items-center mb-4">
          <Camera size={48} className="text-green-600 mr-4" />
          <div>
            <h1 className="text-4xl font-bold text-gray-800">Fruit Classification</h1>
            <p className="text-gray-600 text-lg">Convolutional Neural Network for Image Recognition</p>
          </div>
        </div>
      </div>

      {/* Model Info */}
      {modelInfo && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Test Accuracy</h3>
            <p className="text-3xl font-bold text-green-600">{(modelInfo.test_accuracy * 100).toFixed(2)}%</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Number of Classes</h3>
            <p className="text-3xl font-bold text-blue-600">{modelInfo.num_classes}</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Total Images</h3>
            <p className="text-3xl font-bold text-purple-600">{modelInfo.total_images.toLocaleString()}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload/Browse Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {/* Tab Switcher */}
          <div className="flex mb-6 border-b border-gray-200">
            <button
              onClick={() => {
                setUploadMode('dataset')
                setPrediction(null)
                setError(null)
                setSelectedFile(null)
                setPreviewUrl(null)
              }}
              className={`flex-1 py-3 px-4 font-semibold transition-colors ${
                uploadMode === 'dataset'
                  ? 'text-green-600 border-b-2 border-green-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <ImageIcon className="inline mr-2" size={20} />
              Browse Dataset
            </button>
            <button
              onClick={() => {
                setUploadMode('upload')
                setPrediction(null)
                setError(null)
                setSelectedSampleImage(null)
              }}
              className={`flex-1 py-3 px-4 font-semibold transition-colors ${
                uploadMode === 'upload'
                  ? 'text-green-600 border-b-2 border-green-600'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <Upload className="inline mr-2" size={20} />
              Upload Image
            </button>
          </div>

          {/* Upload Mode */}
          {uploadMode === 'upload' && (
            <div>
              <div className="mb-6">
                <label className="block text-gray-700 font-semibold mb-2">Select Fruit Image</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
                />
              </div>

              {/* Image Preview */}
              {previewUrl && (
                <div className="mb-6">
                  <p className="text-gray-700 font-semibold mb-2">Preview</p>
                  <div className="border-2 border-gray-300 rounded-lg p-4 bg-gray-50">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="max-w-full h-auto max-h-64 mx-auto rounded"
                    />
                  </div>
                </div>
              )}

              <button
                onClick={handlePredict}
                disabled={loading || !selectedFile}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Classifying...' : 'Classify Image'}
              </button>
            </div>
          )}

          {/* Dataset Browse Mode */}
          {uploadMode === 'dataset' && (
            <div>
              <p className="text-gray-700 font-semibold mb-4">Select an image from the dataset</p>

              {/* Sample Images Grid */}
              <div className="max-h-96 overflow-y-auto mb-6">
                {fruits.map((fruit) => (
                  <div key={fruit} className="mb-6">
                    <h4 className="text-lg font-bold text-gray-800 mb-3">{fruit}</h4>
                    <div className="grid grid-cols-5 gap-2">
                      {sampleImages
                        .filter((img) => img.fruit === fruit)
                        .map((sampleImage) => {
                          const [fruitDir, filename] = sampleImage.id.split('/')
                          const imageUrl = cnnAPI.getSampleImageUrl(fruitDir, filename)
                          const isSelected = selectedSampleImage?.id === sampleImage.id

                          return (
                            <div
                              key={sampleImage.id}
                              onClick={() => handleSampleImageSelect(sampleImage)}
                              className={`cursor-pointer border-2 rounded-lg p-1 transition-all ${
                                isSelected
                                  ? 'border-green-600 ring-2 ring-green-300'
                                  : 'border-gray-200 hover:border-green-400'
                              }`}
                            >
                              <img
                                src={imageUrl}
                                alt={sampleImage.fruit}
                                className="w-full h-16 object-cover rounded"
                              />
                            </div>
                          )
                        })}
                    </div>
                  </div>
                ))}
              </div>

              {/* Selected Image Preview */}
              {selectedSampleImage && (
                <div className="mb-6">
                  <p className="text-gray-700 font-semibold mb-2">Selected Image</p>
                  <div className="border-2 border-green-300 rounded-lg p-4 bg-gray-50">
                    <img
                      src={cnnAPI.getSampleImageUrl(...selectedSampleImage.id.split('/'))}
                      alt="Selected"
                      className="max-w-full h-auto max-h-48 mx-auto rounded"
                    />
                    <p className="text-center mt-2 text-sm text-gray-600">
                      Actual: <span className="font-bold text-green-700">{selectedSampleImage.fruit}</span>
                    </p>
                  </div>
                </div>
              )}

              <button
                onClick={handlePredictSample}
                disabled={loading || !selectedSampleImage}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Classifying...' : 'Classify Selected Image'}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg flex items-center">
              <AlertCircle className="mr-2" size={20} />
              {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Prediction Results</h2>

          {prediction ? (
            <div>
              {/* Main Prediction */}
              <div className="bg-green-100 border border-green-400 rounded-lg p-6 mb-6">
                <div className="flex items-center mb-2">
                  <CheckCircle className="text-green-600 mr-2" size={24} />
                  <span className="text-gray-700 font-semibold">Predicted Fruit:</span>
                </div>
                <p className="text-4xl font-bold text-green-700 mb-2">{prediction.predicted_class}</p>
                <p className="text-gray-700">
                  Confidence: <span className="font-bold">{(prediction.confidence * 100).toFixed(2)}%</span>
                </p>
              </div>

              {/* Top 3 Predictions */}
              <div>
                <h3 className="text-xl font-bold text-gray-800 mb-4">Top 3 Predictions</h3>
                <div className="space-y-3">
                  {getTopProbabilities().map(([fruit, prob], idx) => (
                    <div key={idx} className="bg-gray-50 rounded-lg p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-semibold text-gray-800">{idx + 1}. {fruit}</span>
                        <span className="text-green-600 font-bold">{(prob * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-green-600 h-2 rounded-full transition-all duration-500"
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
              <Camera size={64} className="mx-auto mb-4 opacity-50" />
              <p>Upload and classify an image to see results</p>
            </div>
          )}
        </div>
      </div>

      {/* Supported Fruits */}
      {fruits.length > 0 && (
        <div className="bg-white rounded-2xl shadow-xl p-8 mt-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Supported Fruits ({fruits.length})</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {fruits.map((fruit, idx) => (
              <div key={idx} className="bg-green-100 rounded-lg px-4 py-2 text-center">
                <span className="text-green-700 font-semibold">{fruit}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {/* Reports Tabs */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mt-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Reports</h2>
        <ReportTabs projectId="cnn" />
      </div>
    </div>
  )
}


function ReportTabs({ projectId }) {
  const [active, setActive] = useState('technical')
  const [techMd, setTechMd] = useState(null)

  useEffect(() => {
    loadTechnical()
  }, [])

  const loadTechnical = async () => {
    try {
      const res = await docsAPI.getTechnical(projectId)
      setTechMd(res.data.markdown)
      setActive('technical')
    } catch (err) {
      setTechMd('# Technical Report\n\nNot available')
    }
  }

  return (
      <div>
        <div className="flex space-x-2 mb-4">
          <button 
            onClick={loadTechnical} 
            className={`px-4 py-2 rounded ${active === 'technical' ? 'bg-green-600 text-white' : 'bg-gray-100'}`}
          >
            Technical Report
          </button>
          <button 
            onClick={() => setActive('optimization')} 
            className={`px-4 py-2 rounded ${active === 'optimization' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
          >
            Optimization Report
          </button>
        </div>

        <div className="prose max-w-none">
          {active === 'technical' && techMd && <ReactMarkdown>{techMd}</ReactMarkdown>}
          {active === 'optimization' && <OptimizationReport />}
        </div>
      </div>
    )
  }

