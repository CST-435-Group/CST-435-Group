import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { ganAPI, annAPI, cnnAPI, nlpAPI, rnnAPI } from '../services/api'
import { Sparkles, Download, RefreshCw, AlertCircle, Settings, Image as ImageIcon, BookOpen } from 'lucide-react'
import { useModelManager } from '../hooks/useModelManager'

export default function GANProject() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('latest_generator.pth')
  const [tanks, setTanks] = useState([])
  const [views, setViews] = useState([])
  const [selectedTank, setSelectedTank] = useState('')
  const [selectedView, setSelectedView] = useState('')
  const [numImages, setNumImages] = useState(4)
  const [seed, setSeed] = useState('')
  const [generatedImages, setGeneratedImages] = useState([])
  const [generating, setGenerating] = useState(false)

  // Preload GAN model and unload others when this page loads
  useModelManager(ganAPI, [annAPI, cnnAPI, nlpAPI, rnnAPI])

  useEffect(() => {
    loadModelInfo()
    loadAvailableModels()
  }, [])

  const loadModelInfo = async () => {
    try {
      const response = await ganAPI.getModelInfo()
      setModelInfo(response.data)
      setTanks(response.data.available_tanks || [])
      setViews(response.data.available_views || [])

      // Set defaults
      if (response.data.available_tanks?.length > 0) {
        setSelectedTank(response.data.available_tanks[0])
      }
      if (response.data.available_views?.length > 0) {
        setSelectedView(response.data.available_views[0])
      }
    } catch (err) {
      console.error('Error loading model info:', err)
      setError('Failed to load model info. Make sure the backend is running.')
    }
  }

  const loadAvailableModels = async () => {
    try {
      const response = await ganAPI.getAvailableModels()
      setAvailableModels(response.data.models || [])
      if (response.data.current_model) {
        setSelectedModel(response.data.current_model)
      }
    } catch (err) {
      console.error('Error loading available models:', err)
    }
  }

  const handleModelSwitch = async (modelName) => {
    setLoading(true)
    setError(null)
    try {
      await ganAPI.switchModel(modelName)
      setSelectedModel(modelName)
      setGeneratedImages([]) // Clear old images
    } catch (err) {
      setError(err.response?.data?.detail || 'Error switching models')
    } finally {
      setLoading(false)
    }
  }

  const handleGenerate = async () => {
    if (!selectedTank || !selectedView) {
      setError('Please select both a tank type and view angle')
      return
    }

    setGenerating(true)
    setError(null)

    try {
      const response = await ganAPI.generate({
        tank_type: selectedTank,
        view_angle: selectedView,
        num_images: numImages,
        seed: seed ? parseInt(seed) : null
      })

      setGeneratedImages(response.data.images)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error generating images')
    } finally {
      setGenerating(false)
    }
  }

  const handleDownloadImage = (imageData, index) => {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${imageData.image_base64}`
    link.download = `${imageData.tank_type}_${imageData.view_angle}_${index}.png`
    link.click()
  }

  const handleDownloadAll = () => {
    generatedImages.forEach((img, idx) => {
      setTimeout(() => handleDownloadImage(img, idx), idx * 100)
    })
  }

  const handleRandomSeed = () => {
    setSeed(Math.floor(Math.random() * 1000000).toString())
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <div className="flex items-center mb-4">
          <Sparkles size={48} className="text-amber-600 mr-4" />
          <div>
            <h1 className="text-4xl font-bold text-gray-800">Military Vehicle GAN</h1>
            <p className="text-gray-600 text-lg">Dual Conditional GAN for Synthetic Image Generation</p>
          </div>
        </div>
      </div>

      {/* Model Stats */}
      {modelInfo && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Tank Types</h3>
            <p className="text-3xl font-bold text-amber-600">{tanks.length}</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">View Angles</h3>
            <p className="text-3xl font-bold text-blue-600">{views.length}</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Latent Dimension</h3>
            <p className="text-3xl font-bold text-purple-600">{modelInfo.latent_dim}</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Image Size</h3>
            <p className="text-3xl font-bold text-green-600">{modelInfo.image_size}x{modelInfo.image_size}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Control Panel */}
        <div className="lg:col-span-1 space-y-6">
          {/* Model Selection */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
              <Settings size={20} className="mr-2" />
              Model Selection
            </h2>
            <select
              value={selectedModel}
              onChange={(e) => handleModelSwitch(e.target.value)}
              disabled={loading || generating}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              {availableModels.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </div>

          {/* Generation Settings */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Generation Settings</h2>

            {/* Tank Type */}
            <div className="mb-4">
              <label className="block text-gray-700 font-semibold mb-2">Tank Type</label>
              <select
                value={selectedTank}
                onChange={(e) => setSelectedTank(e.target.value)}
                disabled={generating}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
              >
                {tanks.map((tank) => (
                  <option key={tank} value={tank}>{tank}</option>
                ))}
              </select>
            </div>

            {/* View Angle */}
            <div className="mb-4">
              <label className="block text-gray-700 font-semibold mb-2">View Angle</label>
              <select
                value={selectedView}
                onChange={(e) => setSelectedView(e.target.value)}
                disabled={generating}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
              >
                {views.map((view) => (
                  <option key={view} value={view}>{view}</option>
                ))}
              </select>
            </div>

            {/* Number of Images */}
            <div className="mb-4">
              <label className="block text-gray-700 font-semibold mb-2">
                Number of Images: {numImages}
              </label>
              <input
                type="range"
                min="1"
                max="16"
                value={numImages}
                onChange={(e) => setNumImages(parseInt(e.target.value))}
                disabled={generating}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-amber-600"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>1</span>
                <span>8</span>
                <span>16</span>
              </div>
            </div>

            {/* Seed */}
            <div className="mb-6">
              <label className="block text-gray-700 font-semibold mb-2">Random Seed (optional)</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={seed}
                  onChange={(e) => setSeed(e.target.value)}
                  placeholder="Leave empty for random"
                  disabled={generating}
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-amber-500"
                />
                <button
                  onClick={handleRandomSeed}
                  disabled={generating}
                  className="px-4 py-3 bg-gray-200 hover:bg-gray-300 rounded-lg transition-colors"
                  title="Generate random seed"
                >
                  <RefreshCw size={20} />
                </button>
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={generating || !selectedTank || !selectedView}
              className="w-full bg-amber-600 hover:bg-amber-700 text-white font-bold py-4 px-8 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {generating ? (
                <>
                  <RefreshCw size={20} className="mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles size={20} className="mr-2" />
                  Generate Images
                </>
              )}
            </button>
          </div>

          {/* Tank Type Info */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Available Tank Types</h2>
            <div className="space-y-2">
              {tanks.map((tank) => (
                <div
                  key={tank}
                  className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                    selectedTank === tank
                      ? 'bg-amber-100 border-2 border-amber-500 text-amber-700'
                      : 'bg-gray-100 hover:bg-gray-200'
                  }`}
                  onClick={() => setSelectedTank(tank)}
                >
                  {tank}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Generated Images</h2>
              {generatedImages.length > 0 && (
                <button
                  onClick={handleDownloadAll}
                  className="flex items-center bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
                >
                  <Download size={18} className="mr-2" />
                  Download All
                </button>
              )}
            </div>

            {error && (
              <div className="mb-6 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg flex items-center">
                <AlertCircle className="mr-2" size={20} />
                {error}
              </div>
            )}

            {generatedImages.length > 0 ? (
              <div>
                <p className="text-gray-600 mb-4">
                  Generated {generatedImages.length} images of <span className="font-bold">{selectedTank}</span> ({selectedView} view)
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {generatedImages.map((img, idx) => (
                    <div key={idx} className="relative group">
                      <img
                        src={`data:image/png;base64,${img.image_base64}`}
                        alt={`Generated ${img.tank_type} ${img.view_angle} ${idx}`}
                        className="w-full aspect-square object-cover rounded-lg border-2 border-gray-200 group-hover:border-amber-400 transition-colors"
                      />
                      <button
                        onClick={() => handleDownloadImage(img, idx)}
                        className="absolute bottom-2 right-2 bg-white bg-opacity-90 hover:bg-opacity-100 p-2 rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Download image"
                      >
                        <Download size={16} className="text-gray-700" />
                      </button>
                      <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white text-xs px-2 py-1 rounded">
                        #{idx + 1}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-16">
                <ImageIcon size={64} className="mx-auto mb-4 opacity-50" />
                <p className="text-lg">No images generated yet</p>
                <p className="text-sm mt-2">Select options and click "Generate Images" to create synthetic military vehicle images</p>
              </div>
            )}
          </div>

          {/* How it Works */}
          <div className="bg-white rounded-2xl shadow-xl p-8 mt-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">How It Works</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-xl font-bold text-amber-600">1</span>
                </div>
                <h3 className="font-bold text-gray-800 mb-2">Select Conditions</h3>
                <p className="text-gray-600 text-sm">Choose the tank type and view angle you want to generate</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-xl font-bold text-amber-600">2</span>
                </div>
                <h3 className="font-bold text-gray-800 mb-2">GAN Generation</h3>
                <p className="text-gray-600 text-sm">The model generates unique images based on learned features from training data</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <span className="text-xl font-bold text-amber-600">3</span>
                </div>
                <h3 className="font-bold text-gray-800 mb-2">Download Results</h3>
                <p className="text-gray-600 text-sm">Save individual images or download all at once for your use</p>
              </div>
            </div>

            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <div className="flex justify-between items-start mb-2">
                <h4 className="font-bold text-gray-800">About the Model</h4>
                <Link
                  to="/gan/model"
                  className="flex items-center text-amber-600 hover:text-amber-700 text-sm font-semibold"
                >
                  <BookOpen size={16} className="mr-1" />
                  Learn More
                </Link>
              </div>
              <p className="text-gray-600 text-sm">
                This is a <strong>Dual Conditional WGAN-GP</strong> (Wasserstein GAN with Gradient Penalty) that generates
                200x200 RGB images. It uses <strong>self-attention mechanisms</strong> to capture long-range dependencies
                and produce coherent images. The model is conditioned on both tank type and view angle, allowing precise
                control over the generated output.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
