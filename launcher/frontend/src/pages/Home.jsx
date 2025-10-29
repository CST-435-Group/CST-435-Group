import { Link } from 'react-router-dom'
import { Brain, Camera, MessageSquare, ArrowRight } from 'lucide-react'

export default function Home() {
  const projects = [
    {
      id: 'ann',
      title: 'ANN - NBA Team Selection',
      description: 'Artificial Neural Network for optimal NBA team composition using multi-layer perceptron',
      icon: Brain,
      color: 'from-blue-500 to-blue-700',
      path: '/ann',
      features: [
        'Player position classification',
        'Team fit score evaluation',
        '3 selection strategies',
        'Performance analytics'
      ]
    },
    {
      id: 'cnn',
      title: 'CNN - Fruit Classification',
      description: 'Convolutional Neural Network for recognizing different types of fruits from images',
      icon: Camera,
      color: 'from-green-500 to-green-700',
      path: '/cnn',
      features: [
        'Image classification',
        'Multiple fruit types',
        'Confidence scores',
        'Upload custom images'
      ]
    },
    {
      id: 'nlp',
      title: 'NLP - Sentiment Analysis',
      description: '7-point scale sentiment analyzer for movie reviews and text using transformer models',
      icon: MessageSquare,
      color: 'from-purple-500 to-purple-700',
      path: '/nlp',
      features: [
        '7-point sentiment scale (-3 to +3)',
        'Movie review analysis',
        'Confidence scores',
        'Batch processing'
      ]
    }
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-6xl font-bold text-white mb-4">
          Machine Learning Projects
        </h1>
        <p className="text-2xl text-white opacity-90">
          CST-435 Neural Networks - Interactive Demonstrations
        </p>
        <p className="text-lg text-white opacity-75 mt-2">
          Select a project below to explore different ML techniques
        </p>
      </div>

      {/* Project Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-7xl mx-auto">
        {projects.map((project) => {
          const Icon = project.icon
          return (
            <Link
              key={project.id}
              to={project.path}
              className="group"
            >
              <div className="bg-white rounded-2xl shadow-2xl overflow-hidden transform transition-all duration-300 hover:scale-105 hover:shadow-3xl h-full">
                {/* Card Header with Gradient */}
                <div className={`bg-gradient-to-r ${project.color} p-8 text-white`}>
                  <Icon size={48} className="mb-4" />
                  <h2 className="text-3xl font-bold mb-2">{project.title}</h2>
                </div>

                {/* Card Body */}
                <div className="p-8">
                  <p className="text-gray-700 text-lg mb-6">
                    {project.description}
                  </p>

                  {/* Features List */}
                  <div className="space-y-3 mb-6">
                    {project.features.map((feature, idx) => (
                      <div key={idx} className="flex items-center text-gray-600">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>

                  {/* Launch Button */}
                  <div className="flex items-center justify-center text-blue-600 font-semibold group-hover:text-blue-800 transition-colors">
                    <span className="mr-2">Launch Project</span>
                    <ArrowRight
                      size={20}
                      className="transform group-hover:translate-x-2 transition-transform"
                    />
                  </div>
                </div>
              </div>
            </Link>
          )
        })}
      </div>

      {/* Info Section */}
      <div className="mt-16 max-w-4xl mx-auto">
        <div className="bg-white bg-opacity-90 rounded-2xl shadow-xl p-8">
          <h3 className="text-3xl font-bold text-gray-800 mb-4">About This Project</h3>
          <p className="text-gray-700 text-lg leading-relaxed mb-4">
            This unified launcher provides access to three different machine learning projects developed for CST-435.
            Each project demonstrates different neural network architectures and use cases:
          </p>
          <ul className="space-y-2 text-gray-700">
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>ANN:</strong> Uses a Multi-Layer Perceptron (MLP) for classification and regression tasks</span>
            </li>
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>CNN:</strong> Implements Convolutional Neural Networks for computer vision</span>
            </li>
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>NLP:</strong> Leverages Transformer models for natural language processing</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}
