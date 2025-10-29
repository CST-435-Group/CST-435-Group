import { Link, useLocation } from 'react-router-dom'
import { Home, Brain, Camera, MessageSquare } from 'lucide-react'

export default function Header() {
  const location = useLocation()

  const isActive = (path) => location.pathname === path

  return (
    <header className="bg-white shadow-lg">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-2">
            <div className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              CST-435 ML Projects
            </div>
          </Link>

          <nav className="flex space-x-6">
            <Link
              to="/"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/')
                  ? 'bg-blue-100 text-blue-700 font-semibold'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Home size={20} />
              <span>Home</span>
            </Link>

            <Link
              to="/ann"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/ann')
                  ? 'bg-blue-100 text-blue-700 font-semibold'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Brain size={20} />
              <span>ANN</span>
            </Link>

            <Link
              to="/cnn"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/cnn')
                  ? 'bg-blue-100 text-blue-700 font-semibold'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Camera size={20} />
              <span>CNN</span>
            </Link>

            <Link
              to="/nlp"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                isActive('/nlp')
                  ? 'bg-blue-100 text-blue-700 font-semibold'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <MessageSquare size={20} />
              <span>NLP</span>
            </Link>
          </nav>
        </div>
      </div>
    </header>
  )
}
