import React, { useState, useEffect } from 'react';
import './App.css';
import TextGenerator from './components/TextGenerator';
import ModelAnalytics from './components/ModelAnalytics';
import { healthCheck, getModelInfo } from './services/api';
import { useTheme } from './contexts/ThemeContext';

type Tab = 'generator' | 'analytics';

function App() {
  const { darkMode, toggleDarkMode } = useTheme();
  const [activeTab, setActiveTab] = useState<Tab>('generator');
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; checking: boolean; neurons?: number }>({
    loaded: false,
    checking: true,
  });

  useEffect(() => {
    checkModelStatus();
  }, []);

  const checkModelStatus = async () => {
    try {
      const health = await healthCheck();
      if (health.model_loaded) {
        // If model is loaded, fetch model info to get neuron count
        try {
          const modelInfo = await getModelInfo();
          setModelStatus({ loaded: true, checking: false, neurons: modelInfo.total_neurons });
        } catch (error) {
          // If we can't fetch model info, still show model as loaded but without neuron count
          setModelStatus({ loaded: true, checking: false });
        }
      } else {
        setModelStatus({ loaded: false, checking: false });
      }
    } catch (error) {
      setModelStatus({ loaded: false, checking: false });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 transition-colors duration-200">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-md transition-colors duration-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                RNN Text Generator
              </h1>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                LSTM-based Text Generation & Analytics
                {modelStatus.loaded && modelStatus.neurons && (
                  <span className="ml-2 text-indigo-600 dark:text-indigo-400 font-medium">
                    • {modelStatus.neurons.toLocaleString()} neurons
                  </span>
                )}
              </p>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${
                modelStatus.loaded
                  ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
                  : modelStatus.checking
                  ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300'
                  : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  modelStatus.loaded
                    ? 'bg-green-500'
                    : modelStatus.checking
                    ? 'bg-yellow-500 animate-pulse'
                    : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-medium">
                  {modelStatus.checking ? 'Checking...' : modelStatus.loaded ? 'Model Ready' : 'Model Not Loaded'}
                </span>
              </div>
              {/* Dark Mode Toggle */}
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                aria-label="Toggle dark mode"
              >
                {darkMode ? (
                  <svg className="w-5 h-5 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 text-gray-700" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                  </svg>
                )}
              </button>
              <button
                onClick={checkModelStatus}
                className="px-4 py-2 text-sm bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 rounded-lg hover:bg-indigo-200 dark:hover:bg-indigo-800 transition-colors"
              >
                Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex space-x-8 border-b border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setActiveTab('generator')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'generator'
                  ? 'border-indigo-600 text-indigo-600 dark:text-indigo-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              Text Generator
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'analytics'
                  ? 'border-purple-600 text-purple-600 dark:text-purple-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              Analytics Dashboard
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="py-8">
        {!modelStatus.loaded && !modelStatus.checking && (
          <div className="max-w-7xl mx-auto px-6 mb-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-300">Model Not Loaded</h3>
                  <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-400">
                    The model is not currently loaded. Please ensure the backend server is running and the model has been trained.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'generator' && <TextGenerator onModelChange={checkModelStatus} />}
        {activeTab === 'analytics' && <ModelAnalytics onModelChange={checkModelStatus} />}
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-12 transition-colors duration-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            RNN Text Generator - CST-435 Project
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
