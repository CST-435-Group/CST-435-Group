import React, { useState, useEffect } from 'react';
import { getModelInfo, testModel, getTrainingHistoryImage, ModelInfo, TestMetrics, getAvailableModels, switchModel, AvailableModel } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ModelAnalyticsProps {
  onModelChange?: () => void;
}

const ModelAnalytics: React.FC<ModelAnalyticsProps> = ({ onModelChange }) => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [testMetrics, setTestMetrics] = useState<TestMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [testingModel, setTestingModel] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [switchingModel, setSwitchingModel] = useState(false);

  useEffect(() => {
    loadModelInfo();
    loadAvailableModels();
  }, []);

  const loadModelInfo = async () => {
    setLoading(true);
    setError(null);

    try {
      const info = await getModelInfo();
      setModelInfo(info);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load model information');
    } finally {
      setLoading(false);
    }
  };

  const loadAvailableModels = async () => {
    try {
      const response = await getAvailableModels();
      setAvailableModels(response.models);
      if (response.current_model) {
        setSelectedModel(response.current_model);
      }
    } catch (err) {
      console.error('Failed to load available models:', err);
    }
  };

  const handleModelChange = async (modelName: string) => {
    if (modelName === selectedModel) return;

    setSwitchingModel(true);
    setError(null);
    setTestMetrics(null); // Clear test metrics when switching models

    try {
      await switchModel(modelName);
      setSelectedModel(modelName);
      // Reload model info after switching
      await loadModelInfo();
      // Notify parent component to refresh model status in header
      if (onModelChange) {
        onModelChange();
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to switch model');
    } finally {
      setSwitchingModel(false);
    }
  };

  const runModelTest = async () => {
    setTestingModel(true);
    setError(null);

    try {
      const metrics = await testModel();
      setTestMetrics(metrics);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to test model');
    } finally {
      setTestingModel(false);
    }
  };

  const formatMetric = (value: number, decimals: number = 4) => {
    return value.toFixed(decimals);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  // Prepare data for charts
  const metricsChartData = testMetrics
    ? [
        { name: 'Accuracy', value: testMetrics.test_accuracy * 100, fill: '#10b981' },
        { name: 'Loss', value: testMetrics.test_loss * 10, fill: '#ef4444' },
        { name: 'Perplexity', value: Math.min(testMetrics.perplexity, 100), fill: '#f59e0b' },
      ]
    : [];

  const modelInfoData = modelInfo
    ? [
        { name: 'Vocab Size', value: modelInfo.vocab_size },
        { name: 'Seq Length', value: modelInfo.sequence_length },
        { name: 'Embedding Dim', value: modelInfo.embedding_dim },
        { name: 'LSTM Units', value: modelInfo.lstm_units },
        { name: 'Layers', value: modelInfo.num_layers },
      ]
    : [];

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2">Model Analytics Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400">Comprehensive model performance metrics and architecture details</p>
      </div>

      {/* Model Selection */}
      {availableModels.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6 transition-colors duration-200">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Current Model:
          </label>
          <select
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={switchingModel}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
          >
            {availableModels.map((model) => (
              <option key={model.name} value={model.name}>
                {model.display_name}
              </option>
            ))}
          </select>
          {switchingModel && (
            <p className="text-sm text-indigo-600 dark:text-indigo-400 mt-2">Switching model and reloading analytics...</p>
          )}
        </div>
      )}

      {error && (
        <div className="mb-6 p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 rounded-lg">
          {error}
        </div>
      )}

      {/* Model Information Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6 transition-colors duration-200">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">Model Architecture</h2>
          <button
            onClick={loadModelInfo}
            disabled={loading}
            className="px-4 py-2 bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 rounded-lg hover:bg-indigo-200 dark:hover:bg-indigo-800 disabled:bg-gray-200 dark:disabled:bg-gray-700 transition-colors"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {modelInfo ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3 text-gray-700 dark:text-gray-300">Configuration</h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded">
                  <span className="font-medium text-gray-600 dark:text-gray-400">Vocabulary Size:</span>
                  <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{modelInfo.vocab_size.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded">
                  <span className="font-medium text-gray-600 dark:text-gray-400">Sequence Length:</span>
                  <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{modelInfo.sequence_length}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded">
                  <span className="font-medium text-gray-600 dark:text-gray-400">Embedding Dimension:</span>
                  <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{modelInfo.embedding_dim}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded">
                  <span className="font-medium text-gray-600 dark:text-gray-400">LSTM Units:</span>
                  <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{modelInfo.lstm_units}</span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded">
                  <span className="font-medium text-gray-600 dark:text-gray-400">Number of Layers:</span>
                  <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{modelInfo.num_layers}</span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3 text-gray-700 dark:text-gray-300">Architecture Visualization</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={modelInfoData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-15} textAnchor="end" height={60} fontSize={12} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#6366f1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">
            {loading ? 'Loading model information...' : 'No model information available'}
          </div>
        )}
      </div>

      {/* Model Testing Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6 transition-colors duration-200">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-green-600 dark:text-green-400">Model Performance Testing</h2>
          <button
            onClick={runModelTest}
            disabled={testingModel}
            className="px-6 py-3 bg-green-600 dark:bg-green-500 text-white rounded-lg font-semibold hover:bg-green-700 dark:hover:bg-green-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 transition-colors"
          >
            {testingModel ? 'Testing Model...' : 'Run Test'}
          </button>
        </div>

        <p className="text-gray-600 dark:text-gray-400 mb-6">
          Test the model on evaluation data to get accuracy, loss, perplexity, and R² metrics.
        </p>

        {testMetrics ? (
          <div className="space-y-6">
            {/* Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border border-green-200">
                <h3 className="text-sm font-medium text-green-700 mb-1">Test Accuracy</h3>
                <p className="text-3xl font-bold text-green-600">{formatPercentage(testMetrics.test_accuracy)}</p>
              </div>

              <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-lg border border-red-200">
                <h3 className="text-sm font-medium text-red-700 mb-1">Test Loss</h3>
                <p className="text-3xl font-bold text-red-600">{formatMetric(testMetrics.test_loss)}</p>
              </div>

              <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-6 rounded-lg border border-yellow-200">
                <h3 className="text-sm font-medium text-yellow-700 mb-1">Perplexity</h3>
                <p className="text-3xl font-bold text-yellow-600">{formatMetric(testMetrics.perplexity, 2)}</p>
              </div>

              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200">
                <h3 className="text-sm font-medium text-blue-700 mb-1">R² Score</h3>
                <p className="text-3xl font-bold text-blue-600">
                  {testMetrics.r_squared !== undefined && testMetrics.r_squared !== null
                    ? formatMetric(testMetrics.r_squared, 4)
                    : 'N/A'}
                </p>
              </div>
            </div>

            {/* Additional Info */}
            <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
              <p className="text-gray-700 dark:text-gray-300">
                <span className="font-semibold">Samples Tested:</span>{' '}
                {testMetrics.samples_tested.toLocaleString()}
              </p>
            </div>

            {/* Metrics Visualization */}
            <div>
              <h3 className="text-lg font-semibold mb-3 text-gray-700 dark:text-gray-300">Performance Metrics Overview</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={metricsChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" />
                </BarChart>
              </ResponsiveContainer>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                Note: Loss and Perplexity values are scaled for visualization purposes
              </p>
            </div>

            {/* Metrics Explanation */}
            <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
              <h3 className="text-lg font-semibold mb-2 text-blue-900 dark:text-blue-300">Understanding the Metrics</h3>
              <ul className="space-y-2 text-sm text-blue-800 dark:text-blue-300">
                <li>
                  <strong>Accuracy:</strong> Percentage of correctly predicted next words
                </li>
                <li>
                  <strong>Loss:</strong> Cross-entropy loss (lower is better)
                </li>
                <li>
                  <strong>Perplexity:</strong> Measure of how well the model predicts the text (lower is better, exp(loss))
                </li>
                <li>
                  <strong>R² Score:</strong> Statistical measure of model fit (closer to 1 is better)
                </li>
              </ul>
            </div>
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500 dark:text-gray-400">
            <p className="mb-4">No test results available yet</p>
            <p className="text-sm">Click "Run Test" to evaluate the model</p>
          </div>
        )}
      </div>

      {/* Training History Visualization */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 transition-colors duration-200">
        <h2 className="text-2xl font-bold text-purple-600 dark:text-purple-400 mb-4">Training History</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Loss and accuracy progression during model training
        </p>
        <div className="flex justify-center">
          <img
            src={getTrainingHistoryImage()}
            alt="Training History"
            className="max-w-full rounded-lg shadow-md"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none';
              const parent = (e.target as HTMLImageElement).parentElement;
              if (parent) {
                parent.innerHTML = '<p class="text-gray-500 dark:text-gray-400 py-8">Training history visualization not available</p>';
              }
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default ModelAnalytics;
