import { useState, useEffect } from 'react';
import { BarChart3, Loader2, Hash, TrendingUp } from 'lucide-react';
import { getStatistics } from '../services/api';

function SimpleStats() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeWordTab, setActiveWordTab] = useState('all');

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const data = await getStatistics();
      setStats(data);
    } catch (error) {
      console.error('Failed to load statistics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-teal-500" />
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5 text-teal-600" />
          <h2 className="text-xl font-bold text-gray-800">Dataset Statistics</h2>
        </div>
        <p className="text-sm text-gray-500">No statistics available yet.</p>
      </div>
    );
  }

  const sentimentDist = stats.sentiment_distribution || {};
  const total = stats.total_reviews || 0;
  
  const positive = sentimentDist.positive || 0;
  const neutral = sentimentDist.neutral || 0;
  const negative = sentimentDist.negative || 0;

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
      <div className="flex items-center gap-2 mb-6">
        <BarChart3 className="w-6 h-6 text-teal-600" />
        <h2 className="text-2xl font-bold text-gray-800">Dataset Statistics</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {/* Total Reviews */}
        <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
          <p className="text-sm text-gray-600">Total Reviews</p>
          <p className="text-3xl font-bold text-gray-800">{total.toLocaleString()}</p>
        </div>

        {/* Positive */}
        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
          <p className="text-sm text-gray-600">😊 Positive</p>
          <p className="text-3xl font-bold text-green-600">{positive}</p>
          <p className="text-xs text-gray-500">{((positive/total)*100).toFixed(1)}%</p>
        </div>

        {/* Neutral */}
        <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
          <p className="text-sm text-gray-600">😐 Neutral</p>
          <p className="text-3xl font-bold text-yellow-600">{neutral}</p>
          <p className="text-xs text-gray-500">{((neutral/total)*100).toFixed(1)}%</p>
        </div>

        {/* Negative */}
        <div className="bg-red-50 rounded-lg p-4 border border-red-200">
          <p className="text-sm text-gray-600">😞 Negative</p>
          <p className="text-3xl font-bold text-red-600">{negative}</p>
          <p className="text-xs text-gray-500">{((negative/total)*100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Sentiment Distribution Bar */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Sentiment Distribution</h3>
        <div className="flex w-full h-8 rounded-lg overflow-hidden border border-gray-300">
          <div 
            className="bg-green-500 flex items-center justify-center text-white text-xs font-bold"
            style={{ width: `${(positive/total)*100}%` }}
          >
            {positive > 0 && `${((positive/total)*100).toFixed(0)}%`}
          </div>
          <div 
            className="bg-yellow-500 flex items-center justify-center text-white text-xs font-bold"
            style={{ width: `${(neutral/total)*100}%` }}
          >
            {neutral > 0 && `${((neutral/total)*100).toFixed(0)}%`}
          </div>
          <div 
            className="bg-red-500 flex items-center justify-center text-white text-xs font-bold"
            style={{ width: `${(negative/total)*100}%` }}
          >
            {negative > 0 && `${((negative/total)*100).toFixed(0)}%`}
          </div>
        </div>
      </div>

      {/* Most Common Words */}
      {stats.top_words && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Hash className="w-5 h-5 text-teal-600" />
            <h3 className="text-lg font-semibold text-gray-800">Most Common Words</h3>
          </div>

          {/* Word Category Tabs */}
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setActiveWordTab('all')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeWordTab === 'all'
                  ? 'bg-teal-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              All Words
            </button>
            <button
              onClick={() => setActiveWordTab('positive')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeWordTab === 'positive'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              😊 Positive
            </button>
            <button
              onClick={() => setActiveWordTab('neutral')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeWordTab === 'neutral'
                  ? 'bg-yellow-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              😐 Neutral
            </button>
            <button
              onClick={() => setActiveWordTab('negative')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeWordTab === 'negative'
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              😞 Negative
            </button>
          </div>

          {/* Word Cloud/List */}
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <div className="flex flex-wrap gap-2">
              {Object.entries(
                activeWordTab === 'all' ? stats.top_words :
                activeWordTab === 'positive' ? stats.positive_words :
                activeWordTab === 'neutral' ? stats.neutral_words :
                stats.negative_words
              ).slice(0, 15).map(([word, count]) => (
                <div
                  key={word}
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    activeWordTab === 'positive' ? 'bg-green-100 text-green-700' :
                    activeWordTab === 'neutral' ? 'bg-yellow-100 text-yellow-700' :
                    activeWordTab === 'negative' ? 'bg-red-100 text-red-700' :
                    'bg-blue-100 text-blue-700'
                  }`}
                  style={{ 
                    fontSize: `${Math.min(16, 10 + (count / 10))}px` 
                  }}
                >
                  {word} ({count})
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default SimpleStats;