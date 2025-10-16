import { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { analyzeSentiment } from '../services/api';

function InputSection({ setResult, setLoading, loading, apiStatus }) {
  const [text, setText] = useState('');
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    if (apiStatus !== 'connected') {
      setError('API is not connected. Please check the backend server.');
      return;
    }

    setError('');
    setLoading(true);
    setResult(null);

    try {
      const result = await analyzeSentiment(text);
      setResult(result);
    } catch (err) {
      setError(err.message || 'Failed to analyze sentiment');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleAnalyze();
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        📝 Enter Your Review
      </h2>

      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setError('');
        }}
        onKeyPress={handleKeyPress}
        placeholder="Type or paste a movie review here...&#10;&#10;Example: 'This movie was incredible! The acting was superb and the plot kept me engaged throughout.'"
        className="w-full h-40 p-4 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none resize-none text-gray-700"
        disabled={loading}
      />

      {error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
          {error}
        </div>
      )}

      <div className="mt-4 flex items-center justify-between">
        <span className="text-sm text-gray-500">
          {text.length} characters • Press Ctrl+Enter to analyze
        </span>

        <button
          onClick={handleAnalyze}
          disabled={loading || !text.trim() || apiStatus !== 'connected'}
          className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Send className="w-5 h-5" />
              Analyze Sentiment
            </>
          )}
        </button>
      </div>
    </div>
  );
}

export default InputSection;
