import { TrendingUp, TrendingDown, Minus, BarChart3 } from 'lucide-react';

function ResultsSection({ result, loading }) {
  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-100">
        <div className="flex items-center justify-center">
          <div className="animate-pulse space-y-4 w-full">
            <div className="h-8 bg-gray-200 rounded w-3/4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
            <div className="h-32 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  // Debug: Log the result to console
  console.log('Result received:', result);

  // Get sentiment from result
  const sentiment = result.sentiment || 'neutral';
  
  // Determine sentiment display
  const getSentimentDisplay = (sentiment) => {
    const sentimentLower = sentiment.toLowerCase();
    
    if (sentimentLower === 'positive') {
      return {
        emoji: '😊',
        icon: TrendingUp,
        color: 'text-green-600',
        bgColor: 'bg-green-50',
        borderColor: 'border-green-200',
        label: 'Positive'
      };
    } else if (sentimentLower === 'negative') {
      return {
        emoji: '😞',
        icon: TrendingDown,
        color: 'text-red-600',
        bgColor: 'bg-red-50',
        borderColor: 'border-red-200',
        label: 'Negative'
      };
    } else {
      return {
        emoji: '😐',
        icon: Minus,
        color: 'text-yellow-600',
        bgColor: 'bg-yellow-50',
        borderColor: 'border-yellow-200',
        label: 'Neutral'
      };
    }
  };

  const display = getSentimentDisplay(sentiment);
  const Icon = display.icon;

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 space-y-6">
      {/* Main Result */}
      <div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center gap-2">
          <BarChart3 className="w-6 h-6" />
          Analysis Results
        </h2>

        <div className={`${display.bgColor} ${display.borderColor} border-2 rounded-xl p-6`}>
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <span className="text-5xl">{display.emoji}</span>
              <div>
                <h3 className={`text-3xl font-bold ${display.color}`}>
                  {display.label}
                </h3>
                <p className="text-sm text-gray-600">Sentiment Classification</p>
              </div>
            </div>
            <Icon className={`w-12 h-12 ${display.color}`} />
          </div>

          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">Confidence:</span>
              <span className={`text-lg font-bold ${display.color}`}>
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all duration-500 ${
                  sentiment === 'positive' ? 'bg-green-500' :
                  sentiment === 'negative' ? 'bg-red-500' :
                  'bg-yellow-500'
                }`}
                style={{ width: `${result.confidence * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Probability Distribution */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          Probability Distribution
        </h3>
        <div className="space-y-3">
          {Object.entries(result.probabilities).map(([sentiment, probability]) => {
            const sentimentDisplay = getSentimentDisplay(sentiment);
            return (
              <div key={sentiment} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <span>{sentimentDisplay.emoji}</span>
                    <span className="font-medium text-gray-700 capitalize">
                      {sentiment}
                    </span>
                  </div>
                  <span className={`font-semibold ${sentimentDisplay.color}`}>
                    {(probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${
                      sentiment === 'positive' ? 'bg-green-500' :
                      sentiment === 'negative' ? 'bg-red-500' :
                      'bg-yellow-500'
                    }`}
                    style={{ width: `${probability * 100}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Review Text */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-2">
          Analyzed Review
        </h3>
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <p className="text-gray-700 italic">&ldquo;{result.text}&rdquo;</p>
        </div>
      </div>
    </div>
  );
}

export default ResultsSection;