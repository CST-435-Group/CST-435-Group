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

  // Get sentiment info from result
  const sentimentScore = result.sentiment_score; // -3 to +3
  const sentimentLabel = result.sentiment_label; // e.g., "positive" (condensed)
  const sentimentLabelVerbose = result.sentiment_label_verbose || null; // optional verbose label
  const emoji = result.emoji; // e.g., "🤩"
  
  // Determine sentiment display based on score
  const getSentimentDisplay = (score) => {
    if (score >= 2) {
      return {
        icon: TrendingUp,
        color: 'text-green-600',
        bgColor: 'bg-green-50',
        borderColor: 'border-green-200',
        barColor: 'bg-green-500'
      };
    } else if (score >= 1) {
      return {
        icon: TrendingUp,
        color: 'text-green-500',
        bgColor: 'bg-green-50',
        borderColor: 'border-green-200',
        barColor: 'bg-green-400'
      };
    } else if (score === 0) {
      return {
        icon: Minus,
        color: 'text-yellow-600',
        bgColor: 'bg-yellow-50',
        borderColor: 'border-yellow-200',
        barColor: 'bg-yellow-500'
      };
    } else if (score >= -1) {
      return {
        icon: TrendingDown,
        color: 'text-red-500',
        bgColor: 'bg-red-50',
        borderColor: 'border-red-200',
        barColor: 'bg-red-400'
      };
    } else {
      return {
        icon: TrendingDown,
        color: 'text-red-600',
        bgColor: 'bg-red-50',
        borderColor: 'border-red-200',
        barColor: 'bg-red-500'
      };
    }
  };

  const display = getSentimentDisplay(sentimentScore);
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
              <span className="text-5xl">{emoji}</span>
              <div>
                <h3 className={`text-3xl font-bold ${display.color}`}>
                  {sentimentLabelVerbose ? sentimentLabelVerbose : (sentimentLabel.charAt(0).toUpperCase() + sentimentLabel.slice(1))}
                </h3>
                <p className="text-sm text-gray-600">
                  Score: {sentimentScore > 0 ? '+' : ''}{sentimentScore}/3
                </p>
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
                className={`h-3 rounded-full transition-all duration-500 ${display.barColor}`}
                style={{ width: `${result.confidence * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Sentiment Scale Visualization */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          Sentiment Scale
        </h3>
        <div className="relative">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-gray-500">Very Negative</span>
            <span className="text-xs text-gray-500">Neutral</span>
            <span className="text-xs text-gray-500">Very Positive</span>
          </div>
          <div className="relative h-8 bg-gradient-to-r from-red-500 via-yellow-400 to-green-500 rounded-full">
            {/* Score marker */}
            <div
              className="absolute top-1/2 -translate-y-1/2 w-6 h-6 bg-white border-4 border-gray-800 rounded-full shadow-lg transition-all duration-500"
              style={{ left: `${((sentimentScore + 3) / 6) * 100}%`, transform: 'translate(-50%, -50%)' }}
            ></div>
          </div>
          <div className="flex items-center justify-between mt-1 text-xs text-gray-600">
            <span>-3</span>
            <span>-2</span>
            <span>-1</span>
            <span>0</span>
            <span>+1</span>
            <span>+2</span>
            <span>+3</span>
          </div>
        </div>
      </div>

      {/* Probability Distribution */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          Probability Distribution
        </h3>
        <div className="space-y-2">
          {Object.entries(result.probabilities).map(([key, probability]) => {
            // probabilities now expected as: { negative, neutral, positive }
            const mapping = { negative: -2, neutral: 0, positive: 2 };
            const representativeScore = mapping[key] ?? 0;
            const display = getSentimentDisplay(representativeScore);
            const labelText = key.charAt(0).toUpperCase() + key.slice(1);

            return (
              <div key={key} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-gray-700">
                    {labelText}
                  </span>
                  <span className={`font-semibold ${display.color}`}>
                    {(probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-500 ${display.barColor}`}
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