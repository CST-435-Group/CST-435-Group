import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, Award, BarChart3 } from 'lucide-react';

function ResultsSection({ result }) {
  if (!result) return null;

  const { sentiment_score, sentiment_label, emoji, confidence, probabilities } = result;

  // Prepare data for chart
  const chartData = Object.entries(probabilities).map(([label, prob]) => ({
    label: label.split(' ')[0], // Get just the score like "+3"
    probability: (prob * 100).toFixed(1),
    fullLabel: label,
  }));

  // Color mapping for sentiment scores
  const getColor = (score) => {
    const scoreNum = parseInt(score);
    if (scoreNum <= -2) return '#ef4444'; // red
    if (scoreNum === -1) return '#f97316'; // orange
    if (scoreNum === 0) return '#eab308'; // yellow
    if (scoreNum === 1) return '#84cc16'; // lime
    return '#22c55e'; // green
  };

  const getSentimentColor = (score) => {
    if (score <= -2) return 'from-red-500 to-red-600';
    if (score === -1) return 'from-orange-500 to-orange-600';
    if (score === 0) return 'from-yellow-500 to-yellow-600';
    if (score === 1) return 'from-lime-500 to-lime-600';
    return 'from-green-500 to-green-600';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 animate-fadeIn">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">
        📊 Analysis Results
      </h2>

      {/* Main Result Card */}
      <div className={`bg-gradient-to-r ${getSentimentColor(sentiment_score)} rounded-xl p-6 text-white mb-6 shadow-lg`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm opacity-90 mb-1">Sentiment Score</p>
            <p className="text-5xl font-bold">
              {sentiment_score > 0 ? '+' : ''}{sentiment_score}/3
            </p>
          </div>
          <div className="text-right">
            <p className="text-6xl mb-2">{emoji}</p>
            <p className="text-xl font-semibold">{sentiment_label}</p>
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 rounded-lg p-4 text-center">
          <Award className="w-8 h-8 text-blue-600 mx-auto mb-2" />
          <p className="text-2xl font-bold text-blue-900">
            {(confidence * 100).toFixed(1)}%
          </p>
          <p className="text-sm text-blue-600">Confidence</p>
        </div>

        <div className="bg-purple-50 rounded-lg p-4 text-center">
          <TrendingUp className="w-8 h-8 text-purple-600 mx-auto mb-2" />
          <p className="text-2xl font-bold text-purple-900">
            {sentiment_score > 0 ? 'Positive' : sentiment_score < 0 ? 'Negative' : 'Neutral'}
          </p>
          <p className="text-sm text-purple-600">Category</p>
        </div>

        <div className="bg-indigo-50 rounded-lg p-4 text-center">
          <BarChart3 className="w-8 h-8 text-indigo-600 mx-auto mb-2" />
          <p className="text-2xl font-bold text-indigo-900">
            7-Point
          </p>
          <p className="text-sm text-indigo-600">Scale</p>
        </div>
      </div>

      {/* Probability Distribution Chart */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Probability Distribution
        </h3>

        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
                      <p className="font-semibold text-gray-800">{payload[0].payload.fullLabel}</p>
                      <p className="text-blue-600">{payload[0].value}%</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="probability" radius={[8, 8, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={getColor(entry.label)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Probabilities */}
      <details className="mt-4">
        <summary className="cursor-pointer text-blue-600 hover:text-blue-800 font-semibold">
          🔬 Show Detailed Probabilities
        </summary>
        <div className="mt-3 space-y-2">
          {Object.entries(probabilities).map(([label, prob]) => (
            <div key={label} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <span className="text-sm font-medium text-gray-700">{label}</span>
              <span className="text-sm text-gray-600">{(prob * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

export default ResultsSection;
