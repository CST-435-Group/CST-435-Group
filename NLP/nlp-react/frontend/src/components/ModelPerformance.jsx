import { TrendingUp, Award, Target, Cpu } from 'lucide-react';

function ModelPerformance() {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
      <div className="flex items-center gap-2 mb-4">
        <Award className="w-6 h-6 text-teal-600" />
        <h2 className="text-2xl font-bold text-gray-800">Model Performance</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {/* Overall Accuracy */}
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-5 border border-green-300 text-center">
          <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-2" />
          <div className="text-3xl font-bold text-green-700">86.5%</div>
          <div className="text-sm text-green-800 font-medium">Overall Accuracy</div>
        </div>

        {/* Positive Recall */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-5 border border-blue-300 text-center">
          <Target className="w-8 h-8 text-blue-600 mx-auto mb-2" />
          <div className="text-3xl font-bold text-blue-700">97%</div>
          <div className="text-sm text-blue-800 font-medium">Positive Detection</div>
        </div>

        {/* Negative Recall */}
        <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-xl p-5 border border-red-300 text-center">
          <Target className="w-8 h-8 text-red-600 mx-auto mb-2" />
          <div className="text-3xl font-bold text-red-700">83%</div>
          <div className="text-sm text-red-800 font-medium">Negative Detection</div>
        </div>

        {/* Processing Speed */}
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-5 border border-purple-300 text-center">
          <Cpu className="w-8 h-8 text-purple-600 mx-auto mb-2" />
          <div className="text-3xl font-bold text-purple-700">&lt;1s</div>
          <div className="text-sm text-purple-800 font-medium">Analysis Time</div>
        </div>
      </div>

      {/* Performance by Category */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Performance Breakdown</h3>
        <div className="space-y-3">
          {/* Positive */}
          <div className="bg-green-50 rounded-lg p-4 border border-green-200">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xl">😊</span>
                <span className="font-semibold text-gray-800">Positive Reviews</span>
              </div>
              <span className="text-sm font-bold text-green-600">Excellent</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-600">Precision: </span>
                <span className="font-bold">89%</span>
              </div>
              <div>
                <span className="text-gray-600">Recall: </span>
                <span className="font-bold">97%</span>
              </div>
              <div>
                <span className="text-gray-600">F1-Score: </span>
                <span className="font-bold">92%</span>
              </div>
            </div>
          </div>

          {/* Negative */}
          <div className="bg-red-50 rounded-lg p-4 border border-red-200">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xl">😞</span>
                <span className="font-semibold text-gray-800">Negative Reviews</span>
              </div>
              <span className="text-sm font-bold text-red-600">Good</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-600">Precision: </span>
                <span className="font-bold">85%</span>
              </div>
              <div>
                <span className="text-gray-600">Recall: </span>
                <span className="font-bold">83%</span>
              </div>
              <div>
                <span className="text-gray-600">F1-Score: </span>
                <span className="font-bold">84%</span>
              </div>
            </div>
          </div>

          {/* Neutral */}
          <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-200">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xl">😐</span>
                <span className="font-semibold text-gray-800">Neutral Reviews</span>
              </div>
              <span className="text-sm font-bold text-yellow-600">Moderate</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <span className="text-gray-600">Precision: </span>
                <span className="font-bold">73%</span>
              </div>
              <div>
                <span className="text-gray-600">Recall: </span>
                <span className="font-bold">44%</span>
              </div>
              <div>
                <span className="text-gray-600">F1-Score: </span>
                <span className="font-bold">55%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 bg-gray-50 rounded-lg p-3 border border-gray-200">
        <p className="text-xs text-gray-600 text-center">
          💡 <strong>Note:</strong> Neutral reviews are harder to classify due to mixed sentiment signals
        </p>
      </div>
    </div>
  );
}

export default ModelPerformance;