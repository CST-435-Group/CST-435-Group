import { Info, Brain, Zap, Globe } from 'lucide-react';

function InfoSection() {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-2 mb-4">
        <Info className="w-5 h-5 text-blue-600" />
        <h3 className="text-xl font-bold text-gray-800">About</h3>
      </div>

      <div className="space-y-4">
        <div className="flex items-start gap-3">
          <Brain className="w-5 h-5 text-purple-600 flex-shrink-0 mt-1" />
          <div>
            <p className="font-semibold text-gray-800 text-sm">AI-Powered</p>
            <p className="text-xs text-gray-600">
              Uses DistilBERT transformer model for accurate sentiment analysis
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Zap className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-1" />
          <div>
            <p className="font-semibold text-gray-800 text-sm">7-Point Scale</p>
            <p className="text-xs text-gray-600">
              Captures nuanced sentiment from -3 (Very Negative) to +3 (Very Positive)
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Globe className="w-5 h-5 text-green-600 flex-shrink-0 mt-1" />
          <div>
            <p className="font-semibold text-gray-800 text-sm">Real-Time</p>
            <p className="text-xs text-gray-600">
              Instant analysis with confidence scores and probability distributions
            </p>
          </div>
        </div>
      </div>

      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <p className="text-xs text-gray-700 leading-relaxed">
          <strong>How it works:</strong> This app uses a fine-tuned language model to analyze
          the sentiment of movie reviews. It goes beyond simple positive/negative classification
          by providing a nuanced 7-point scale.
        </p>
      </div>

      <div className="mt-4 p-4 bg-purple-50 rounded-lg">
        <p className="text-xs text-gray-700">
          <strong>Tech Stack:</strong> React, FastAPI, PyTorch, Transformers, Tailwind CSS
        </p>
      </div>
    </div>
  );
}

export default InfoSection;
