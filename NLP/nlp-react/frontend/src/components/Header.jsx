import { Activity, CheckCircle, XCircle, Clock } from 'lucide-react';

function Header({ apiStatus }) {
  return (
    <div className="text-center">
      <div className="flex items-center justify-center gap-3 mb-4">
        <Activity className="w-12 h-12 text-teal-600" />
        <h1 className="text-5xl font-bold bg-gradient-to-r from-teal-600 to-blue-600 bg-clip-text text-transparent">
          Hospital Review Analyzer
        </h1>
      </div>
      
      <p className="text-xl text-gray-600 mb-4">
        AI-Powered Patient Feedback Analysis using NLTK & Machine Learning
      </p>

      {/* API Status Indicator */}
      <div className="flex items-center justify-center gap-2 text-sm">
        {apiStatus === 'checking' && (
          <>
            <Clock className="w-4 h-4 text-yellow-500 animate-spin" />
            <span className="text-yellow-600">Checking API...</span>
          </>
        )}
        {apiStatus === 'connected' && (
          <>
            <CheckCircle className="w-4 h-4 text-green-500" />
            <span className="text-green-600">Model Ready</span>
          </>
        )}
        {apiStatus === 'disconnected' && (
          <>
            <XCircle className="w-4 h-4 text-red-500" />
            <span className="text-red-600">API Disconnected</span>
          </>
        )}
      </div>

      {/* Remove the old 7-point scale, add simple 3-category display */}
      <div className="mt-6 flex items-center justify-center gap-4">
        <div className="flex items-center gap-2 px-4 py-2 bg-green-50 rounded-lg border border-green-200">
          <span className="text-2xl">😊</span>
          <span className="text-sm font-medium text-green-700">Positive</span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-yellow-50 rounded-lg border border-yellow-200">
          <span className="text-2xl">😐</span>
          <span className="text-sm font-medium text-yellow-700">Neutral</span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-red-50 rounded-lg border border-red-200">
          <span className="text-2xl">😞</span>
          <span className="text-sm font-medium text-red-700">Negative</span>
        </div>
      </div>
    </div>
  );
}

export default Header;