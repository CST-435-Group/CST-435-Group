import { Film, Activity } from 'lucide-react';

function Header({ apiStatus }) {
  return (
    <header className="text-center">
      <div className="flex items-center justify-center gap-3 mb-4">
        <Film className="w-12 h-12 text-blue-600" />
        <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Sentiment Analyzer
        </h1>
      </div>

      <p className="text-xl text-gray-700 mb-4">
        Analyze movie reviews on a 7-point sentiment scale
      </p>

      <div className="flex items-center justify-center gap-2 text-sm">
        <Activity className={`w-4 h-4 ${
          apiStatus === 'connected' ? 'text-green-500' :
          apiStatus === 'disconnected' ? 'text-red-500' :
          'text-yellow-500'
        }`} />
        <span className={`${
          apiStatus === 'connected' ? 'text-green-600' :
          apiStatus === 'disconnected' ? 'text-red-600' :
          'text-yellow-600'
        }`}>
          {apiStatus === 'connected' && 'AI Model Ready'}
          {apiStatus === 'disconnected' && 'API Disconnected'}
          {apiStatus === 'checking' && 'Connecting...'}
        </span>
      </div>

      <div className="mt-6 bg-white rounded-lg shadow-md p-4 max-w-3xl mx-auto">
        <div className="grid grid-cols-7 gap-2 text-center text-xs">
          {[
            { score: '-3', emoji: '😢', label: 'Very Negative' },
            { score: '-2', emoji: '😞', label: 'Negative' },
            { score: '-1', emoji: '😐', label: 'Slightly Negative' },
            { score: '0', emoji: '😶', label: 'Neutral' },
            { score: '+1', emoji: '🙂', label: 'Slightly Positive' },
            { score: '+2', emoji: '😊', label: 'Positive' },
            { score: '+3', emoji: '🤩', label: 'Very Positive' },
          ].map((item) => (
            <div key={item.score} className="flex flex-col items-center">
              <span className="text-2xl mb-1">{item.emoji}</span>
              <span className="font-semibold text-gray-700">{item.score}</span>
              <span className="text-gray-500 hidden sm:block">{item.label}</span>
            </div>
          ))}
        </div>
      </div>
    </header>
  );
}

export default Header;
