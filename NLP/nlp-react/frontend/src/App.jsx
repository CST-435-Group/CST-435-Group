import { useState, useEffect } from 'react';
import { Film } from 'lucide-react';
import Header from './components/Header';
import InputSection from './components/InputSection';
import ResultsSection from './components/ResultsSection';
import ExamplesSection from './components/ExamplesSection';
import InfoSection from './components/InfoSection';
import { checkHealth } from './services/api';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    // Check API health on mount
    checkHealth()
      .then(() => setApiStatus('connected'))
      .catch(() => setApiStatus('disconnected'));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* API Status Banner */}
      {apiStatus === 'disconnected' && (
        <div className="bg-red-500 text-white px-4 py-2 text-center text-sm">
          ⚠️ Backend API is not responding. Please ensure the FastAPI server is running on port 8000.
        </div>
      )}

      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <Header apiStatus={apiStatus} />

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
          {/* Left Column - Input and Results */}
          <div className="lg:col-span-2 space-y-6">
            <InputSection
              setResult={setResult}
              setLoading={setLoading}
              loading={loading}
              apiStatus={apiStatus}
            />

            {result && (
              <ResultsSection result={result} loading={loading} />
            )}
          </div>

          {/* Right Column - Examples and Info */}
          <div className="space-y-6">
            <ExamplesSection setResult={setResult} setLoading={setLoading} />
            <InfoSection />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-600 text-sm">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Film className="w-4 h-4" />
            <span>Multi-Scale Sentiment Analyzer</span>
          </div>
          <p>Built with React + FastAPI + Transformers 🤗</p>
          <p className="mt-2 text-xs text-gray-500">AIT-204 Course Project</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
