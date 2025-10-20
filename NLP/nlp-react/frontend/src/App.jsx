import { useState, useEffect } from 'react';
import { Activity, FileText, Brain } from 'lucide-react';
import Header from './components/Header';
import InputSection from './components/InputSection';
import ResultsSection from './components/ResultsSection';
import ExamplesSection from './components/ExamplesSection';
import SimpleStats from './components/SimpleStats';
import InfoSection from './components/InfoSection';
import AboutNLP from './components/AboutNLP';
import { checkHealth } from './services/api';
import ModelPerformance from './components/ModelPerformance';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [activeTab, setActiveTab] = useState('analysis');

  useEffect(() => {
    checkHealth()
      .then(() => setApiStatus('connected'))
      .catch(() => setApiStatus('disconnected'));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-blue-50 to-teal-50">
      {apiStatus === 'disconnected' && (
        <div className="bg-red-500 text-white px-4 py-2 text-center text-sm">
          ⚠️ Backend API is not responding. Please ensure the FastAPI server is running on port 8000.
        </div>
      )}

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <Header apiStatus={apiStatus} />

        {/* Tab Navigation */}
        <div className="flex items-center justify-center gap-4 mt-8">
          <button
            onClick={() => setActiveTab('analysis')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'analysis'
                ? 'bg-teal-600 text-white shadow-lg'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            <FileText className="w-5 h-5" />
            Analysis Tool
          </button>
          <button
            onClick={() => setActiveTab('about')}
            className={`flex items-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all ${
              activeTab === 'about'
                ? 'bg-teal-600 text-white shadow-lg'
                : 'bg-white text-gray-600 hover:bg-gray-50'
            }`}
          >
            <Brain className="w-5 h-5" />
            About NLP
          </button>
        </div>

        {/* Analysis Tab Content */}
        {activeTab === 'analysis' && (
          <div className="space-y-6 mt-8">
            {/* Top Row: Input & Results (LEFT) | Examples & Info (RIGHT) */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* LEFT COLUMN - Input and Results */}
              <div className="lg:col-span-2 space-y-6">
                <InputSection
                  setResult={setResult}
                  setLoading={setLoading}
                  loading={loading}
                  apiStatus={apiStatus}
                />
                {result && <ResultsSection result={result} loading={loading} />}
              </div>

              {/* RIGHT COLUMN - Examples and Info */}
              <div className="space-y-6">
                <ExamplesSection setResult={setResult} setLoading={setLoading} />
                <InfoSection />
              </div>
            </div>

            {/* Bottom Section: Stats and Performance */}
            <div className="space-y-6">
              <SimpleStats />
              <ModelPerformance />
            </div>
          </div>
        )}

        {/* About NLP Tab Content */}
        {activeTab === 'about' && (
          <div className="mt-8">
            <AboutNLP />
          </div>
        )}

        <footer className="mt-16 text-center text-gray-600 text-sm">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Activity className="w-4 h-4" />
            <span>Hospital Review Sentiment Analyzer</span>
          </div>
          <p>Built with React + FastAPI + NLTK + Scikit-learn</p>
          <p className="mt-2 text-xs text-gray-500">AIT-204 Course Project</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
