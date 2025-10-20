import { useState } from 'react';
import { Search, List, BarChart3, Info } from 'lucide-react';
import InputSection from './InputSection';
import ResultsSection from './ResultsSection';
import ExamplesSection from './ExamplesSection';
import SimpleStats from './SimpleStats';
import InfoSection from './InfoSection';

export default function Dashboard({ apiStatus, result, setResult, loading, setLoading }) {
  const [activeTab, setActiveTab] = useState('analyze');

  const tabs = [
    { id: 'analyze', label: 'Analyze', icon: Search },
    { id: 'examples', label: 'Examples', icon: List },
    { id: 'stats', label: 'Statistics', icon: BarChart3 },
    { id: 'info', label: 'Info', icon: Info },
  ];

  return (
    <div className="mt-8">
      {/* Tab Navigation */}
      <div className="flex space-x-4 border-b border-gray-300 mb-4">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-white border-x border-t border-gray-300 text-blue-600'
                  : 'text-gray-600 hover:text-blue-500'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="bg-white p-6 rounded-b-xl shadow">
        {activeTab === 'analyze' && (
          <>
            <InputSection setResult={setResult} setLoading={setLoading} loading={loading} apiStatus={apiStatus} />
            {result && <ResultsSection result={result} loading={loading} />}
          </>
        )}

        {activeTab === 'examples' && <ExamplesSection setResult={setResult} setLoading={setLoading} />}
        {activeTab === 'stats' && <SimpleStats />}
        {activeTab === 'info' && <InfoSection />}
      </div>
    </div>
  );
} 
