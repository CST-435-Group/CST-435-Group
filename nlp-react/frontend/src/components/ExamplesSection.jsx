import { useState, useEffect } from 'react';
import { Sparkles } from 'lucide-react';
import { getExamples, analyzeSentiment } from '../services/api';

function ExamplesSection({ setResult, setLoading }) {
  const [examples, setExamples] = useState(null);

  useEffect(() => {
    getExamples()
      .then(setExamples)
      .catch(console.error);
  }, []);

  const handleExampleClick = async (text) => {
    setLoading(true);
    try {
      const result = await analyzeSentiment(text);
      setResult(result);
    } catch (error) {
      console.error('Failed to analyze example:', error);
    } finally {
      setLoading(false);
    }
  };

  if (!examples) return null;

  const exampleButtons = [
    { score: '3', emoji: '🤩', color: 'from-green-500 to-green-600', hoverColor: 'hover:from-green-600 hover:to-green-700' },
    { score: '2', emoji: '😊', color: 'from-lime-500 to-lime-600', hoverColor: 'hover:from-lime-600 hover:to-lime-700' },
    { score: '0', emoji: '😶', color: 'from-yellow-500 to-yellow-600', hoverColor: 'hover:from-yellow-600 hover:to-yellow-700' },
    { score: '-2', emoji: '😞', color: 'from-orange-500 to-orange-600', hoverColor: 'hover:from-orange-600 hover:to-orange-700' },
    { score: '-3', emoji: '😢', color: 'from-red-500 to-red-600', hoverColor: 'hover:from-red-600 hover:to-red-700' },
  ];

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-purple-600" />
        <h3 className="text-xl font-bold text-gray-800">Try Examples</h3>
      </div>

      <p className="text-sm text-gray-600 mb-4">
        Click to analyze pre-written reviews
      </p>

      <div className="space-y-3">
        {exampleButtons.map(({ score, emoji, color, hoverColor }) => {
          const exampleText = examples[score]?.[0];
          if (!exampleText) return null;

          return (
            <button
              key={score}
              onClick={() => handleExampleClick(exampleText)}
              className={`w-full text-left p-4 bg-gradient-to-r ${color} ${hoverColor} text-white rounded-lg transition-all shadow-md hover:shadow-lg`}
            >
              <div className="flex items-start gap-3">
                <span className="text-2xl flex-shrink-0">{emoji}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold mb-1">
                    Score: {score}/3
                  </p>
                  <p className="text-xs opacity-90 line-clamp-2">
                    {exampleText}
                  </p>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default ExamplesSection;
