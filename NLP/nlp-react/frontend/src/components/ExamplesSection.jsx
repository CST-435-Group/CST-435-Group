import { useState, useEffect } from 'react';
import { Sparkles, Loader2 } from 'lucide-react';
import { getExamples, analyzeSentiment } from '../services/api';

function ExamplesSection({ setResult, setLoading }) {
  const [examples, setExamples] = useState(null);
  const [loadingExamples, setLoadingExamples] = useState(true);

  useEffect(() => {
    loadExamples();
  }, []);

  const loadExamples = async () => {
    try {
      const data = await getExamples();
      // Transform backend format {"-3": [...], "0": [...], "3": [...]}
      // to frontend format {negative: [...], neutral: [...], positive: [...]}
      const transformed = {
        positive: [...(data["2"] || []), ...(data["3"] || [])],
        neutral: [...(data["-1"] || []), ...(data["0"] || []), ...(data["1"] || [])],
        negative: [...(data["-3"] || []), ...(data["-2"] || [])]
      };
      setExamples(transformed);
    } catch (error) {
      console.error('Failed to load examples:', error);
    } finally {
      setLoadingExamples(false);
    }
  };

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

  if (loadingExamples) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-teal-500" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100">
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-teal-600" />
        <h2 className="text-xl font-bold text-gray-800">
          Try Examples
        </h2>
      </div>

      <p className="text-sm text-gray-600 mb-4">
        Click to analyze hospital review examples
      </p>

      <div className="space-y-3">
        {/* Positive Examples */}
        {examples?.positive && examples.positive.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-green-600 mb-2 flex items-center gap-1">
              😊 Positive
            </h3>
            <div className="space-y-2">
              {examples.positive.map((example, index) => (
                <button
                  key={`pos-${index}`}
                  onClick={() => handleExampleClick(example)}
                  className="w-full text-left p-3 bg-green-50 hover:bg-green-100 border border-green-200 rounded-lg text-sm text-gray-700 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Neutral Examples */}
        {examples?.neutral && examples.neutral.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-yellow-600 mb-2 flex items-center gap-1">
              😐 Neutral
            </h3>
            <div className="space-y-2">
              {examples.neutral.map((example, index) => (
                <button
                  key={`neu-${index}`}
                  onClick={() => handleExampleClick(example)}
                  className="w-full text-left p-3 bg-yellow-50 hover:bg-yellow-100 border border-yellow-200 rounded-lg text-sm text-gray-700 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Negative Examples */}
        {examples?.negative && examples.negative.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-red-600 mb-2 flex items-center gap-1">
              😞 Negative
            </h3>
            <div className="space-y-2">
              {examples.negative.map((example, index) => (
                <button
                  key={`neg-${index}`}
                  onClick={() => handleExampleClick(example)}
                  className="w-full text-left p-3 bg-red-50 hover:bg-red-100 border border-red-200 rounded-lg text-sm text-gray-700 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ExamplesSection;