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
      // Backend may return either 7-point buckets ("-3".."3") or three buckets (negative/neutral/positive).
      let negativeExamples = [];
      let neutralExamples = [];
      let positiveExamples = [];

      if (data && (data.negative || data.neutral || data.positive)) {
        // New backend shape
        negativeExamples = data.negative || [];
        neutralExamples = data.neutral || [];
        positiveExamples = data.positive || [];
      } else {
        // Old backend shape: aggregate -3..+3
        negativeExamples = [
          ...(data["-3"] || []),
          ...(data["-2"] || []),
          ...(data["-1"] || []),
        ];
        neutralExamples = [...(data["0"] || [])];
        positiveExamples = [
          ...(data["1"] || []),
          ...(data["2"] || []),
          ...(data["3"] || []),
        ];
      }

      // Build three buckets (positive, neutral, negative). Backend now returns these keys.
      const positive = data.positive || positiveExamples || [];
      const neutral = data.neutral || neutralExamples || [];
      const negative = data.negative || negativeExamples || [];

      const transformed = [
        { key: "positive", label: "😊 Positive", color: "green", examples: positive },
        { key: "neutral", label: "😐 Neutral", color: "yellow", examples: neutral },
        { key: "negative", label: "😞 Negative", color: "red", examples: negative }
      ];

      // Filter out empty sections but keep order
      setExamples(transformed.filter(s => s.examples && s.examples.length > 0));
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
        {examples.map((section) => (
          section.examples.length > 0 && (
            <div key={section.key}>
              <h3 className={`text-sm font-semibold text-${section.color}-600 mb-2 flex items-center gap-1`}>
                {section.label}
              </h3>
              <div className="space-y-2">
                {section.examples.map((example, idx) => (
                  <button
                    key={`${section.key}-${idx}`}
                    onClick={() => handleExampleClick(example)}
                    className={`w-full text-left p-3 bg-${section.color}-50 hover:bg-${section.color}-100 border border-${section.color}-200 rounded-lg text-sm text-gray-700 transition-colors`}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )
        ))}
      </div>
    </div>
  );
}

export default ExamplesSection;
