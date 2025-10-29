import React, { useState, useEffect } from 'react';
import { generateText, GenerateResponse, getAvailableModels, switchModel, AvailableModel } from '../services/api';

interface TextGeneratorProps {
  onModelChange?: () => void;
}

const TextGenerator: React.FC<TextGeneratorProps> = ({ onModelChange }) => {
  const [seedText, setSeedText] = useState('');
  const [numWords, setNumWords] = useState(50);
  const [temperature, setTemperature] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [switchingModel, setSwitchingModel] = useState(false);

  // Beam search state
  const [useBeamSearch, setUseBeamSearch] = useState(false);
  const [beamWidth, setBeamWidth] = useState(5);
  const [lengthPenalty, setLengthPenalty] = useState(1.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.2);
  const [beamTemperature, setBeamTemperature] = useState(0.0);

  // Post-processing
  const [addPunctuation, setAddPunctuation] = useState(false);
  const [validateGrammar, setValidateGrammar] = useState(false);

  // Load available models on component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await getAvailableModels();
        setAvailableModels(response.models);
        if (response.current_model) {
          setSelectedModel(response.current_model);
        }
      } catch (err) {
        console.error('Failed to load available models:', err);
      }
    };
    loadModels();
  }, []);

  const handleModelChange = async (modelName: string) => {
    if (modelName === selectedModel) return;

    setSwitchingModel(true);
    setError(null);

    try {
      await switchModel(modelName);
      setSelectedModel(modelName);
      // Notify parent component to refresh model status
      if (onModelChange) {
        onModelChange();
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to switch model');
    } finally {
      setSwitchingModel(false);
    }
  };

  const handleGenerate = async () => {
    if (!seedText.trim()) {
      setError('Please enter some seed text');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await generateText({
        seed_text: seedText,
        num_words: numWords,
        temperature: temperature,
        use_beam_search: useBeamSearch,
        beam_width: useBeamSearch ? beamWidth : undefined,
        length_penalty: useBeamSearch ? lengthPenalty : undefined,
        repetition_penalty: useBeamSearch ? repetitionPenalty : undefined,
        beam_temperature: useBeamSearch ? beamTemperature : undefined,
        add_punctuation: addPunctuation,
        validate_grammar: validateGrammar,
      });
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate text');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 transition-colors duration-200">
        <h2 className="text-3xl font-bold mb-6 text-indigo-600 dark:text-indigo-400">
          Generate Text
        </h2>

        {/* Model Selection */}
        {availableModels.length > 0 && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Select Model:
            </label>
            <select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={switchingModel}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            >
              {availableModels.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.display_name}
                </option>
              ))}
            </select>
            {switchingModel && (
              <p className="text-sm text-indigo-600 dark:text-indigo-400 mt-2">Switching model...</p>
            )}
          </div>
        )}

        {/* Seed Text Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Seed Text (starting words):
          </label>
          <textarea
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            rows={4}
            value={seedText}
            onChange={(e) => setSeedText(e.target.value)}
            placeholder="Enter some text to start with..."
          />
        </div>

        {/* Number of Words */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Number of Words: {numWords}
          </label>
          <input
            type="range"
            min="10"
            max="200"
            value={numWords}
            onChange={(e) => setNumWords(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Generation Method Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Generation Method:
          </label>
          <div className="flex gap-4">
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                name="generationMethod"
                checked={!useBeamSearch}
                onChange={() => setUseBeamSearch(false)}
                className="w-4 h-4 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="ml-2 text-gray-700 dark:text-gray-300">
                Sampling (Random)
              </span>
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                name="generationMethod"
                checked={useBeamSearch}
                onChange={() => setUseBeamSearch(true)}
                className="w-4 h-4 text-indigo-600 focus:ring-indigo-500"
              />
              <span className="ml-2 text-gray-700 dark:text-gray-300">
                Beam Search (Deterministic)
              </span>
            </label>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
            {useBeamSearch
              ? 'Beam search finds the most probable sequence (deterministic, more coherent)'
              : 'Sampling randomly selects words based on probability (varies each run, more creative)'}
          </p>
        </div>

        {/* Temperature Slider (only for sampling) */}
        {!useBeamSearch && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Creativity (Temperature): {temperature.toFixed(1)}
            </label>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Lower = more predictable, Higher = more creative
            </p>
          </div>
        )}

        {/* Beam Search Parameters (only for beam search) */}
        {useBeamSearch && (
          <>
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Beam Width: {beamWidth}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                value={beamWidth}
                onChange={(e) => setBeamWidth(parseInt(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Number of candidate sequences to explore (higher = more thorough search)
              </p>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Length Penalty: {lengthPenalty.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={lengthPenalty}
                onChange={(e) => setLengthPenalty(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Lower = prefer shorter sequences, Higher = prefer longer sequences
              </p>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Repetition Penalty: {repetitionPenalty.toFixed(1)}
              </label>
              <input
                type="range"
                min="1.0"
                max="3.0"
                step="0.1"
                value={repetitionPenalty}
                onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Higher = less repetition (prevents loops, default: 1.2)
              </p>
            </div>

            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Randomness (Beam Temperature): {beamTemperature.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.0"
                max="1.5"
                step="0.1"
                value={beamTemperature}
                onChange={(e) => setBeamTemperature(parseFloat(e.target.value))}
                className="w-full"
              />
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                0.0 = deterministic (same result every time), 0.5-1.0 = varied outputs
              </p>
            </div>
          </>
        )}

        {/* Post-processing Options */}
        <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600">
          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">Post-Processing & Validation</h3>

          <label className="flex items-center cursor-pointer mb-3">
            <input
              type="checkbox"
              checked={addPunctuation}
              onChange={(e) => setAddPunctuation(e.target.checked)}
              className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
              Add punctuation & capitalization
            </span>
          </label>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-4 ml-6">
            Adds periods, commas, and capitalizes sentences using AI model
          </p>

          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={validateGrammar}
              onChange={(e) => setValidateGrammar(e.target.checked)}
              className="w-4 h-4 text-indigo-600 rounded focus:ring-indigo-500"
            />
            <span className="ml-2 text-sm text-gray-700 dark:text-gray-300">
              Validate grammar during generation
            </span>
          </label>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 ml-6">
            Checks grammatical structure (verbs, nouns, word order) - slower but more grammatical output
          </p>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={loading}
          className="w-full bg-indigo-600 dark:bg-indigo-500 text-white py-3 px-6 rounded-lg font-semibold hover:bg-indigo-700 dark:hover:bg-indigo-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 transition-colors"
        >
          {loading ? 'Generating...' : 'Generate Text'}
        </button>

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-100 dark:bg-red-900/30 border border-red-400 dark:border-red-800 text-red-700 dark:text-red-300 rounded-lg">
            {error}
          </div>
        )}

        {/* Generated Text */}
        {result && (
          <div className="mt-6 p-6 bg-gray-50 dark:bg-gray-700/50 rounded-lg border-l-4 border-indigo-500 dark:border-indigo-400">
            <div className="flex justify-between items-start mb-3">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Generated Text:</h3>
              <span className="text-xs bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 px-2 py-1 rounded">
                {result.use_beam_search ? 'Beam Search' : 'Sampling'}
              </span>
            </div>
            {result.use_beam_search && result.beam_width && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                Beam Width: {result.beam_width} | Length Penalty: {result.length_penalty?.toFixed(1)} | Repetition Penalty: {result.repetition_penalty?.toFixed(1)}
                {result.beam_temperature !== undefined && result.beam_temperature > 0 && (
                  <> | Randomness: {result.beam_temperature.toFixed(1)}</>
                )}
              </p>
            )}
            {!result.use_beam_search && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                Temperature: {result.temperature.toFixed(1)}
              </p>
            )}
            <p className="text-gray-800 dark:text-gray-200 leading-relaxed whitespace-pre-wrap">
              {result.generated_text}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TextGenerator;
