import React, { useState } from 'react';
import '../styles/ControlPanel.css';

interface ControlPanelProps {
  onStart: (target: string, config: any) => void;
  onStop: () => void;
  onReset: () => void;
  isRunning: boolean;
  isComplete: boolean;
}

const SHAKESPEARE_QUOTES = [
  "TO BE OR NOT TO BE",
  "ALL THE WORLD'S A STAGE",
  "BREVITY IS THE SOUL OF WIT",
  "THE COURSE OF TRUE LOVE",
  "SOME ARE BORN GREAT",
  "NOW IS THE WINTER",
  "FRIENDS ROMANS COUNTRYMEN",
  "WHEREFORE ART THOU ROMEO",
  "A HORSE A HORSE",
  "DOUBLE DOUBLE TOIL",
];

const ControlPanel: React.FC<ControlPanelProps> = ({
  onStart,
  onStop,
  onReset,
  isRunning,
  isComplete,
}) => {
  const [selectedQuote, setSelectedQuote] = useState(SHAKESPEARE_QUOTES[0]);
  const [customPhrase, setCustomPhrase] = useState('');
  const [useCustom, setUseCustom] = useState(false);
  const [populationSize, setPopulationSize] = useState(200);
  const [mutationRate, setMutationRate] = useState(1);
  const [crossoverRate, setCrossoverRate] = useState(80);
  const [elitismCount, setElitismCount] = useState(2);

  const handleStart = () => {
    const target = useCustom ? customPhrase.toUpperCase() : selectedQuote;

    if (!target || target.trim().length === 0) {
      alert('Please enter a target phrase');
      return;
    }

    const config = {
      populationSize,
      mutationRate,
      crossoverRate,
      elitismCount,
      tournamentSize: 5,
      charset: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ',
    };

    onStart(target, config);
  };

  return (
    <div className="control-panel">
      <h1>🧬 Genetic Algorithm: Shakespeare Evolution</h1>

      <div className="section">
        <h3>Target Phrase</h3>
        <div className="phrase-selection">
          <label>
            <input
              type="radio"
              checked={!useCustom}
              onChange={() => setUseCustom(false)}
            />
            Shakespeare Quotes
          </label>
          <select
            value={selectedQuote}
            onChange={(e) => setSelectedQuote(e.target.value)}
            disabled={useCustom || isRunning}
          >
            {SHAKESPEARE_QUOTES.map((quote, index) => (
              <option key={index} value={quote}>
                {quote}
              </option>
            ))}
          </select>
        </div>

        <div className="phrase-selection">
          <label>
            <input
              type="radio"
              checked={useCustom}
              onChange={() => setUseCustom(true)}
            />
            Custom Phrase
          </label>
          <input
            type="text"
            value={customPhrase}
            onChange={(e) => setCustomPhrase(e.target.value)}
            disabled={!useCustom || isRunning}
            placeholder="Enter your custom phrase..."
            maxLength={50}
          />
        </div>
      </div>

      <div className="section">
        <h3>Algorithm Parameters</h3>

        <div className="parameter">
          <label>
            Population Size: <strong>{populationSize}</strong>
          </label>
          <input
            type="range"
            min="50"
            max="500"
            step="50"
            value={populationSize}
            onChange={(e) => setPopulationSize(Number(e.target.value))}
            disabled={isRunning}
          />
        </div>

        <div className="parameter">
          <label>
            Mutation Rate: <strong>{mutationRate}%</strong>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            step="0.5"
            value={mutationRate}
            onChange={(e) => setMutationRate(Number(e.target.value))}
            disabled={isRunning}
          />
        </div>

        <div className="parameter">
          <label>
            Crossover Rate: <strong>{crossoverRate}%</strong>
          </label>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            value={crossoverRate}
            onChange={(e) => setCrossoverRate(Number(e.target.value))}
            disabled={isRunning}
          />
        </div>

        <div className="parameter">
          <label>
            Elitism Count: <strong>{elitismCount}</strong>
          </label>
          <input
            type="range"
            min="0"
            max="10"
            step="1"
            value={elitismCount}
            onChange={(e) => setElitismCount(Number(e.target.value))}
            disabled={isRunning}
          />
        </div>
      </div>

      <div className="controls">
        {!isRunning && !isComplete && (
          <button className="btn btn-start" onClick={handleStart}>
            Start Evolution
          </button>
        )}
        {isRunning && (
          <button className="btn btn-stop" onClick={onStop}>
            Stop
          </button>
        )}
        {(isComplete || isRunning) && (
          <button className="btn btn-reset" onClick={onReset}>
            Reset
          </button>
        )}
      </div>

      {isComplete && (
        <div className="completion-message">
          🎉 Evolution Complete! Target achieved!
        </div>
      )}
    </div>
  );
};

export default ControlPanel;
