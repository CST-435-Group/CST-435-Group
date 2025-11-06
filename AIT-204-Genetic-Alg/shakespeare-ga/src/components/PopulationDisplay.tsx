import React from 'react';
import { IndividualClass } from '../types/Individual';
import '../styles/PopulationDisplay.css';

interface PopulationDisplayProps {
  population: IndividualClass[];
  target: string;
  generation: number;
  bestEver: IndividualClass | null;
  averageFitness: number;
  isComplete: boolean;
}

const PopulationDisplay: React.FC<PopulationDisplayProps> = ({
  population,
  target,
  generation,
  bestEver,
  averageFitness,
  isComplete,
}) => {
  const renderColoredText = (genes: string, target: string) => {
    return genes.split('').map((char, index) => {
      const isMatch = char === target[index];
      return (
        <span
          key={index}
          className={isMatch ? 'char-match' : 'char-mismatch'}
        >
          {char}
        </span>
      );
    });
  };

  const displayCount = Math.min(population.length, 20);
  const topIndividuals = population.slice(0, displayCount);

  return (
    <div className="population-display">
      <div className="stats-section">
        <h2>Evolution Status</h2>
        <div className="stats">
          <div className="stat">
            <span className="stat-label">Generation:</span>
            <span className="stat-value">{generation}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Best Fitness:</span>
            <span className="stat-value">
              {bestEver ? bestEver.fitness.toFixed(2) : 0}%
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Average Fitness:</span>
            <span className="stat-value">{averageFitness.toFixed(2)}%</span>
          </div>
        </div>
      </div>

      <div className="target-section">
        <h3>Target Phrase:</h3>
        <div className="target-phrase">{target}</div>
      </div>

      {bestEver && (
        <div className="best-section">
          <h3>Best Individual Ever:</h3>
          <div className="individual best-individual">
            <div className="genes">
              {renderColoredText(bestEver.genes, target)}
            </div>
            <div className="fitness">{bestEver.fitness.toFixed(2)}%</div>
          </div>
        </div>
      )}

      {isComplete && (
        <div className="complete-message">
          <h2>🎉 Evolution Complete!</h2>
          <p>Successfully evolved to match the target phrase!</p>
        </div>
      )}

      {population.length > 0 && (
        <div className="population-section">
          <h3>Top {displayCount} Individuals (Generation {generation}):</h3>
          <div className="population-list">
            {topIndividuals.map((individual, index) => (
              <div key={index} className="individual">
                <div className="rank">#{index + 1}</div>
                <div className="genes">
                  {renderColoredText(individual.genes, target)}
                </div>
                <div className="fitness">{individual.fitness.toFixed(2)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PopulationDisplay;
