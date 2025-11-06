export default function PopulationDisplay({
  population,
  target,
  generation,
  bestEver,
  averageFitness,
  isComplete,
}) {
  const renderColoredText = (genes, target) => {
    return genes.split('').map((char, index) => {
      const isMatch = char === target[index];
      return (
        <span
          key={index}
          className={isMatch ? 'ga-char-match' : 'ga-char-mismatch'}
        >
          {char}
        </span>
      );
    });
  };

  const displayCount = Math.min(population.length, 20);
  const topIndividuals = population.slice(0, displayCount);

  return (
    <div className="ga-population-display">
      <div className="ga-stats-section">
        <h2>Evolution Status</h2>
        <div className="ga-stats">
          <div className="ga-stat">
            <span className="ga-stat-label">Generation:</span>
            <span className="ga-stat-value">{generation}</span>
          </div>
          <div className="ga-stat">
            <span className="ga-stat-label">Best Fitness:</span>
            <span className="ga-stat-value">
              {bestEver ? bestEver.fitness.toFixed(2) : 0}%
            </span>
          </div>
          <div className="ga-stat">
            <span className="ga-stat-label">Average Fitness:</span>
            <span className="ga-stat-value">{averageFitness.toFixed(2)}%</span>
          </div>
        </div>
      </div>

      <div className="ga-target-section">
        <h3>Target Phrase:</h3>
        <div className="ga-target-phrase">{target}</div>
      </div>

      {bestEver && (
        <div className="ga-best-section">
          <h3>Best Individual Ever:</h3>
          <div className="ga-individual ga-best-individual">
            <div className="ga-genes">
              {renderColoredText(bestEver.genes, target)}
            </div>
            <div className="ga-fitness">{bestEver.fitness.toFixed(2)}%</div>
          </div>
        </div>
      )}

      {isComplete && (
        <div className="ga-complete-message">
          <h2>🎉 Evolution Complete!</h2>
          <p>Successfully evolved to match the target phrase!</p>
        </div>
      )}

      {population.length > 0 && (
        <div className="ga-population-section">
          <h3>Top {displayCount} Individuals (Generation {generation}):</h3>
          <div className="ga-population-list">
            {topIndividuals.map((individual, index) => (
              <div key={index} className="ga-individual">
                <div className="ga-rank">#{index + 1}</div>
                <div className="ga-genes">
                  {renderColoredText(individual.genes, target)}
                </div>
                <div className="ga-fitness">{individual.fitness.toFixed(2)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
