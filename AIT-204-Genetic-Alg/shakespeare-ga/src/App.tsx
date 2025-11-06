import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { GeneticAlgorithm, GAConfig } from './algorithms/GeneticAlgorithm';
import { IndividualClass } from './types/Individual';
import ControlPanel from './components/ControlPanel';
import PopulationDisplay from './components/PopulationDisplay';
import FitnessChart, { DataPoint } from './components/FitnessChart';

function App() {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [ga, setGa] = useState<GeneticAlgorithm | null>(null);
  const [population, setPopulation] = useState<IndividualClass[]>([]);
  const [generation, setGeneration] = useState(0);
  const [bestEver, setBestEver] = useState<IndividualClass | null>(null);
  const [averageFitness, setAverageFitness] = useState(0);
  const [chartData, setChartData] = useState<DataPoint[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [target, setTarget] = useState('');
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const startEvolution = (targetPhrase: string, config: GAConfig) => {
    // Create and initialize GA
    const newGa = new GeneticAlgorithm(config);
    newGa.initialize(targetPhrase);

    setGa(newGa);
    setTarget(targetPhrase);
    setPopulation(newGa.getPopulation());
    setGeneration(newGa.getGeneration());
    setBestEver(newGa.getBestEver());
    setAverageFitness(newGa.getAverageFitness());
    setChartData([
      {
        generation: 0,
        bestFitness: newGa.getBestEver()?.fitness || 0,
        averageFitness: newGa.getAverageFitness(),
      },
    ]);
    setIsRunning(true);
    setIsComplete(false);

    // Start evolution loop
    intervalRef.current = window.setInterval(() => {
      evolveGeneration(newGa);
    }, 50);
  };

  const evolveGeneration = (gaInstance: GeneticAlgorithm) => {
    if (gaInstance.isComplete()) {
      stopEvolution();
      setIsComplete(true);
      return;
    }

    gaInstance.evolve();

    // Update state
    setPopulation([...gaInstance.getPopulation()]);
    setGeneration(gaInstance.getGeneration());
    setBestEver(gaInstance.getBestEver());
    setAverageFitness(gaInstance.getAverageFitness());

    // Update chart data
    setChartData((prevData) => [
      ...prevData,
      {
        generation: gaInstance.getGeneration(),
        bestFitness: gaInstance.getBestEver()?.fitness || 0,
        averageFitness: gaInstance.getAverageFitness(),
      },
    ]);
  };

  const stopEvolution = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setIsRunning(false);
  };

  const resetEvolution = () => {
    stopEvolution();
    setGa(null);
    setPopulation([]);
    setGeneration(0);
    setBestEver(null);
    setAverageFitness(0);
    setChartData([]);
    setIsComplete(false);
    setTarget('');
  };

  return (
    <div className="App">
      <div className="container">
        <div className="left-panel">
          <ControlPanel
            onStart={startEvolution}
            onStop={stopEvolution}
            onReset={resetEvolution}
            isRunning={isRunning}
            isComplete={isComplete}
          />
          {chartData.length > 0 && <FitnessChart data={chartData} />}
        </div>
        <div className="right-panel">
          {population.length > 0 && (
            <PopulationDisplay
              population={population}
              target={target}
              generation={generation}
              bestEver={bestEver}
              averageFitness={averageFitness}
              isComplete={isComplete}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
