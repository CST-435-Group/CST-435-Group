import { useState, useEffect, useRef } from 'react';
import '../styles/GeneticAlgorithm.css';
import { GeneticAlgorithm } from '../utils/ga/GeneticAlgorithm';
import ControlPanel from '../components/ga/ControlPanel';
import PopulationDisplay from '../components/ga/PopulationDisplay';
import FitnessChart from '../components/ga/FitnessChart';

export default function GAProject() {
  const [ga, setGa] = useState(null);
  const [population, setPopulation] = useState([]);
  const [generation, setGeneration] = useState(0);
  const [bestEver, setBestEver] = useState(null);
  const [averageFitness, setAverageFitness] = useState(0);
  const [chartData, setChartData] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [target, setTarget] = useState('');
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const startEvolution = (targetPhrase, config) => {
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

  const evolveGeneration = (gaInstance) => {
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
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 py-8">
      <div className="ga-container">
        <div className="ga-left-panel">
          <ControlPanel
            onStart={startEvolution}
            onStop={stopEvolution}
            onReset={resetEvolution}
            isRunning={isRunning}
            isComplete={isComplete}
          />
          {chartData.length > 0 && <FitnessChart data={chartData} />}
        </div>
        <div className="ga-right-panel">
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
