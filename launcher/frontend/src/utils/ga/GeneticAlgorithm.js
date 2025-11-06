import { Individual } from './Individual';

export class GeneticAlgorithm {
  constructor(config) {
    this.config = config;
    this.population = [];
    this.generation = 0;
    this.target = '';
    this.bestEver = null;
  }

  initialize(target) {
    this.target = target;
    this.generation = 0;
    this.population = [];
    this.bestEver = null;

    // Create initial random population
    for (let i = 0; i < this.config.populationSize; i++) {
      const individual = Individual.createRandom(target.length, this.config.charset);
      individual.calculateFitness(target);
      this.population.push(individual);
    }

    // Set initial best
    this.updateBestEver();
  }

  evolve() {
    // Calculate fitness for all individuals
    this.population.forEach(individual => {
      individual.calculateFitness(this.target);
    });

    // Sort population by fitness (descending)
    this.population.sort((a, b) => b.fitness - a.fitness);

    // Update best ever
    this.updateBestEver();

    // Create new population
    const newPopulation = [];

    // Apply elitism - keep the best individuals
    for (let i = 0; i < this.config.elitismCount && i < this.population.length; i++) {
      newPopulation.push(this.population[i]);
    }

    // Fill the rest of the population with offspring
    while (newPopulation.length < this.config.populationSize) {
      const parent1 = this.selectParent();
      const parent2 = this.selectParent();

      let offspring;

      // Apply crossover based on crossover rate
      if (Math.random() < this.config.crossoverRate / 100) {
        offspring = this.crossover(parent1, parent2);
      } else {
        // No crossover, just copy parents
        offspring = [
          new Individual(parent1.genes),
          new Individual(parent2.genes)
        ];
      }

      // Mutate offspring
      offspring.forEach(child => {
        const mutated = child.mutate(this.config.mutationRate, this.config.charset);
        mutated.calculateFitness(this.target);
        if (newPopulation.length < this.config.populationSize) {
          newPopulation.push(mutated);
        }
      });
    }

    this.population = newPopulation;
    this.generation++;
  }

  selectParent() {
    // Tournament selection
    let best = null;

    for (let i = 0; i < this.config.tournamentSize; i++) {
      const randomIndex = Math.floor(Math.random() * this.population.length);
      const individual = this.population[randomIndex];

      if (best === null || individual.fitness > best.fitness) {
        best = individual;
      }
    }

    return best;
  }

  crossover(parent1, parent2) {
    // Single-point crossover
    const crossoverPoint = Math.floor(Math.random() * (parent1.genes.length - 1)) + 1;

    const child1Genes = parent1.genes.slice(0, crossoverPoint) + parent2.genes.slice(crossoverPoint);
    const child2Genes = parent2.genes.slice(0, crossoverPoint) + parent1.genes.slice(crossoverPoint);

    return [
      new Individual(child1Genes),
      new Individual(child2Genes)
    ];
  }

  updateBestEver() {
    const currentBest = this.population[0]; // Already sorted
    if (!this.bestEver || currentBest.fitness > this.bestEver.fitness) {
      this.bestEver = new Individual(currentBest.genes, currentBest.fitness);
    }
  }

  // Getter methods
  getPopulation() {
    return this.population;
  }

  getGeneration() {
    return this.generation;
  }

  getBestIndividual() {
    return this.population.length > 0 ? this.population[0] : null;
  }

  getBestEver() {
    return this.bestEver;
  }

  getAverageFitness() {
    if (this.population.length === 0) return 0;
    const sum = this.population.reduce((acc, ind) => acc + ind.fitness, 0);
    return sum / this.population.length;
  }

  isComplete() {
    const best = this.getBestIndividual();
    return best !== null && best.fitness >= 100;
  }

  getTarget() {
    return this.target;
  }
}
