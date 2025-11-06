import { IndividualClass } from '../types/Individual';

export interface GAConfig {
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
  elitismCount: number;
  tournamentSize: number;
  charset: string;
}

export class GeneticAlgorithm {
  private config: GAConfig;
  private population: IndividualClass[];
  private generation: number;
  private target: string;
  private bestEver: IndividualClass | null;

  constructor(config: GAConfig) {
    this.config = config;
    this.population = [];
    this.generation = 0;
    this.target = '';
    this.bestEver = null;
  }

  initialize(target: string): void {
    this.target = target;
    this.generation = 0;
    this.population = [];
    this.bestEver = null;

    // Create initial random population
    for (let i = 0; i < this.config.populationSize; i++) {
      const individual = IndividualClass.createRandom(target.length, this.config.charset);
      individual.calculateFitness(target);
      this.population.push(individual);
    }

    // Set initial best
    this.updateBestEver();
  }

  evolve(): void {
    // Calculate fitness for all individuals
    this.population.forEach(individual => {
      individual.calculateFitness(this.target);
    });

    // Sort population by fitness (descending)
    this.population.sort((a, b) => b.fitness - a.fitness);

    // Update best ever
    this.updateBestEver();

    // Create new population
    const newPopulation: IndividualClass[] = [];

    // Apply elitism - keep the best individuals
    for (let i = 0; i < this.config.elitismCount && i < this.population.length; i++) {
      newPopulation.push(this.population[i]);
    }

    // Fill the rest of the population with offspring
    while (newPopulation.length < this.config.populationSize) {
      const parent1 = this.selectParent();
      const parent2 = this.selectParent();

      let offspring: IndividualClass[];

      // Apply crossover based on crossover rate
      if (Math.random() < this.config.crossoverRate / 100) {
        offspring = this.crossover(parent1, parent2);
      } else {
        // No crossover, just copy parents
        offspring = [
          new IndividualClass(parent1.genes),
          new IndividualClass(parent2.genes)
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

  private selectParent(): IndividualClass {
    // Tournament selection
    let best: IndividualClass | null = null;

    for (let i = 0; i < this.config.tournamentSize; i++) {
      const randomIndex = Math.floor(Math.random() * this.population.length);
      const individual = this.population[randomIndex];

      if (best === null || individual.fitness > best.fitness) {
        best = individual;
      }
    }

    return best!;
  }

  private crossover(parent1: IndividualClass, parent2: IndividualClass): IndividualClass[] {
    // Single-point crossover
    const crossoverPoint = Math.floor(Math.random() * (parent1.genes.length - 1)) + 1;

    const child1Genes = parent1.genes.slice(0, crossoverPoint) + parent2.genes.slice(crossoverPoint);
    const child2Genes = parent2.genes.slice(0, crossoverPoint) + parent1.genes.slice(crossoverPoint);

    return [
      new IndividualClass(child1Genes),
      new IndividualClass(child2Genes)
    ];
  }

  private updateBestEver(): void {
    const currentBest = this.population[0]; // Already sorted
    if (!this.bestEver || currentBest.fitness > this.bestEver.fitness) {
      this.bestEver = new IndividualClass(currentBest.genes, currentBest.fitness);
    }
  }

  // Getter methods
  getPopulation(): IndividualClass[] {
    return this.population;
  }

  getGeneration(): number {
    return this.generation;
  }

  getBestIndividual(): IndividualClass | null {
    return this.population.length > 0 ? this.population[0] : null;
  }

  getBestEver(): IndividualClass | null {
    return this.bestEver;
  }

  getAverageFitness(): number {
    if (this.population.length === 0) return 0;
    const sum = this.population.reduce((acc, ind) => acc + ind.fitness, 0);
    return sum / this.population.length;
  }

  isComplete(): boolean {
    const best = this.getBestIndividual();
    return best !== null && best.fitness >= 100;
  }

  getTarget(): string {
    return this.target;
  }
}
