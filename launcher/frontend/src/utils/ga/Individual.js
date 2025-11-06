export class Individual {
  constructor(genes, fitness = 0) {
    this.genes = genes;
    this.fitness = fitness;
  }

  static createRandom(length, charset) {
    let genes = '';
    for (let i = 0; i < length; i++) {
      const randomIndex = Math.floor(Math.random() * charset.length);
      genes += charset[randomIndex];
    }
    return new Individual(genes);
  }

  calculateFitness(target) {
    let matches = 0;
    for (let i = 0; i < this.genes.length; i++) {
      if (this.genes[i] === target[i]) {
        matches++;
      }
    }
    this.fitness = (matches / target.length) * 100;
  }

  mutate(mutationRate, charset) {
    let newGenes = '';
    for (let i = 0; i < this.genes.length; i++) {
      if (Math.random() < mutationRate / 100) {
        // Mutate this character
        const randomIndex = Math.floor(Math.random() * charset.length);
        newGenes += charset[randomIndex];
      } else {
        // Keep the same character
        newGenes += this.genes[i];
      }
    }
    return new Individual(newGenes);
  }
}
