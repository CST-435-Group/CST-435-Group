# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational activity package for teaching genetic algorithms through an interactive Shakespeare text evolution project. Students implement a React + TypeScript application that uses genetic algorithms to evolve random text into Shakespeare quotes.

**Important**: The `shakespeare-ga/` directory is initially empty. Students create the React application from scratch following the instructions in `STUDENT-INSTRUCTIONS.md`. This is a learning exercise, not a pre-built application.

## Development Workflow

### Initial Setup (For Students)
```bash
# Create React app with TypeScript template
npx create-react-app shakespeare-ga --template typescript
cd shakespeare-ga
npm start
```

### Running the Application
```bash
cd shakespeare-ga
npm start          # Development server at http://localhost:3000
npm test           # Run tests (if implemented)
npm run build      # Production build
```

### Deployment
```bash
# Option 1: Vercel
npm install -g vercel
vercel --prod

# Option 2: Render.com
# Push to GitHub, then deploy via Render dashboard
# Build: npm install && npm run build
# Publish: build
```

## Architecture

### Core Components Structure

The application follows this architecture (once built by students):

```
shakespeare-ga/
├── src/
│   ├── types/
│   │   └── Individual.ts          # Individual/Chromosome class with genes and fitness
│   ├── algorithms/
│   │   └── GeneticAlgorithm.ts    # Core GA implementation (population, selection, crossover, mutation)
│   ├── components/
│   │   ├── ControlPanel.tsx       # User controls for starting/stopping, parameter adjustment
│   │   ├── PopulationDisplay.tsx  # Real-time evolution visualization with color-coded characters
│   │   └── FitnessChart.tsx       # Canvas-based fitness progress chart
│   └── App.tsx                    # Main component coordinating evolution loop and state
```

### Genetic Algorithm Implementation

**Individual Class** (`types/Individual.ts`):
- `genes`: String representing the chromosome (candidate solution)
- `fitness`: Percentage of characters matching target (0-100)
- `createRandom()`: Generates random individual from charset
- `calculateFitness()`: Compares genes to target character-by-character
- `mutate()`: Returns new individual with random character changes based on mutation rate

**GeneticAlgorithm Class** (`algorithms/GeneticAlgorithm.ts`):
- Manages population of individuals across generations
- **Selection**: Tournament selection (pick N random, return fittest)
- **Crossover**: Single-point crossover (random split point, swap tails)
- **Mutation**: Random character replacement based on mutation rate
- **Elitism**: Top N individuals automatically survive to next generation
- **Evolution cycle**: Calculate fitness → Sort → Elitism → Selection → Crossover → Mutation

### Key Parameters

- **Population Size**: 50-500 (default: 200) - Number of individuals per generation
- **Mutation Rate**: 0-10% (default: 1%) - Probability each character mutates
- **Crossover Rate**: 0-100% (default: 80%) - Probability parents produce offspring
- **Elitism Count**: 0-10 (default: 2) - Top individuals preserved unchanged
- **Tournament Size**: 5 - Number of individuals competing in selection
- **Charset**: Uppercase A-Z + space (27 characters total)

### React State Management

The main App component maintains:
- GA instance (GeneticAlgorithm object)
- Current population array
- Generation counter
- Chart data (best/average fitness per generation)
- Running/complete status flags
- Evolution interval/animation frame ID

Evolution loop typically runs via `setInterval()` or `requestAnimationFrame()` calling `evolve()` repeatedly until target fitness reaches 100%.

## Problem Domain

**The Challenge**: Evolve random text into Shakespeare quotes
- Example target: "TO BE OR NOT TO BE" (18 characters)
- Search space: 27^18 ≈ 7.6 × 10^25 possible combinations
- Brute force: ~2.4 billion years at 1 billion attempts/second
- Genetic algorithm: Typically solves in 200-500 generations (~seconds)

**Why Genetic Algorithms Excel Here**:
- Massive search space makes exhaustive search impossible
- Simple fitness function (character matching percentage)
- Population explores multiple solution paths simultaneously
- No domain knowledge needed - pure evolutionary optimization

## Typical Performance

For "TO BE OR NOT TO BE":
- Population 100, Mutation 1%: ~300-500 generations
- Population 200, Mutation 1%: ~200-400 generations
- Population 500, Mutation 1%: ~150-300 generations

Lower mutation rates converge faster but risk local optima.
Higher mutation rates maintain diversity but converge slower.

## Common Implementation Pitfalls

1. **Off-by-one errors** in crossover point selection (ensure valid split)
2. **Mutation always changing** instead of probabilistic (check random < rate)
3. **Forgetting to recalculate fitness** after crossover/mutation
4. **Not preserving elites** before generating new population
5. **Character case sensitivity** (target and genes must match case)
6. **Infinite loops** if evolution runs without completion check

## Extension Ideas (For Advanced Students)

- Roulette wheel selection (fitness-proportionate)
- Rank-based selection
- Two-point or uniform crossover
- Adaptive mutation rate (decrease over time)
- Parallel populations with migration
- Web Workers for background evolution
- CSV export of statistics

## Troubleshooting

**Evolution stuck at 90-95% fitness**: Increase mutation rate to 1.5-2% to escape local optimum

**Very slow convergence**: Reduce population size or increase elitism count

**App crashes with large populations**: Limit display to top N individuals, not entire population

**Build fails on deployment**: Ensure all imports use correct relative paths and file extensions
