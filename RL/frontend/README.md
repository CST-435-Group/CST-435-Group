# RL Platformer - Frontend

Side-scrolling platformer where humans race against trained AI agents.

## Setup

```bash
cd frontend
npm install
```

## Development

```bash
npm start
```

Opens at http://localhost:3000

## Features

- **1920x1080 HD gameplay**
- **Procedurally generated levels** - different map every race
- **Human vs AI racing** - compete against trained RL agent
- **Real-time AI inference** - AI runs in browser via TensorFlow.js
- **Smooth 60 FPS rendering**

## Controls

- **Arrow Keys / WASD**: Move left/right, jump
- **Shift**: Sprint (run faster)
- **Down Arrow / S**: Duck/slide
- **Space**: Jump (alternative)

## How It Works

1. Game generates random map using procedural generation
2. Both human and AI start at same position
3. Human controls player with keyboard
4. AI "sees" game state and decides actions in real-time
5. First to reach the goal wins!

## Project Structure

- `src/components/` - React components (GameCanvas, RaceTracker)
- `src/game/` - Game engine (MapGenerator, Player, Physics, Renderer)
- `src/ai/` - AI integration (ModelLoader, VisionProcessor)
- `src/utils/` - Constants and utilities

## Model Loading

The AI model must be exported from the backend training:

1. Train model: `python backend/training/train_agent.py`
2. Export: `python backend/training/export_model.py`
3. Copy `backend/models/tfjs_model/` to `frontend/public/models/`

The model loads automatically when the game starts.

## Building for Production

```bash
npm run build
```

Output in `build/` directory, ready for deployment.
