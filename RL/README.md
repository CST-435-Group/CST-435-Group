# 🎮 RL Platformer - AI vs Human Racing Game

A side-scrolling platformer where you race against a trained reinforcement learning agent on procedurally generated levels.

## 🎯 Features

- **Procedural Generation**: Every race is a new, random level (like 2D Minecraft)
- **Smart Map Design**: Levels flow naturally and are always completable
- **Visual AI**: Agent "sees" the game like a human player
- **Competitive Racing**: Human vs AI to the finish line
- **HD Graphics**: 1920x1080 gameplay with smooth 60 FPS
- **Browser-Based**: No installation needed, runs in web browser

## 🏗️ Project Structure

```
RL/
├── backend/                    # Python training environment
│   ├── training/
│   │   ├── environment.py      # Custom Gym environment
│   │   ├── map_generator.py    # Procedural level generation
│   │   ├── train_agent.py      # RL training script
│   │   └── export_model.py     # Export to TensorFlow.js
│   ├── models/                 # Saved models (created during training)
│   └── utils/
│       └── config.py           # Game constants
│
└── frontend/                   # React + TypeScript game
    ├── src/
    │   ├── components/         # React components
    │   │   ├── GameCanvas.tsx  # Main game rendering
    │   │   └── RaceTracker.tsx # Race progress UI
    │   ├── game/               # Game engine
    │   │   ├── MapGenerator.ts # Client-side map generation
    │   │   ├── Player.ts       # Player class
    │   │   ├── Physics.ts      # Physics engine
    │   │   └── Renderer.ts     # Canvas rendering
    │   ├── ai/                 # AI integration
    │   │   ├── ModelLoader.ts  # TensorFlow.js model loader
    │   │   └── VisionProcessor.ts # Game state to AI observation
    │   └── utils/
    │       └── constants.ts    # Game constants
    └── public/
        └── models/             # Exported TensorFlow.js model
```

## 🚀 Quick Start

### 1. Train the AI Agent (Backend)

```bash
cd backend
pip install -r requirements.txt
python training/train_agent.py
```

**Training time:** 4-8 hours on GPU for 1M steps

### 2. Export Model for Web

```bash
python training/export_model.py
```

This converts the PyTorch model to TensorFlow.js format.

### 3. Setup Frontend

```bash
cd frontend
npm install
```

Copy exported model:
```bash
cp -r ../backend/models/tfjs_model/ public/models/
```

### 4. Run the Game

```bash
npm start
```

Open http://localhost:3000 and race against the AI!

## 🎮 How to Play

1. **Click "Start Race"** to begin
2. **Use keyboard controls:**
   - Arrow Keys / WASD: Move and jump
   - Shift: Sprint faster
   - Down: Duck under obstacles
3. **Race the AI** to the goal flag
4. **Click "New Race"** to generate a new random level

## 🧠 How the AI Works

### Training Phase (Backend - Python):
1. Custom Gym environment with procedural levels
2. Agent receives visual observation (84x84 downscaled view)
3. PPO algorithm learns optimal movement strategy
4. Trained on hundreds of random levels
5. Model saved as PyTorch checkpoint

### Inference Phase (Frontend - Browser):
1. Model loaded via TensorFlow.js
2. AI "sees" game state through VisionProcessor
3. Real-time predictions (60+ FPS)
4. Actions applied same way as human input
5. No server needed - runs entirely in browser!

## 🗺️ Procedural Generation

Maps are randomly generated but guaranteed to be playable:

- **Perlin noise** for smooth terrain
- **Smart platform placement** - gaps are always jumpable
- **Playability verification** - goal is reachable from spawn
- **Difficulty scaling** - later sections get harder
- **Collectibles & enemies** placed strategically

Same generation algorithm in Python (training) and TypeScript (racing).

## 🎨 Current State (Placeholders)

- Players: White (human) and red (AI) rectangles
- Platforms: Green rectangles
- Coins: Yellow circles
- Enemies: Purple squares
- Goal: Green star/flag

**TODO:** Replace with sprite sheets later!

## 📊 Technologies

**Backend:**
- Python 3.8+
- PyTorch
- Stable-Baselines3 (RL library)
- OpenAI Gym (environment framework)
- Pygame (rendering during training)

**Frontend:**
- React 18 with TypeScript
- TensorFlow.js (AI inference)
- HTML5 Canvas (rendering)
- No external game engines

## 🎓 Why This is Cool

1. **Real AI learning** - not scripted behavior
2. **Actually sees the game** - visual input like humans
3. **Adapts to randomness** - not memorizing levels
4. **Browser-based ML** - no backend needed for gameplay
5. **Fair competition** - AI has same abilities as human

## 🐛 Development Status

**Completed:**
- ✅ Full project structure
- ✅ All file scaffolding with function signatures
- ✅ Architecture design

**TODO (Implementation):**
- ⬜ Backend: Implement environment, map generation, training
- ⬜ Frontend: Implement game engine, rendering, AI integration
- ⬜ Test and tune RL training
- ⬜ Export and test model in browser
- ⬜ Add sprites and polish

## 📝 Next Steps

1. **Implement backend training first**
   - Start with `map_generator.py` (test map generation)
   - Then `environment.py` (test with random actions)
   - Finally `train_agent.py` (full training)

2. **Export and test model**
   - Convert to TensorFlow.js
   - Test in Node.js first

3. **Implement frontend**
   - Start with map rendering
   - Add player movement
   - Integrate AI last

## 🤝 Contributing

This is part of the CST-435 Neural Networks project showcasing 7 different AI models.

---

**Ready to race against AI?** 🏁

*Note: All files are scaffolded with function signatures - implementation coming next!*
