# 🎮 RL Platformer - Testing Guide

## ✅ All Changes Pushed to Git!

The single-player platformer is ready to test. All code has been committed and pushed to the repository.

**Git Commit:** `b4c8cad4` - "Add RL Platformer - Single Player Game"

---

## 🚀 How to Test the Game

### Step 1: Start the Launcher Backend

```bash
cd launcher/backend
python main.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start the Launcher Frontend

Open a **new terminal**:

```bash
cd launcher/frontend
npm start
```

Browser should open at `http://localhost:3000`

### Step 3: Navigate to RL Project

1. Click the **"RL"** tab in the header (gamepad icon)
2. Or go directly to: `http://localhost:3000/rl`

### Step 4: Start Playing!

1. Click the **"🎮 Start Game"** button
2. Use the controls to play

---

## 🎮 Game Controls

- **← →** or **A D** - Move left/right
- **Space** or **↑** or **W** - Jump
- **Shift** - Sprint (move faster)
- **↓** or **S** - Duck (not fully implemented yet)

---

## 🎯 Game Objectives

1. **Reach the goal flag** 🏁 (green with gold flag)
2. **Collect yellow coins** 💰 for points
3. **Avoid purple enemies** 👾 (they kill you on contact)
4. **Don't fall off** the platforms

---

## 🎨 What You'll See

### Visuals (Simple Placeholders):
- **White square** = You (the player)
- **Green rectangles** = Platforms (you can stand on these)
- **Brown rectangle** = Ground
- **Yellow circles** = Coins (collect for +10 points)
- **Purple squares** = Enemies (avoid these!)
- **Green rectangle with 🏁** = Goal (reach this to win!)

### UI Elements:
- **Top left:** Score and Distance traveled
- **Top right:** Control hints
- **On win/death:** Overlay with final stats and restart option

---

## ✨ Features to Test

### Core Gameplay:
- ✅ **Movement** - Walk left and right
- ✅ **Jumping** - Jump over gaps and onto platforms
- ✅ **Sprinting** - Hold Shift for faster movement
- ✅ **Gravity & Physics** - Realistic falling
- ✅ **Platform collision** - Stand on platforms correctly

### Game Mechanics:
- ✅ **Coin collection** - Touch coins to collect (+10 points)
- ✅ **Enemy collision** - Touch enemy = death
- ✅ **Goal detection** - Reach flag = win
- ✅ **Death by falling** - Fall off map = death
- ✅ **Score tracking** - Real-time score display
- ✅ **Distance tracking** - How far you traveled

### Map Generation:
- ✅ **Random platforms** - Different layout every game
- ✅ **Jumpable gaps** - All platforms are reachable
- ✅ **Varied heights** - Platforms at different levels
- ✅ **Long level** - ~50 platforms to the goal

### UI/UX:
- ✅ **Smooth camera** - Follows player horizontally
- ✅ **Win screen** - Shows when you reach goal
- ✅ **Death screen** - Shows when you die
- ✅ **Restart** - Click "Play Again" to restart
- ✅ **Back to menu** - Return to intro screen

---

## 🐛 Known Issues / Things to Note

### Current State:
1. **No AI yet** - This is single-player only
2. **Simple graphics** - Just colored rectangles (easy to replace later)
3. **Basic enemies** - They just walk back and forth
4. **No sounds** - Audio not implemented
5. **No power-ups** - Just coins for now

### Expected Behavior:
- Game runs at **60 FPS** in browser
- **Random map** generated each time you restart
- **Smooth controls** and responsive jumping
- **Instant restart** on death/win

---

## 🎯 Testing Checklist

Try these scenarios:

### Basic Movement:
- [ ] Walk left and right
- [ ] Jump in place
- [ ] Jump while moving
- [ ] Sprint (should be noticeably faster)

### Platforming:
- [ ] Jump from one platform to another
- [ ] Jump across gaps
- [ ] Land on narrow platforms
- [ ] Jump to higher platforms

### Collectibles:
- [ ] Collect a coin (score should increase by 10)
- [ ] Collect multiple coins
- [ ] Watch distance increase as you move right

### Enemies:
- [ ] Touch an enemy (should die and see game over)
- [ ] Try to jump over an enemy

### Win/Lose:
- [ ] Die by touching enemy (game over screen)
- [ ] Die by falling off map (game over screen)
- [ ] Reach the goal flag (win screen)
- [ ] Click "Play Again" (new random map)
- [ ] Click "Back to Menu" (return to intro)

### Edge Cases:
- [ ] Try to jump off left side of screen
- [ ] Jump very high
- [ ] Sprint and jump together
- [ ] Fall through small gaps

---

## 📊 Performance Check

The game should:
- ✅ Run smoothly at 60 FPS
- ✅ No lag or stuttering
- ✅ Instant key response
- ✅ Smooth camera movement
- ✅ Fast restart (<1 second)

If you see lag:
- Check browser console (F12) for errors
- Try in Chrome/Edge (best performance)
- Close other tabs

---

## 🔧 If Something Breaks

### Game won't load:
1. Check browser console (F12 → Console tab)
2. Look for JavaScript errors
3. Make sure both backend and frontend are running

### Controls don't work:
1. Click on the game canvas to focus it
2. Try refreshing the page
3. Make sure you're using arrow keys or WASD

### Player falls through platforms:
- This is a known physics bug with fast falling
- Should be rare, but might need tweaking

### Camera feels weird:
- Camera follows player with slight smoothing
- This is intentional for cinematic feel

---

## 📝 Feedback to Provide

When testing, please note:

1. **Does it feel fun?** Is the jumping satisfying?
2. **Is it too hard/easy?** Should gaps be bigger/smaller?
3. **Controls responsive?** Do keys feel good?
4. **Graphics clear?** Can you tell what everything is?
5. **Any bugs?** List anything that breaks
6. **Performance?** Does it run smoothly?

---

## 🎓 Next Steps (After Testing)

Once you confirm the game works well:

### 1. Implement Backend Training Environment
- Match the frontend game mechanics exactly
- Same physics, same collision detection
- Visual observation generation (84x84 grayscale)

### 2. Train the AI Agent
```bash
cd RL/backend
python training/train_agent.py --timesteps 1000000
```
- Uses GPU if available
- Takes 4-8 hours on GPU
- Model learns to play by watching pixels

### 3. Export Model for Web
```bash
python training/export_model.py
```
- Converts PyTorch → TensorFlow.js

### 4. Add AI to Frontend
- Load TensorFlow.js model
- Add AI player rendering
- Implement race mode
- Split screen or side-by-side view

---

## 🎉 Have Fun Testing!

This is a fully playable platformer game. Try to beat it!

**Remember:** Each restart generates a completely new random level.

---

**Questions or Issues?**
- Check browser console for errors
- Make sure backend is running
- Try restarting both backend and frontend

**Ready to train AI?**
- After testing, we'll implement the training environment
- Then train an agent that can beat this game!
