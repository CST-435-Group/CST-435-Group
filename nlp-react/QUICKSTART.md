# 🚀 Quick Start Guide

Get the app running in 5 minutes!

## Prerequisites
- Node.js installed
- Python 3.8+ installed
- Git installed

## Steps

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/nlp-react.git
cd nlp-react
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 3. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Start Backend Server
```bash
cd backend
python main.py
```
Leave this terminal running!

### 5. Start Frontend Server (New Terminal)
```bash
cd frontend
npm run dev
```

### 6. Open in Browser
Go to: http://localhost:3000

## That's It!

You should now see the Sentiment Analyzer app running locally.

## Next Steps
- Test with example reviews
- Read the full README.md for modification instructions
- Deploy to Vercel when ready

## Troubleshooting

**Backend won't start?**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Frontend won't start?**
```bash
cd frontend
rm -rf node_modules
npm install
```

**Port already in use?**
```bash
# Use different port
npm run dev -- --port 3001
```
