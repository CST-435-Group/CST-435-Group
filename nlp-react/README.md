# 🎬 Multi-Scale Sentiment Analyzer - Class Activity

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [What is React?](#what-is-react)
- [Learning Objectives](#learning-objectives)
- [Prerequisites](#prerequisites)
- [Part 1: Understanding the Project](#part-1-understanding-the-project)
- [Part 2: Setting Up Your Environment](#part-2-setting-up-your-environment)
- [Part 3: Running the Application Locally](#part-3-running-the-application-locally)
- [Part 4: Making Your Modifications](#part-4-making-your-modifications)
- [Part 5: Deploying to Vercel](#part-5-deploying-to-vercel)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

---

## 🎯 Project Overview

This is a **full-stack web application** that analyzes movie reviews and determines their sentiment on a 7-point scale from -3 (Very Negative) to +3 (Very Positive).

### What Does It Do?

**Input:** "This movie was absolutely phenomenal! Best film of the decade!"

**Output:**
- Sentiment Score: **+3/3**
- Label: **Very Positive 🤩**
- Confidence: **87%**
- Probability distribution across all sentiment levels

### Technology Stack

- **Frontend:** React with Vite, Tailwind CSS
- **Backend:** FastAPI (Python)
- **AI Model:** DistilBERT (Hugging Face Transformers)
- **Deployment:** Vercel Cloud

---

## ⚛️ What is React?

### Understanding React

**React** is a JavaScript library for building user interfaces, particularly for web applications. Think of it as a tool that helps you create interactive websites efficiently.

#### Key Concepts:

1. **Components:** React apps are built from small, reusable pieces called components
   - Like LEGO blocks - each component is a self-contained piece
   - Example: A button, a form, a navigation bar

2. **JSX:** A syntax that looks like HTML but works in JavaScript
   ```jsx
   const greeting = <h1>Hello, World!</h1>;
   ```

3. **State:** Data that can change in your app
   - When state changes, React automatically updates what users see
   - Example: The text in an input box, whether a button is clicked

4. **Props:** How components pass data to each other
   - Like function parameters
   - Parent components send data to child components

#### Why React?

- ✅ **Fast:** Only updates the parts of the page that change
- ✅ **Modular:** Break complex UIs into simple, reusable components
- ✅ **Popular:** Huge community, lots of resources
- ✅ **Industry Standard:** Used by Facebook, Netflix, Instagram, etc.

### Traditional Web vs React

**Traditional HTML/JavaScript:**
```html
<!-- You manually update the DOM -->
<div id="greeting">Hello</div>
<script>
  document.getElementById('greeting').textContent = 'Hello, User!';
</script>
```

**React:**
```jsx
// React automatically updates when state changes
function Greeting() {
  const [name, setName] = useState('User');
  return <div>Hello, {name}!</div>;
}
```

---

## 🎓 Learning Objectives

By completing this activity, you will:

1. ✅ Understand the basics of React and modern web development
2. ✅ Learn how frontend (React) and backend (FastAPI) communicate
3. ✅ Gain hands-on experience with Git and GitHub
4. ✅ Deploy a full-stack application to the cloud
5. ✅ Modify and customize an existing codebase
6. ✅ Understand RESTful APIs and HTTP requests

---

## 📋 Prerequisites

### Required Accounts
- [ ] **GitHub Account** - [Sign up here](https://github.com/signup)
- [ ] **Vercel Account** - [Sign up here](https://vercel.com/signup) (use your GitHub account)

### Required Software

#### 1. **Node.js** (JavaScript Runtime)
- **What it is:** Allows you to run JavaScript on your computer
- **Download:** https://nodejs.org/ (LTS version)
- **Verify installation:**
  ```bash
  node --version  # Should show v18.0.0 or higher
  npm --version   # Should show v9.0.0 or higher
  ```

#### 2. **Python** (for Backend)
- **What it is:** Programming language for the AI backend
- **Download:** https://www.python.org/ (3.8 or higher)
- **Verify installation:**
  ```bash
  python --version  # Should show 3.8 or higher
  pip --version     # Should show pip version
  ```

#### 3. **Git** (Version Control)
- **What it is:** Tracks changes to your code
- **Download:** https://git-scm.com/
- **Verify installation:**
  ```bash
  git --version  # Should show git version
  ```

#### 4. **Code Editor**
- **Recommended:** Visual Studio Code (VS Code)
- **Download:** https://code.visualstudio.com/
- **Alternative:** Any text editor works

### Recommended VS Code Extensions
- ESLint (JavaScript linting)
- Prettier (Code formatting)
- Tailwind CSS IntelliSense (CSS autocomplete)
- Python (Python support)

---

## 📚 Part 1: Understanding the Project

### Project Structure

```
nlp-react/
│
├── frontend/                    # React application
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── Header.jsx       # Top navigation/title
│   │   │   ├── InputSection.jsx # Text input area
│   │   │   ├── ResultsSection.jsx # Shows analysis results
│   │   │   ├── ExamplesSection.jsx # Example reviews
│   │   │   └── InfoSection.jsx  # About information
│   │   ├── services/
│   │   │   └── api.js           # API communication functions
│   │   ├── App.jsx              # Main application component
│   │   ├── main.jsx             # Entry point
│   │   └── index.css            # Global styles
│   ├── package.json             # Dependencies & scripts
│   └── vite.config.js           # Build configuration
│
├── backend/                     # FastAPI server
│   ├── main.py                  # API endpoints
│   ├── model.py                 # AI model logic
│   └── requirements.txt         # Python dependencies
│
├── vercel.json                  # Deployment configuration
├── package.json                 # Root package file
└── README.md                    # This file!
```

### How It Works

```
┌─────────┐      HTTP Request      ┌─────────┐      Model Inference      ┌─────────┐
│         │ ───────────────────────>│         │ ───────────────────────>│         │
│ React   │  { "text": "movie" }   │ FastAPI │  Process with AI Model  │ DistilBERT │
│ Frontend│ <───────────────────────│ Backend │ <───────────────────────│  Model  │
│         │  { "score": +3, ... }  │         │  Return predictions     │         │
└─────────┘                        └─────────┘                         └─────────┘
```

1. **User types** a movie review in the React frontend
2. **Frontend sends** the text to the FastAPI backend via HTTP
3. **Backend processes** the text using the AI model
4. **Model analyzes** and returns sentiment scores
5. **Backend sends** results back to frontend
6. **Frontend displays** results with charts and visualizations

---

## 🔧 Part 2: Setting Up Your Environment

### Step 1: Fork and Clone the Repository

#### What is Forking?
Forking creates your own copy of the project on GitHub. This allows you to make changes without affecting the original project.

#### Instructions:

1. **Go to the original repository** (your instructor will provide the link)

2. **Click the "Fork" button** (top-right corner)
   - This creates a copy under your GitHub account

3. **Clone your fork to your computer:**
   ```bash
   # Replace YOUR-USERNAME with your GitHub username
   git clone https://github.com/YOUR-USERNAME/nlp-react.git

   # Navigate into the project
   cd nlp-react
   ```

### Step 2: Install Dependencies

#### Frontend Dependencies (React)

```bash
# Navigate to frontend folder
cd frontend

# Install Node.js packages (this will take 2-3 minutes)
npm install

# Go back to root directory
cd ..
```

**What this does:** Downloads all the JavaScript libraries needed for React, Vite, Tailwind, etc.

#### Backend Dependencies (Python)

```bash
# Navigate to backend folder
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install Python packages (this will take 5-10 minutes)
pip install -r requirements.txt

# Go back to root directory
cd ..
```

**What this does:** Downloads Python libraries including FastAPI, PyTorch, and Transformers.

**Note:** The AI model (DistilBERT) will be downloaded automatically the first time you run the backend. This is about 250MB and may take a few minutes.

---

## 🚀 Part 3: Running the Application Locally

### Step 1: Start the Backend Server

```bash
# From the root directory, navigate to backend
cd backend

# Make sure your virtual environment is activated
# Then run the server
python main.py
```

**You should see:**
```
Loading model: distilbert-base-uncased
✅ Model loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal window open!** The backend needs to keep running.

### Step 2: Start the Frontend Development Server

Open a **NEW terminal window** (keep the backend running in the first one).

```bash
# From the root directory, navigate to frontend
cd frontend

# Start the development server
npm run dev
```

**You should see:**
```
  VITE v5.0.11  ready in 500 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

### Step 3: Open in Browser

1. Open your web browser
2. Go to: **http://localhost:3000**
3. You should see the Sentiment Analyzer app!

### Step 4: Test the Application

1. **Enter a movie review** in the text box
2. **Click "Analyze Sentiment"**
3. **View the results** showing sentiment score, confidence, and probability distribution

**Test with these examples:**
- "This movie was absolutely phenomenal! Best film of the decade!"
- "It was okay, nothing special."
- "Terrible movie. Complete waste of time."

---

## ✏️ Part 4: Making Your Modifications

### Required Modifications

You must make **2-3 meaningful modifications** to the application. Choose from the options below or create your own!

### Modification Ideas

#### 🎨 **Option 1: Change the Color Scheme** (Easy)

**File:** `frontend/tailwind.config.js`

**What to do:** Change the primary colors used throughout the app.

**Current code:**
```javascript
colors: {
  primary: {
    500: '#3b82f6',  // Blue
    600: '#2563eb',
  }
}
```

**Modify to:**
```javascript
colors: {
  primary: {
    500: '#8b5cf6',  // Purple
    600: '#7c3aed',
  }
}
```

**Also update:** `frontend/src/components/Header.jsx`
```jsx
// Change from blue to your new color
<Film className="w-12 h-12 text-purple-600" />
```

---

#### 📝 **Option 2: Add a New Example Review** (Easy)

**File:** `backend/main.py`

**What to do:** Add your own example reviews to the `/examples` endpoint.

**Find this section** (around line 110):
```python
examples = {
    "3": [
        "Absolutely amazing! Best movie I've seen this year!",
        "Masterpiece! Incredible in every way!",
    ],
    # ... other examples
}
```

**Add new examples:**
```python
"3": [
    "Absolutely amazing! Best movie I've seen this year!",
    "Masterpiece! Incredible in every way!",
    "YOUR NEW VERY POSITIVE REVIEW HERE!",  # Add this
],
```

**Repeat for other sentiment levels** (-3, -2, -1, 0, 1, 2).

---

#### 🎯 **Option 3: Add Character Count Limit** (Medium)

**File:** `frontend/src/components/InputSection.jsx`

**What to do:** Add a maximum character limit with a warning.

**Find the textarea** (around line 40):
```jsx
<textarea
  value={text}
  onChange={(e) => {
    setText(e.target.value);
    setError('');
  }}
  // ... other props
```

**Modify to:**
```jsx
<textarea
  value={text}
  onChange={(e) => {
    const newText = e.target.value;
    if (newText.length <= 1000) {  // Add limit
      setText(newText);
      setError('');
    } else {
      setError('Maximum 1000 characters allowed');
    }
  }}
  // ... other props
```

**Update the character counter** to show limit:
```jsx
<span className="text-sm text-gray-500">
  {text.length}/1000 characters • Press Ctrl+Enter to analyze
</span>
```

---

#### 📊 **Option 4: Add More Metrics** (Medium)

**File:** `frontend/src/components/ResultsSection.jsx`

**What to do:** Add a new metric showing the sentiment category.

**Find the metrics grid** (around line 95):
```jsx
<div className="grid grid-cols-3 gap-4 mb-6">
  {/* Existing metrics */}
</div>
```

**Change to 4 columns and add:**
```jsx
<div className="grid grid-cols-4 gap-4 mb-6">
  {/* Existing metrics */}

  {/* Add this new metric */}
  <div className="bg-green-50 rounded-lg p-4 text-center">
    <span className="text-2xl mb-2">📈</span>
    <p className="text-2xl font-bold text-green-900">
      {sentiment_score >= 0 ? 'Positive' : 'Negative'}
    </p>
    <p className="text-sm text-green-600">Overall</p>
  </div>
</div>
```

---

#### 🌟 **Option 5: Add Animation on Results** (Medium)

**File:** `frontend/src/index.css`

**What to do:** Add a fade-in animation when results appear.

**Add this CSS:**
```css
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fadeIn {
  animation: fadeIn 0.5s ease-in-out;
}
```

**File:** `frontend/src/components/ResultsSection.jsx`

**Already has `animate-fadeIn` class** - just verify it works!

---

#### 🎨 **Option 6: Customize the Header** (Easy)

**File:** `frontend/src/components/Header.jsx`

**What to do:** Change the title, description, or add your name.

**Find:**
```jsx
<h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
  Sentiment Analyzer
</h1>
```

**Change to:**
```jsx
<h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
  Movie Sentiment AI by [Your Name]
</h1>
```

---

#### 🚀 **Option 7: Add a "Clear" Button** (Medium)

**File:** `frontend/src/components/InputSection.jsx`

**What to do:** Add a button to clear the input and results.

**Find the analyze button section:**
```jsx
<button
  onClick={handleAnalyze}
  // ... props
>
  Analyze Sentiment
</button>
```

**Add a clear button:**
```jsx
<div className="flex gap-3">
  <button
    onClick={() => {
      setText('');
      setResult(null);
      setError('');
    }}
    className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
  >
    Clear
  </button>

  <button
    onClick={handleAnalyze}
    // ... existing props
  >
    Analyze Sentiment
  </button>
</div>
```

---

#### 💾 **Option 8: Add History Feature** (Advanced)

**File:** `frontend/src/App.jsx`

**What to do:** Store previous analyses and display them.

**Add state for history:**
```jsx
const [history, setHistory] = useState([]);
```

**Update when analysis completes:**
```jsx
// In InputSection, after getting result:
setHistory(prev => [result, ...prev].slice(0, 5)); // Keep last 5
```

**Display history:**
```jsx
{history.length > 0 && (
  <div className="bg-white rounded-xl shadow-lg p-6 mt-6">
    <h3 className="text-xl font-bold mb-4">Recent Analyses</h3>
    {history.map((item, idx) => (
      <div key={idx} className="p-3 bg-gray-50 rounded mb-2">
        <p className="text-sm text-gray-600">{item.text.substring(0, 50)}...</p>
        <p className="text-lg font-bold">{item.sentiment_score}/3 {item.emoji}</p>
      </div>
    ))}
  </div>
)}
```

---

### Testing Your Modifications

After making changes:

1. **Save all files**
2. **Check your terminals** - both frontend and backend should still be running
3. **Refresh your browser** (or it may auto-refresh)
4. **Test your changes** thoroughly

---

## 🚀 Part 5: Deploying to the Cloud

### Important: Two-Platform Deployment

⚠️ **Critical Information:**
**Vercel does NOT support Python backends!**

Your application needs to be deployed in **two parts**:

1. **Frontend (React)** → **Vercel** (Best for React apps)
2. **Backend (FastAPI/Python)** → **Render.com OR Railway.app** (Support Python)

### Quick Overview

```
┌─────────────────┐         ┌─────────────────┐
│   VERCEL        │◄───────►│  RENDER/RAILWAY │
│   (Frontend)    │  HTTPS  │  (Backend API)  │
│   React App     │         │  Python+AI      │
└─────────────────┘         └─────────────────┘
```

### Time Estimate
- **Frontend deployment:** 10-15 minutes
- **Backend deployment:** 15-20 minutes
- **Total:** 30-45 minutes

---

## 📖 Full Deployment Instructions

**For complete step-by-step instructions, see:**

### → [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) ←

The deployment guide includes:
- ✅ Platform comparison (Render vs Railway)
- ✅ Detailed screenshots and explanations
- ✅ Step-by-step instructions for each platform
- ✅ Environment variable setup
- ✅ Troubleshooting common issues
- ✅ Testing your deployed app

---

## Quick Start Deployment

If you want a quick overview, here's the condensed version:

### Step 1: Prepare Your Code

```bash
# Commit all changes
git add .
git commit -m "Prepare for deployment"
git push origin main
```

### Step 2: Deploy Backend (Choose ONE)

**Option A: Render.com** (No credit card required)

1. Go to [render.com](https://render.com) → Sign up with GitHub
2. New + → Web Service
3. Connect your `nlp-react` repository
4. Configure:
   ```
   Name: sentiment-analyzer-api
   Language: Python
   Root Directory: backend
   Build: pip install -r requirements.txt
   Start: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
5. Deploy (takes 10-15 minutes first time)
6. Copy your URL: `https://your-api.onrender.com`

**Option B: Railway.app** (Requires credit card for $5/month free credit)

1. Go to [railway.app](https://railway.app) → Sign up with GitHub
2. New Project → Deploy from GitHub repo
3. Select `nlp-react` → Configure root directory: `backend`
4. Generate domain after deployment
5. Copy your URL: `https://your-api.railway.app`

### Step 3: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com) → Sign up with GitHub
2. New Project → Import `nlp-react`
3. **Configure:**
   ```
   Framework: Vite
   Root Directory: frontend (MUST SET THIS!)
   Build: npm run build
   Output: dist
   ```
4. **Add Environment Variable:**
   ```
   VITE_API_URL = https://your-backend-url.onrender.com
   (NO trailing slash!)
   ```
5. Deploy → Get your URL: `https://your-app.vercel.app`

### Step 4: Test Your App

1. Visit your Vercel URL
2. Wait 30-60 seconds for backend to wake up (first time)
3. Test sentiment analysis
4. Share your URL!

---

### Step 5: Update Your Repository

Add your URLs to README:

```markdown
# 🎬 Multi-Scale Sentiment Analyzer

## 🌐 Live Demo
- **Frontend:** https://your-app.vercel.app
- **Backend API:** https://your-backend-url.onrender.com

## 👤 Created By
[Your Name] - AIT-204 Course Project
```

Then commit:
```bash
git add README.md
git commit -m "Add deployment URLs"
git push origin main
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Frontend Issues

**Problem:** "npm install" fails
```bash
# Solution: Clear npm cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**Problem:** "Cannot find module" errors
```bash
# Solution: Reinstall dependencies
cd frontend
rm -rf node_modules
npm install
```

**Problem:** Port 3000 is already in use
```bash
# Solution: Use a different port
npm run dev -- --port 3001
```

#### Backend Issues

**Problem:** "torch not found" or model download fails
```bash
# Solution: Install PyTorch separately
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Problem:** "ModuleNotFoundError: No module named 'fastapi'"
```bash
# Solution: Make sure you're in the virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

**Problem:** Backend takes too long to start
```
# This is normal! The first time, it downloads the AI model (~250MB)
# Subsequent starts will be faster (2-3 seconds)
```

#### Deployment Issues

**Problem:** Vercel build fails
```
# Check build logs in Vercel dashboard
# Common fixes:
# 1. Make sure all dependencies are in package.json
# 2. Check that build command is correct: npm run build
# 3. Verify output directory is: dist
```

**Problem:** API requests fail after deployment
```
# Check:
# 1. VITE_API_URL environment variable is set correctly
# 2. Backend is deployed and running
# 3. CORS is configured in backend main.py (already done)
# 4. Browser console for specific error messages
```

**Problem:** "Backend API is not responding"
```
# Solutions:
# 1. Check if backend is deployed and awake (visit backend URL)
# 2. Free tier services sleep after inactivity - wait 30 seconds
# 3. Check environment variables are set correctly
# 4. Verify backend URL doesn't have trailing slash
```

---

## 📚 Additional Resources

### Learning React
- [React Official Tutorial](https://react.dev/learn) - Best place to start
- [React in 100 Seconds](https://www.youtube.com/watch?v=Tn6-PIqc4UM) - Quick overview
- [Full React Course](https://www.youtube.com/watch?v=bMknfKXIFA8) - 10+ hours comprehensive

### Learning FastAPI
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Excellent docs
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) - Step-by-step guide

### Understanding Transformers & NLP
- [Hugging Face Course](https://huggingface.co/course) - Free NLP course
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Transformers Explained](https://www.youtube.com/watch?v=SZorAJ4I-sA) - Video tutorial

### Git & GitHub
- [Git Handbook](https://guides.github.com/introduction/git-handbook/) - Git basics
- [GitHub Desktop](https://desktop.github.com/) - GUI for Git (easier than command line)

### Deployment
- [Vercel Documentation](https://vercel.com/docs) - Deployment guides
- [Render Documentation](https://render.com/docs) - Backend deployment

### Design & UI
- [Tailwind CSS Documentation](https://tailwindcss.com/docs) - Styling reference
- [Tailwind UI Components](https://tailwindui.com/components) - Pre-built components
- [Lucide Icons](https://lucide.dev/) - Icon library used in this project

---

## 🎓 Submission Requirements

### What to Submit

1. **GitHub Repository URL**
   - Your forked repository with all modifications
   - README updated with your deployment URLs

2. **Live Deployment URLs**
   - Frontend (Vercel): `https://your-app.vercel.app`
   - Backend (Render/Railway): `https://your-backend.onrender.com`

3. **Modification Summary** (in a comment or separate file)
   - List the 2-3 modifications you made
   - Explain what each modification does
   - Include before/after screenshots if applicable

### Grading Criteria

- ✅ **Functionality** (40%): App works correctly after modifications
- ✅ **Modifications** (30%): 2-3 meaningful changes implemented
- ✅ **Deployment** (20%): Successfully deployed to Vercel
- ✅ **Documentation** (10%): README updated, code is clean

---

## 🌟 Extension Challenges

Want to go further? Try these advanced features:

### Challenge 1: Batch Analysis
Add ability to analyze multiple reviews at once
- Add file upload for CSV
- Display results in a table
- Export results as JSON

### Challenge 2: Sentiment Trends
Track sentiment over time
- Store analysis history in localStorage
- Create line chart showing sentiment trends
- Add date/time to each analysis

### Challenge 3: Custom Emojis
Allow users to customize sentiment emojis
- Add settings panel
- Let users choose their own emojis
- Save preferences in localStorage

### Challenge 4: Dark Mode
Implement a dark mode toggle
- Add theme switcher component
- Use Tailwind's dark: prefix
- Persist theme choice

### Challenge 5: Multi-Language Support
Extend to analyze reviews in other languages
- Add language selector
- Use multilingual models
- Display language-specific examples

---

## 💡 Tips for Success

1. **Read Error Messages Carefully**
   - They usually tell you exactly what's wrong
   - Google the error message if you don't understand

2. **Test Frequently**
   - Make one change at a time
   - Test immediately
   - Don't make multiple changes before testing

3. **Use Browser DevTools**
   - Press F12 to open
   - Check Console for JavaScript errors
   - Check Network tab for API request issues

4. **Commit Often**
   - Save your progress with git commits
   - Use descriptive commit messages
   - Push to GitHub regularly

5. **Ask for Help**
   - Check documentation first
   - Google your specific error
   - Ask classmates or instructor
   - Post on forums (Stack Overflow, Reddit)

6. **Have Fun!**
   - Experiment with different modifications
   - Try to break things (safely) and learn
   - Make the project your own
   - Be creative!

---

## 📞 Support

### Getting Help

1. **Check this README** thoroughly - most answers are here
2. **Review the code comments** - they explain what each part does
3. **Check the browser console** - F12 → Console tab
4. **Search the error message** on Google or Stack Overflow
5. **Ask your instructor** or classmates

### Common Questions

**Q: Can I use this project in my portfolio?**
A: Yes! That's encouraged. Make sure to credit the original project.

**Q: Can I modify more than 2-3 things?**
A: Absolutely! The more you customize, the better.

**Q: Do I need to deploy the backend for grading?**
A: Check with your instructor. The frontend alone demonstrates React skills.

**Q: Can I work in a group?**
A: Check with your instructor. Usually, individual work is required.

---

## 📜 License

This project is for educational purposes as part of the AIT-204 course.

---

## 🎉 Congratulations!

You've completed the Multi-Scale Sentiment Analyzer project! You now have:

- ✅ A working full-stack web application
- ✅ Experience with React and modern web development
- ✅ A deployed project you can show to potential employers
- ✅ Understanding of how AI models integrate with web apps

**Share your deployed app and celebrate your achievement!** 🎊

---

**Questions? Issues? Feedback?**
Contact your course instructor or create an issue in the GitHub repository.

**Happy Coding!** 💻✨
