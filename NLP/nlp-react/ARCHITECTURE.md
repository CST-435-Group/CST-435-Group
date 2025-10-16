# 🏗️ Project Architecture

Visual guide to understand how everything connects.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER'S BROWSER                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    React Frontend                         │ │
│  │                  (Port 3000 / Vercel)                     │ │
│  │                                                           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │ │
│  │  │  Header  │  │  Input   │  │ Results  │               │ │
│  │  │Component │  │Component │  │Component │               │ │
│  │  └──────────┘  └──────────┘  └──────────┘               │ │
│  │                                                           │ │
│  │  ┌──────────┐  ┌──────────┐                              │ │
│  │  │Examples  │  │   Info   │                              │ │
│  │  │Component │  │Component │                              │ │
│  │  └──────────┘  └──────────┘                              │ │
│  │                                                           │ │
│  │         │                                                 │ │
│  │         │ API Service Layer (api.js)                     │ │
│  │         ▼                                                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          │
                          │ HTTP/HTTPS
                          │ (JSON)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│                   (Port 8000 / Render)                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                      main.py                               │ │
│  │                   (API Endpoints)                          │ │
│  │                                                            │ │
│  │  POST /analyze      - Analyze single text                 │ │
│  │  POST /analyze/batch - Analyze multiple texts             │ │
│  │  GET  /examples     - Get example reviews                 │ │
│  │  GET  /health       - Health check                        │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     model.py                               │ │
│  │              (Sentiment Analyzer Class)                    │ │
│  │                                                            │ │
│  │  • Load DistilBERT model                                  │ │
│  │  • Tokenize input text                                    │ │
│  │  • Run inference                                          │ │
│  │  • Convert outputs to sentiment scores                    │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                      │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   DistilBERT Model                         │ │
│  │              (Hugging Face Transformers)                   │ │
│  │                                                            │ │
│  │  • Pre-trained language model                             │ │
│  │  • 66M parameters                                         │ │
│  │  • Analyzes text sentiment                                │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Request Flow (Analyzing Text)

```
1. USER INPUT
   ┌────────────────────────────────────┐
   │ User types: "Great movie!"         │
   │ Clicks: "Analyze Sentiment"        │
   └─────────────────┬──────────────────┘
                     │
                     ▼
2. REACT FRONTEND
   ┌────────────────────────────────────┐
   │ InputSection.jsx                   │
   │ - Validates input                  │
   │ - Calls api.analyzeSentiment()     │
   └─────────────────┬──────────────────┘
                     │
                     ▼
3. API SERVICE
   ┌────────────────────────────────────┐
   │ api.js                             │
   │ - POST request to backend          │
   │ - JSON: { "text": "Great movie!" } │
   └─────────────────┬──────────────────┘
                     │
                     │ HTTP POST
                     │ http://backend/analyze
                     ▼
4. FASTAPI ENDPOINT
   ┌────────────────────────────────────┐
   │ main.py: /analyze                  │
   │ - Receives request                 │
   │ - Validates data                   │
   │ - Calls analyzer.analyze()         │
   └─────────────────┬──────────────────┘
                     │
                     ▼
5. MODEL PROCESSING
   ┌────────────────────────────────────┐
   │ model.py: SentimentAnalyzer        │
   │ - Tokenizes text                   │
   │ - Runs through DistilBERT          │
   │ - Gets predictions                 │
   │ - Converts to sentiment score      │
   └─────────────────┬──────────────────┘
                     │
                     ▼
6. RESPONSE
   ┌────────────────────────────────────┐
   │ JSON Response:                     │
   │ {                                  │
   │   "text": "Great movie!",          │
   │   "sentiment_score": 2,            │
   │   "sentiment_label": "Positive",   │
   │   "emoji": "😊",                   │
   │   "confidence": 0.89,              │
   │   "probabilities": {...}           │
   │ }                                  │
   └─────────────────┬──────────────────┘
                     │
                     │ HTTP 200 OK
                     ▼
7. REACT FRONTEND
   ┌────────────────────────────────────┐
   │ ResultsSection.jsx                 │
   │ - Receives data                    │
   │ - Displays results                 │
   │ - Shows charts and metrics         │
   └────────────────────────────────────┘
                     │
                     ▼
8. USER SEES RESULTS
   ┌────────────────────────────────────┐
   │ ✓ Score: +2/3                      │
   │ ✓ Label: Positive 😊               │
   │ ✓ Confidence: 89%                  │
   │ ✓ Probability Chart                │
   └────────────────────────────────────┘
```

---

## Component Hierarchy

```
App.jsx (Main Container)
│
├── Header.jsx
│   ├── Title & Description
│   ├── API Status Indicator
│   └── Sentiment Scale Display
│
├── Main Content (2-column grid)
│   │
│   ├── Left Column
│   │   ├── InputSection.jsx
│   │   │   ├── Text Area
│   │   │   ├── Character Counter
│   │   │   └── Analyze Button
│   │   │
│   │   └── ResultsSection.jsx (conditional)
│   │       ├── Score Display
│   │       ├── Metrics Grid
│   │       ├── Probability Chart
│   │       └── Detailed Breakdown
│   │
│   └── Right Column
│       ├── ExamplesSection.jsx
│       │   └── Example Buttons (5)
│       │
│       └── InfoSection.jsx
│           ├── About Info
│           ├── Tech Stack
│           └── How It Works
│
└── Footer
    └── Credits & Links
```

---

## File Structure with Purpose

```
nlp-react/
│
├── 📖 Documentation
│   ├── README.md                    # Main comprehensive guide
│   ├── QUICKSTART.md                # 5-minute setup guide
│   ├── DEPLOYMENT_GUIDE.md          # Vercel deployment steps
│   ├── MODIFICATION_TASKS.md        # Student assignment tasks
│   ├── PROJECT_SUMMARY.md           # Instructor overview
│   └── ARCHITECTURE.md              # This file!
│
├── ⚛️ Frontend (React)
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx           # Top section with title
│   │   │   ├── InputSection.jsx     # User input area
│   │   │   ├── ResultsSection.jsx   # Results display
│   │   │   ├── ExamplesSection.jsx  # Example buttons
│   │   │   └── InfoSection.jsx      # Information panel
│   │   │
│   │   ├── services/
│   │   │   └── api.js               # Backend communication
│   │   │
│   │   ├── App.jsx                  # Main component
│   │   ├── main.jsx                 # Entry point
│   │   └── index.css                # Global styles
│   │
│   ├── package.json                 # Dependencies
│   ├── vite.config.js               # Dev server config
│   └── tailwind.config.js           # Styling config
│
├── 🐍 Backend (FastAPI)
│   ├── main.py                      # API endpoints
│   ├── model.py                     # AI model wrapper
│   └── requirements.txt             # Python packages
│
└── ⚙️ Configuration
    ├── vercel.json                  # Vercel deployment
    ├── package.json                 # Root package file
    └── .gitignore                   # Git ignore rules
```

---

## Technology Stack Details

### Frontend Technologies

```
React (v18.2)
├── Purpose: UI library for building components
├── Why: Industry standard, large community
└── Key Features: Components, hooks, virtual DOM

Vite (v5.0)
├── Purpose: Build tool and dev server
├── Why: Fast, modern, better than Create React App
└── Key Features: Hot reload, fast builds

Tailwind CSS (v3.4)
├── Purpose: Utility-first CSS framework
├── Why: Fast styling, responsive design
└── Key Features: Pre-built utilities, customizable

Axios (v1.6)
├── Purpose: HTTP client for API calls
├── Why: Better than fetch, easier error handling
└── Key Features: Interceptors, timeout, JSON auto-parse

Recharts (v2.10)
├── Purpose: Charting library for React
├── Why: Easy to use, customizable
└── Key Features: Bar charts, responsive
```

### Backend Technologies

```
FastAPI (v0.109)
├── Purpose: Modern Python web framework
├── Why: Fast, automatic API docs, type validation
└── Key Features: Async support, Pydantic validation

PyTorch (v2.0+)
├── Purpose: Deep learning framework
├── Why: Required for transformer models
└── Key Features: GPU support, model inference

Transformers (v4.30+)
├── Purpose: Pre-trained NLP models
├── Why: State-of-the-art sentiment analysis
└── Key Features: DistilBERT, easy to use

Uvicorn (v0.27)
├── Purpose: ASGI server for FastAPI
├── Why: Fast, production-ready
└── Key Features: Async, WebSocket support
```

---

## State Management

### Frontend State

```
App.jsx (Top-level state)
├── result: Current analysis result
├── loading: Is analysis in progress?
└── apiStatus: Is backend connected?
     │
     ├── Passed to InputSection as props
     ├── Passed to ResultsSection as props
     └── Passed to Header as props

InputSection.jsx (Local state)
├── text: User's input text
└── error: Validation error message

ExamplesSection.jsx (Local state)
└── examples: Example reviews from API
```

### Why This Pattern?

- ✅ **Top-level state** for data shared between components
- ✅ **Local state** for component-specific data
- ✅ **Props** for passing data down
- ✅ **Simple and clear** - no Redux needed

---

## API Endpoints

### Backend API Routes

```
GET /
└── Returns: API info and version

GET /health
└── Returns: { status, model_loaded }

POST /analyze
├── Input: { text: string }
└── Returns: {
      text, sentiment_score, sentiment_label,
      emoji, confidence, probabilities
    }

POST /analyze/batch
├── Input: { texts: string[] }
└── Returns: Array of results

GET /examples
└── Returns: { "-3": [...], ..., "3": [...] }

GET /sentiment-scale
└── Returns: { -3: {label, emoji}, ..., 3: {label, emoji} }
```

---

## Deployment Architecture

### Development (Local)

```
┌──────────────┐         ┌──────────────┐
│   Frontend   │◄───────►│   Backend    │
│ localhost:   │  Proxy  │ localhost:   │
│    3000      │         │    8000      │
└──────────────┘         └──────────────┘
```

### Production (Cloud)

```
┌─────────────────┐         ┌─────────────────┐
│     Vercel      │◄───────►│     Render      │
│   (Frontend)    │  HTTPS  │   (Backend)     │
│  Static Files   │         │   Python API    │
│   CDN Global    │         │   Serverless    │
└─────────────────┘         └─────────────────┘
```

---

## Security Considerations

### CORS Configuration
```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[...],  # Specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables
```bash
# Frontend (.env)
VITE_API_URL=https://backend-url.com

# Backend (.env)
CORS_ORIGINS=https://frontend-url.com
MODEL_NAME=distilbert-base-uncased
```

### Best Practices
- ✅ No API keys in frontend code
- ✅ HTTPS only in production
- ✅ Input validation on backend
- ✅ Rate limiting (not implemented, but recommended)
- ✅ Error handling everywhere

---

## Performance Optimization

### Frontend
- ✅ Code splitting (Vite automatic)
- ✅ Lazy loading components (can be added)
- ✅ Memoization for expensive renders
- ✅ Production build minification

### Backend
- ✅ Model caching (loaded once)
- ✅ Async endpoints (FastAPI)
- ✅ Batch processing support
- ✅ Response compression

### Model
- ✅ DistilBERT (smaller than BERT)
- ✅ GPU support if available
- ✅ Batch inference capability
- ✅ Efficient tokenization

---

## Testing Strategy (Not Implemented, but Recommended)

### Frontend Tests
```javascript
// Unit tests (Jest)
- Component rendering
- User interactions
- API service functions

// Integration tests (React Testing Library)
- User flows
- API mocking
- Error handling
```

### Backend Tests
```python
# Unit tests (pytest)
- Model inference
- Endpoint responses
- Input validation

# Integration tests
- Full API workflow
- Error scenarios
- Performance
```

---

## Monitoring & Debugging

### Frontend Debugging
```
Browser DevTools:
├── Console: JavaScript errors
├── Network: API requests/responses
├── React DevTools: Component state
└── Performance: Rendering issues
```

### Backend Debugging
```
Logs:
├── Uvicorn access logs
├── FastAPI route logging
├── Model inference timing
└── Error stack traces
```

---

## Scalability Considerations

### Current Limitations
- Backend sleeps on free tier
- No database (stateless)
- No caching layer
- Single model instance

### Potential Improvements
```
High Traffic:
├── Add Redis for caching
├── Implement request queuing
├── Use CDN for static assets
└── Load balancing for backend

Data Storage:
├── Add PostgreSQL database
├── Store analysis history
├── User authentication
└── Analytics tracking

Model Optimization:
├── Quantized model version
├── ONNX conversion
├── Model caching
└── GPU acceleration
```

---

## Future Enhancement Ideas

### Features
1. User accounts & history
2. Compare multiple reviews
3. Export results (PDF, CSV)
4. API key for developers
5. Webhook notifications
6. Sentiment trends over time
7. Multi-language support
8. Voice input

### Technical
1. WebSocket for real-time
2. GraphQL API
3. Mobile app (React Native)
4. Browser extension
5. Slack/Discord bot
6. Email reports
7. A/B testing framework
8. Analytics dashboard

---

This architecture is designed to be:
- ✅ **Educational** - Easy to understand for beginners
- ✅ **Modular** - Components are independent
- ✅ **Scalable** - Can grow with more features
- ✅ **Modern** - Uses current best practices
- ✅ **Deployable** - Cloud-ready from day one

---

**Questions about the architecture?**
Refer back to this document or check the other guides!
