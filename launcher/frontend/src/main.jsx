import React from 'react'
import ReactDOM from 'react-dom/client'
import './components/rl/onnxInit'  // Configure ONNX Runtime BEFORE any components load
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
