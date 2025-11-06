import { Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { Brain, Camera, MessageSquare, MessageCircle, Dna, ArrowRight } from 'lucide-react'
import { docsAPI } from '../services/api'

export default function Home() {
  const projects = [
    {
      id: 'ann',
      title: 'ANN - NBA Team Selection',
      description: 'Artificial Neural Network for optimal NBA team composition using multi-layer perceptron',
      icon: Brain,
      color: 'from-blue-500 to-blue-700',
      path: '/ann',
      features: [
        'Player position classification',
        'Team fit score evaluation',
        '3 selection strategies',
        'Performance analytics'
      ]
    },
    {
      id: 'cnn',
      title: 'CNN - Fruit Classification',
      description: 'Convolutional Neural Network for recognizing different types of fruits from images',
      icon: Camera,
      color: 'from-green-500 to-green-700',
      path: '/cnn',
      features: [
        'Image classification',
        'Multiple fruit types',
        'Confidence scores',
        'Upload custom images'
      ]
    },
    {
      id: 'nlp',
      title: 'NLP - Sentiment Analysis',
      description: '3-point scale sentiment analyzer for hospital reviews and healthcare feedback using transformer models',
      icon: MessageSquare,
      color: 'from-purple-500 to-purple-700',
      path: '/nlp',
      features: [
        '3-point sentiment scale (1-3)',
        'Hospital review analysis',
        'Confidence scores',
        'Batch processing'
      ]
    },
    {
      id: 'rnn',
      title: 'RNN - Text Generation',
      description: 'LSTM-based Recurrent Neural Network for next-word prediction and text generation trained on classical literature',
      icon: MessageCircle,
      color: 'from-indigo-500 to-indigo-700',
      path: '/rnn',
      features: [
        'Classical literature training',
        'Sampling & Beam search',
        'Temperature control',
        'Technical report viewer'
      ]
    },
    {
      id: 'ga',
      title: 'GA - Shakespeare Evolution',
      description: 'Genetic Algorithm that evolves random text into Shakespeare quotes using selection, crossover, and mutation',
      icon: Dna,
      color: 'from-pink-500 to-rose-700',
      path: '/ga',
      features: [
        'Text evolution visualization',
        'Real-time fitness tracking',
        'Adjustable GA parameters',
        'Interactive evolution control'
      ]
    }
  ]

  return (
    <div className="container mx-auto px-4 py-12">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <h1 className="text-6xl font-bold text-white mb-4">
          Machine Learning Projects
        </h1>
        <p className="text-2xl text-white opacity-90">
          CST-435 Neural Networks - Interactive Demonstrations
        </p>
        <p className="text-lg text-white opacity-75 mt-2">
          Select a project below to explore different ML techniques
        </p>
      </div>

      {/* Project Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-8 max-w-7xl mx-auto">
        {projects.map((project) => {
          const Icon = project.icon
          return (
            <Link
              key={project.id}
              to={project.path}
              className="group"
            >
              <div className="bg-white rounded-2xl shadow-2xl overflow-hidden transform transition-all duration-300 hover:scale-105 hover:shadow-3xl h-full">
                {/* Card Header with Gradient */}
                <div className={`bg-gradient-to-r ${project.color} p-8 text-white`}>
                  <Icon size={48} className="mb-4" />
                  <h2 className="text-3xl font-bold mb-2">{project.title}</h2>
                </div>

                {/* Card Body */}
                <div className="p-8">
                  <p className="text-gray-700 text-lg mb-6">
                    {project.description}
                  </p>

                  {/* Features List */}
                  <div className="space-y-3 mb-6">
                    {project.features.map((feature, idx) => (
                      <div key={idx} className="flex items-center text-gray-600">
                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-3"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>

                  {/* Launch Button */}
                  <div className="flex items-center justify-center text-blue-600 font-semibold group-hover:text-blue-800 transition-colors">
                    <span className="mr-2">Launch Project</span>
                    <ArrowRight
                      size={20}
                      className="transform group-hover:translate-x-2 transition-transform"
                    />
                  </div>
                </div>
              </div>
            </Link>
          )
        })}
      </div>

      {/* Info Section */}
      <div className="mt-16 max-w-4xl mx-auto">
        <div className="bg-white bg-opacity-90 rounded-2xl shadow-xl p-8">
          <h3 className="text-3xl font-bold text-gray-800 mb-4">About This Project</h3>
          <p className="text-gray-700 text-lg leading-relaxed mb-4">
            This unified launcher provides access to five different machine learning and AI projects developed for CST-435.
            Each project demonstrates different AI techniques and architectures:
          </p>
          <ul className="space-y-2 text-gray-700">
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>ANN:</strong> Uses a Multi-Layer Perceptron (MLP) for classification and regression tasks</span>
            </li>
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>CNN:</strong> Implements Convolutional Neural Networks for computer vision</span>
            </li>
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>NLP:</strong> Leverages Transformer models for natural language processing</span>
            </li>
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>RNN:</strong> Utilizes LSTM networks for sequential learning and text generation</span>
            </li>
            <li className="flex items-start">
              <span className="font-semibold mr-2">•</span>
              <span><strong>GA:</strong> Implements Genetic Algorithms for evolutionary optimization and text evolution</span>
            </li>
          </ul>
        </div>
      </div>
      {/* Cost Analysis */}
      <div className="mt-12 max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h3 className="text-3xl font-bold text-gray-800 mb-4">Cost Analysis (per project)</h3>
          <p className="text-gray-600 mb-4">Estimated training and deployment costs. Values are auto-generated if no project-specific cost file exists.</p>

          <CostTable projects={projects} />
          <ReportViewer />
        </div>
      </div>
    </div>
  )
}


function CostTable({ projects }) {
  const [costs, setCosts] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    const fetchAll = async () => {
      const result = {}
      for (const p of projects) {
        try {
          const res = await docsAPI.getCostJson(p.id)
          result[p.id] = res.data
        } catch (err) {
          result[p.id] = null
        }
      }
      if (mounted) {
        setCosts(result)
        setLoading(false)
      }
    }
    fetchAll()
    return () => { mounted = false }
  }, [projects])

  const fmt = (v) => (typeof v === 'number' ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(v) : '—')

  return (
    <div>
      {loading ? (
        <div className="text-center py-8">Loading cost data...</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="bg-gray-100">
                <th className="px-4 py-3">Project</th>
                <th className="px-4 py-3">Training (one-time)</th>
                <th className="px-4 py-3">Monthly Inference</th>
                <th className="px-4 py-3">Monthly Data Transfer</th>
                <th className="px-4 py-3">Monthly Misc</th>
                <th className="px-4 py-3">Total (deployment)</th>
              </tr>
            </thead>
            <tbody>
              {projects.map((p) => {
                const c = costs[p.id]
                return (
                  <tr key={p.id} className="border-b hover:bg-gray-50">
                    <td className="px-4 py-3 font-semibold">{p.title}</td>
                    <td className="px-4 py-3">{c ? fmt(c.training_cost) : 'N/A'}</td>
                    <td className="px-4 py-3">{c ? fmt(c.monthly_inference_cost) : 'N/A'}</td>
                    <td className="px-4 py-3">{c ? fmt(c.monthly_data_transfer_cost) : 'N/A'}</td>
                    <td className="px-4 py-3">{c ? fmt(c.monthly_misc) : 'N/A'}</td>
                    <td className="px-4 py-3">{c ? fmt(c.total_over_deployment) : 'N/A'}</td>
                    <td className="px-4 py-3">
                      <ReportButton projectId={p.id} projectTitle={p.title} />
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}


function ReportButton({ projectId, projectTitle }) {
  const openReport = async () => {
    // Prefer the structured JSON cost report and render it as a friendly HTML page
    try {
      let res
      try {
        res = await docsAPI.getCostJson(projectId)
      } catch (e) {
        // fallback to markdown endpoint
        const mdRes = await docsAPI.getCost(projectId)
        const md = mdRes.data.markdown || JSON.stringify(mdRes.data, null, 2)
        const w = window.open('', `${projectId}-cost-report`)
        if (w) {
          w.document.title = `${projectTitle} - Cost Report`
          const pre = w.document.createElement('pre')
          pre.style.whiteSpace = 'pre-wrap'
          pre.style.fontFamily = 'Menlo, Monaco, monospace'
          pre.textContent = md
          w.document.body.appendChild(pre)
          return
        }
        throw new Error('Unable to open window for report (popup blocked)')
      }

      const data = res.data || {}
      const title = `${projectTitle} — Deployment Cost Analysis`

      // Build scenario rows if present
      const scenarios = []
      const scenarioKeys = Object.keys(data).filter(k => k.startsWith('cost_') && k.endsWith('_per_day'))
      scenarioKeys.sort((a,b) => {
        const na = parseInt(a.split('_')[1],10)
        const nb = parseInt(b.split('_')[1],10)
        return na - nb
      })
      for (const k of scenarioKeys) {
        const n = k.split('_')[1]
        const cost = data[k]
        const instances = data[`instances_${n}_per_day`] || '—'
        scenarios.push({requests_per_day: parseInt(n,10), cost, instances})
      }

      const html = `
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8" />
          <title>${title}</title>
          <meta name="viewport" content="width=device-width,initial-scale=1" />
          <style>
            body { font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color:#111; padding:24px; }
            h1 { margin-bottom:6px }
            .muted { color:#556; margin-bottom:18px }
            .card { border:1px solid #e6e6e6; border-radius:8px; padding:16px; margin-bottom:12px; box-shadow:0 1px 4px rgba(0,0,0,0.03) }
            table { width:100%; border-collapse:collapse; }
            th, td { padding:8px 12px; text-align:left; border-bottom:1px solid #f1f1f1 }
            th { background:#fafafa }
            .k { font-weight:600 }
            .mono { font-family: SFMono-Regular, Menlo, Monaco, monospace; color:#0b6; }
          </style>
        </head>
        <body>
          <h1>${title}</h1>
          <div class="muted">Auto-generated cost summary. Values are estimates based on measured median latency and conservative defaults. Click "View report" in-app for full markdown.</div>

          <div class="card">
            <h3>Summary</h3>
            <table>
              <tbody>
                <tr><th>Project</th><td>${projectTitle}</td></tr>
                <tr><th>Chosen instance</th><td>${data.chosen_instance || 't3.medium'}</td></tr>
                <tr><th>Median inference latency</th><td>${data.median_inference_s ? data.median_inference_s.toFixed(4) + ' s' : '—'}</td></tr>
                <tr><th>Model size</th><td>${data.model_size_gb ? data.model_size_gb + ' GB' : '—'}</td></tr>
                <tr><th>Training (one-time)</th><td>${data.training_cost != null ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(data.training_cost) : '—'}</td></tr>
                <tr><th>Storage (monthly)</th><td>${data.storage_cost != null ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(data.storage_cost) : '—'}</td></tr>
              </tbody>
            </table>
          </div>

          <div class="card">
            <h3>Scenario estimates</h3>
            <table>
              <thead><tr><th>Requests / day</th><th>Instances needed</th><th>Monthly cost (compute + storage + transfer)</th></tr></thead>
              <tbody>
                ${scenarios.map(s => `
                  <tr>
                    <td class="k">${s.requests_per_day.toLocaleString()}</td>
                    <td>${s.instances}</td>
                    <td>${s.cost != null ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(s.cost) : '—'}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>

          <div class="card">
            <h3>Detailed breakdown (10k/day sample)</h3>
            <table>
              <tbody>
                <tr><th>Monthly inference (compute)</th><td>${data.monthly_inference_cost != null ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(data.monthly_inference_cost) : '—'}</td></tr>
                <tr><th>Monthly data transfer</th><td>${data.monthly_data_transfer_cost != null ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(data.monthly_data_transfer_cost) : '—'}</td></tr>
                <tr><th>Monthly misc</th><td>${data.monthly_misc != null ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(data.monthly_misc) : '—'}</td></tr>
                <tr><th>Total (deployment)</th><td>${data.total_over_deployment != null ? new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(data.total_over_deployment) : '—'}</td></tr>
              </tbody>
            </table>
          </div>

          <div style="font-size:13px;color:#666">Generated on ${new Date().toLocaleString()}</div>
        </body>
        </html>
      `

      const w = window.open('', `${projectId}-cost-report`)
      if (!w) {
        alert('Unable to open new window to show the report (popup blocked).')
        return
      }
      w.document.open()
      w.document.write(html)
      w.document.close()
    } catch (err) {
      alert('Failed to fetch report: ' + (err?.message || err))
    }
  }

  return (
    <button onClick={openReport} className="text-sm text-blue-600 hover:underline">
      View report
    </button>
  )
}


function ReportViewer() {
  // kept for possible inline viewer in the future; reserved placeholder
  return null
}
