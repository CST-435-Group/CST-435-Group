import { useState, useEffect } from 'react'
import { annAPI } from '../services/api'
import { Brain, Users, TrendingUp, AlertCircle } from 'lucide-react'

export default function ANNProject() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dataInfo, setDataInfo] = useState(null)
  const [team, setTeam] = useState(null)
  const [topPlayers, setTopPlayers] = useState([])
  const [method, setMethod] = useState('balanced')

  useEffect(() => {
    loadDataInfo()
    loadTopPlayers()
  }, [])

  const loadDataInfo = async () => {
    try {
      const response = await annAPI.getDataInfo()
      setDataInfo(response.data)
    } catch (err) {
      console.error('Error loading data info:', err)
    }
  }

  const loadTopPlayers = async () => {
    try {
      const response = await annAPI.getPlayers(10)
      setTopPlayers(response.data)
    } catch (err) {
      console.error('Error loading players:', err)
    }
  }

  const handleSelectTeam = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await annAPI.selectTeam({ method })
      setTeam(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Error selecting team')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <div className="flex items-center mb-4">
          <Brain size={48} className="text-blue-600 mr-4" />
          <div>
            <h1 className="text-4xl font-bold text-gray-800">NBA Team Selection</h1>
            <p className="text-gray-600 text-lg">Artificial Neural Network for Optimal Team Composition</p>
          </div>
        </div>
      </div>

      {/* Data Info */}
      {dataInfo && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Total Players</h3>
            <p className="text-3xl font-bold text-blue-600">{dataInfo.total_players}</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Avg Points</h3>
            <p className="text-3xl font-bold text-green-600">{dataInfo.avg_points.toFixed(1)}</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Avg Rebounds</h3>
            <p className="text-3xl font-bold text-purple-600">{dataInfo.avg_rebounds.toFixed(1)}</p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h3 className="text-gray-600 text-sm font-semibold mb-2">Avg Assists</h3>
            <p className="text-3xl font-bold text-orange-600">{dataInfo.avg_assists.toFixed(1)}</p>
          </div>
        </div>
      )}

      {/* Team Selection */}
      <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <Users className="mr-3 text-blue-600" />
          Team Selection
        </h2>

        <div className="mb-6">
          <label className="block text-gray-700 font-semibold mb-2">Selection Method</label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="w-full md:w-1/2 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="greedy">Greedy (Top 5 Players)</option>
            <option value="balanced">Balanced (Position-Aware)</option>
            <option value="exhaustive">Exhaustive Search</option>
          </select>
        </div>

        <button
          onClick={handleSelectTeam}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {loading ? 'Selecting Team...' : 'Select Optimal Team'}
        </button>

        {error && (
          <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg flex items-center">
            <AlertCircle className="mr-2" size={20} />
            {error}
          </div>
        )}
      </div>

      {/* Selected Team */}
      {team && (
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Selected Team</h2>

          <div className="mb-6">
            <p className="text-gray-700 mb-2"><strong>Method:</strong> {team.method}</p>
            <p className="text-gray-700 mb-4"><strong>Analysis:</strong> {team.composition_analysis}</p>

            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-blue-100 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-600">Guards</p>
                <p className="text-2xl font-bold text-blue-700">{team.position_distribution.Guard || 0}</p>
              </div>
              <div className="bg-green-100 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-600">Forwards</p>
                <p className="text-2xl font-bold text-green-700">{team.position_distribution.Forward || 0}</p>
              </div>
              <div className="bg-purple-100 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-600">Centers</p>
                <p className="text-2xl font-bold text-purple-700">{team.position_distribution.Center || 0}</p>
              </div>
            </div>
          </div>

          <h3 className="text-xl font-bold text-gray-800 mb-4">Team Roster</h3>
          <div className="space-y-3">
            {team.players.map((player, idx) => (
              <div key={idx} className="bg-gray-50 rounded-lg p-4 flex justify-between items-center">
                <div>
                  <p className="font-bold text-lg">{idx + 1}. {player.player_name}</p>
                  <p className="text-gray-600">Position: {player.predicted_position}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-600">Team Fit: <span className="font-semibold">{player.team_fit_score.toFixed(3)}</span></p>
                  <p className="text-sm text-gray-600">Overall: <span className="font-semibold">{player.overall_score.toFixed(3)}</span></p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Top Players */}
      {topPlayers.length > 0 && (
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
            <TrendingUp className="mr-3 text-blue-600" />
            Top 10 Players
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-100">
                  <th className="px-4 py-3 text-left">Rank</th>
                  <th className="px-4 py-3 text-left">Player</th>
                  <th className="px-4 py-3 text-left">Position</th>
                  <th className="px-4 py-3 text-right">Team Fit</th>
                  <th className="px-4 py-3 text-right">Overall Score</th>
                </tr>
              </thead>
              <tbody>
                {topPlayers.map((player, idx) => (
                  <tr key={idx} className="border-b hover:bg-gray-50">
                    <td className="px-4 py-3">{idx + 1}</td>
                    <td className="px-4 py-3 font-semibold">{player.player_name}</td>
                    <td className="px-4 py-3">{player.predicted_position}</td>
                    <td className="px-4 py-3 text-right">{player.team_fit_score.toFixed(3)}</td>
                    <td className="px-4 py-3 text-right font-bold text-blue-600">{player.overall_score.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
