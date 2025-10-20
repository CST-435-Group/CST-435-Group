import { useEffect, useState } from 'react';
import { getStatistics } from '../../services/api';
import { BarChart3, Loader2 } from 'lucide-react';

export default function StatisticsTab() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function fetchStats() {
      try {
        const data = await getStatistics();
        setStats(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchStats();
  }, []);

  if (loading) return <div className="flex justify-center py-6"><Loader2 className="animate-spin" /></div>;
  if (error) return <div className="text-red-500">{error}</div>;

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-blue-500" />
        Dataset Statistics
      </h2>

      <div className="bg-white p-4 rounded-xl shadow">
        <p className="text-gray-700 mb-2">Total Reviews: <strong>{stats.total_reviews}</strong></p>

        <h3 className="text-gray-800 font-semibold mt-4 mb-2">Sentiment Distribution</h3>
        <ul className="text-gray-700">
          {Object.entries(stats.sentiment_distribution).map(([sentiment, count]) => (
            <li key={sentiment}>
              {sentiment}: <strong>{count}</strong>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
