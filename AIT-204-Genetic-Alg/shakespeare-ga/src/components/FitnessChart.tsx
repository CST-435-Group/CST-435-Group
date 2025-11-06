import React, { useEffect, useRef } from 'react';
import '../styles/FitnessChart.css';

export interface DataPoint {
  generation: number;
  bestFitness: number;
  averageFitness: number;
}

interface FitnessChartProps {
  data: DataPoint[];
}

const FitnessChart: React.FC<FitnessChartProps> = ({ data }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const padding = 50;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;

    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;

    // Horizontal grid lines (fitness %)
    for (let i = 0; i <= 10; i++) {
      const y = padding + (chartHeight * i) / 10;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(canvas.width - padding, y);
      ctx.stroke();

      // Y-axis labels
      ctx.fillStyle = '#666';
      ctx.font = '12px Arial';
      ctx.textAlign = 'right';
      ctx.fillText(`${100 - i * 10}%`, padding - 10, y + 4);
    }

    // Find max generation for scaling
    const maxGen = Math.max(...data.map(d => d.generation));
    const genStep = Math.max(1, Math.floor(maxGen / 10));

    // Vertical grid lines (generations)
    for (let i = 0; i <= 10; i++) {
      const gen = i * genStep;
      if (gen > maxGen) break;

      const x = padding + (chartWidth * gen) / maxGen;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, canvas.height - padding);
      ctx.stroke();

      // X-axis labels
      ctx.fillStyle = '#666';
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(gen.toString(), x, canvas.height - padding + 20);
    }

    // Draw axes
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#333';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Generation', canvas.width / 2, canvas.height - 10);

    ctx.save();
    ctx.translate(15, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Fitness (%)', 0, 0);
    ctx.restore();

    // Helper function to convert data point to canvas coordinates
    const toCanvasX = (generation: number) => {
      return padding + (chartWidth * generation) / maxGen;
    };

    const toCanvasY = (fitness: number) => {
      return padding + chartHeight - (chartHeight * fitness) / 100;
    };

    // Draw best fitness line
    ctx.strokeStyle = '#2ecc71';
    ctx.lineWidth = 3;
    ctx.beginPath();
    data.forEach((point, index) => {
      const x = toCanvasX(point.generation);
      const y = toCanvasY(point.bestFitness);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw average fitness line
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((point, index) => {
      const x = toCanvasX(point.generation);
      const y = toCanvasY(point.averageFitness);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw legend
    const legendX = canvas.width - padding - 150;
    const legendY = padding + 20;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(legendX - 10, legendY - 15, 140, 50);
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;
    ctx.strokeRect(legendX - 10, legendY - 15, 140, 50);

    // Best fitness legend
    ctx.strokeStyle = '#2ecc71';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 30, legendY);
    ctx.stroke();

    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText('Best Fitness', legendX + 40, legendY + 4);

    // Average fitness legend
    ctx.strokeStyle = '#3498db';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(legendX, legendY + 20);
    ctx.lineTo(legendX + 30, legendY + 20);
    ctx.stroke();

    ctx.fillText('Avg Fitness', legendX + 40, legendY + 24);
  }, [data]);

  return (
    <div className="fitness-chart">
      <h3>Fitness Progress</h3>
      <canvas ref={canvasRef} width={800} height={400} />
    </div>
  );
};

export default FitnessChart;
