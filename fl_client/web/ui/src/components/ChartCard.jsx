import { useRef, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Tooltip, Filler
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler);

function hexToRgba(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export default function ChartCard({ title, icon: Icon, labels = [], data = [], color = '#6366f1' }) {
  const chartData = {
    labels,
    datasets: [{
      data,
      borderColor: color,
      backgroundColor: hexToRgba(color, 0.06),
      fill: true,
      tension: 0.35,
      pointRadius: 3,
      pointHoverRadius: 6,
      pointBackgroundColor: color,
      pointBorderColor: 'transparent',
      pointHoverBorderColor: '#fff',
      pointHoverBorderWidth: 2,
      borderWidth: 2,
    }],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    scales: {
      x: {
        ticks: { color: '#475569', font: { size: 10, family: 'Inter' } },
        grid: { color: 'rgba(51,65,85,0.15)', drawBorder: false },
        border: { display: false },
      },
      y: {
        ticks: { color: '#475569', font: { size: 10, family: 'Inter' } },
        grid: { color: 'rgba(51,65,85,0.15)', drawBorder: false },
        border: { display: false },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(99,102,241,0.2)',
        borderWidth: 1,
        cornerRadius: 6,
        padding: 8,
        titleFont: { family: 'Inter', weight: '600', size: 12 },
        bodyFont: { family: 'JetBrains Mono', size: 11 },
      },
    },
    interaction: { intersect: false, mode: 'index' },
  };

  return (
    <div className="card">
      {title && (
        <div className="flex items-center gap-2 mb-3">
          {Icon && <Icon size={14} className="text-slate-500" strokeWidth={1.75} />}
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</h3>
        </div>
      )}
      <div className="h-48">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
