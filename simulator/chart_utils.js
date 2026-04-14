/**
 * chart_utils.js – Chart.js helpers for the Inference Dashboard
 */

const ChartUtils = (() => {
  let mainChart = null;
  let errorChart = null;

  const COLORS = {
    actual:    'rgba(99, 102, 241, 1)',
    predicted: 'rgba(16, 185, 129, 1)',
    error:     'rgba(239, 68, 68, 0.75)',
    errorBg:   'rgba(239, 68, 68, 0.15)',
    actualBg:  'rgba(99, 102, 241, 0.08)',
    predictedBg:'rgba(16, 185, 129, 0.08)',
  };

  const BASE_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
      legend: {
        position: 'top',
        labels: { font: { size: 12 }, padding: 16 },
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
        },
      },
    },
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        ticks: { maxTicksLimit: 12, font: { size: 11 } },
        grid: { color: 'rgba(0,0,0,0.04)' },
        title: { display: true, text: 'Sample Index', font: { size: 11 } },
      },
      y: {
        ticks: { font: { size: 11 } },
        grid: { color: 'rgba(0,0,0,0.04)' },
        title: { display: true, text: 'Traffic Value', font: { size: 11 } },
      },
    },
  };

  /**
   * Render (or update) the actual-vs-predicted line chart.
   * @param {HTMLCanvasElement} canvas
   * @param {number[]} labels  – sample indices
   * @param {number[]} actual
   * @param {number[]} predicted
   */
  function renderMainChart(canvas, labels, actual, predicted) {
    if (mainChart) mainChart.destroy();

    mainChart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Actual',
            data: actual,
            borderColor: COLORS.actual,
            backgroundColor: COLORS.actualBg,
            borderWidth: 2,
            pointRadius: labels.length > 200 ? 0 : 2,
            tension: 0.3,
            fill: true,
          },
          {
            label: 'Predicted',
            data: predicted,
            borderColor: COLORS.predicted,
            backgroundColor: COLORS.predictedBg,
            borderWidth: 2,
            pointRadius: labels.length > 200 ? 0 : 2,
            tension: 0.3,
            fill: true,
            borderDash: [5, 3],
          },
        ],
      },
      options: {
        ...BASE_OPTS,
        plugins: {
          ...BASE_OPTS.plugins,
          title: {
            display: true,
            text: 'Actual vs Predicted Traffic',
            font: { size: 14, weight: 'bold' },
            padding: { bottom: 10 },
          },
        },
      },
    });
  }

  /**
   * Render (or update) the prediction-error chart.
   * @param {HTMLCanvasElement} canvas
   * @param {number[]} labels
   * @param {number[]} errors
   */
  function renderErrorChart(canvas, labels, errors) {
    if (errorChart) errorChart.destroy();

    errorChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Prediction Error (Actual − Predicted)',
            data: errors,
            backgroundColor: errors.map(e =>
              e >= 0 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(16, 185, 129, 0.6)'
            ),
            borderColor: errors.map(e =>
              e >= 0 ? 'rgba(239, 68, 68, 1)' : 'rgba(16, 185, 129, 1)'
            ),
            borderWidth: 1,
          },
        ],
      },
      options: {
        ...BASE_OPTS,
        plugins: {
          ...BASE_OPTS.plugins,
          title: {
            display: true,
            text: 'Prediction Error Distribution',
            font: { size: 14, weight: 'bold' },
            padding: { bottom: 10 },
          },
        },
        scales: {
          ...BASE_OPTS.scales,
          y: {
            ...BASE_OPTS.scales.y,
            title: { display: true, text: 'Error', font: { size: 11 } },
          },
        },
      },
    });
  }

  /**
   * Destroy all charts (call before rendering new results).
   */
  function destroyAll() {
    if (mainChart) { mainChart.destroy(); mainChart = null; }
    if (errorChart) { errorChart.destroy(); errorChart = null; }
  }

  return { renderMainChart, renderErrorChart, destroyAll };
})();
