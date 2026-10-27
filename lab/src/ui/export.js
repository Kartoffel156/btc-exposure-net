/**
 * export.js — exporta resultados del laboratorio a JSON/CSV.
 */

/**
 * Exporta los resultados actuales del store a un archivo JSON descargable.
 * @param {object} store
 */
export function exportResults(store) {
  const result = {
    assetId: store.assetId,
    step: store.step,
    lastLoss: store.lastLoss,
    metrics: store.metrics ? {
      sharpe: store.metrics.sharpe,
      cagr: store.metrics.cagr,
      mdd: store.metrics.mdd,
      sortino: store.metrics.sortino,
      turnover: store.metrics.turnover,
    } : null,
    attribution: store.attribution ? Array.from(store.attribution) : null,
    exportedAt: new Date().toISOString(),
  };

  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `btcx-lab-${store.assetId || 'export'}-step${store.step}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Exporta la curva de equity a CSV.
 * @param {Float64Array|number[]} equity
 * @param {string[]} dates
 * @param {string} filename
 */
export function exportEquityCSV(equity, dates, filename = 'equity.csv') {
  const lines = ['date,equity'];
  for (let i = 0; i < equity.length; i++) {
    lines.push(`${dates[i] || i},${equity[i].toFixed(6)}`);
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
