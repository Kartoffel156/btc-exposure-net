/**
 * alerts.js — notificaciones no bloqueantes.
 */

let alertContainer = null;

function getContainer() {
  if (!alertContainer) {
    alertContainer = document.createElement('div');
    alertContainer.id = 'btcx-alerts';
    alertContainer.style.cssText = `
      position: fixed; top: 10px; right: 10px; z-index: 9999;
      display: flex; flex-direction: column; gap: 8px;
    `;
    document.body.appendChild(alertContainer);
  }
  return alertContainer;
}

/**
 * Muestra una alerta temporal.
 * @param {string} message
 * @param {'info'|'warn'|'error'|'success'} type
 * @param {number} duration   ms (default 3000)
 */
export function showAlert(message, type = 'info', duration = 3000) {
  const container = getContainer();
  const colors = {
    info:    { bg: '#161b22', border: '#3498db', text: '#c9d1d9' },
    warn:    { bg: '#1c1700', border: '#f39c12', text: '#f39c12' },
    error:   { bg: '#1c0000', border: '#e74c3c', text: '#e74c3c' },
    success: { bg: '#001c0a', border: '#2ecc71', text: '#2ecc71' },
  };
  const c = colors[type] || colors.info;

  const el = document.createElement('div');
  el.style.cssText = `
    background: ${c.bg}; border: 1px solid ${c.border}; color: ${c.text};
    padding: 8px 14px; border-radius: 4px; font: 12px monospace;
    max-width: 300px; opacity: 1; transition: opacity 0.3s;
  `;
  el.textContent = message;
  container.appendChild(el);

  setTimeout(() => {
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 300);
  }, duration);
}

export const info = (msg, d) => showAlert(msg, 'info', d);
export const warn = (msg, d) => showAlert(msg, 'warn', d);
export const error = (msg, d) => showAlert(msg, 'error', d);
export const success = (msg, d) => showAlert(msg, 'success', d);
