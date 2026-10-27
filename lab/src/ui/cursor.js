/**
 * cursor.js — cursor sincronizado entre paneles.
 */

/**
 * Configura el seguimiento del cursor sobre todos los paneles canvas.
 * Actualiza store.cursorIndex con el índice de tiempo bajo el mouse.
 * @param {object} store
 * @param {object} panels   { a: { canvas }, b: { canvas }, ... }
 */
export function setupCursor(store, panels) {
  for (const panel of Object.values(panels)) {
    if (!panel.canvas) continue;
    panel.canvas.addEventListener('mousemove', (e) => {
      const rect = panel.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const w = rect.width;

      // Estimar n del activo actual
      const asset = store.data && store.assetId ? store.data[store.assetId] : null;
      const n = asset && asset.r ? asset.r.length : 0;
      if (n === 0) return;

      const idx = Math.min(Math.max(0, Math.round((x / w) * (n - 1))), n - 1);
      if (idx !== store.cursorIndex) {
        store.cursorIndex = idx;
        store.markDirty('b');
        store.markDirty('c');
        store.markDirty('h');
      }
    });

    panel.canvas.addEventListener('mouseleave', () => {
      store.cursorIndex = null;
      store.markDirty('b');
      store.markDirty('c');
      store.markDirty('h');
    });
  }
}
