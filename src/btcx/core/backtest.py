"""ÚNICO lugar del proyecto donde una decisión del día t se aplica al retorno de t+1 (pre-plan §5).

Motor de panel desde el primer día: recibe una matriz de exposiciones (T, A) sobre A activos.
Un solo activo es el caso A = 1, no una función distinta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcx.core.types import AssetData, BacktestResult, CostModel, PanelData


def _as_frame(w: pd.DataFrame | pd.Series, ids: tuple[str, ...]) -> pd.DataFrame:
    if isinstance(w, pd.Series):
        if len(ids) != 1:
            raise ValueError(
                f"w_target es una Series pero el panel tiene {len(ids)} activos. "
                "Pasa un DataFrame con columnas = ids del panel."
            )
        return w.to_frame(name=ids[0])
    return w


def _validate_inputs(W: pd.DataFrame, panel: PanelData) -> None:
    # El índice debe ser DatetimeIndex monótono sin duplicados (antes de comparar con panel)
    if not isinstance(W.index, pd.DatetimeIndex):
        raise ValueError("El índice de w_target debe ser DatetimeIndex")
    if not W.index.is_monotonic_increasing:
        raise ValueError("El índice de w_target no es monótono creciente")
    if W.index.duplicated().any():
        raise ValueError("El índice de w_target tiene fechas duplicadas")
    if not W.index.equals(panel.r.index):
        raise ValueError("w_target y panel tienen índices distintos")
    if list(W.columns) != list(panel.ids):
        raise ValueError(
            f"Columnas de w_target {list(W.columns)} != ids del panel {list(panel.ids)}"
        )
    if W.isna().any().any():
        raise ValueError("w_target contiene NaN")
    if (W < 0).any().any():
        raise ValueError("w_target contiene valores negativos (posiciones cortas no soportadas)")


def _pack(
    W: pd.DataFrame,
    panel: PanelData,
    held: np.ndarray,
    turn: np.ndarray,
    comp: np.ndarray,
    contrib: np.ndarray,
    costs: CostModel,
) -> BacktestResult:
    """Construye BacktestResult eliminando la fila t=0 donde el retorno no está definido."""
    idx = W.index
    ids = list(panel.ids)

    # El retorno del día t (t >= 1) se realiza usando held[t-1]; comp[0] es NaN por diseño.
    # Devolvemos retornos desde idx[1] (primer retorno realizable).
    ret_idx = idx[1:]
    ret = pd.Series(
        comp[1:, 0] + comp[1:, 1] + comp[1:, 2] + comp[1:, 3],
        index=ret_idx,
        name="returns",
    )
    return BacktestResult(
        returns=ret,
        w_target=pd.DataFrame(W.to_numpy(), index=idx, columns=ids),
        w_held=pd.DataFrame(held, index=idx, columns=ids),
        turnover=pd.DataFrame(turn, index=idx, columns=ids),
        components=pd.DataFrame(
            comp[1:],
            index=ret_idx,
            columns=["asset", "cash", "financing", "trading"],
        ),
        asset_contrib=pd.DataFrame(contrib[1:], index=ret_idx, columns=ids),
        costs=costs,
    )


def backtest(
    w_target: pd.DataFrame | pd.Series,
    data: PanelData | AssetData,
    costs: CostModel,
    w_init: np.ndarray | float = 0.0,
) -> BacktestResult:
    """Convierte exposiciones objetivo en retornos netos de la cartera.

    Parameters
    ----------
    w_target:
        Exposiciones objetivo decididas con información ≤ cierre de cada día t.
        DataFrame (T, A) o Series (T,) si A = 1.
    data:
        Datos del activo o panel. Si es AssetData se convierte a PanelData internamente.
    costs:
        Modelo de costos: buffer, cash_yield, drift, bps_multiplier.
    w_init:
        Exposición inicial antes del primer día (por defecto 0 = entrada desde efectivo).

    Returns
    -------
    BacktestResult con retornos indexados en el día de realización (t+1).
    """
    panel = data.as_panel() if isinstance(data, AssetData) else data
    W = _as_frame(w_target, panel.ids)
    _validate_inputs(W, panel)

    Wv = W.to_numpy(dtype=float)
    R = panel.r.to_numpy(dtype=float)
    trad = panel.tradable.to_numpy(dtype=bool)
    rf = panel.rf.to_numpy(dtype=float)
    fin = panel.fin.to_numpy(dtype=float)
    c = panel.trade_bps.to_numpy(dtype=float) * costs.bps_multiplier / 1e4
    T, A = Wv.shape
    cash_on = 1.0 if costs.cash_yield else 0.0

    held = np.empty((T, A), dtype=float)
    turn = np.empty((T, A), dtype=float)
    comp = np.full((T, 4), np.nan, dtype=float)  # asset, cash, financing, trading
    contrib = np.full((T, A), np.nan, dtype=float)

    pre = np.broadcast_to(np.asarray(w_init, dtype=float), (A,)).copy()
    cost_prev = 0.0

    for t in range(T):
        if t > 0:
            h = held[t - 1]
            a = h * R[t]
            gross_exposure = float(h.sum())
            cash = rf[t] * max(1.0 - gross_exposure, 0.0) * cash_on
            financing = -fin[t] * max(gross_exposure - 1.0, 0.0)
            comp[t] = (float(a.sum()), cash, financing, -cost_prev)
            contrib[t] = a
            if costs.drift:
                gross = float(a.sum()) + cash + financing
                denom = 1.0 + gross
                pre = h * (1.0 + R[t]) / denom if denom != 0.0 else h.copy()
            else:
                pre = h.copy()

        tgt = Wv[t]
        # Rebalancear si: tradable Y (pre == 0 O desviación supera el buffer)
        rebalance = trad[t] & ((pre <= 0.0) | (np.abs(tgt - pre) > costs.buffer * pre))
        new = np.where(rebalance, tgt, pre)
        turn[t] = np.abs(new - pre)
        held[t] = new
        cost_prev = float((c * turn[t]).sum())

    return _pack(W, panel, held, turn, comp, contrib, costs)
