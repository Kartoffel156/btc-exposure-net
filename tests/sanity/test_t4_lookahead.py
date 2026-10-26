"""Sanidad T4: estrategia tramposa obtiene Sharpe > 5; estrategia honesta ~1.

Una estrategia con acceso al retorno futuro (w_t = 1 si r_{t+1} > 0, si no 0.2)
debe obtener un Sharpe claramente superior al buy & hold.
Si este test falla, el desfase está mal implementado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcx.core.backtest import backtest
from btcx.core.types import CostModel, PanelData
from btcx.metrics.performance import sharpe


def _synthetic_panel(n: int = 2000, seed: int = 0) -> PanelData:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.025, n)
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    rf = pd.Series(0.0001, index=idx)
    fin = rf + 0.005 / 365
    return PanelData(
        ids=("BTC",),
        r=pd.DataFrame({"BTC": r}, index=idx),
        tradable=pd.DataFrame({"BTC": [True] * n}, index=idx),
        low=pd.DataFrame({"BTC": np.nan * np.ones(n)}, index=idx),
        close=pd.DataFrame({"BTC": np.cumprod(1 + r)}, index=idx),
        rf=rf,
        fin=fin,
        trade_bps=pd.Series({"BTC": 0.0}),  # sin costos para maximizar señal
        panel_hash="synthetic_t4",
    )


def test_cheating_strategy_has_high_sharpe():
    panel = _synthetic_panel()
    idx = panel.r.index
    r_arr = panel.r["BTC"].to_numpy()
    n = len(r_arr)

    # Estrategia tramposa: usa r_{t+1} para decidir w_t (lookahead explícito en el TEST)
    # Se construye aquí con shift(-1) — solo permitido en tests/training, no en src/
    w_cheat = np.where(np.roll(r_arr, -1) > 0, 1.0, 0.2)
    w_cheat[-1] = 0.2  # último día sin información

    w_cheat_df = pd.DataFrame({"BTC": w_cheat}, index=idx)
    w_honest_df = pd.DataFrame({"BTC": np.ones(n)}, index=idx)

    costs = CostModel(buffer=0.0, cash_yield=True)
    res_cheat = backtest(w_cheat_df, panel, costs)
    res_honest = backtest(w_honest_df, panel, costs)

    rf_r = panel.rf.reindex(res_cheat.returns.index).fillna(0)
    sharpe_cheat = sharpe(res_cheat.returns, rf_r)
    sharpe_honest = sharpe(res_honest.returns, rf_r)

    # La estrategia tramposa debe superar claramente al buy & hold
    assert sharpe_cheat > 5.0, (
        f"Sharpe tramposo {sharpe_cheat:.2f} debería ser > 5 (el desfase puede estar mal)"
    )
    assert sharpe_cheat > sharpe_honest * 3, (
        f"La estrategia tramposa ({sharpe_cheat:.2f}) debería ser "
        f">> buy & hold ({sharpe_honest:.2f})"
    )
