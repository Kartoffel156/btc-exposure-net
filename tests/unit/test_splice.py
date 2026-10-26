"""Tests de empalme de segmentos."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from btcx.data.splice import check_overlap, splice_returns, splice_segments


def _make_df(start: str, end: str, start_price: float = 100.0, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0001, 0.01, len(idx))
    close = start_price * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close, "volume": 1e6, "source": "test"}, index=idx)


class TestSplice:
    def test_check_overlap_passes(self):
        df1 = _make_df("2020-01-01", "2020-06-30", seed=1)
        start_price = df1.loc["2020-01-01", "close"] * 1.001
        df2 = _make_df("2020-01-01", "2021-12-31", start_price=start_price, seed=1)
        # mismo seed y precio inicial casi igual → diff mínima
        report = check_overlap(df1, df2, date(2020, 1, 1), date(2020, 6, 30), tol=0.1)
        assert report.overlap_days > 0
        assert report.passed

    def test_check_overlap_fails(self):
        df1 = _make_df("2020-01-01", "2020-06-30", start_price=100.0, seed=1)
        df2 = _make_df("2020-01-01", "2021-12-31", start_price=200.0, seed=2)
        report = check_overlap(df1, df2, date(2020, 1, 1), date(2020, 6, 30), tol=0.005)
        assert not report.passed

    def test_splice_returns_preserves_each_side(self):
        df_old = _make_df("2019-01-01", "2020-12-31", seed=10)
        df_new = _make_df("2020-01-01", "2021-12-31", seed=11)
        cut = pd.Timestamp("2020-07-01")
        result = splice_returns(df_old, df_new, cut)

        # Antes del corte: retornos de old (reescalados)
        before_cut = result.loc[: cut - pd.Timedelta(days=1)]
        assert len(before_cut) > 0
        # Desde el corte: new sin cambios
        after_cut = result.loc[cut:]
        pd.testing.assert_series_equal(
            after_cut["close"],
            df_new.loc[cut:, "close"],
            check_names=False,
        )

    def test_splice_returns_continuity(self):
        df_old = _make_df("2019-01-01", "2020-12-31", seed=10)
        df_new = _make_df("2020-01-01", "2021-12-31", seed=11)
        cut = pd.Timestamp("2020-07-01")
        result = splice_returns(df_old, df_new, cut)
        prev = cut - pd.Timedelta(days=1)
        # El close del día anterior al corte debe estar muy cerca del close de new en ese día
        if prev in df_new.index:
            diff = abs(result.loc[prev, "close"] / df_new.loc[prev, "close"] - 1)
            assert diff < 1e-9

    def test_splice_n3_segments(self):
        """3 segmentos con solapamiento en el rango declarado deben empalmarse correctamente.

        Los tres segmentos deben solaparse en el rango overlap del SpliceSpec.
        Se usan seeds con precio inicial casi igual para que check_overlap pase.
        """
        df1 = _make_df("2018-01-01", "2019-06-30", seed=1, start_price=100.0)
        # df2 y df3 empiezan en 2019-01-01 (dentro del overlap) con precios similares
        df2 = _make_df("2019-01-01", "2020-06-30", seed=1, start_price=100.0)
        df3 = _make_df("2019-01-01", "2021-12-31", seed=1, start_price=100.0)

        from btcx.assets.spec import (
            AssetPeriods,
            AssetSpec,
            LeverageSpec,
            SourceSegment,
            SpliceSpec,
        )

        spec = AssetSpec(
            id="TST",
            name="Test",
            asset_class="crypto",
            quote_currency="USD",
            rate="USD_TBILL_3M",
            calendar="continuous",
            periods=AssetPeriods(
                history_start=date(2018, 1, 1),
                dev_start=date(2019, 1, 1),
                dev_end=date(2021, 6, 30),
                test_start=date(2021, 7, 1),
            ),
            sources=[
                SourceSegment(adapter="synthetic"),
                SourceSegment(adapter="synthetic"),
                SourceSegment(adapter="synthetic"),
            ],
            splice=SpliceSpec(
                overlap=(date(2019, 1, 1), date(2019, 6, 30)), max_median_abs_diff=0.5
            ),
            columns=["close"],
            outlier_abs_return=0.4,
            trade_bps=15.0,
            leverage=LeverageSpec(
                instrument="spot_margin", maintenance_margin=0.10, max_leverage=2.0
            ),
        )
        result, _ = splice_segments([df1, df2, df3], spec)
        assert len(result) > 0
        assert result.index.is_monotonic_increasing
