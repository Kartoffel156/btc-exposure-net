"""Tests unitarios del core de activos: spec, registry, calendars, rates, candado."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from btcx.assets.calendars import apply_calendar
from btcx.assets.registry import UnknownAssetClassError, UnknownAssetError, load_asset
from btcx.assets.spec import AssetPeriods, AssetSpec, LeverageSpec, SourceSegment
from btcx.config import load_settings
from btcx.data.rates import dtb3_to_daily, to_calendar_asof

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_settings(tmp_path: Path, extra_yaml: str = "") -> object:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    rates_dir = tmp_path / "rates"
    rates_dir.mkdir()
    # Crear un base.yaml temporal que apunta a los directorios tmp
    base = (Path("config") / "base.yaml").read_text()
    override = f"\ndata:\n  assets_dir: {assets_dir}\n  rates_dir: {rates_dir}\n  universe: []\n"
    cfg = tmp_path / "base.yaml"
    cfg.write_text(base + override + extra_yaml)
    return load_settings(cfg)


def _write_asset(assets_dir: Path, data: dict) -> None:
    path = assets_dir / f"{data['id']}.yaml"
    path.write_text(yaml.dump(data))


# ── AssetSpec ─────────────────────────────────────────────────────────────────


class TestAssetSpec:
    def test_valid_spec(self):
        spec = AssetSpec(
            id="TST",
            name="Test",
            asset_class="crypto",
            quote_currency="USD",
            rate="USD_TBILL_3M",
            calendar="continuous",
            periods=AssetPeriods(
                history_start=date(2020, 1, 1),
                dev_start=date(2021, 1, 1),
                dev_end=date(2022, 12, 31),
                test_start=date(2023, 1, 1),
            ),
            sources=[SourceSegment(adapter="synthetic", params={"seed": 42})],
            columns=["open", "high", "low", "close", "volume"],
            outlier_abs_return=0.4,
            trade_bps=15.0,
            leverage=LeverageSpec(
                instrument="spot_margin", maintenance_margin=0.10, max_leverage=2.0
            ),
        )
        assert spec.id == "TST"

    def test_periods_order_enforced(self):
        with pytest.raises(ValueError, match="orden"):
            AssetPeriods(
                history_start=date(2022, 1, 1),
                dev_start=date(2021, 1, 1),  # anterior a history_start
                dev_end=date(2022, 12, 31),
                test_start=date(2023, 1, 1),
            )

    def test_no_close_column_fails(self):
        with pytest.raises(ValueError, match="close"):
            AssetSpec(
                id="TST",
                name="Test",
                asset_class="crypto",
                quote_currency="USD",
                rate="USD_TBILL_3M",
                calendar="continuous",
                periods=AssetPeriods(
                    history_start=date(2020, 1, 1),
                    dev_start=date(2021, 1, 1),
                    dev_end=date(2022, 12, 31),
                    test_start=date(2023, 1, 1),
                ),
                sources=[SourceSegment(adapter="synthetic")],
                columns=["open", "high", "low", "volume"],  # sin close
                outlier_abs_return=0.4,
                trade_bps=15.0,
                leverage=LeverageSpec(
                    instrument="spot_margin", maintenance_margin=0.10, max_leverage=2.0
                ),
            )

    def test_multiple_sources_require_splice(self):
        with pytest.raises(ValueError, match="splice"):
            AssetSpec(
                id="TST",
                name="Test",
                asset_class="crypto",
                quote_currency="USD",
                rate="USD_TBILL_3M",
                calendar="continuous",
                periods=AssetPeriods(
                    history_start=date(2020, 1, 1),
                    dev_start=date(2021, 1, 1),
                    dev_end=date(2022, 12, 31),
                    test_start=date(2023, 1, 1),
                ),
                sources=[
                    SourceSegment(adapter="synthetic"),
                    SourceSegment(adapter="synthetic"),
                ],
                splice=None,
                columns=["close"],
                outlier_abs_return=0.4,
                trade_bps=15.0,
                leverage=LeverageSpec(
                    instrument="spot_margin", maintenance_margin=0.10, max_leverage=2.0
                ),
            )

    def test_invalid_id_pattern(self):
        with pytest.raises(Exception, match=r"\S"):
            AssetSpec(
                id="btc-usd",  # minúsculas y guión no permitidos
                name="Test",
                asset_class="crypto",
                quote_currency="USD",
                rate="USD_TBILL_3M",
                calendar="continuous",
                periods=AssetPeriods(
                    history_start=date(2020, 1, 1),
                    dev_start=date(2021, 1, 1),
                    dev_end=date(2022, 12, 31),
                    test_start=date(2023, 1, 1),
                ),
                sources=[SourceSegment(adapter="synthetic")],
                columns=["close"],
                outlier_abs_return=0.4,
                trade_bps=15.0,
                leverage=LeverageSpec(
                    instrument="spot_margin", maintenance_margin=0.10, max_leverage=2.0
                ),
            )


# ── Registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_load_asset_inherits_class_defaults(self, tmp_path):
        settings = _make_settings(tmp_path)
        _write_asset(
            Path(settings.data.assets_dir),
            {
                "id": "TST",
                "name": "Test",
                "asset_class": "synthetic",
                "quote_currency": "USD",
                "rate": "USD_TBILL_3M",
                "sources": [{"adapter": "synthetic", "params": {"seed": 42}}],
                "columns": ["open", "high", "low", "close", "volume"],
            },
        )
        spec = load_asset("TST", settings)
        # debe heredar calendar=continuous de la clase synthetic
        assert spec.calendar == "continuous"
        assert spec.trade_bps == 15.0

    def test_override_wins(self, tmp_path):
        settings = _make_settings(tmp_path)
        _write_asset(
            Path(settings.data.assets_dir),
            {
                "id": "TST",
                "name": "Test",
                "asset_class": "synthetic",
                "quote_currency": "USD",
                "rate": "USD_TBILL_3M",
                "trade_bps": 99.0,  # override
                "sources": [{"adapter": "synthetic", "params": {"seed": 42}}],
                "columns": ["close"],
            },
        )
        spec = load_asset("TST", settings)
        assert spec.trade_bps == 99.0

    def test_unknown_asset_raises(self, tmp_path):
        settings = _make_settings(tmp_path)
        with pytest.raises(UnknownAssetError):
            load_asset("NOEXISTE", settings)

    def test_unknown_class_raises(self, tmp_path):
        settings = _make_settings(tmp_path)
        _write_asset(
            Path(settings.data.assets_dir),
            {
                "id": "TST",
                "name": "Test",
                "asset_class": "clase_inexistente",
                "quote_currency": "USD",
                "rate": "USD_TBILL_3M",
                "sources": [{"adapter": "synthetic"}],
                "columns": ["close"],
            },
        )
        with pytest.raises(UnknownAssetClassError):
            load_asset("TST", settings)

    def test_periods_inherited_from_base(self, tmp_path):
        settings = _make_settings(tmp_path)
        _write_asset(
            Path(settings.data.assets_dir),
            {
                "id": "TST",
                "name": "Test",
                "asset_class": "synthetic",
                "quote_currency": "USD",
                "rate": "USD_TBILL_3M",
                "sources": [{"adapter": "synthetic", "params": {"seed": 42}}],
                "columns": ["close"],
                # sin bloque periods → hereda de base.yaml
            },
        )
        spec = load_asset("TST", settings)
        assert str(spec.periods.history_start) == "2014-01-01"


# ── Calendarios ───────────────────────────────────────────────────────────────


class TestCalendars:
    def _base_df(self, dates: list[str]) -> pd.DataFrame:
        idx = pd.to_datetime(dates)
        return pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1e6,
                "source": "test",
            },
            index=idx,
        )

    def test_continuous_fills_gap(self):
        df = self._base_df(["2021-01-01", "2021-01-03"])  # falta el 02
        result = apply_calendar(df, "continuous", "2021-01-01", "2021-01-03")
        assert pd.Timestamp("2021-01-02") in result.index
        assert result.loc[pd.Timestamp("2021-01-02"), "volume"] == 0.0
        assert result.loc[pd.Timestamp("2021-01-02"), "tradable"]

    def test_business_marks_weekend_not_tradable(self):
        # 2021-01-04 es lunes, 2021-01-02/03 son sab/dom
        df = self._base_df(["2021-01-01", "2021-01-04"])
        result = apply_calendar(df, "business", "2021-01-01", "2021-01-04")
        assert pd.Timestamp("2021-01-02") in result.index
        assert not result.loc[pd.Timestamp("2021-01-02"), "tradable"]
        assert result.loc[pd.Timestamp("2021-01-04"), "tradable"]

    def test_business_weekend_return_zero(self):
        df = self._base_df(["2021-01-01", "2021-01-04"])
        result = apply_calendar(df, "business", "2021-01-01", "2021-01-04")
        # precio del sábado = precio del viernes (retorno 0)
        fri_close = result.loc[pd.Timestamp("2021-01-01"), "close"]
        sat_close = result.loc[pd.Timestamp("2021-01-02"), "close"]
        assert fri_close == sat_close

    def test_unknown_calendar_raises(self):
        df = self._base_df(["2021-01-01"])
        with pytest.raises(ValueError, match="Calendario"):
            apply_calendar(df, "quarterly", "2021-01-01", "2021-01-01")


# ── Tasas ─────────────────────────────────────────────────────────────────────


class TestRates:
    def test_dtb3_to_daily_5pct(self):
        series = pd.Series([5.0])
        result = dtb3_to_daily(series)
        # y = 365*0.05 / (360 - 91*0.05) ≈ 0.05134
        # rf_d = (1 + y)^(1/365) - 1 ≈ 1.37e-4
        assert abs(result.iloc[0] - 1.3695e-4) < 1e-6

    def test_to_calendar_asof_no_lookahead(self):
        idx = pd.date_range("2021-01-04", periods=5, freq="D")
        rf = pd.Series(
            [0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
            index=pd.date_range("2021-01-04", periods=5, freq="D"),
        )
        result = to_calendar_asof(rf, idx)
        # El 2021-01-04 debe usar el dato del 2021-01-03 (no existe) → NaN
        assert pd.isna(result.iloc[0]) or result.iloc[0] != rf.iloc[0]

    def test_to_calendar_asof_ffill(self):
        idx = pd.date_range("2021-01-05", periods=3, freq="D")
        rf = pd.Series(
            [0.0001, 0.0002],
            index=pd.to_datetime(["2021-01-03", "2021-01-04"]),
        )
        result = to_calendar_asof(rf, idx)
        # 2021-01-05 usa dato del 2021-01-04 (≤ t-1 = 2021-01-04)
        assert result.iloc[0] == pytest.approx(0.0002)


# ── Candado del test ──────────────────────────────────────────────────────────


class TestTestLock:
    def test_load_asset_stops_at_dev_end(self, tmp_path):
        """Por defecto load_asset no incluye el test."""
        from btcx.data.access import load_asset as access_load_asset
        from btcx.data.sources.synthetic import SyntheticAdapter

        settings = _make_settings(tmp_path)
        _write_asset(
            Path(settings.data.assets_dir),
            {
                "id": "SYN2",
                "name": "Synthetic test lock",
                "asset_class": "synthetic",
                "quote_currency": "USD",
                "rate": "USD_TBILL_3M",
                "sources": [{"adapter": "synthetic", "params": {"seed": 99}}],
                "columns": ["open", "high", "low", "close", "volume"],
            },
        )

        # Construir los canónicos necesarios
        from btcx.assets.registry import load_asset as reg_load
        from btcx.data import store

        spec = reg_load("SYN2", settings)
        adapter = SyntheticAdapter()
        df = adapter.fetch(
            spec.sources[0].params, spec.periods.history_start, date(2023, 12, 31), None
        )
        df["filled"] = False
        df["tradable"] = True

        processed = tmp_path / "processed"
        store.save_asset(df, spec, {}, processed)

        # Crear tasa dummy: un dia antes de history_start para cubrir t-1
        from datetime import timedelta

        rf_start = spec.periods.history_start - timedelta(days=1)
        rf = pd.Series(
            0.0001,
            index=pd.date_range(rf_start, date(2023, 12, 31), freq="D"),
        )
        (processed / "rates" / "USD_TBILL_3M").mkdir(parents=True, exist_ok=True)
        store.save_rate(rf, "USD_TBILL_3M", processed)

        frame = access_load_asset(settings, "SYN2", processed_root=str(processed))
        assert frame.prices.index.max() <= pd.Timestamp(spec.periods.dev_end)

    def test_include_test_without_env_raises(self, tmp_path, monkeypatch):
        from btcx.data.access import TestLockedError, _end_date

        monkeypatch.delenv("UNLOCK_TEST", raising=False)
        settings = _make_settings(tmp_path)
        _write_asset(
            Path(settings.data.assets_dir),
            {
                "id": "SYN3",
                "name": "Test",
                "asset_class": "synthetic",
                "quote_currency": "USD",
                "rate": "USD_TBILL_3M",
                "sources": [{"adapter": "synthetic", "params": {"seed": 1}}],
                "columns": ["close"],
            },
        )
        from btcx.assets.registry import load_asset as reg_load

        spec = reg_load("SYN3", settings)
        with pytest.raises(TestLockedError):
            _end_date(spec, include_test=True)
