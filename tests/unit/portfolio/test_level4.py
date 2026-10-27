"""Tests del Nivel 4 y cartera multi-activo (Plan 17)."""
import numpy as np
import pandas as pd
import pytest

from btcx.portfolio.covariance import (
    ewma_corr,
    idm,
    inverse_vol_weights,
    weekly_returns,
)
from btcx.portfolio.universe import UniverseSpec, load_universe, validate_universe_assets


def test_idm_un_activo():
    """IDM = 1.0 con un solo activo."""
    weights = np.array([1.0])
    rho = np.array([[1.0]])
    assert idm(weights, rho, (1.0, 2.5)) == pytest.approx(1.0)


def test_idm_aumenta_con_baja_correlacion():
    """IDM aumenta cuando la correlacion baja."""
    w = np.array([0.5, 0.5])
    rho_high = np.array([[1.0, 0.9], [0.9, 1.0]])
    rho_low = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert idm(w, rho_low, (1.0, 2.5)) > idm(w, rho_high, (1.0, 2.5))


def test_inverse_vol_weights_suma_uno():
    """Los pesos de inverse_vol suman 1."""
    sigma = np.array([0.1, 0.2, 0.3])
    w = inverse_vol_weights(sigma)
    assert abs(w.sum() - 1.0) < 1e-10


def test_dos_activos_correlacion_1_reproduce_nivel3():
    """Dos copias del mismo activo (correlacion 1) → Nivel 4 reproduce mitad de peso en cada copia."""
    sigma = np.array([0.2, 0.2])
    w = inverse_vol_weights(sigma)
    rho = np.array([[1.0, 1.0], [1.0, 1.0]])
    # IDM con correlacion perfecta = 1/sqrt(w^T rho w) = 1/sqrt(1) = 1
    idm_val = idm(w, rho, (1.0, 2.5))
    assert idm_val == pytest.approx(1.0, abs=1e-6)
    # Pesos iguales
    assert w[0] == pytest.approx(0.5, abs=1e-10)


def test_weekly_returns_forma_correcta():
    """weekly_returns reduce el numero de filas."""
    # 30 dias de retornos diarios
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    r = pd.DataFrame(
        {"A": np.random.randn(30) * 0.01, "B": np.random.randn(30) * 0.01},
        index=dates,
    )
    wr = weekly_returns(r)
    assert len(wr) <= len(r)
    assert list(wr.columns) == ["A", "B"]


def test_load_universe_crypto2(tmp_path):
    """load_universe carga crypto2.yaml correctamente."""
    yaml_content = """
id: CRYPTO2
assets: [BTC, ETH]
rate: USD_TBILL_3M
weights:
  method: inverse_vol
correlation:
  returns: weekly
  halflife_weeks: 52
idm_bounds: [1.0, 2.5]
portfolio_vol_target: train_median
per_asset_overrides: {}
"""
    (tmp_path / "crypto2.yaml").write_text(yaml_content)
    spec = load_universe(tmp_path / "crypto2.yaml")
    assert spec.id == "CRYPTO2"
    assert "BTC" in spec.assets
    assert spec.idm_bounds == (1.0, 2.5)


def test_validate_universe_rate_distinta_lanza_error():
    """validate_universe_assets lanza ValueError si los activos tienen rates distintos."""
    from unittest.mock import MagicMock

    spec = UniverseSpec(
        id="TEST",
        assets=("BTC", "ETH"),
        rate="USD_TBILL_3M",
        weights_method="inverse_vol",
        correlation_returns="weekly",
        correlation_halflife_weeks=52,
        idm_bounds=(1.0, 2.5),
        portfolio_vol_target="train_median",
        per_asset_overrides={},
    )

    btc_spec = MagicMock()
    btc_spec.rate = "USD_TBILL_3M"
    eth_spec = MagicMock()
    eth_spec.rate = "EUR_RATE"  # diferente!

    with pytest.raises(ValueError, match="rate"):
        validate_universe_assets(spec, {"BTC": btc_spec, "ETH": eth_spec})


def test_idm_recortado_a_bounds():
    """IDM queda dentro de los bounds especificados."""
    # Con baja correlacion, IDM natural = sqrt(2) ~ 1.41, dentro de (1.0, 2.5)
    w = np.array([0.5, 0.5])
    rho_zero = np.array([[1.0, 0.0], [0.0, 1.0]])
    val = idm(w, rho_zero, (1.0, 1.2))
    assert val <= 1.2
    assert val >= 1.0


def test_ewma_corr_diagonal_uno():
    """La diagonal de la matriz de correlacion EWMA es siempre 1."""
    dates = pd.date_range("2020-01-01", periods=60, freq="W")
    r = pd.DataFrame(
        {
            "A": np.random.randn(60) * 0.02,
            "B": np.random.randn(60) * 0.01,
        },
        index=dates,
    )
    corr = ewma_corr(r, halflife_weeks=10)
    assert corr.shape == (2, 2)
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr[1, 1] == pytest.approx(1.0)
    assert abs(corr[0, 1]) <= 1.0
