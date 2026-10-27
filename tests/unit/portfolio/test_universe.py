"""Tests de carga y validacion de universos (Plan 17)."""
import pytest

from btcx.portfolio.universe import UniverseSpec, load_universe, validate_universe_assets


def test_load_universe_global5(tmp_path):
    """load_universe carga global5 con per_asset_overrides correctamente."""
    yaml_content = """
id: GLOBAL5
assets: [BTC, ETH, GLD, SPY, IEF]
rate: USD_TBILL_3M
weights:
  method: inverse_vol
correlation:
  returns: weekly
  halflife_weeks: 52
idm_bounds: [1.0, 2.5]
portfolio_vol_target: train_median
per_asset_overrides:
  IEF: {w_min: 0.0}
"""
    (tmp_path / "global5.yaml").write_text(yaml_content)
    spec = load_universe(tmp_path / "global5.yaml")
    assert spec.id == "GLOBAL5"
    assert "GLD" in spec.assets
    assert "IEF" in spec.assets
    assert len(spec.assets) == 5
    assert spec.per_asset_overrides.get("IEF", {}).get("w_min") == 0.0
    assert spec.idm_bounds == (1.0, 2.5)
    assert spec.correlation_halflife_weeks == 52


def test_universe_activo_faltante_lanza_error():
    """Activo no en asset_specs lanza ValueError."""
    from unittest.mock import MagicMock

    spec = UniverseSpec(
        id="TEST",
        assets=("BTC", "MISSING"),
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

    with pytest.raises(ValueError, match="MISSING"):
        validate_universe_assets(spec, {"BTC": btc_spec})


def test_universe_assets_como_tupla(tmp_path):
    """Los assets del universo se almacenan como tupla."""
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
    (tmp_path / "c.yaml").write_text(yaml_content)
    spec = load_universe(tmp_path / "c.yaml")
    assert isinstance(spec.assets, tuple)


def test_universe_idm_bounds_como_tupla(tmp_path):
    """Los idm_bounds del universo se almacenan como tupla de floats."""
    yaml_content = """
id: TEST
assets: [BTC]
rate: USD_TBILL_3M
weights:
  method: inverse_vol
correlation:
  returns: weekly
  halflife_weeks: 26
idm_bounds: [1.0, 3.0]
portfolio_vol_target: train_median
per_asset_overrides: {}
"""
    (tmp_path / "t.yaml").write_text(yaml_content)
    spec = load_universe(tmp_path / "t.yaml")
    assert spec.idm_bounds == (1.0, 3.0)
    assert isinstance(spec.idm_bounds[0], float)
    assert isinstance(spec.idm_bounds[1], float)
