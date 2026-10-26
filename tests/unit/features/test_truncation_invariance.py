"""Test de invarianza a truncación (anti-lookahead). Ver pre-plan §7."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcx.core.types import AssetData
from btcx.features.pipeline import FeaturePipeline

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE,
    reason="hypothesis no disponible",
)


@pytest.mark.slow
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(cut=st.integers(min_value=500, max_value=2800))
def test_features_truncation_invariant(
    cut: int,
    md_syn_long: AssetData,
    default_pipeline: FeaturePipeline,
) -> None:
    """Las features en t no cambian si se trunca la serie en t.

    Verifica que no haya lookahead: calcular sobre la historia completa
    hasta t produce el mismo resultado que calcular sobre la serie truncada en t.

    Args:
        cut: Índice de corte en la serie temporal.
        md_syn_long: AssetData sintético de ~3000 días.
        default_pipeline: FeaturePipeline con benchmark TrendVolTarget.
    """
    train_pos = np.arange(400)
    # Si el corte está dentro del período de entrenamiento, saltar
    if cut <= int(train_pos[-1]):
        return

    # Verificar que hay datos suficientes
    if cut >= len(md_syn_long.r):
        return

    # Calcular sobre la serie completa
    full = default_pipeline.build(md_syn_long, train_pos, "full").X

    # Calcular sobre la serie truncada
    date_at_cut = md_syn_long.r.index[cut]
    start_date = md_syn_long.r.index[0]
    trunc_md = md_syn_long.slice(start_date, date_at_cut)

    # El período de entrenamiento debe estar dentro de la serie truncada
    if train_pos[-1] >= len(trunc_md.r):
        return

    trunc = default_pipeline.build(trunc_md, train_pos, "trunc").X

    # Comparar en la fecha del corte
    if date_at_cut not in trunc.index or date_at_cut not in full.index:
        return

    full_row = full.loc[date_at_cut].dropna()
    trunc_row = trunc.loc[date_at_cut].dropna()

    if full_row.empty or trunc_row.empty:
        return

    # Las columnas comunes deben ser idénticas
    common = full_row.index.intersection(trunc_row.index)
    if common.empty:
        return

    pd.testing.assert_series_equal(
        full_row[common],
        trunc_row[common],
        check_names=False,
        rtol=1e-10,
        atol=1e-10,
    )
