"""Tests unitarios para walk-forward validation."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from btcx.validation.walkforward import walkforward_splits


@pytest.fixture()
def dev_dates() -> pd.DatetimeIndex:
    """Periodo de desarrollo: 2016-01-01 a 2022-12-31."""
    return pd.date_range("2016-01-01", "2022-12-31", freq="D")


class TestWalkforwardSplits:
    def test_correct_number_of_splits(self, dev_dates: pd.DatetimeIndex) -> None:
        years = [2019, 2020, 2021, 2022]
        splits = walkforward_splits(dev_dates, years, gap_days=30)
        assert len(splits) == 4

    def test_last_train_day_is_before_gap(self, dev_dates: pd.DatetimeIndex) -> None:
        """El último día de train = 1 enero Y − gap_days − 1 día."""
        years = [2021]
        gap_days = 30
        splits = walkforward_splits(dev_dates, years, gap_days=gap_days)
        assert len(splits) == 1
        sp = splits[0]

        # Último día de train en DatetimeIndex
        last_train_date = dev_dates[sp.train_pos[-1]]

        # El cutoff es 1 enero 2021 - 30 días = 2 diciembre 2020
        expected_cutoff = pd.Timestamp("2021-01-01") - pd.Timedelta(days=gap_days)
        # El último día de train debe ser estrictamente antes del cutoff
        assert last_train_date < expected_cutoff, (
            f"Último día train={last_train_date}, cutoff={expected_cutoff}"
        )

    def test_no_test_year_days_in_train(self, dev_dates: pd.DatetimeIndex) -> None:
        """Ningún día del año Y aparece en train."""
        years = [2020, 2021]
        gap_days = 30
        splits = walkforward_splits(dev_dates, years, gap_days=gap_days)

        for sp in splits:
            # Extraer año del bloque test
            test_dates = dev_dates[sp.test_pos]
            test_years = set(test_dates.year.tolist())
            assert len(test_years) == 1
            test_year = test_years.pop()

            # Verificar que ningún día del año test está en train
            train_dates = dev_dates[sp.train_pos]
            train_in_test_year = train_dates[train_dates.year == test_year]
            assert len(train_in_test_year) == 0, (
                f"Año {test_year}: {len(train_in_test_year)} días del año test en train"
            )

    def test_train_is_expansive(self, dev_dates: pd.DatetimeIndex) -> None:
        """La ventana de train es expansiva: el train del año Y+1 es más largo que el de Y."""
        years = [2019, 2020, 2021, 2022]
        splits = walkforward_splits(dev_dates, years, gap_days=30)
        train_sizes = [len(sp.train_pos) for sp in splits]
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i] < train_sizes[i + 1], (
                f"Train no expansivo: año {years[i]} tiene {train_sizes[i]}, "
                f"año {years[i+1]} tiene {train_sizes[i+1]}"
            )

    def test_test_positions_in_range(self, dev_dates: pd.DatetimeIndex) -> None:
        """Posiciones de test están en rango [0, len(dates))."""
        splits = walkforward_splits(dev_dates, [2020], gap_days=30)
        for sp in splits:
            assert sp.test_pos.min() >= 0
            assert sp.test_pos.max() < len(dev_dates)

    def test_gap_respected(self, dev_dates: pd.DatetimeIndex) -> None:
        """Hay al menos gap_days días entre el último train y el primer test."""
        years = [2021]
        gap_days = 30
        splits = walkforward_splits(dev_dates, years, gap_days=gap_days)
        sp = splits[0]

        last_train_date = dev_dates[sp.train_pos[-1]]
        first_test_date = dev_dates[sp.test_pos[0]]

        gap_actual = (first_test_date - last_train_date).days
        assert gap_actual >= gap_days, (
            f"Gap actual={gap_actual} días, requerido>={gap_days}"
        )

    def test_year_not_in_dates_skipped(self) -> None:
        """Un año fuera del rango de dates simplemente se salta."""
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        splits = walkforward_splits(dates, [2019, 2020, 2021, 2025], gap_days=30)
        # 2019 y 2025 están fuera del rango → 2 splits (2020, 2021)
        assert len(splits) == 2
