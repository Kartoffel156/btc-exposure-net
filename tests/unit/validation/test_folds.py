"""Tests unitarios para folds.py."""

from __future__ import annotations

import numpy as np
import pytest

from btcx.validation.folds import Fold, to_absolute


class TestToAbsolute:
    def test_basic_shift(self) -> None:
        """to_absolute suma warmup a cada posición."""
        pos = np.array([0, 1, 2, 10, 50])
        result = to_absolute(pos, warmup=365)
        expected = np.array([365, 366, 367, 375, 415])
        np.testing.assert_array_equal(result, expected)

    def test_warmup_365(self) -> None:
        """Con warmup=365, el índice 0 del dev corresponde al índice 365 absoluto."""
        pos = np.arange(100)
        result = to_absolute(pos, warmup=365)
        assert result[0] == 365
        assert result[-1] == 464

    def test_warmup_zero(self) -> None:
        """Con warmup=0, las posiciones no cambian."""
        pos = np.array([5, 10, 100])
        result = to_absolute(pos, warmup=0)
        np.testing.assert_array_equal(result, pos)

    def test_empty_array(self) -> None:
        result = to_absolute(np.array([], dtype=np.int64), warmup=100)
        assert len(result) == 0

    def test_dtype_preserved(self) -> None:
        pos = np.array([0, 1, 2], dtype=np.int64)
        result = to_absolute(pos, warmup=100)
        assert result.dtype == np.int64 or np.issubdtype(result.dtype, np.integer)

    def test_large_warmup(self) -> None:
        """Warmup grande no causa overflow en int64."""
        pos = np.array([0, 1000], dtype=np.int64)
        result = to_absolute(pos, warmup=10000)
        assert result[0] == 10000
        assert result[1] == 11000


class TestFold:
    def test_fold_creation(self) -> None:
        """Fold se puede crear correctamente."""
        train = np.arange(100)
        test = np.arange(100, 150)
        fold = Fold(
            fold_id="BTC/cpcv-03",
            asset_ids=("BTC",),
            kind="cpcv",
            train_pos=train,
            test_pos=test,
            test_blocks=(0, 1),
        )
        assert fold.fold_id == "BTC/cpcv-03"
        assert fold.kind == "cpcv"
        assert len(fold.train_pos) == 100
        assert len(fold.test_pos) == 50

    def test_fold_is_frozen(self) -> None:
        """Fold es inmutable (frozen dataclass)."""
        train = np.arange(50)
        test = np.arange(50, 100)
        fold = Fold(
            fold_id="BTC/wf-2021",
            asset_ids=("BTC",),
            kind="walkforward",
            train_pos=train,
            test_pos=test,
        )
        with pytest.raises((AttributeError, TypeError)):
            fold.fold_id = "otro"  # type: ignore[misc]

    def test_fold_kinds(self) -> None:
        """Soporta todos los kinds definidos."""
        train = np.arange(50)
        test = np.arange(50, 100)
        for kind in ["cpcv", "walkforward", "lab", "final"]:
            fold = Fold(
                fold_id=f"test/{kind}",
                asset_ids=("BTC",),
                kind=kind,  # type: ignore[arg-type]
                train_pos=train,
                test_pos=test,
            )
            assert fold.kind == kind

    def test_fold_multiple_assets(self) -> None:
        """Fold puede tener múltiples assets."""
        train = np.arange(100)
        test = np.arange(100, 150)
        fold = Fold(
            fold_id="pooled/cpcv-00",
            asset_ids=("BTC", "ETH"),
            kind="cpcv",
            train_pos=train,
            test_pos=test,
        )
        assert fold.asset_ids == ("BTC", "ETH")
