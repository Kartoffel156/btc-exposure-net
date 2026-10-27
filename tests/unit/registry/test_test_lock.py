"""Tests unitarios para test_lock.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from btcx.registry.test_lock import TestBurnedError as BurnedError
from btcx.registry.test_lock import assert_unlock_allowed


@pytest.fixture()
def lock_file(tmp_path: Path) -> Path:
    return tmp_path / "test_lock.json"


def _mock_git_clean(clean: bool = True, sha: str = "abc123def456") -> dict:
    """Helpers de mocking para git."""
    return {
        "btcx.registry.test_lock._git_is_clean": lambda *_: clean,
        "btcx.registry.test_lock._git_sha": lambda *_: sha,
    }


class TestAssertUnlockAllowed:
    def test_no_env_var_raises(self, lock_file: Path) -> None:
        """Sin UNLOCK_TEST=1, lanza RuntimeError."""
        env = os.environ.copy()
        env.pop("UNLOCK_TEST", None)
        with patch.dict(os.environ, {"UNLOCK_TEST": "0"}):
            with pytest.raises(RuntimeError, match="UNLOCK_TEST"):
                assert_unlock_allowed("test", "hash1", lock_file)

    def test_dirty_git_raises(self, lock_file: Path) -> None:
        """Con git sucio, lanza RuntimeError."""
        with (
            patch.dict(os.environ, {"UNLOCK_TEST": "1"}),
            patch("btcx.registry.test_lock._git_is_clean", return_value=False),
            patch("btcx.registry.test_lock._git_sha", return_value="sha123"),
        ):
            with pytest.raises(RuntimeError, match="limpio"):
                assert_unlock_allowed("test", "hash1", lock_file)

    def test_first_unlock_succeeds(self, lock_file: Path) -> None:
        """Primera apertura: permite y registra."""
        with (
            patch.dict(os.environ, {"UNLOCK_TEST": "1"}),
            patch("btcx.registry.test_lock._git_is_clean", return_value=True),
            patch("btcx.registry.test_lock._git_sha", return_value="sha123abc456"),
        ):
            # No debe lanzar excepción
            assert_unlock_allowed("key1", "config_hash_A", lock_file)
            # El lock file debe existir
            assert lock_file.exists()
            state = json.loads(lock_file.read_text())
            assert "key1" in state
            assert state["key1"]["config_hash"] == "config_hash_A"
            assert state["key1"]["burned"] is False

    def test_second_unlock_same_hash_succeeds(self, lock_file: Path) -> None:
        """Segunda apertura con mismo config_hash y git_sha: permite."""
        sha = "sha123abc456"
        with (
            patch.dict(os.environ, {"UNLOCK_TEST": "1"}),
            patch("btcx.registry.test_lock._git_is_clean", return_value=True),
            patch("btcx.registry.test_lock._git_sha", return_value=sha),
        ):
            assert_unlock_allowed("key1", "config_hash_A", lock_file)
            # Segunda apertura con mismo hash y sha
            assert_unlock_allowed("key1", "config_hash_A", lock_file)
            # No debe haberse quemado
            state = json.loads(lock_file.read_text())
            assert state["key1"]["burned"] is False

    def test_second_unlock_different_config_hash_burns(self, lock_file: Path) -> None:
        """Segunda apertura con diferente config_hash → burned=True + BurnedError."""
        sha = "sha123abc456"
        with (
            patch.dict(os.environ, {"UNLOCK_TEST": "1"}),
            patch("btcx.registry.test_lock._git_is_clean", return_value=True),
            patch("btcx.registry.test_lock._git_sha", return_value=sha),
        ):
            # Primera apertura
            assert_unlock_allowed("key1", "config_hash_A", lock_file)
            # Segunda apertura con diferente config_hash
            with pytest.raises(BurnedError):
                assert_unlock_allowed("key1", "config_hash_B", lock_file)

        # Verificar que burned=True en el lock file
        state = json.loads(lock_file.read_text())
        assert state["key1"]["burned"] is True

    def test_second_unlock_different_sha_burns(self, lock_file: Path) -> None:
        """Segunda apertura con diferente git_sha → burned=True + BurnedError."""
        with (
            patch.dict(os.environ, {"UNLOCK_TEST": "1"}),
            patch("btcx.registry.test_lock._git_is_clean", return_value=True),
        ):
            # Primera apertura con sha1
            with patch("btcx.registry.test_lock._git_sha", return_value="sha_original"):
                assert_unlock_allowed("key2", "config_hash_A", lock_file)

            # Segunda apertura con sha diferente
            with patch("btcx.registry.test_lock._git_sha", return_value="sha_diferente"):
                with pytest.raises(BurnedError):
                    assert_unlock_allowed("key2", "config_hash_A", lock_file)

        state = json.loads(lock_file.read_text())
        assert state["key2"]["burned"] is True

    def test_burned_key_always_raises(self, lock_file: Path) -> None:
        """Una vez burned=True, siempre lanza BurnedError."""
        sha = "sha123abc456"
        with (
            patch.dict(os.environ, {"UNLOCK_TEST": "1"}),
            patch("btcx.registry.test_lock._git_is_clean", return_value=True),
            patch("btcx.registry.test_lock._git_sha", return_value=sha),
        ):
            # Quemar
            assert_unlock_allowed("key3", "hash_A", lock_file)
            with pytest.raises(BurnedError):
                assert_unlock_allowed("key3", "hash_B", lock_file)

            # Incluso con el hash original, ya está quemado
            with pytest.raises(BurnedError):
                assert_unlock_allowed("key3", "hash_A", lock_file)

    def test_different_keys_independent(self, lock_file: Path) -> None:
        """Diferentes keys son independientes."""
        sha = "sha123"
        with (
            patch.dict(os.environ, {"UNLOCK_TEST": "1"}),
            patch("btcx.registry.test_lock._git_is_clean", return_value=True),
            patch("btcx.registry.test_lock._git_sha", return_value=sha),
        ):
            # Quemar key1
            assert_unlock_allowed("key1", "hash_A", lock_file)
            with pytest.raises(BurnedError):
                assert_unlock_allowed("key1", "hash_B", lock_file)

            # key2 debe funcionar normalmente
            assert_unlock_allowed("key2", "hash_C", lock_file)
            state = json.loads(lock_file.read_text())
            assert state["key2"]["burned"] is False
