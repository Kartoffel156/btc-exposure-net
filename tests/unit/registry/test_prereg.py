"""Tests unitarios para prereg.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from btcx.registry.prereg import (
    PreregStatus,
    PreregViolation,
    assert_preregistered,
    _git_tag_exists,
    _git_is_clean,
)


class TestGitTagExists:
    def test_nonexistent_tag(self, tmp_path: Path) -> None:
        """Tag que no existe retorna False."""
        result = _git_tag_exists("nonexistent-tag-xyz-123", tmp_path)
        assert result is False


class TestAssertPreregistered:
    def test_tag_not_exists_raises(self, tmp_path: Path) -> None:
        """Si el tag no existe, lanza PreregViolation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value")
        with pytest.raises(PreregViolation, match="no existe"):
            assert_preregistered(
                config_path=config_file,
                tag="nonexistent-tag-xyz",
                repo_root=tmp_path,
            )

    def test_config_not_exists_raises(self, tmp_path: Path) -> None:
        """Si el archivo de config no existe, lanza PreregViolation."""
        config_file = tmp_path / "nonexistent.yaml"

        # Mockear que el tag existe
        with patch("btcx.registry.prereg._git_tag_exists", return_value=True):
            with pytest.raises(PreregViolation, match="no encontrado"):
                assert_preregistered(
                    config_path=config_file,
                    tag="my-prereg-tag",
                    repo_root=tmp_path,
                )

    def test_config_modified_after_tag_raises(self, tmp_path: Path) -> None:
        """Si el archivo de config ha sido modificado tras el tag, lanza PreregViolation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: modified_value")

        tagged_content = b"key: original_value"

        with (
            patch("btcx.registry.prereg._git_tag_exists", return_value=True),
            patch("btcx.registry.prereg._git_file_at_tag", return_value=tagged_content),
        ):
            with pytest.raises(PreregViolation, match="modificado"):
                assert_preregistered(
                    config_path=config_file,
                    tag="my-prereg-tag",
                    repo_root=tmp_path,
                )

    def test_config_identical_to_tag_succeeds(self, tmp_path: Path) -> None:
        """Si el archivo es idéntico al del tag, retorna PreregStatus."""
        config_file = tmp_path / "config.yaml"
        content = b"key: original_value\n"
        config_file.write_bytes(content)

        with (
            patch("btcx.registry.prereg._git_tag_exists", return_value=True),
            patch("btcx.registry.prereg._git_file_at_tag", return_value=content),
        ):
            result = assert_preregistered(
                config_path=config_file,
                tag="my-prereg-tag",
                repo_root=tmp_path,
                budget=30,
                project="BTC",
            )
            assert isinstance(result, PreregStatus)
            assert result.tag == "my-prereg-tag"
            assert result.budget == 30
            assert result.project == "BTC"

    def test_prereg_status_fields(self, tmp_path: Path) -> None:
        """PreregStatus tiene los campos correctos."""
        config_file = tmp_path / "config.yaml"
        content = b"model: M1\n"
        config_file.write_bytes(content)

        with (
            patch("btcx.registry.prereg._git_tag_exists", return_value=True),
            patch("btcx.registry.prereg._git_file_at_tag", return_value=content),
        ):
            status = assert_preregistered(
                config_path=config_file,
                tag="v1.0-prereg",
                budget=15,
                project="ETH",
            )
            assert status.config_path == config_file
            assert status.attempts_used == 0
