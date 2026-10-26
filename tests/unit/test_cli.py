"""Tests de la CLI. Ver Plan 01 §4."""

from typer.testing import CliRunner

from btcx.cli import app

runner = CliRunner()


def test_info_exits_zero() -> None:
    """btcx info devuelve código 0."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output


def test_info_shows_version() -> None:
    """btcx info muestra la versión."""
    result = runner.invoke(app, ["info"])
    assert "btcx version" in result.output


def test_info_shows_hash() -> None:
    """btcx info muestra config_hash."""
    result = runner.invoke(app, ["info"])
    assert "config_hash" in result.output
