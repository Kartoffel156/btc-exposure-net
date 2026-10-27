"""Tests para el comando btcx test open (Plan 14)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from btcx.cli import app

runner = CliRunner()


def _make_decision_go(output_dir: Path, config_hash: str = "abc123") -> Path:
    """Crea un decision.json con decisión GO."""
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "decision": "GO",
        "project": "BTC-v1",
        "asset": "BTC",
        "strategy": "P05",
        "config_hash": config_hash,
        "n_trials": 12,
        "git_sha": "deadbeef",
        "data_hash": "abc",
        "generated_at": "2026-10-26T00:00:00+00:00",
        "checks": {},
    }
    p = output_dir / "decision.json"
    p.write_text(json.dumps(decision), encoding="utf-8")
    return p


def test_falla_sin_decision_json(tmp_path):
    """btcx test open falla si no hay decision.json GO."""
    result = runner.invoke(app, [
        "test", "open",
        "--asset", "BTC",
        "--decision-dir", str(tmp_path / "empty"),
        "--models-dir", str(tmp_path / "models"),
        "--unlock-registry", str(tmp_path / "unlock.json"),
    ])
    assert result.exit_code != 0
    assert "decision.json" in result.output.lower() or "decision.json" in str(result.exception or "")


def test_falla_decision_no_go(tmp_path):
    """btcx test open falla si decision.json no es GO."""
    decision_dir = tmp_path / "decision"
    decision_dir.mkdir()
    decision = {"decision": "NO-GO", "config_hash": "abc"}
    (decision_dir / "decision.json").write_text(json.dumps(decision))

    result = runner.invoke(app, [
        "test", "open",
        "--asset", "BTC",
        "--decision-dir", str(decision_dir),
        "--models-dir", str(tmp_path / "models"),
        "--unlock-registry", str(tmp_path / "unlock.json"),
    ])
    assert result.exit_code != 0
    assert "no-go" in result.output.lower() or "go" in result.output.lower()


def test_falla_sin_unlock_test(tmp_path):
    """btcx test open falla si UNLOCK_TEST=1 no está activo."""
    decision_dir = tmp_path / "decision"
    _make_decision_go(decision_dir)

    models_dir = tmp_path / "models" / "BTC"
    models_dir.mkdir(parents=True)

    env = {k: v for k, v in os.environ.items() if k != "UNLOCK_TEST"}
    result = runner.invoke(app, [
        "test", "open",
        "--asset", "BTC",
        "--decision-dir", str(decision_dir),
        "--models-dir", str(tmp_path / "models"),
        "--unlock-registry", str(tmp_path / "unlock.json"),
    ], env=env)
    assert result.exit_code != 0
    assert "unlock" in result.output.lower()


def test_falla_con_arbol_git_sucio(tmp_path):
    """btcx test open falla con árbol git sucio (mock)."""
    decision_dir = tmp_path / "decision"
    _make_decision_go(decision_dir)

    models_dir = tmp_path / "models" / "BTC"
    models_dir.mkdir(parents=True)

    with patch("subprocess.run") as mock_run:
        # Simular UNLOCK_TEST=1 activo y git sucio
        def side_effect(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = "M modified_file.py\n"  # árbol sucio
            return Result()

        mock_run.side_effect = side_effect

        env = {**os.environ, "UNLOCK_TEST": "1"}
        result = runner.invoke(app, [
            "test", "open",
            "--asset", "BTC",
            "--decision-dir", str(decision_dir),
            "--models-dir", str(tmp_path / "models"),
            "--unlock-registry", str(tmp_path / "unlock.json"),
        ], env=env)
    assert result.exit_code != 0
    assert "limpio" in result.output.lower() or "git" in result.output.lower()


def test_falla_si_ya_esta_abierto(tmp_path):
    """btcx test open falla si el test ya está abierto."""
    decision_dir = tmp_path / "decision"
    _make_decision_go(decision_dir)

    models_dir = tmp_path / "models" / "BTC"
    models_dir.mkdir(parents=True)

    # Pre-crear unlock con el asset ya abierto
    unlock_data = {
        "BTC/track_a": {
            "opened_at": "2026-10-26T00:00:00+00:00",
            "git_sha": "abc",
            "config_hash": "abc",
            "asset": "BTC",
            "track": "a",
        }
    }
    unlock_path = tmp_path / "unlock.json"
    unlock_path.write_text(json.dumps(unlock_data))

    env = {**os.environ, "UNLOCK_TEST": "1"}

    with patch("subprocess.run") as mock_run:
        def side_effect(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = ""  # árbol limpio
            return Result()
        mock_run.side_effect = side_effect

        result = runner.invoke(app, [
            "test", "open",
            "--asset", "BTC",
            "--decision-dir", str(decision_dir),
            "--models-dir", str(tmp_path / "models"),
            "--unlock-registry", str(unlock_path),
        ], env=env)

    assert result.exit_code != 0
    assert "abierto" in result.output.lower() or "opened" in result.output.lower()


def test_falla_si_modelo_no_existe(tmp_path):
    """btcx test open falla si el modelo congelado no existe."""
    decision_dir = tmp_path / "decision"
    _make_decision_go(decision_dir)

    # No crear el directorio del modelo
    env = {**os.environ, "UNLOCK_TEST": "1"}
    result = runner.invoke(app, [
        "test", "open",
        "--asset", "BTC",
        "--decision-dir", str(decision_dir),
        "--models-dir", str(tmp_path / "models"),  # BTC/ no existe
        "--unlock-registry", str(tmp_path / "unlock.json"),
    ], env=env)
    assert result.exit_code != 0
    assert "model" in result.output.lower() or "frozen" in result.output.lower() or "congelado" in result.output.lower()
