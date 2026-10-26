# PLAN 01 · Andamiaje, tooling, configuración, CI y repo en GitHub

**Versión al cerrar:** v0.1.0 · **Rama:** `plan/01-andamiaje` · **Depende de:** — · **Tamaño:** S

## 1. Objetivo

Dejar un repo vacío pero completo: paquete instalable, configuración tipada, CLI, calidad automatizada, CI en GitHub y el script que cierra cada plan. Al terminar, `make check` pasa en local y en CI.

## 2. Entregables

```
pyproject.toml  uv.lock  Makefile  .pre-commit-config.yaml  .gitignore  .editorconfig
README.md  CHANGELOG.md
.github/workflows/ci.yml
config/base.yaml
config/assets/.gitkeep   config/rates/.gitkeep
docs/adr/ADR-000-plantilla.md
plans/  (todos los PLAN_*.md)
scripts/close_plan.sh
src/btcx/{__init__,constants,config,logging,cli}.py
src/btcx/<subpaquetes>/__init__.py   (vacíos, según PLAN_00 §3)
tests/unit/test_config.py  tests/unit/test_cli.py
registry/.gitkeep  reports/.gitkeep  artifacts/.gitkeep  data/raw/.gitkeep
```

## 3. Tareas

### 3.1 Crear el repo con `main` inicial

```bash
gh repo create btc-exposure-net --private --clone --description "Exposición dinámica en BTC con red residual"
cd btc-exposure-net
echo "# btc-exposure-net" > README.md
git add README.md && git commit -m "chore: commit inicial"
git push -u origin main
git checkout -b plan/01-andamiaje
```

Opcional pero recomendado: proteger `main` (Settings → Branches → require status checks `ci / python`).

### 3.2 `pyproject.toml`

```toml
[project]
name = "btcx"
version = "0.0.0"
description = "Exposición dinámica en BTC con red residual + laboratorio visual"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pandas>=2.2",
  "scipy>=1.12",
  "pyarrow>=15",
  "pydantic>=2.6",
  "pyyaml>=6",
  "typer>=0.12",
  "requests>=2.31",
  "tenacity>=8.2",
  "matplotlib>=3.8",
  "joblib>=1.3",
]

[project.scripts]
btcx = "btcx.cli:app"

[dependency-groups]
dev = [
  "ruff>=0.5", "mypy>=1.10", "pytest>=8", "pytest-cov>=5", "hypothesis>=6.100",
  "pre-commit>=3.7", "pandas-stubs", "types-requests", "types-PyYAML", "freezegun>=1.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/btcx"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "N", "SIM", "RUF", "PD", "NPY", "PL", "PT"]
ignore = ["PLR0913", "PLR2004", "PD901"]

[tool.mypy]
strict = true
python_version = "3.11"
plugins = ["pydantic.mypy"]
mypy_path = "src"

[[tool.mypy.overrides]]
module = ["scipy.*", "joblib.*", "matplotlib.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
  "unit: rápido, sin IO",
  "integration: usa datos canónicos del repo",
  "sanity: pruebas de cordura T1–T9",
  "slow: > 30 s; fuera del CI de PR, corre en nightly",
]
```

`torch` se añade en el Plan 07 (índice CPU).

### 3.3 Árbol de paquetes

Crear `src/btcx/` y los subpaquetes del PLAN_00 §3 con `__init__.py` vacío. `src/btcx/__init__.py`:

```python
"""btcx: exposición dinámica en BTC con red residual. Ver pre-plan v0.2."""
from importlib.metadata import version

__version__ = version("btcx")
```

### 3.4 `src/btcx/constants.py`

```python
"""Constantes que NO son parámetros de estrategia (esos van en config/base.yaml)."""
from typing import Final

ANN_FACTOR: Final[int] = 365          # pre-plan §4.7
DATE_FMT: Final[str] = "%Y-%m-%d"
UNLOCK_ENV: Final[str] = "UNLOCK_TEST"  # pre-plan §11.10
```

### 3.5 `config/base.yaml` (fuente única de parámetros)

Todos los números del pre-plan viven aquí. Los planes siguientes añaden secciones, nunca valores sueltos en el código.

```yaml
project:
  seed: 20260101
  ann_factor: 365

periods:                       # pre-plan §11.3 · DEFAULTS: cada activo puede declarar los suyos
  history_start: "2014-01-01"
  dev_start: "2015-01-01"
  dev_end: "2022-12-31"
  test_start: "2023-01-01"
  lab_train_end: "2020-12-31"  # división solo para intuición en el laboratorio

data:                          # multi-activo: los activos viven en config/assets/*.yaml (Plan 02)
  assets_dir: config/assets
  rates_dir: config/rates
  universe: [BTC]
  binance_base_urls: ["https://api.binance.com", "https://data-api.binance.vision"]

asset_classes: {}              # Plan 02 (defaults por clase: calendario, costos, margen)

exposure:
  w_min: 0.20                        # D1
  w_max: 1.00
  buffer: 0.10
  drift: false                       # H4

costs:                               # pre-plan §4.5 · trade_bps vive en cada clase/activo (Plan 02)
  trade_bps_multipliers: [0.3333, 1.0, 2.0]   # sensibilidad T6: en BTC (15 pb) = 5 / 15 / 30 pb
  fin_spread_annual: 0.05
  fin_spread_sensitivity: [0.02, 0.05, 0.10]
  cash_yield: true

benchmarks:                          # pre-plan §6 (FIJOS, no se optimizan)
  vol_halflife: 20
  vol_min_periods: 60
  sigma_target: train_median         # D2
  trend_sma: 50
  ewmac_pairs: [[16, 64], [32, 128]] # D7
  ewmac_weights: [0.5, 0.5]
  fdm: 1.1
  forecast_abs_target: 10.0
  forecast_cap: 20.0
  random_paths: 1000

scaling: {}        # Plan 05
features: {}       # Plan 06
model: {}          # Plan 07
training: {}       # Plan 07
evolution: {}      # Plan 08
validation: {}     # Plan 10
stats: {}          # Plan 11
```

### 3.6 `src/btcx/config.py`

Modelos `pydantic` congelados con `extra="forbid"` (un typo en el YAML rompe en carga, no en silencio). Las secciones vacías se modelan como `dict[str, Any]` hasta que su plan las tipa.

```python
"""Carga y validación de configuración. Las configs de experimento se fusionan sobre base.yaml."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectCfg(_Frozen):
    seed: int
    ann_factor: int


class PeriodsCfg(_Frozen):
    """Valores por defecto; AssetSpec (Plan 02) puede sobrescribirlos por activo."""
    history_start: str
    dev_start: str
    dev_end: str
    test_start: str
    lab_train_end: str


class DataCfg(_Frozen):
    assets_dir: Path
    rates_dir: Path
    universe: list[str]
    binance_base_urls: list[str]


class ExposureCfg(_Frozen):
    w_min: float
    w_max: float
    buffer: float
    drift: bool

    @model_validator(mode="after")
    def _check(self) -> ExposureCfg:
        if not 0 < self.w_min < self.w_max:
            raise ValueError("se requiere 0 < w_min < w_max")
        return self


class CostsCfg(_Frozen):
    trade_bps_multipliers: list[float]
    fin_spread_annual: float
    fin_spread_sensitivity: list[float]
    cash_yield: bool


class BenchmarksCfg(_Frozen):
    vol_halflife: int
    vol_min_periods: int
    sigma_target: Literal["train_median"] | float
    trend_sma: int
    ewmac_pairs: list[tuple[int, int]]
    ewmac_weights: list[float]
    fdm: float
    forecast_abs_target: float
    forecast_cap: float
    random_paths: int


class Settings(_Frozen):
    project: ProjectCfg
    periods: PeriodsCfg
    data: DataCfg
    exposure: ExposureCfg
    costs: CostsCfg
    benchmarks: BenchmarksCfg
    asset_classes: dict[str, Any] = {}   # tipado en el Plan 02
    scaling: dict[str, Any] = {}
    features: dict[str, Any] = {}
    model: dict[str, Any] = {}
    training: dict[str, Any] = {}
    evolution: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    stats: dict[str, Any] = {}

    def config_hash(self) -> str:
        """SHA256 del JSON canónico: identifica una configuración en el registro."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_settings(*overrides: Path, base: Path = Path("config/base.yaml")) -> Settings:
    """Carga base.yaml y fusiona encima, en orden, cada YAML de experimento."""
    raw: dict[str, Any] = yaml.safe_load(base.read_text(encoding="utf-8"))
    for p in overrides:
        raw = _deep_merge(raw, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    return Settings.model_validate(raw)
```

### 3.7 `src/btcx/logging.py` y `src/btcx/cli.py`

`logging.py`: función `setup_logging(level: str = "INFO")` con formato `%(asctime)s %(levelname)s %(name)s | %(message)s` en UTC.

`cli.py`: app `typer` con subcomandos vacíos que cada plan rellena: `asset`, `data`, `rates`, `bench`, `report`, `features`, `train`, `evolve`, `lab`, `validate`, `stats`, `decide`, `test`, `paper`. Comando `btcx info` que imprime versión, `config_hash`, universo de activos y si `UNLOCK_TEST` está activo.

**Convención multi-activo de la CLI (desde ya):** todo comando que trabaja sobre datos acepta `--asset <ID>` (repetible) y `--universe <archivo>`; por defecto usa `data.universe`. Ningún comando asume un activo concreto.

### 3.8 `.gitignore`, `.editorconfig`, pre-commit

`.gitignore`: `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `.coverage`, `data/raw/*` (con `!data/raw/.gitkeep`), `artifacts/*` (con `!artifacts/.gitkeep`), `lab/node_modules/`, `.ipynb_checkpoints/`.

`.pre-commit-config.yaml`: hooks `ruff` + `ruff-format` (astral-sh/ruff-pre-commit), `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `check-added-large-files --maxkb=2000` (protege contra subir volcados crudos).

### 3.9 `Makefile`

```make
.PHONY: setup fmt lint type test test-all check
setup:     ; uv sync && uv run pre-commit install
fmt:       ; uv run ruff format . && uv run ruff check --fix .
lint:      ; uv run ruff check . && uv run ruff format --check .
type:      ; uv run mypy src
test:      ; uv run pytest -m "not slow" --cov=btcx --cov-report=term-missing
test-all:  ; uv run pytest
check: lint type test
```

### 3.10 CI: `.github/workflows/ci.yml`

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request:
jobs:
  python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { enable-cache: true }
      - run: uv python install 3.11
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest -m "not slow" --cov=btcx
```

### 3.11 `scripts/close_plan.sh` (el cierre estándar de todos los planes)

```bash
#!/usr/bin/env bash
# Cierra un plan: verifica, versiona, sube, abre PR, espera CI, mergea, etiqueta y publica release.
# Uso: scripts/close_plan.sh <NN> <vX.Y.Z> "<Título>" [tag_adicional]
set -euo pipefail

NN="${1:?falta NN}"; VERSION="${2:?falta versión}"; TITLE="${3:?falta título}"; EXTRA_TAG="${4:-}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

[[ "$BRANCH" == plan/${NN}-* ]] || { echo "✗ La rama '$BRANCH' no es del plan $NN"; exit 1; }
[[ "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "✗ Versión inválida: $VERSION"; exit 1; }
git fetch --tags --quiet
if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then echo "✗ El tag $VERSION ya existe"; exit 1; fi
grep -q "\[$VERSION\]" CHANGELOG.md || { echo "✗ CHANGELOG.md no tiene entrada para $VERSION"; exit 1; }
command -v gh >/dev/null || { echo "✗ Falta GitHub CLI (gh)"; exit 1; }

echo "→ Verificación local"
make check

echo "→ Versión del paquete = ${VERSION#v}"
uv run python - "$VERSION" <<'PY'
import pathlib, re, sys
p = pathlib.Path("pyproject.toml"); v = sys.argv[1].lstrip("v")
p.write_text(re.sub(r'(?m)^version = ".*"$', f'version = "{v}"', p.read_text(), count=1))
PY
uv lock --quiet

echo "→ Commit y push"
git add -A
git diff --cached --quiet || git commit -m "chore(release): ${VERSION} · Plan ${NN} · ${TITLE}"
git push -u origin "$BRANCH"

echo "→ Pull request"
if ! gh pr view "$BRANCH" >/dev/null 2>&1; then
  gh pr create --base main --head "$BRANCH" \
    --title "Plan ${NN} · ${TITLE} (${VERSION})" \
    --body "Cierra el Plan ${NN}. Detalle en \`plans/\` y en CHANGELOG.md."
fi
sleep 15   # dar tiempo a que GitHub registre los checks
gh pr checks "$BRANCH" --watch --fail-fast

echo "→ Merge (squash)"
gh pr merge "$BRANCH" --squash --delete-branch
git checkout main
git pull --ff-only

echo "→ Tag y release"
git tag -a "$VERSION" -m "Plan ${NN} · ${TITLE}"
git push origin "$VERSION"
if [[ -n "$EXTRA_TAG" ]]; then
  git tag -a "$EXTRA_TAG" -m "Hito: ${EXTRA_TAG} (Plan ${NN})"
  git push origin "$EXTRA_TAG"
fi
gh release create "$VERSION" --title "${VERSION} · Plan ${NN} · ${TITLE}" --generate-notes

echo "✓ Plan ${NN} cerrado como ${VERSION}${EXTRA_TAG:+ (+ $EXTRA_TAG)}"
```

`chmod +x scripts/close_plan.sh`.

### 3.12 Documentación mínima

`README.md`: qué es (resumen §0 del pre-plan en 10 líneas), cómo instalar (`make setup`), cómo se trabaja (planes + `close_plan.sh`), advertencia de que el periodo de test está bloqueado.

`CHANGELOG.md` (formato Keep a Changelog):

```markdown
# Changelog
## [v0.1.0] · Plan 01 · Andamiaje
- Paquete btcx, configuración tipada, CLI, CI y script de cierre de planes.
```

`docs/adr/ADR-000-plantilla.md`: secciones Contexto · Opciones · Decisión · Consecuencias · Referencia al pre-plan.

## 4. Tests

- `test_config.py`: `base.yaml` carga; un campo desconocido lanza `ValidationError`; `w_min >= w_max` lanza error; el `config_hash` es estable entre dos cargas y cambia si cambia un valor; el merge de un override anidado solo toca la clave indicada.
- `test_cli.py`: `btcx info` devuelve código 0 (usar `typer.testing.CliRunner`).

## 5. Definición de terminado

- [ ] `make setup && make check` pasa en una máquina limpia.
- [ ] CI verde en el PR.
- [ ] `btcx info` muestra versión y hash.
- [ ] `plans/` copiado en el repo.
- [ ] `close_plan.sh` ejecutado con éxito (este propio plan es su primera prueba).

## 6. Cierre

```bash
./scripts/close_plan.sh 01 v0.1.0 "Andamiaje, tooling y CI"
```
