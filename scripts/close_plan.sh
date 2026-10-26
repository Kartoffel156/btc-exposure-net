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
sleep 15
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
