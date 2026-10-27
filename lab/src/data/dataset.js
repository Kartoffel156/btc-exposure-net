/**
 * Acceso tipado a embedded.json.
 * Los datos tienen estructura: { [assetId]: { spec, dates, r, rf, sigma, features, scalars, meta } }
 */

export function getAssetIds(data) {
  return Object.keys(data);
}

export function getAsset(data, id) {
  return data[id] || null;
}

export function getSpec(data, id) {
  const asset = getAsset(data, id);
  return asset ? asset.spec : null;
}

export function getDates(data, id) {
  const asset = getAsset(data, id);
  return asset ? asset.dates : [];
}

export function getReturns(data, id) {
  const asset = getAsset(data, id);
  if (!asset || !asset.r) return new Float64Array(0);
  return new Float64Array(asset.r);
}

export function getRf(data, id) {
  const asset = getAsset(data, id);
  if (!asset || !asset.rf) return new Float64Array(0);
  return new Float64Array(asset.rf);
}

export function getSigma(data, id) {
  const asset = getAsset(data, id);
  if (!asset || !asset.sigma) return new Float64Array(0);
  return new Float64Array(asset.sigma);
}

export function getFeatures(data, id) {
  const asset = getAsset(data, id);
  return asset ? asset.features || {} : {};
}

export function getScalars(data, id) {
  const asset = getAsset(data, id);
  return asset ? asset.scalars || {} : {};
}

export function getMeta(data, id) {
  const asset = getAsset(data, id);
  return asset ? asset.meta || {} : {};
}

export function getFeatureMatrix(data, id) {
  const asset = getAsset(data, id);
  if (!asset || !asset.features) return null;
  const names = Object.keys(asset.features);
  if (names.length === 0) return null;
  const n = asset.features[names[0]].length;
  const F = names.length;
  const matrix = new Float64Array(n * F);
  for (let f = 0; f < F; f++) {
    const col = asset.features[names[f]];
    for (let t = 0; t < n; t++) {
      matrix[t * F + f] = col[t];
    }
  }
  return { names, matrix, n, F };
}
