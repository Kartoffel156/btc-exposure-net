import * as esbuild from 'esbuild';
import { readFileSync, writeFileSync } from 'fs';

await esbuild.build({
  entryPoints: ['src/main.js'],
  bundle: true,
  format: 'iife',
  minify: false,
  outfile: '/tmp/lab-bundle.js',
  external: [],
});

const js = readFileSync('/tmp/lab-bundle.js', 'utf8');
const css = readFileSync('src/ui/theme.css', 'utf8');
const template = readFileSync('template.html', 'utf8');
const html = template
  .replace('<!--INJECT_JS-->', `<script>${js}</script>`)
  .replace('<!--INJECT_CSS-->', `<style>${css}</style>`);
writeFileSync('index.html', html);
console.log('Built lab/index.html');
