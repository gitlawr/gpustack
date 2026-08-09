#!/usr/bin/env node
// Parse an SGLang cookbook config `.jsx` (a single `export const config = {...}`
// static object literal) and emit it as JSON on stdout.
// Usage: node parse_sglang.js <path-to.jsx>
const fs = require('fs');

const path = process.argv[2];
if (!path) {
  console.error('usage: node parse_sglang.js <file.jsx>');
  process.exit(1);
}

let text = fs.readFileSync(path, 'utf8');

// Strip everything up to and including the real `export const config =`
// declaration (a leading comment mentions the phrase without `=`, so anchor on
// the assignment), drop the trailing `;`, then eval the object literal.
// Comments and template literals eval fine.
const decl = /export\s+const\s+config\s*=/.exec(text);
if (!decl) {
  console.error('no `export const config =` declaration found');
  process.exit(1);
}
let body = text.slice(decl.index + decl[0].length).trim().replace(/;\s*$/, '');

// eslint-disable-next-line no-eval
const config = (0, eval)('(' + body + ')');
process.stdout.write(JSON.stringify(config, null, 2));
