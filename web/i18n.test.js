import assert from 'node:assert/strict';
import { dictionaries, LANGS, t, setLang, getLang, fmtNum } from './i18n.js';

const [base, ...rest] = LANGS;
const baseKeys = Object.keys(dictionaries[base]).sort();

// Key parity is the one failure mode that is silent in the browser: a missing
// string just renders its own key, in one language only.
for (const lang of rest) {
  const keys = Object.keys(dictionaries[lang]).sort();
  const missing = baseKeys.filter(key => !keys.includes(key));
  const extra = keys.filter(key => !baseKeys.includes(key));
  assert.deepEqual(missing, [], `${lang} is missing keys: ${missing.join(', ')}`);
  assert.deepEqual(extra, [], `${lang} has keys ${base} does not: ${extra.join(', ')}`);
}

// Placeholders must match across languages, or one language silently drops a value.
const placeholders = text => (text.match(/\{\w+\}/g) || []).sort();
for (const key of baseKeys) {
  for (const lang of rest) {
    assert.deepEqual(
      placeholders(dictionaries[lang][key]),
      placeholders(dictionaries[base][key]),
      `placeholder mismatch for "${key}"`,
    );
  }
}

setLang('zh');
assert.equal(t('ctrl.pause'), '暂停');
assert.equal(t('chart.steps', { left: 3, right: 4 }), '触地 3 / 4');
setLang('en');
assert.equal(t('ctrl.pause'), 'Pause');
assert.equal(t('chart.steps', { left: 3, right: 4 }), 'Contacts 3 / 4');

// Unfilled placeholders stay visible rather than becoming "undefined".
assert.equal(t('chart.steps', { left: 3 }), 'Contacts 3 / {right}');
// Unknown keys fall through to the key itself instead of throwing.
assert.equal(t('nope.not.a.key'), 'nope.not.a.key');
assert.equal(getLang(), 'en');
assert.equal(fmtNum(166700), '166,700');

console.log(`i18n: ${baseKeys.length} keys, zh/en parity, placeholders and fallback verified.`);
