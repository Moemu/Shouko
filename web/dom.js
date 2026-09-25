// DOM and display helpers shared by both pages and by the view renderers they
// have in common. main.js and preview.js each carried private copies; the shared
// evaluation renderer would have been a third, so they live here instead.
import { t, getLang } from './i18n.js';
import { finite } from './training-series.js';

export const $ = (id) => document.getElementById(id);
/** Writers compare before writing: these run at stream rate, and a blind write
 *  breaks text selection and focus. */
export const setText = (el, value) => { if (el && el.textContent !== value) el.textContent = value; };
export const setHTML = (el, value) => { if (el && el.innerHTML !== value) el.innerHTML = value; };
/** Sets the value AND re-points data-i18n, so a later applyStatic() renders it in the new language. */
export const setKey = (el, key, vars) => { if (!el) return; el.dataset.i18n = key; el.textContent = t(key, vars); };
/** Rail navigation: reveal one `.view` and mark its `.nav` button active. The page
 *  keeps its own record of which view is showing; this is only the DOM half. */
export const showView = (name) => {
  document.querySelectorAll('.view').forEach((el) => el.classList.toggle('active', el.id === `${name}View`));
  document.querySelectorAll('.nav').forEach((el) => el.classList.toggle('active', el.dataset.view === name));
};
export const escapeHTML = value => String(value ?? '--').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
export const number = (value, digits = 3) => finite(value) ? value.toLocaleString(getLang(), { maximumFractionDigits: digits }) : '--';
