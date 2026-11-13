#!/usr/bin/env node
// Banc d'essai à blanc des workflows « Atelier ».
// Exécute un script de workflow Claude Code avec des agents simulés : aucun token dépensé.
// Usage : node tools/atelier/dryrun.mjs <script.js> [arguments JSON]
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;

// API refusées par le moteur de workflows (rejeu déterministe, pas d'accès système).
const APIS_INTERDITES = [
  /\bDate\.now\s*\(/,
  /\bMath\.random\s*\(/,
  /\bnew\s+Date\s*\(\s*\)/,
  /\bimport\s*\(/,
  /^\s*import\s/m,
  /\brequire\s*\(/,
];

const CHAINES = /'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"/g;

/** Fabrique une valeur factice conforme à un schéma JSON (sous-ensemble utilisé par les workflows). */
export function fabriquer(schema, nom = 'valeur', niveau = 0) {
  if (!schema || typeof schema !== 'object') return null;
  if (Array.isArray(schema.enum) && schema.enum.length) return schema.enum[0];
  if (schema.type === 'object') {
    const proprietes = schema.properties || {};
    return Object.fromEntries(
      Object.keys(proprietes).map((cle) => [cle, fabriquer(proprietes[cle], cle, niveau + 1)]),
    );
  }
  if (schema.type === 'array') {
    return niveau > 8 ? [] : [fabriquer(schema.items || { type: 'string' }, nom, niveau + 1)];
  }
  if (schema.type === 'number' || schema.type === 'integer') {
    return typeof schema.minimum === 'number' ? schema.minimum : 1;
  }
  if (schema.type === 'boolean') return true;
  return `<${nom}>`;
}

/** Contraintes du moteur : racine objet avec properties, required inclus dans properties. */
export function verifierSchema(schema, chemin = 'racine') {
  if (!schema || typeof schema !== 'object') return [`${chemin} : schéma absent`];
  const erreurs = [];
  if (chemin === 'racine' && (schema.type !== 'object' || !schema.properties)) {
    erreurs.push('racine : il faut { type: "object", properties: {...} }');
  }
  if (schema.type === 'object') {
    const proprietes = schema.properties || {};
    const inconnues = (schema.required || []).filter((cle) => !(cle in proprietes));
    if (inconnues.length) erreurs.push(`${chemin} : required hors properties (${inconnues.join(', ')})`);
    const enfants = Object.keys(proprietes).flatMap((cle) => verifierSchema(proprietes[cle], `${chemin}.${cle}`));
    return erreurs.concat(enfants);
  }
  if (schema.type === 'array') return erreurs.concat(verifierSchema(schema.items, `${chemin}[]`));
  return erreurs;
}

/** Valide une valeur contre un schéma (type, required, enum, items). */
export function valider(schema, valeur, chemin = 'racine') {
  if (!schema || typeof schema !== 'object') return [];
  if (Array.isArray(schema.enum) && !schema.enum.includes(valeur)) {
    return [`${chemin} : « ${valeur} » hors de l'énumération`];
  }
  if (schema.type === 'object') {
    if (!valeur || typeof valeur !== 'object' || Array.isArray(valeur)) return [`${chemin} : objet attendu`];
    const proprietes = schema.properties || {};
    const manquantes = (schema.required || []).filter((cle) => valeur[cle] === undefined);
    const erreurs = manquantes.map((cle) => `${chemin}.${cle} : champ requis manquant`);
    const enfants = Object.keys(proprietes)
      .filter((cle) => valeur[cle] !== undefined)
      .flatMap((cle) => valider(proprietes[cle], valeur[cle], `${chemin}.${cle}`));
    return erreurs.concat(enfants);
  }
  if (schema.type === 'array') {
    if (!Array.isArray(valeur)) return [`${chemin} : tableau attendu`];
    return valeur.flatMap((element, index) => valider(schema.items, element, `${chemin}[${index}]`));
  }
  if (schema.type === 'string' && typeof valeur !== 'string') return [`${chemin} : chaîne attendue`];
  if ((schema.type === 'number' || schema.type === 'integer') && typeof valeur !== 'number') {
    return [`${chemin} : nombre attendu`];
  }
  if (schema.type === 'boolean' && typeof valeur !== 'boolean') return [`${chemin} : booléen attendu`];
  return [];
}

/** Sépare le littéral `meta` du corps du script et contrôle que c'est un littéral pur. */
export function decouper(source) {
  const propre = source.replace(/^﻿/, '');
  const trouve = /^export const meta = (\{[\s\S]*?\n\});?[ \t]*\r?\n/.exec(propre);
  if (!trouve) throw new Error('`export const meta = {...}` doit être la toute première instruction du script');
  const litteral = trouve[1];
  const sansChaines = litteral.replace(CHAINES, '""');
  if (/[`()+]|\.\.\./.test(sansChaines)) {
    throw new Error('meta doit être un littéral pur (ni appel, ni gabarit, ni concaténation, ni spread)');
  }
  const valeurNue = /:\s*(?!true\b|false\b|null\b)[A-Za-z_$][\w$]*\s*[,}\]\n]/.exec(sansChaines);
  if (valeurNue) throw new Error(`meta doit être un littéral pur (variable détectée : ${valeurNue[0].trim()})`);
  const meta = new Function(`"use strict"; return (${litteral});`)();
  return { meta, corps: propre.slice(trouve[0].length) };
}

function nouvelleTrace() {
  return { appels: [], journal: [], phases: [], avalees: [] };
}

function simulerAgent(etatLocal, trace, options) {
  return async (prompt, opts = {}) => {
    if (typeof prompt !== 'string' || !prompt.trim()) throw new Error('agent() : prompt vide');
    if (opts.schema) {
      const defauts = verifierSchema(opts.schema);
      if (defauts.length) throw new Error(`schéma refusé par le moteur : ${defauts.join(' ; ')}`);
    }
    const appel = {
      n: trace.appels.length + 1,
      workflow: etatLocal.nom,
      label: opts.label || '',
      phase: opts.phase || etatLocal.phase,
      model: opts.model || null,
      agentType: opts.agentType || null,
      schema: Boolean(opts.schema),
      prompt,
    };
    if (opts.agentType && (options.indisponibles || []).includes(opts.agentType)) {
      trace.appels.push({ ...appel, echec: true });
      throw new Error(`Unknown agent type: ${opts.agentType}`);
    }
    trace.appels.push(appel);
    const imposee = options.scenario ? await options.scenario(appel, opts) : undefined;
    const reponse = imposee !== undefined ? imposee : opts.schema ? fabriquer(opts.schema) : `réponse simulée (${appel.label})`;
    if (reponse !== null && opts.schema) {
      const defauts = valider(opts.schema, reponse);
      if (defauts.length) throw new Error(`réponse simulée non conforme pour « ${appel.label} » : ${defauts.join(' ; ')}`);
    }
    return reponse;
  };
}

function simulerConcurrence(trace) {
  const proteger = async (tache) => {
    try {
      return await tache();
    } catch (erreur) {
      // Le vrai moteur avale l'erreur et renvoie null : on la garde pour que les tests la voient.
      trace.avalees.push(erreur);
      return null;
    }
  };
  const parallel = (taches) => Promise.all(taches.map((tache) => proteger(tache)));
  const pipeline = (elements, ...etapes) =>
    Promise.all(
      elements.map((element, index) =>
        proteger(async () => {
          let courant = element;
          for (const etape of etapes) courant = await etape(courant, element, index);
          return courant;
        }),
      ),
    );
  return { parallel, pipeline };
}

/**
 * Exécute un script de workflow avec des agents simulés.
 * options : args, scenario(appel, opts) → réponse imposée ou undefined, indisponibles (agentTypes inconnus).
 */
export async function executer(fichier, options = {}) {
  const absolu = path.resolve(fichier);
  const source = await readFile(absolu, 'utf8');
  const { meta, corps } = decouper(source);
  const interdites = APIS_INTERDITES.filter((motif) => motif.test(corps)).map(String);
  if (interdites.length) {
    throw new Error(`${path.basename(absolu)} : API interdite dans un workflow (${interdites.join(', ')})`);
  }
  const trace = options.trace || nouvelleTrace();
  const etatLocal = { nom: meta.name, phase: null };
  const agent = simulerAgent(etatLocal, trace, options);
  const { parallel, pipeline } = simulerConcurrence(trace);
  const phase = (titre) => {
    etatLocal.phase = titre;
    trace.phases.push({ workflow: meta.name, titre });
  };
  const log = (message) => {
    trace.journal.push(String(message));
  };
  const budget = { total: null, spent: () => 0, remaining: () => Infinity };
  const workflow = async (reference, sousArgs) => {
    if (options.imbrique) throw new Error('workflow() : un seul niveau d’imbrication autorisé');
    const cible = typeof reference === 'string' ? path.join(path.dirname(absolu), `${reference}.js`) : reference.scriptPath;
    const enfant = await executer(cible, { ...options, args: sousArgs, imbrique: true, trace });
    return enfant.resultat;
  };
  const fonction = new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'args', 'budget', 'workflow', corps);
  const resultat = await fonction(agent, parallel, pipeline, phase, log, options.args, budget, workflow);
  return { meta, resultat, ...trace };
}

async function principal(argv) {
  const [fichier, brut] = argv;
  if (!fichier) {
    console.error('Usage : node tools/atelier/dryrun.mjs <script.js> [arguments JSON]');
    return 2;
  }
  const args = brut ? JSON.parse(brut) : undefined;
  const { meta, resultat, appels, journal, avalees } = await executer(fichier, { args });
  console.log(`# ${meta.name} — ${appels.length} agent(s) simulé(s)`);
  appels.forEach((a) => {
    const qui = a.agentType || a.model || 'modèle de session';
    console.log(`${String(a.n).padStart(3)}  [${a.phase || '-'}] ${a.label || '(sans label)'}  →  ${qui}${a.echec ? '  (indisponible)' : ''}`);
  });
  journal.forEach((ligne) => console.log(`  log : ${ligne}`));
  avalees.forEach((erreur) => console.log(`  ERREUR AVALÉE : ${erreur.stack || erreur}`));
  console.log(JSON.stringify(resultat, null, 2).slice(0, 4000));
  return avalees.length ? 1 : 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  principal(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (erreur) => {
      console.error(erreur.stack || erreur);
      process.exit(1);
    },
  );
}
