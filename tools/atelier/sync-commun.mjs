#!/usr/bin/env node
// Injecte tools/atelier/commun.fragment.js entre les balises <atelier:commun> de chaque script de phase.
// Le moteur de workflows exige des scripts autonomes (pas d'import) : le bloc commun est donc recopié, jamais édité à la main.
// Usage : node tools/atelier/sync-commun.mjs [--check]
import { readFile, writeFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ICI = path.dirname(fileURLToPath(import.meta.url));
export const DOSSIER_WORKFLOWS = path.resolve(ICI, '../../claude-home/workflows');
export const FRAGMENT = path.join(ICI, 'commun.fragment.js');

const BLOC = /(\/\/ <atelier:commun>[^\n]*\n)[\s\S]*?(\/\/ <\/atelier:commun>)/;

const normaliser = (texte) => texte.replace(/^﻿/, '').replace(/\r\n/g, '\n');

/** Renvoie la source avec le bloc commun remplacé, ou null si le script n'a pas de balises. */
export function injecter(source, commun) {
  if (!BLOC.test(source)) return null;
  return source.replace(BLOC, (_tout, debut, fin) => `${debut}${commun.trimEnd()}\n${fin}`);
}

/** Renvoie le contenu actuel du bloc commun d'un script, ou null. */
export function extraire(source) {
  const trouve = /\/\/ <atelier:commun>[^\n]*\n([\s\S]*?)\/\/ <\/atelier:commun>/.exec(source);
  return trouve ? trouve[1] : null;
}

export async function listerScripts() {
  const noms = await readdir(DOSSIER_WORKFLOWS);
  return noms.filter((nom) => /^atelier-.*\.js$/.test(nom)).sort();
}

export async function synchroniser({ verifierSeulement = false } = {}) {
  const commun = normaliser(await readFile(FRAGMENT, 'utf8'));
  const resultats = [];
  for (const nom of await listerScripts()) {
    const chemin = path.join(DOSSIER_WORKFLOWS, nom);
    const source = normaliser(await readFile(chemin, 'utf8'));
    const attendu = injecter(source, commun);
    if (attendu === null) {
      resultats.push({ nom, statut: 'sans bloc commun' });
    } else if (attendu === source) {
      resultats.push({ nom, statut: 'à jour' });
    } else {
      if (!verifierSeulement) await writeFile(chemin, attendu, 'utf8');
      resultats.push({ nom, statut: verifierSeulement ? 'désynchronisé' : 'mis à jour' });
    }
  }
  return resultats;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const verifierSeulement = process.argv.includes('--check');
  synchroniser({ verifierSeulement }).then(
    (resultats) => {
      resultats.forEach(({ nom, statut }) => console.log(`${statut.padEnd(18)} ${nom}`));
      process.exit(resultats.some(({ statut }) => statut === 'désynchronisé') ? 1 : 0);
    },
    (erreur) => {
      console.error(erreur.stack || erreur);
      process.exit(1);
    },
  );
}
