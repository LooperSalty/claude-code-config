// Tests à blanc des workflows « Atelier » : aucun agent réel, aucun token.
// Lancement : node --test tools/atelier/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { executer, decouper } from './dryrun.mjs';
import { DOSSIER_WORKFLOWS, listerScripts, synchroniser } from './sync-commun.mjs';

const chemin = (nom) => path.join(DOSSIER_WORKFLOWS, `${nom}.js`);
const PHASES = ['atelier-cadrage', 'atelier-architecture', 'atelier-plan', 'atelier-build', 'atelier-revue', 'atelier-recette'];

const ETAT = {
  version: 1,
  nom: 'Démo',
  type: 'webapp',
  traits: ['3d', 'donnees'],
  resume: 'projet de test',
  exigences: [
    { id: 'REQ-001', titre: 'A', priorite: 'P0' },
    { id: 'REQ-002', titre: 'B', priorite: 'P1' },
    { id: 'REQ-003', titre: 'C', priorite: 'P2' },
  ],
  skills: ['s1'],
  lentilles: ['correction', 'securite', 'conformite'],
  recette: ['criteres', 'e2e'],
  consignes: [],
  env: 'win32',
  stack: ['Langage : TypeScript 5', 'UI : React 19'],
  zones: ['apps/api', 'apps/web'],
  commandes: { install: 'npm ci', test: 'npm test', build: 'npm run build' },
  phasesFaites: ['cadrage', 'architecture', 'plan'],
  vaguesFaites: [],
};

const tache = (id, extra = {}) => ({
  id,
  titre: `Tâche ${id}`,
  objectif: 'objectif',
  fichiers: [`src/${id}/`],
  dependances: [],
  exigences: ['REQ-001'],
  tests: ['un test'],
  verification: 'npm test',
  complexite: 'routine',
  ...extra,
});

const PLAN = {
  vagues: [
    { id: 0, objectif: 'socle', taches: [tache('T0.1', { echafaudage: true, exigences: ['REQ-001', 'REQ-002'], fichiers: ['package.json', 'src/shared/'] })] },
    {
      id: 1,
      objectif: 'tranche',
      taches: [tache('T1.1', { dependances: ['T0.1'] }), tache('T1.2', { dependances: ['T0.1'], complexite: 'complexe', exigences: ['REQ-002'] })],
    },
  ],
  couverture: [{ exigence: 'REQ-001', taches: ['T0.1'] }],
};

const CONTROLE_OK = { ok: true, etapes: [], echecs: [] };
const CONTROLE_KO = { ok: false, etapes: [], echecs: [{ etape: 'tests', resume: 'rouge', fichiers: [] }] };
const SANS_PROBLEME = { problemes: [] };
const RECETTE_OK = { ok: true, resultats: [{ element: 'CA-01', statut: 'valide', preuve: 'test' }], echecs: [] };
const RAPPORT_FAIT = { statut: 'fait', resume: 'ok', fichiers: ['src/a.ts'], tests: ['a.test.ts'], verification: 'vert', horsPerimetre: [], dependancesManquantes: [] };

/** Construit un scénario : première règle dont le motif touche le label ; la réponse peut dépendre du nombre d'appels. */
function scenario(regles) {
  const compteurs = new Map();
  return (appel, opts) => {
    for (const [motif, reponse] of regles) {
      const touche = typeof motif === 'string' ? appel.label.startsWith(motif) : motif.test(appel.label);
      if (!touche) continue;
      const rang = (compteurs.get(motif) || 0) + 1;
      compteurs.set(motif, rang);
      return typeof reponse === 'function' ? reponse(rang, appel, opts) : reponse;
    }
    return undefined;
  };
}

const labels = (trace, prefixe) => trace.appels.filter((appel) => appel.label.startsWith(prefixe));
const sansErreurAvalee = (trace) => assert.deepEqual(trace.avalees.map(String), []);

test('chaque script commence par un meta littéral dont le nom est celui du fichier', async () => {
  for (const nom of await listerScripts()) {
    const { meta, corps } = decouper(await readFile(path.join(DOSSIER_WORKFLOWS, nom), 'utf8'));
    assert.equal(`${meta.name}.js`, nom);
    assert.ok(meta.description && meta.description.length > 20, `${nom} : description`);
    assert.ok(Array.isArray(meta.phases) && meta.phases.length > 0, `${nom} : phases`);
    assert.ok(corps.split('\n').length < 800, `${nom} : moins de 800 lignes`);
  }
});

test('le bloc commun est synchronisé dans tous les scripts de phase', async () => {
  const resultats = await synchroniser({ verifierSeulement: true });
  const parNom = Object.fromEntries(resultats.map(({ nom, statut }) => [nom, statut]));
  for (const phase of PHASES) assert.equal(parNom[`${phase}.js`], 'à jour', `${phase} : lancer node tools/atelier/sync-commun.mjs`);
  assert.equal(parNom['atelier-auto.js'], 'sans bloc commun');
});

test('chaque phase tourne à blanc sans erreur avalée et n’utilise que des phases déclarées', async () => {
  const args = {
    'atelier-cadrage': 'un brief en texte libre',
    'atelier-build': { etat: ETAT, plan: PLAN },
  };
  for (const nom of PHASES) {
    const trace = await executer(chemin(nom), { args: args[nom] || { etat: ETAT } });
    sansErreurAvalee(trace);
    const declarees = trace.meta.phases.map((phase) => phase.title);
    for (const appel of trace.appels) assert.ok(declarees.includes(appel.phase), `${nom} : phase « ${appel.phase} » non déclarée (${appel.label})`);
    assert.ok(trace.appels.length > 0 && trace.appels.length < 60, `${nom} : ${trace.appels.length} agents`);
  }
});

test('les clés des profils existent dans la revue et la recette ; le schéma de tâche est partagé', async () => {
  const tables = async (nom) => (await executer(chemin(nom), { args: { __tables: true } })).resultat;
  const { PROFILS, TRAITS, SOCLE } = await tables('atelier-cadrage');
  const { LENTILLES } = await tables('atelier-revue');
  const { RECETTES } = await tables('atelier-recette');
  for (const [nom, profil] of Object.entries({ ...PROFILS, ...TRAITS })) {
    for (const lentille of profil.lentilles) assert.ok(LENTILLES.includes(lentille), `${nom} : lentille inconnue « ${lentille} »`);
    for (const recette of profil.recette) assert.ok(RECETTES.includes(recette), `${nom} : recette inconnue « ${recette} »`);
  }
  assert.ok(SOCLE.length > 0);
  assert.deepEqual((await tables('atelier-plan')).TACHE, (await tables('atelier-build')).TACHE);
});

test('repli sur un agent générique quand les types d’agents des plugins manquent', async () => {
  const indisponibles = ['eco-router:scout', 'ecc:architect', 'ecc:security-reviewer', 'ecc:planner', 'ecc:code-reviewer', 'ecc:build-error-resolver'];
  const trace = await executer(chemin('atelier-architecture'), { args: { etat: ETAT }, indisponibles });
  sansErreurAvalee(trace);
  assert.ok(trace.journal.some((ligne) => ligne.includes('repli sur un agent générique')));
  assert.ok(trace.resultat.etat.phasesFaites.includes('architecture'));
});

test('cadrage : profil fusionné avec les traits, pas de révision si la critique est satisfaite', async () => {
  const trace = await executer(chemin('atelier-cadrage'), {
    args: { brief: 'cahier.docx', profondeur: 'max' },
    scenario: scenario([
      ['critique:', { complet: true, manques: [], inventions: [], criteresVagues: [] }],
      ['rédaction SPEC', { nom: 'CRM', resume: 'r', type: 'webapp', traits: ['3d', 'auth'], exigences: ETAT.exigences, hypotheses: ['h'], questionsOuvertes: [] }],
    ]),
  });
  sansErreurAvalee(trace);
  const { etat, revisee } = trace.resultat;
  assert.equal(revisee, false);
  assert.equal(labels(trace, 'révision SPEC').length, 0);
  assert.equal(labels(trace, 'analyste:').length, 4);
  assert.equal(labels(trace, 'critique:').length, 2);
  assert.ok(etat.skills.includes('threejs-performance') && etat.skills.includes('superpowers:test-driven-development'));
  assert.ok(etat.lentilles.includes('performance-3d') && etat.recette.includes('matrice-droits'));
  assert.deepEqual(etat.phasesFaites, ['cadrage']);
});

test('cadrage : une critique insatisfaite déclenche une révision ; le type peut être imposé', async () => {
  const trace = await executer(chemin('atelier-cadrage'), {
    args: { brief: 'un jeu', type: 'game', profondeur: 'eco' },
    scenario: scenario([['critique:', { complet: false, manques: [{ element: 'x', gravite: 'majeur', correction: 'y' }], inventions: [], criteresVagues: [] }]]),
  });
  assert.equal(labels(trace, 'révision SPEC').length, 1);
  assert.equal(labels(trace, 'analyste:').length, 1);
  assert.equal(trace.resultat.etat.type, 'game');
  assert.ok(trace.resultat.etat.consignes.some((consigne) => consigne.includes('Jeu original')));
});

test('architecture : le jury retient la meilleure moyenne, un veto rétrograde', async () => {
  const noter = (notes) => ({ notes: Object.entries(notes).map(([proposition, [note, redhibitoire]]) => ({ proposition, note, forces: [], faiblesses: [], redhibitoire: redhibitoire || '' })), aGreffer: [] });
  const jouer = (notes) =>
    executer(chemin('atelier-architecture'), {
      args: { etat: ETAT },
      scenario: scenario([
        ['juré:', noter(notes)],
        ['vérification adverse', { ok: true, problemes: [] }],
      ]),
    });
  const sansVeto = await jouer({ mvp: [5], robustesse: [9], experience: [7] });
  assert.equal(sansVeto.resultat.classement[0].cle, 'robustesse');
  assert.equal(labels(sansVeto, 'révision ARCHITECTURE').length, 0);
  const avecVeto = await jouer({ mvp: [5], robustesse: [9, 'incompatible'], experience: [7] });
  assert.equal(avecVeto.resultat.classement[0].cle, 'experience');
  assert.deepEqual(avecVeto.resultat.etat.zones, ['<chemin>']);
});

test('plan : la validation mécanique détecte chevauchements, dépendances et couverture', async () => {
  const valider = async (plan) => (await executer(chemin('atelier-plan'), { args: { __valider: plan, __exigences: ETAT.exigences } })).resultat.erreurs;
  assert.deepEqual(await valider(PLAN), []);
  const fautif = {
    vagues: [
      { id: 0, objectif: 's', taches: [tache('A', { fichiers: ['src/'] }), tache('B', { fichiers: ['src/x.ts'], dependances: ['A', 'Z'] }), tache('B', { fichiers: ['lib/*.ts'], tests: [] })] },
      { id: 1, objectif: 'vide', taches: [] },
    ],
    couverture: [],
  };
  const erreurs = (await valider(fautif)).join('\n');
  for (const attendu of ['en double : B', 'se disputent src/', 'dépendance inconnue Z', 'pas dans une vague antérieure', 'motif générique', 'aucun test', 'vague 1 sans tâche', 'REQ-002 (P1)']) {
    assert.ok(erreurs.includes(attendu), `erreur attendue : ${attendu}\n${erreurs}`);
  }
  assert.ok(!erreurs.includes('REQ-003'), 'les P2 ne sont pas exigées');
});

test('plan : un plan invalide est révisé jusqu’à passer la validation', async () => {
  const invalide = { vagues: [{ id: 0, objectif: 's', taches: [tache('A', { fichiers: ['src/'] }), tache('B', { fichiers: ['src/b.ts'] })] }], couverture: [] };
  const trace = await executer(chemin('atelier-plan'), {
    args: { etat: ETAT },
    scenario: scenario([
      ['découpage', invalide],
      ['critique:', SANS_PROBLEME],
      ['révision du plan', PLAN],
    ]),
  });
  sansErreurAvalee(trace);
  assert.equal(labels(trace, 'révision du plan').length, 1);
  assert.deepEqual(trace.resultat.erreurs, []);
  assert.equal(trace.resultat.plan.vagues.length, 2);
  assert.ok(trace.resultat.etat.phasesFaites.includes('plan'));
});

test('plan : horsPerimetre retire des exigences de la couverture exigée, contraintes transmises', async () => {
  const partiel = { vagues: [{ id: 0, objectif: 's', taches: [tache('A', { echafaudage: true })] }], couverture: [] };
  const jouer = (args) =>
    executer(chemin('atelier-plan'), {
      args: { etat: ETAT, profondeur: 'eco', ...args },
      scenario: scenario([
        ['découpage', partiel],
        ['critique:', SANS_PROBLEME],
        ['révision du plan', partiel],
      ]),
    });
  assert.ok((await jouer({})).resultat.erreurs.some((erreur) => erreur.includes('REQ-002')));
  const restreint = await jouer({ horsPerimetre: ['REQ-002'], contraintes: 'quatre vagues maximum' });
  assert.deepEqual(restreint.resultat.erreurs, []);
  assert.ok(labels(restreint, 'découpage')[0].prompt.includes('quatre vagues maximum'));
  assert.ok(labels(restreint, 'découpage')[0].prompt.includes('Hors périmètre de cette livraison'));
});

test('build : livre la prochaine vague, un contrôle rouge déclenche une réparation', async () => {
  const trace = await executer(chemin('atelier-build'), {
    args: { etat: { ...ETAT, vaguesFaites: [0] }, plan: PLAN },
    scenario: scenario([
      ['contrôle', (rang) => (rang === 1 ? CONTROLE_KO : CONTROLE_OK)],
      ['relecture', SANS_PROBLEME],
      [/^T1\./, RAPPORT_FAIT],
    ]),
  });
  sansErreurAvalee(trace);
  assert.equal(trace.resultat.ok, true);
  assert.deepEqual(trace.resultat.etat.vaguesFaites, [0, 1]);
  assert.deepEqual(trace.resultat.restantes, []);
  assert.ok(trace.resultat.etat.phasesFaites.includes('build'));
  assert.equal(labels(trace, 'réparation').length, 1);
  assert.equal(labels(trace, 'clôture').length, 1);
  const cloture = labels(trace, 'clôture')[0].prompt;
  assert.ok(cloture.includes('"vaguesFaites"') && !cloture.includes('"lentilles"'), 'la clôture ne renvoie que les champs modifiés');
  const parTache = Object.fromEntries(trace.appels.filter((appel) => /^T1\./.test(appel.label)).map((appel) => [appel.label.slice(0, 4), appel.model]));
  assert.deepEqual(parTache, { 'T1.1': 'sonnet', 'T1.2': null });
});

test('build : intégration durablement rouge → arrêt sans commit', async () => {
  const trace = await executer(chemin('atelier-build'), {
    args: { etat: ETAT, plan: PLAN, vagues: 'toutes' },
    scenario: scenario([['contrôle', CONTROLE_KO]]),
  });
  assert.equal(trace.resultat.ok, false);
  assert.equal(labels(trace, 'clôture').length, 0);
  assert.deepEqual(trace.resultat.restantes, [0, 1]);
  assert.equal(trace.resultat.vagues.length, 1, 'la vague 1 n’est pas tentée');
});

test('build : une tâche bloquée obtient une seconde chance après intégration ; eco saute la relecture', async () => {
  const bloque = { ...RAPPORT_FAIT, statut: 'bloque', dependancesManquantes: ['zod'] };
  const trace = await executer(chemin('atelier-build'), {
    args: { etat: { ...ETAT, vaguesFaites: [0] }, plan: PLAN, profondeur: 'eco' },
    scenario: scenario([
      ['T1.1', (rang) => (rang === 1 ? bloque : RAPPORT_FAIT)],
      ['T1.2', RAPPORT_FAIT],
      ['contrôle', CONTROLE_OK],
    ]),
  });
  sansErreurAvalee(trace);
  assert.equal(labels(trace, 'T1.1').length, 2);
  assert.equal(labels(trace, 'intégration vague 1').length, 1);
  assert.equal(labels(trace, 'relecture').length, 0);
  assert.equal(trace.resultat.vagues[0].rapports.find((rapport) => rapport.id === 'T1.1').statut, 'fait');
});

test('build : sans plan en argument, la vague est relue depuis plan.json', async () => {
  const trace = await executer(chemin('atelier-build'), {
    args: { etat: ETAT },
    scenario: scenario([
      ['chargement du plan', { trouve: true, vagues: [PLAN.vagues[0]], ids: [0, 1] }],
      ['contrôle', CONTROLE_OK],
    ]),
  });
  assert.deepEqual(trace.resultat.etat.vaguesFaites, [0]);
  assert.deepEqual(trace.resultat.restantes, [1]);
  assert.ok(!trace.resultat.etat.phasesFaites.includes('build'));
});

test('revue : fusion des doublons, réfutés non corrigés, faibles gravités non vérifiées', async () => {
  const constat = (titre, gravite, ligne, fichier = 'apps/api/src/stock.ts') => ({ titre, gravite, fichier, ligne, preuve: 'p', scenario: 's' });
  const trace = await executer(chemin('atelier-revue'), {
    args: { etat: ETAT },
    scenario: scenario([
      ['constats:correction', { constats: [constat('course', 'haute', 10), constat('style', 'basse', 200)] }],
      ['constats:securite', { constats: [constat('course bis', 'critique', 12), constat('fuite', 'haute', 90, 'apps/web/src/a.tsx')] }],
      [/^constats:/, { constats: [] }],
      ['contre-expertise', (_rang, appel) => ({
        verdicts: [...appel.prompt.matchAll(/"id":"(C1-\d+)","titre":"([^"]+)"/g)].map(([, id, titre]) => ({ id, reel: titre !== 'fuite', raison: 'vu' })),
      })],
      ['correction:', (_rang, appel) => ({ corriges: [...appel.prompt.matchAll(/"id":"(C1-\d+)"/g)].map(([, id]) => id), reports: [], fichiers: ['apps/api/src/stock.ts'], tests: [] })],
      ['contrôle', CONTROLE_OK],
    ]),
  });
  sansErreurAvalee(trace);
  const { confirmes, refutes, corriges, restants, nonVerifies, ok } = trace.resultat;
  assert.equal(confirmes.length, 1, 'les deux constats voisins sont fusionnés');
  assert.equal(confirmes[0].gravite, 'critique');
  assert.deepEqual(confirmes[0].lentilles.sort(), ['correction', 'securite']);
  assert.deepEqual(refutes.map((c) => c.titre), ['fuite']);
  assert.deepEqual(nonVerifies.map((c) => c.titre), ['style']);
  assert.equal(corriges.length, 1);
  assert.equal(restants.length, 0);
  assert.equal(ok, true);
  assert.equal(labels(trace, 'correction:').length, 1, 'seule la zone du défaut confirmé est corrigée');
});

test('recette : tout au vert → SHIP ; un échec corrigé repasse seul ; intégration rouge → BLOCKED', async () => {
  const vert = await executer(chemin('atelier-recette'), { args: { etat: ETAT }, scenario: scenario([['recette:', RECETTE_OK], ['contrôle', CONTROLE_OK]]) });
  sansErreurAvalee(vert);
  assert.equal(vert.resultat.verdict, 'SHIP');
  assert.equal(labels(vert, 'correction recette').length, 0);
  assert.equal(vert.resultat.criteres.length, 1);

  const echec = { ok: false, resultats: [], echecs: [{ titre: 'parcours cassé', gravite: 'haute', description: 'd' }] };
  const repare = await executer(chemin('atelier-recette'), {
    args: { etat: ETAT },
    scenario: scenario([
      ['recette:e2e', (rang) => (rang === 1 ? echec : RECETTE_OK)],
      ['recette:', RECETTE_OK],
      ['contrôle', CONTROLE_OK],
    ]),
  });
  assert.equal(repare.resultat.verdict, 'SHIP');
  assert.equal(labels(repare, 'correction recette:e2e').length, 1);
  assert.equal(labels(repare, 'recette:e2e').length, 2);
  assert.equal(labels(repare, 'recette:criteres').length, 1);

  const bloque = await executer(chemin('atelier-recette'), { args: { etat: ETAT, profondeur: 'eco' }, scenario: scenario([['recette:', RECETTE_OK], ['contrôle', CONTROLE_KO]]) });
  assert.equal(bloque.resultat.verdict, 'BLOCKED');
  const insuffisant = await executer(chemin('atelier-recette'), { args: { etat: ETAT, profondeur: 'eco' }, scenario: scenario([['recette:', echec], ['contrôle', CONTROLE_OK]]) });
  assert.equal(insuffisant.resultat.verdict, 'NEEDS WORK');
});

test('atelier-auto : enchaîne les six phases jusqu’au verdict', async () => {
  const trace = await executer(chemin('atelier-auto'), {
    args: { brief: 'un brief', profondeur: 'eco' },
    scenario: scenario([
      ['critique:', (_rang, appel) => (appel.workflow === 'atelier-plan' ? SANS_PROBLEME : { complet: true, manques: [], inventions: [], criteresVagues: [] })],
      ['rédaction SPEC', { nom: 'CRM', resume: 'r', type: 'webapp', traits: [], exigences: ETAT.exigences, hypotheses: [], questionsOuvertes: [] }],
      ['vérification adverse', { ok: true, problemes: [] }],
      ['découpage', PLAN],
      ['contrôle', CONTROLE_OK],
      [/^constats:/, { constats: [] }],
      ['recette:', RECETTE_OK],
    ]),
  });
  sansErreurAvalee(trace);
  assert.equal(trace.resultat.verdict, 'SHIP');
  assert.deepEqual(trace.resultat.vaguesLivrees, [0, 1]);
  assert.deepEqual(trace.resultat.etat.phasesFaites, ['cadrage', 'architecture', 'plan', 'build', 'revue', 'recette']);
  assert.deepEqual([...new Set(trace.appels.map((appel) => appel.workflow))], PHASES);
});

test('atelier-auto : s’arrête proprement quand une vague reste rouge', async () => {
  const trace = await executer(chemin('atelier-auto'), {
    args: { brief: 'un brief', profondeur: 'eco' },
    scenario: scenario([
      ['critique:', (_rang, appel) => (appel.workflow === 'atelier-plan' ? SANS_PROBLEME : { complet: true, manques: [], inventions: [], criteresVagues: [] })],
      ['rédaction SPEC', { nom: 'CRM', resume: 'r', type: 'webapp', traits: [], exigences: ETAT.exigences, hypotheses: [], questionsOuvertes: [] }],
      ['vérification adverse', { ok: true, problemes: [] }],
      ['découpage', PLAN],
      ['contrôle', CONTROLE_KO],
    ]),
  });
  assert.equal(trace.resultat.verdict, 'BLOCKED');
  assert.equal(trace.resultat.etape, 'build');
  assert.ok(!trace.appels.some((appel) => appel.workflow === 'atelier-revue'));
});
