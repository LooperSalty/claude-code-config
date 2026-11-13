export const meta = {
  name: 'atelier-build',
  description: 'Atelier 4/6 · Build : réalise une vague du plan — tâches en parallèle en TDD, relecture par tâche, intégration contrôlée, commit',
  whenToUse: 'Après atelier-plan, une fois par vague (par défaut la prochaine vague non livrée ; vagues: "toutes" pour tout enchaîner).',
  phases: [
    { title: 'État' },
    { title: 'Implémentation', detail: 'un agent par tâche, test d’abord, périmètre de fichiers exclusif' },
    { title: 'Relecture', detail: 'relecture de chaque tâche, correction des défauts graves' },
    { title: 'Intégration', detail: 'demandes transverses, contrôle, réparation, commit' },
  ],
};

// <atelier:commun> Bloc injecté par tools/atelier/sync-commun.mjs — ne pas modifier ici.
const DOCS = 'docs/atelier';
const A = (() => {
  if (args && typeof args === 'object' && !Array.isArray(args)) return args;
  if (typeof args === 'string' && args.trim().startsWith('{')) {
    try {
      return JSON.parse(args);
    } catch (erreur) {
      log(`Arguments JSON illisibles (${erreur.message}) : traités comme du texte libre.`);
    }
  }
  return {};
})();
const TEXTE_LIBRE = typeof args === 'string' && !Object.keys(A).length ? args.trim() : '';
const PROFONDEUR = ['eco', 'standard', 'max'].includes(A.profondeur) ? A.profondeur : 'standard';
const selon = (eco, standard, max) => ({ eco, standard, max })[PROFONDEUR];
const liste = (elements) => (elements && elements.length ? elements.join(', ') : 'aucun');

// Rôle → agent. Les types ecc:* et eco-router:* viennent des plugins installés ; s'ils manquent, repli sur un agent générique.
// Modèles : éclaireur = Haiku, ouvrier = Sonnet, expert = modèle de session (règles performance.md et eco-router).
const AGENTS = Object.assign(
  {
    eclaireur: { agentType: 'eco-router:scout' },
    ouvrier: { model: 'sonnet' },
    expert: {},
    planificateur: { agentType: 'ecc:planner' },
    architecte: { agentType: 'ecc:architect' },
    relecteur: { agentType: 'ecc:code-reviewer' },
    securite: { agentType: 'ecc:security-reviewer' },
    depanneur: { agentType: 'ecc:build-error-resolver' },
    e2e: { agentType: 'ecc:e2e-runner' },
    documentaliste: { agentType: 'ecc:doc-updater', model: 'sonnet' },
  },
  A.agents || {},
);

const REGLES = [
  'Tu es un agent de l’« Atelier », le workflow de création de projet d’ANAKIN.',
  `- Racine du projet : le répertoire de travail courant. Dossier de pilotage : ${DOCS}/ (BRIEF.md, SPEC.md, ARCHITECTURE.md, PLAN.md, etat.json).`,
  '- Les règles globales déjà injectées (CLAUDE.md, rules/*.md) s’appliquent : immutabilité, petits fichiers (200-400 lignes, 800 max), fonctions de moins de 50 lignes, erreurs gérées explicitement, validation aux frontières, aucun secret en dur, test d’abord (TDD), couverture visée 80 %.',
  '- Langue : documents et commentaires de code en français ; identifiants de code et messages de commit en anglais (commits conventionnels, sans ligne d’attribution).',
  '- Autonomie totale : ne pose aucune question. Face à une ambiguïté, retiens l’option la plus raisonnable et consigne-la comme hypothèse.',
  '- Fidélité : n’affirme rien que tu n’as pas vérifié ; un test rouge ou une étape sautée se rapporte tel quel, sortie à l’appui.',
  '- Coordination : consulte le tableau agent-link (topic « atelier ») en arrivant et publies-y toute découverte utile aux autres agents (contrat modifié, piège, commande cassée).',
  '- Portabilité : poste Windows (PowerShell et Git Bash). Passe par les scripts du projet (npm run …) et des chemins relatifs.',
].join('\n');

// Lance un agent pour un rôle (clé de AGENTS ou configuration directe), avec repli si le type d'agent est inconnu.
async function lancer(role, prompt, opts) {
  const conf = typeof role === 'string' ? AGENTS[role] || {} : role || {};
  const base = Object.assign({}, opts || {});
  if (conf.model && !base.model) base.model = conf.model;
  const texte = `${REGLES}\n\n${prompt}`;
  if (conf.agentType) {
    try {
      return await agent(texte, Object.assign({}, base, { agentType: conf.agentType }));
    } catch (erreur) {
      log(`Agent « ${conf.agentType} » indisponible (${String((erreur && erreur.message) || erreur).slice(0, 120)}) : repli sur un agent générique.`);
    }
  }
  return agent(texte, base);
}

const ETAT_SCHEMA = {
  type: 'object',
  properties: {
    version: { type: 'number' },
    nom: { type: 'string' },
    type: { type: 'string' },
    traits: { type: 'array', items: { type: 'string' } },
    resume: { type: 'string' },
    exigences: {
      type: 'array',
      items: {
        type: 'object',
        properties: { id: { type: 'string' }, titre: { type: 'string' }, priorite: { type: 'string' } },
        required: ['id', 'titre', 'priorite'],
      },
    },
    skills: { type: 'array', items: { type: 'string' } },
    lentilles: { type: 'array', items: { type: 'string' } },
    recette: { type: 'array', items: { type: 'string' } },
    consignes: { type: 'array', items: { type: 'string' } },
    env: { type: 'string' },
    stack: { type: 'array', items: { type: 'string' } },
    zones: { type: 'array', items: { type: 'string' } },
    commandes: {
      type: 'object',
      properties: {
        install: { type: 'string' },
        dev: { type: 'string' },
        test: { type: 'string' },
        typecheck: { type: 'string' },
        lint: { type: 'string' },
        build: { type: 'string' },
        e2e: { type: 'string' },
      },
    },
    phasesFaites: { type: 'array', items: { type: 'string' } },
    vaguesFaites: { type: 'array', items: { type: 'number' } },
  },
  required: ['nom', 'type', 'traits', 'exigences', 'skills', 'lentilles', 'recette'],
};

// État du projet : fourni par l'appelant (args.etat) ou relu depuis docs/atelier/etat.json par un éclaireur.
async function chargerEtat() {
  if (A.etat && typeof A.etat === 'object') return A.etat;
  const lu = await lancer(
    'eclaireur',
    `Lis ${DOCS}/etat.json et renvoie son contenu champ par champ, sans rien inventer ni compléter. S’il est absent ou illisible, renvoie trouve=false.`,
    {
      label: 'chargement de l’état',
      phase: 'État',
      schema: { type: 'object', properties: { trouve: { type: 'boolean' }, etat: ETAT_SCHEMA }, required: ['trouve'] },
    },
  );
  if (!lu || !lu.trouve || !lu.etat) {
    throw new Error(`État introuvable : lance d’abord atelier-cadrage (attendu : ${DOCS}/etat.json).`);
  }
  return lu.etat;
}

const marquer = (etat, phaseFaite, patch) =>
  Object.assign({}, etat, patch || {}, {
    phasesFaites: Array.from(new Set((etat.phasesFaites || []).concat(phaseFaite))),
  });

const CONTROLE_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    etapes: {
      type: 'array',
      items: {
        type: 'object',
        properties: { commande: { type: 'string' }, ok: { type: 'boolean' }, extrait: { type: 'string' } },
        required: ['commande', 'ok'],
      },
    },
    echecs: {
      type: 'array',
      items: {
        type: 'object',
        properties: { etape: { type: 'string' }, resume: { type: 'string' }, fichiers: { type: 'array', items: { type: 'string' } } },
        required: ['etape', 'resume'],
      },
    },
    couverture: { type: 'string' },
  },
  required: ['ok', 'etapes', 'echecs'],
};

// Contrôle d'intégration : un agent exécute les commandes du projet et rapporte, sans rien corriger.
async function controler(etat, contexte, titrePhase) {
  return lancer(
    'ouvrier',
    [
      `Tu es le contrôleur d’intégration (${contexte}). Tu ne corriges rien : tu exécutes et tu rapportes.`,
      `Commandes du projet, depuis la racine : ${JSON.stringify(etat.commandes || {})}. Si l’une manque, déduis-la de la configuration (package.json, pyproject.toml, Cargo.toml…).`,
      'Exécute dans l’ordre, jusqu’au bout même après un échec : installation des dépendances si un manifeste a changé, vérification de types, lint, tests unitaires et d’intégration (avec couverture si le projet la mesure), build.',
      'Pour chaque étape : la commande exacte, son résultat, et en cas d’échec l’extrait utile de la sortie (messages, fichiers, lignes). ok=true seulement si toutes les étapes passent réellement.',
    ].join('\n'),
    { label: `contrôle ${contexte}`, phase: titrePhase, schema: CONTROLE_SCHEMA },
  );
}

// Boucle contrôle → réparation jusqu'au vert, bornée par la profondeur.
async function stabiliser(etat, contexte, titrePhase) {
  const maximum = selon(1, 3, 4);
  let controle = await controler(etat, contexte, titrePhase);
  for (let tentative = 1; controle && !controle.ok && tentative <= maximum; tentative += 1) {
    await lancer(
      'depanneur',
      [
        `L’intégration (${contexte}) est rouge. Répare avec des modifications minimales : pas de changement d’architecture, aucun test affaibli, désactivé ou supprimé (un test juste qui échoue se corrige côté implémentation).`,
        `Échecs relevés par le contrôleur : ${JSON.stringify(controle.echecs)}`,
        'Relance les commandes concernées jusqu’au vert, puis dis ce que tu as changé et ce qui reste rouge.',
      ].join('\n'),
      { label: `réparation ${contexte} #${tentative}`, phase: titrePhase },
    );
    controle = await controler(etat, contexte, titrePhase);
  }
  return controle || { ok: false, etapes: [], echecs: [{ etape: 'contrôle', resume: 'le contrôleur n’a pas répondu', fichiers: [] }] };
}

// Clôture d'étape : met l'état à jour (seulement les champs listés, s'il y en a), passe les contrôles de sécurité
// d'avant-commit et committe.
async function cloturer(etat, sujet, titrePhase, champs) {
  const partiel = Array.isArray(champs) && champs.length > 0;
  const contenu = partiel ? Object.fromEntries(champs.map((cle) => [cle, etat[cle]])) : etat;
  const compteRendu = await lancer(
    'ouvrier',
    [
      'Clôture cette étape de l’Atelier :',
      partiel
        ? `1. Dans ${DOCS}/etat.json, remplace uniquement les champs ci-dessous par ces valeurs exactes et laisse tous les autres champs intacts (lis le fichier, fusionne, réécris), puis vérifie qu’il se parse :`
        : `1. Écris exactement ce JSON dans ${DOCS}/etat.json (crée le dossier au besoin), sans changer une valeur, puis vérifie qu’il se parse :`,
      JSON.stringify(contenu, null, 2),
      '2. Si le dépôt git n’existe pas encore : git init, et assure-toi qu’un .gitignore écarte dépendances, builds, .env et secrets.',
      '3. Avant de committer, applique la liste « Mandatory Security Checks » de rules/security.md au diff : au minimum aucun secret, clé ou mot de passe en dur, et aucun .env suivi.',
      `4. git add -A puis un commit conventionnel dont le sujet, en anglais, reprend : « ${sujet} ». Pas de ligne d’attribution. Ne pousse rien.`,
      'Renvoie le hash court et le sujet du commit, ainsi que toute alerte (secret détecté, fichier volumineux, rien à committer).',
    ].join('\n'),
    {
      label: `clôture ${titrePhase}`,
      phase: titrePhase,
      schema: {
        type: 'object',
        properties: { commit: { type: 'string' }, alertes: { type: 'array', items: { type: 'string' } } },
        required: ['commit', 'alertes'],
      },
    },
  );
  return compteRendu || { commit: '', alertes: ['la clôture n’a pas répondu'] };
}
// </atelier:commun>

const texte = { type: 'string' };
const textes = { type: 'array', items: texte };

// Schéma d'une tâche : identique dans atelier-plan.js (contrôlé par les tests).
const TACHE = {
  type: 'object',
  properties: {
    id: texte,
    titre: texte,
    objectif: texte,
    fichiers: textes,
    dependances: textes,
    exigences: textes,
    tests: textes,
    verification: texte,
    complexite: { type: 'string', enum: ['routine', 'complexe'] },
    echafaudage: { type: 'boolean' },
    skills: textes,
  },
  required: ['id', 'titre', 'objectif', 'fichiers', 'dependances', 'exigences', 'tests', 'verification', 'complexite'],
};

const VAGUE = {
  type: 'object',
  properties: { id: { type: 'number' }, objectif: texte, taches: { type: 'array', items: TACHE } },
  required: ['id', 'objectif', 'taches'],
};

const RAPPORT = {
  type: 'object',
  properties: {
    statut: { type: 'string', enum: ['fait', 'partiel', 'bloque'] },
    resume: texte,
    fichiers: textes,
    tests: textes,
    verification: texte,
    horsPerimetre: textes,
    dependancesManquantes: textes,
  },
  required: ['statut', 'resume', 'fichiers', 'tests', 'verification', 'horsPerimetre', 'dependancesManquantes'],
};

const REVUE = {
  type: 'object',
  properties: {
    problemes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          gravite: { type: 'string', enum: ['critique', 'haute', 'moyenne', 'basse'] },
          fichier: texte,
          ligne: { type: 'number' },
          description: texte,
          correction: texte,
        },
        required: ['gravite', 'fichier', 'description', 'correction'],
      },
    },
  },
  required: ['problemes'],
};

if (A.__tables) return { TACHE };

const unique = (elements) => Array.from(new Set(elements));
const RAPPORT_VIDE = {
  statut: 'bloque',
  resume: 'agent sans réponse',
  fichiers: [],
  tests: [],
  verification: '',
  horsPerimetre: [],
  dependancesManquantes: [],
};

phase('État');
const etatInitial = await chargerEtat();
const dejaLivrees = etatInitial.vaguesFaites || [];

// Plan fourni par l'appelant, ou relu depuis plan.json (seulement les vagues utiles, pour limiter la recopie).
async function chargerVagues() {
  if (A.plan && Array.isArray(A.plan.vagues)) return { vagues: A.plan.vagues, ids: A.plan.vagues.map((vague) => vague.id) };
  const voulu = Array.isArray(A.vagues)
    ? `les vagues ${A.vagues.join(', ')}`
    : typeof A.vagues === 'number'
      ? `la vague ${A.vagues}`
      : A.vagues === 'toutes'
        ? `toutes les vagues non livrées (déjà livrées : ${liste(dejaLivrees)})`
        : `la première vague non livrée dans l’ordre des identifiants (déjà livrées : ${liste(dejaLivrees)})`;
  const lu = await lancer(
    'ouvrier',
    `Lis ${DOCS}/plan.json. Renvoie dans « vagues » ${voulu}, recopiée(s) champ par champ sans rien modifier, et dans « ids » les identifiants de TOUTES les vagues du plan. Si le fichier est absent ou illisible : trouve=false.`,
    {
      label: 'chargement du plan',
      phase: 'État',
      schema: {
        type: 'object',
        properties: { trouve: { type: 'boolean' }, vagues: { type: 'array', items: VAGUE }, ids: { type: 'array', items: { type: 'number' } } },
        required: ['trouve', 'vagues', 'ids'],
      },
    },
  );
  if (!lu || !lu.trouve) throw new Error(`Plan introuvable : lance d’abord atelier-plan (attendu : ${DOCS}/plan.json).`);
  return { vagues: lu.vagues, ids: lu.ids };
}

const source = await chargerVagues();
const ordonnees = source.vagues.slice().sort((a, b) => a.id - b.id);
function choisir() {
  if (Array.isArray(A.vagues)) return ordonnees.filter((vague) => A.vagues.includes(vague.id));
  if (typeof A.vagues === 'number') return ordonnees.filter((vague) => vague.id === A.vagues);
  const restantes = ordonnees.filter((vague) => !dejaLivrees.includes(vague.id));
  return A.vagues === 'toutes' ? restantes : restantes.slice(0, 1);
}
const selection = choisir();
const restantesApres = (etat) => source.ids.filter((id) => !(etat.vaguesFaites || []).includes(id)).sort((a, b) => a - b);
if (!selection.length) return { ok: true, etat: etatInitial, vagues: [], restantes: restantesApres(etatInitial), message: 'aucune vague à livrer' };

const roleDe = (tache) => (tache.complexite === 'complexe' ? 'expert' : 'ouvrier');

function consigneTache(tache, vague, precedent) {
  return [
    `Tu réalises la tâche ${tache.id} « ${tache.titre} » de la vague ${vague.id} (${vague.objectif}).`,
    `Objectif : ${tache.objectif}`,
    `Exigences couvertes : ${liste(tache.exigences)} (détail dans ${DOCS}/SPEC.md). Architecture et contrats : ${DOCS}/ARCHITECTURE.md — lis les sections qui te concernent avant d’écrire.`,
    `Périmètre exclusif (toi seul écris ici) : ${liste(tache.fichiers)}`,
    tache.echafaudage
      ? 'Tâche d’échafaudage : tu peux créer la configuration racine et installer les dépendances prévues par l’architecture (toutes, pour que les vagues suivantes n’aient rien à installer).'
      : 'D’autres agents travaillent en même temps dans ce dépôt : ne modifie AUCUN fichier hors de ton périmètre (ni manifeste de paquets, ni verrou, ni configuration partagée, ni contrat), n’installe aucune dépendance, ne lance aucun serveur persistant. Si un changement hors périmètre est indispensable, ne le fais pas : décris-le précisément dans « horsPerimetre » (ou « dependancesManquantes »).',
    'Méthode — TDD strict :',
    `1. Écris d’abord ces tests et constate qu’ils échouent pour la bonne raison : ${(tache.tests || []).join(' ; ') || 'les tests qui prouvent l’objectif'}.`,
    '2. Écris l’implémentation minimale qui les fait passer, puis refactorise (petits fichiers, fonctions courtes, pas de mutation, erreurs explicites, entrées validées aux frontières).',
    `3. Vérifie avec : ${tache.verification} — seulement tes tests ciblés, pas toute la suite (les autres tâches sont en cours).`,
    'Git : n’exécute ni add, ni commit, ni stash, ni checkout ; l’intégrateur s’en charge.',
    `Skills à charger avant de coder si pertinents : ${liste(unique((tache.skills || []).concat(etatInitial.skills || [])))}.`,
    (etatInitial.consignes || []).length ? `Consignes du profil : ${etatInitial.consignes.join(' ')}` : '',
    precedent ? `Tentative précédente (à reprendre, les demandes transverses ont été traitées entre-temps) : ${JSON.stringify(precedent)}` : '',
    'Rapporte honnêtement : statut « fait » seulement si ta commande de vérification passe réellement ; « verification » contient son résultat réel.',
  ]
    .filter(Boolean)
    .join('\n');
}

async function implementer(tache, vague, precedent) {
  const rapport = await lancer(roleDe(tache), consigneTache(tache, vague, precedent), {
    label: `${tache.id} ${tache.titre}`.slice(0, 60),
    phase: 'Implémentation',
    schema: RAPPORT,
  });
  return rapport || RAPPORT_VIDE;
}

async function relire(rapport, tache) {
  if (PROFONDEUR === 'eco' || rapport.statut === 'bloque') return { rapport, problemes: [] };
  const revue = await lancer(
    'relecteur',
    [
      `Relis le travail de la tâche ${tache.id} « ${tache.titre} » — objectif : ${tache.objectif}`,
      `Fichiers annoncés : ${liste(rapport.fichiers)} (périmètre : ${liste(tache.fichiers)}). Exigences : ${liste(tache.exigences)}. Références : ${DOCS}/SPEC.md, ${DOCS}/ARCHITECTURE.md.`,
      'Vérifie : conformité à l’objectif et aux exigences ; tests présents, pertinents, cas limites couverts ; bugs de logique ; erreurs avalées ; entrées non validées ; secrets en dur ; mutations ; fichiers de plus de 800 lignes ou fonctions de plus de 50 ; écart avec l’architecture ; écriture hors périmètre.',
      'Ne modifie rien. Ne signale que des problèmes réels et précis (fichier, ligne, scénario d’échec, correction) ; gravité critique, haute, moyenne ou basse.',
    ].join('\n'),
    { label: `relecture ${tache.id}`, phase: 'Relecture', schema: REVUE },
  );
  return { rapport, problemes: revue ? revue.problemes : [] };
}

async function corriger(resultat, tache, vague) {
  const graves = resultat.problemes.filter((probleme) => probleme.gravite === 'critique' || probleme.gravite === 'haute');
  const mineurs = resultat.problemes.filter((probleme) => !graves.includes(probleme));
  if (!graves.length) return Object.assign({}, resultat.rapport, { defautsRestants: mineurs, defautsCorriges: 0 });
  const corrige = await lancer(
    roleDe(tache),
    [
      consigneTache(tache, vague, resultat.rapport),
      `La relecture a trouvé des défauts graves dans ton périmètre. Pour chacun : un test qui échoue d’abord, puis la correction. Défauts : ${JSON.stringify(graves)}`,
    ].join('\n'),
    { label: `correction ${tache.id}`, phase: 'Relecture', schema: RAPPORT },
  );
  return Object.assign({}, corrige || resultat.rapport, { defautsRestants: mineurs, defautsCorriges: corrige ? graves.length : 0 });
}

// Livre une vague : tâches en pipeline (implémenter → relire → corriger), demandes transverses, seconde chance, contrôle, commit.
async function livrer(vague, etat) {
  log(`Vague ${vague.id} — ${vague.objectif} : ${vague.taches.length} tâche(s).`);
  const bruts = await pipeline(
    vague.taches,
    (tache) => implementer(tache, vague),
    (rapport, tache) => relire(rapport, tache),
    (resultat, tache) => corriger(resultat, tache, vague),
  );
  const premiers = bruts.map((rapport, index) => Object.assign({ id: vague.taches[index].id }, rapport || RAPPORT_VIDE));

  const demandes = premiers.flatMap((rapport) => (rapport.horsPerimetre || []).map((demande) => `[${rapport.id}] ${demande}`));
  const dependances = unique(premiers.flatMap((rapport) => rapport.dependancesManquantes || []));
  if (demandes.length || dependances.length) {
    await lancer(
      'ouvrier',
      [
        `Tu es l’intégrateur de la vague ${vague.id} : les tâches parallèles sont terminées, tu es seul sur le dépôt.`,
        dependances.length ? `Installe les dépendances manquantes signalées : ${dependances.join(', ')} (versions compatibles avec ${DOCS}/ARCHITECTURE.md).` : '',
        demandes.length ? `Applique les changements hors périmètre demandés par les tâches (contrats partagés, configuration, branchements) :\n- ${demandes.join('\n- ')}` : '',
        'Reste minimal et cohérent avec l’architecture ; dis ce que tu refuses de faire et pourquoi.',
      ]
        .filter(Boolean)
        .join('\n'),
      { label: `intégration vague ${vague.id}`, phase: 'Intégration' },
    );
  }

  const aReprendre = premiers.map((rapport, index) => ({ rapport, tache: vague.taches[index] })).filter(({ rapport }) => rapport.statut !== 'fait');
  const reprises = aReprendre.length
    ? await parallel(aReprendre.map(({ rapport, tache }) => () => implementer(tache, vague, rapport).then((nouveau) => Object.assign({ id: tache.id, reprise: true }, nouveau))))
    : [];
  const rapports = premiers.map((rapport) => reprises.filter(Boolean).find((reprise) => reprise.id === rapport.id) || rapport);

  const controle = await stabiliser(etat, `vague ${vague.id}`, 'Intégration');
  if (!controle.ok) {
    log(`Vague ${vague.id} : intégration toujours rouge après réparations — arrêt.`);
    return { etat, compteRendu: { id: vague.id, ok: false, rapports, controle, cloture: null } };
  }
  const faites = unique((etat.vaguesFaites || []).concat(vague.id)).sort((a, b) => a - b);
  const avecVague = Object.assign({}, etat, { vaguesFaites: faites });
  const suivant = restantesApres(avecVague).length ? avecVague : marquer(avecVague, 'build');
  const cloture = await cloturer(suivant, `feat: wave ${vague.id} — ${vague.objectif}`, 'Intégration', ['vaguesFaites', 'phasesFaites']);
  return { etat: suivant, compteRendu: { id: vague.id, ok: true, rapports, controle, cloture } };
}

let etat = etatInitial;
let livrees = [];
for (const vague of selection) {
  const resultat = await livrer(vague, etat);
  etat = resultat.etat;
  livrees = livrees.concat(resultat.compteRendu);
  if (!resultat.compteRendu.ok) break;
}

return { ok: livrees.every((vague) => vague.ok), etat, vagues: livrees, restantes: restantesApres(etat) };
