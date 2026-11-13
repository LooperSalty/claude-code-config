export const meta = {
  name: 'atelier-plan',
  description: 'Atelier 3/6 · Plan : découpage en vagues de tâches parallèles à périmètre de fichiers exclusif, critiqué puis validé mécaniquement',
  whenToUse: 'Après atelier-architecture, pour obtenir PLAN.md et plan.json avant le build.',
  phases: [
    { title: 'État' },
    { title: 'Découpage', detail: 'le planificateur propose vagues et tâches' },
    { title: 'Critique', detail: 'couverture, parallélisme, testabilité' },
    { title: 'Plan final', detail: 'révision, validation mécanique, écriture' },
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

// Clôture d'étape : écrit l'état, passe les contrôles de sécurité d'avant-commit et committe.
async function cloturer(etat, sujet, titrePhase) {
  const compteRendu = await lancer(
    'ouvrier',
    [
      'Clôture cette étape de l’Atelier :',
      `1. Écris exactement ce JSON dans ${DOCS}/etat.json (crée le dossier au besoin), sans changer une valeur, puis vérifie qu’il se parse :`,
      JSON.stringify(etat, null, 2),
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

// Schéma d'une tâche : identique dans atelier-build.js (contrôlé par les tests).
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

const PLAN = {
  type: 'object',
  properties: {
    vagues: {
      type: 'array',
      items: {
        type: 'object',
        properties: { id: { type: 'number' }, objectif: texte, taches: { type: 'array', items: TACHE } },
        required: ['id', 'objectif', 'taches'],
      },
    },
    couverture: {
      type: 'array',
      items: { type: 'object', properties: { exigence: texte, taches: textes }, required: ['exigence', 'taches'] },
    },
  },
  required: ['vagues', 'couverture'],
};

const CRITIQUE = {
  type: 'object',
  properties: {
    problemes: {
      type: 'array',
      items: {
        type: 'object',
        properties: { gravite: { type: 'string', enum: ['bloquant', 'majeur', 'mineur'] }, description: texte, correction: texte },
        required: ['gravite', 'description', 'correction'],
      },
    },
  },
  required: ['problemes'],
};

const normaliser = (chemin) => String(chemin).replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase();

// Deux chemins se chevauchent s'ils sont identiques ou si l'un est un dossier (terminé par /) contenant l'autre.
function seChevauchent(a, b) {
  const x = normaliser(a);
  const y = normaliser(b);
  if (x === y) return true;
  return (x.endsWith('/') && y.startsWith(x)) || (y.endsWith('/') && x.startsWith(y));
}

function conflitsDeVague(vague) {
  const lot = vague.taches || [];
  return lot.flatMap((a, index) =>
    lot.slice(index + 1).flatMap((b) =>
      (a.fichiers || []).flatMap((fa) =>
        (b.fichiers || [])
          .filter((fb) => seChevauchent(fa, fb))
          .map((fb) => `vague ${vague.id} : ${a.id} et ${b.id} se disputent ${fa}${fa === fb ? '' : ` / ${fb}`}`),
      ),
    ),
  );
}

// Validation mécanique : ce que le script peut prouver sans agent.
function validerPlan(plan, exigences) {
  const vagues = plan.vagues || [];
  const taches = vagues.flatMap((vague) => (vague.taches || []).map((tache) => ({ tache, vague: vague.id })));
  const rang = new Map(taches.map(({ tache, vague }) => [tache.id, vague]));
  const ids = taches.map(({ tache }) => tache.id);
  const idsVagues = vagues.map((vague) => vague.id);
  const couvertes = new Set(taches.flatMap(({ tache }) => tache.exigences || []));
  return [].concat(
    ids.filter((id, index) => ids.indexOf(id) !== index).map((id) => `identifiant de tâche en double : ${id}`),
    idsVagues.filter((id, index) => idsVagues.indexOf(id) !== index).map((id) => `identifiant de vague en double : ${id}`),
    vagues.filter((vague) => !(vague.taches || []).length).map((vague) => `vague ${vague.id} sans tâche`),
    taches.flatMap(({ tache }) =>
      (tache.fichiers || [])
        .filter((fichier) => fichier.includes('*'))
        .map((fichier) => `${tache.id} : motif générique interdit dans « fichiers » (${fichier})`),
    ),
    taches.filter(({ tache }) => !(tache.fichiers || []).length).map(({ tache }) => `${tache.id} : aucun fichier possédé`),
    taches
      .filter(({ tache }) => !tache.echafaudage && !(tache.tests || []).length)
      .map(({ tache }) => `${tache.id} : aucun test à écrire d’abord`),
    taches.flatMap(({ tache, vague }) =>
      (tache.dependances || []).flatMap((dependance) => {
        if (!rang.has(dependance)) return [`${tache.id} : dépendance inconnue ${dependance}`];
        if (rang.get(dependance) >= vague) return [`${tache.id} : dépend de ${dependance}, qui n’est pas dans une vague antérieure`];
        return [];
      }),
    ),
    vagues.flatMap(conflitsDeVague),
    (exigences || [])
      .filter((exigence) => exigence.priorite !== 'P2' && !couvertes.has(exigence.id))
      .map((exigence) => `exigence ${exigence.id} (${exigence.priorite}) couverte par aucune tâche`),
  );
}

if (A.__tables) return { TACHE };
if (A.__valider) return { erreurs: validerPlan(A.__valider, A.__exigences || []) };

phase('État');
const etat = await chargerEtat();

const reglesDecoupage = [
  'Règles de découpage :',
  '- Vague 0 = échafaudage : dépôt, configuration, TOUTES les dépendances, outillage de test, contrats partagés (types, schémas, interfaces). Ses tâches portent echafaudage=true.',
  '- Ensuite des tranches verticales : chaque vague livre un incrément démontrable ; P0 avant P1 ; P2 seulement si la SPEC les inclut dans le MVP.',
  '- Dans une vague, les tâches tournent EN MÊME TEMPS dans le même répertoire : chaque fichier ou dossier appartient à UNE seule tâche (« fichiers » : chemins relatifs sans motif générique, dossiers terminés par /), et une tâche ne dépend que de tâches de vagues antérieures.',
  '- Une tâche = un agent, une session : objectif unique, 3 à 12 fichiers, vérifiable par une commande ciblée (« verification ») qui ne lance que ses propres tests.',
  '- Chaque tâche liste les tests à écrire D’ABORD (comportements attendus, cas limites) et les exigences REQ qu’elle réalise ; « skills » nomme les skills utiles parmi ceux du projet.',
  '- complexite « complexe » pour le cœur délicat (transactions, concurrence, sécurité, rendu, algorithmes) ; « routine » sinon.',
  '- Dernière vague : finitions transverses (jeu de démonstration, tests de bout en bout des parcours clés, documentation d’exploitation).',
  '- « couverture » : pour chaque exigence P0/P1, les tâches qui la réalisent.',
].join('\n');
const contexte = [
  `Projet : ${etat.nom} (${etat.type} ; ${liste(etat.traits)}). Pile : ${liste(etat.stack)}.`,
  `Exigences : ${(etat.exigences || []).map((exigence) => `${exigence.id} [${exigence.priorite}] ${exigence.titre}`).join(' ; ')}`,
  `Skills du projet : ${liste(etat.skills)}.`,
].join('\n');

phase('Découpage');
const brouillon = await lancer(
  'planificateur',
  [
    `Tu es le planificateur. Lis ${DOCS}/SPEC.md et ${DOCS}/ARCHITECTURE.md, puis découpe la réalisation en vagues de tâches exécutables par des agents en parallèle.`,
    contexte,
    reglesDecoupage,
  ].join('\n'),
  { label: 'découpage', phase: 'Découpage', schema: PLAN },
);
if (!brouillon) throw new Error('atelier-plan : le planificateur n’a rien renvoyé.');

phase('Critique');
const CRITIQUES = [
  { cle: 'couverture', consigne: 'chaque exigence P0/P1 et chaque critère d’acceptation (y compris globaux) de la SPEC est-il réalisé ET testé par une tâche identifiable ? Rien d’inventé hors SPEC ?' },
  { cle: 'parallelisme', consigne: 'périmètres de fichiers réellement disjoints dans chaque vague (y compris fichiers d’index, de routes, de configuration que plusieurs tâches voudront toucher) ; contrats définis avant leurs consommateurs ; ordre des vagues' },
  { cle: 'testabilite', consigne: 'chaque tâche a des tests à écrire d’abord qui prouvent son objectif, une commande de vérification exécutable, une taille tenable en une session ; les trois niveaux de tests (unitaire, intégration, bout en bout) sont couverts' },
];
const critiques = (
  await parallel(
    selon(CRITIQUES.slice(0, 1), CRITIQUES, CRITIQUES).map((critique) => () =>
      lancer(
        'expert',
        [
          `Tu es critique du plan (angle : ${critique.cle}). Lis ${DOCS}/SPEC.md et ${DOCS}/ARCHITECTURE.md, puis examine le plan ci-dessous.`,
          `Question : ${critique.consigne}`,
          'Signale des problèmes précis (tâche, fichier, exigence) avec leur correction ; gravité bloquant, majeur ou mineur. Ne modifie aucun fichier.',
          `Plan : ${JSON.stringify(brouillon)}`,
        ].join('\n'),
        { label: `critique:${critique.cle}`, phase: 'Critique', schema: CRITIQUE },
      ).then((resultat) => resultat && Object.assign({ angle: critique.cle }, resultat)),
    ),
  )
).filter(Boolean);

// --- Plan final : révision sur critiques, puis boucle tant que la validation mécanique échoue.
phase('Plan final');
const reproches = critiques.flatMap((critique) =>
  critique.problemes
    .filter((probleme) => probleme.gravite !== 'mineur')
    .map((probleme) => `[${critique.angle}] ${probleme.description} → ${probleme.correction}`),
);
let plan = brouillon;
let erreurs = validerPlan(plan, etat.exigences);
for (let tour = 1; tour <= selon(1, 2, 3) && (erreurs.length || (tour === 1 && reproches.length)); tour += 1) {
  const revise = await lancer(
    'planificateur',
    [
      `Tu es le planificateur. Révise le plan ci-dessous (références : ${DOCS}/SPEC.md, ${DOCS}/ARCHITECTURE.md) et renvoie le plan COMPLET corrigé.`,
      contexte,
      reglesDecoupage,
      erreurs.length ? `Erreurs de validation mécanique à corriger impérativement :\n- ${erreurs.join('\n- ')}` : '',
      tour === 1 && reproches.length ? `Critiques à traiter :\n- ${reproches.join('\n- ')}` : '',
      `Plan actuel : ${JSON.stringify(plan)}`,
    ]
      .filter(Boolean)
      .join('\n'),
    { label: `révision du plan #${tour}`, phase: 'Plan final', schema: PLAN },
  );
  if (!revise) break;
  plan = revise;
  erreurs = validerPlan(plan, etat.exigences);
}
if (erreurs.length) log(`Plan : ${erreurs.length} erreur(s) de validation restante(s) — ${erreurs.slice(0, 3).join(' | ')}`);

await lancer(
  'ouvrier',
  [
    'Écris deux fichiers à partir du plan ci-dessous, sans en changer le contenu :',
    `- ${DOCS}/plan.json : exactement ce JSON (vérifie qu’il se parse) ;`,
    `- ${DOCS}/PLAN.md : version lisible — pour chaque vague son objectif, un tableau des tâches (id, titre, complexité, fichiers possédés, dépendances, exigences, vérification), les tests à écrire d’abord ; puis la matrice de couverture exigence → tâches.`,
    JSON.stringify(plan),
  ].join('\n'),
  { label: 'écriture du plan', phase: 'Plan final' },
);
const suivant = marquer(etat, 'plan', { vaguesFaites: etat.vaguesFaites || [] });
const cloture = await cloturer(suivant, 'docs: add implementation plan', 'Plan final');

const nombreTaches = plan.vagues.reduce((total, vague) => total + vague.taches.length, 0);
log(`Plan : ${plan.vagues.length} vague(s), ${nombreTaches} tâche(s).`);
return { etat: suivant, plan, erreurs, critiques, cloture };
