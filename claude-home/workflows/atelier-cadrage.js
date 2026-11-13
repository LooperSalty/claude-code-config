export const meta = {
  name: 'atelier-cadrage',
  description: 'Atelier 1/6 · Cadrage : brief ou cahier des charges → SPEC.md vérifiable, hypothèses, profil du projet et état initial',
  whenToUse: 'Au démarrage de tout projet (logiciel, application web, site, jeu), avant toute architecture. Argument : le brief ou le chemin du cahier des charges.',
  phases: [
    { title: 'Lecture', detail: 'brief normalisé dans docs/atelier/BRIEF.md, relevé de l’environnement' },
    { title: 'Extraction', detail: 'quatre analystes, un angle chacun' },
    { title: 'Spécification', detail: 'SPEC.md, critique de complétude, état initial' },
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

// Catalogue des profils : skills à charger, lentilles de revue et vérifications de recette, par type de projet puis par trait.
const SOCLE = [
  'superpowers:test-driven-development',
  'superpowers:systematic-debugging',
  'superpowers:verification-before-completion',
  'agent-skills:incremental-implementation',
  'agent-skills:source-driven-development',
  'agent-skills:security-and-hardening',
  'ecc:coding-standards',
];

const PROFILS = {
  webapp: {
    libelle: 'Application web (interface, API, données)',
    skills: [
      'frontend-design:frontend-design',
      'ui-ux-pro-max:ui-ux-pro-max',
      'agent-skills:frontend-ui-engineering',
      'agent-skills:api-and-interface-design',
      'ecc:backend-patterns',
      'ecc:frontend-patterns',
      'ecc:react-patterns',
      'ecc:react-testing',
      'ecc:e2e-testing',
      'webapp-testing',
      'vercel:react-best-practices',
    ],
    lentilles: ['correction', 'securite', 'conformite', 'tests', 'echecs-silencieux', 'performance', 'accessibilite'],
    recette: ['criteres', 'e2e', 'demo', 'accessibilite', 'installation'],
    consignes: ['Chaque écran prévoit ses états vide, chargement et erreur.'],
  },
  website: {
    libelle: 'Site web (vitrine, contenu, marketing)',
    skills: [
      'design-taste-frontend',
      'frontend-design:frontend-design',
      'antislop-ui',
      'antislop-copywriting',
      'web-design-guidelines',
      'ui-ux-pro-max:ui-ux-pro-max',
      'gsap-core',
      'gsap-scrolltrigger',
      'motion-design',
      'ecc:seo',
      'ecc:accessibility',
      'agent-skills:performance-optimization',
    ],
    lentilles: ['correction', 'conformite', 'accessibilite', 'performance', 'seo', 'contenu'],
    recette: ['criteres', 'e2e', 'demo', 'lighthouse', 'responsive', 'accessibilite'],
    consignes: ['Textes rédigés pour de vrais lecteurs : pas de remplissage, pas de formules génériques.'],
  },
  game: {
    libelle: 'Jeu (web, moteur ou plateforme)',
    skills: ['agent-skills:performance-optimization', 'improve-animations', 'animation-performance'],
    lentilles: ['correction', 'conformite', 'tests', 'echecs-silencieux', 'performance', 'jouabilite'],
    recette: ['criteres', 'jouabilite', 'budget-fps', 'installation'],
    consignes: [
      'Jeu original uniquement : jamais de recréation d’un jeu existant (nom, personnages, art, musique, niveaux).',
      'Logique de jeu séparée du rendu et testable sans affichage (pas de temps fixe, états explicites).',
    ],
  },
  software: {
    libelle: 'Logiciel (CLI, bibliothèque, service, application de bureau)',
    skills: ['agent-skills:api-and-interface-design', 'ecc:error-handling', 'agent-skills:documentation-and-adrs'],
    lentilles: ['correction', 'securite', 'conformite', 'tests', 'echecs-silencieux', 'types'],
    recette: ['criteres', 'cli', 'installation'],
    consignes: ['Codes de sortie, messages d’erreur et aide en ligne font partie de l’interface : ils se testent.'],
  },
};

const TRAITS = {
  '3d': {
    skills: [
      'threejs-react-three-fiber',
      'threejs-fundamentals',
      'threejs-performance',
      'threejs-interaction',
      'threejs-geometry',
      'threejs-materials',
      'threejs-lighting',
    ],
    lentilles: ['performance-3d'],
    recette: ['budget-fps'],
    consignes: ['Rendu 3D : instancing, pas d’allocation dans la boucle de rendu, ressources libérées (dispose).'],
  },
  donnees: {
    skills: ['ecc:postgres-patterns', 'ecc:database-migrations'],
    lentilles: ['donnees', 'concurrence'],
    recette: ['integrite-donnees'],
    consignes: ['Chaque invariant de données est garanti par la base ou une transaction, pas seulement par le code appelant.'],
  },
  auth: {
    skills: ['ecc:security-review'],
    lentilles: ['securite'],
    recette: ['matrice-droits'],
    consignes: ['Les droits se vérifient côté serveur à chaque requête ; l’interface ne fait que les refléter.'],
  },
  api: {
    skills: ['ecc:api-design', 'agent-skills:api-and-interface-design'],
    lentilles: [],
    recette: ['api'],
    consignes: ['Réponses d’API sous enveloppe uniforme (succès, données, erreur, métadonnées de pagination).'],
  },
  ia: { skills: ['claude-api', 'ecc:cost-aware-llm-pipeline'], lentilles: [], recette: [], consignes: [] },
  'temps-reel': { skills: [], lentilles: ['concurrence'], recette: [], consignes: [] },
  animation: { skills: ['gsap-core', 'gsap-timeline', 'motion-design', 'improve-animations'], lentilles: [], recette: [], consignes: [] },
  mobile: { skills: ['ecc:react-native-patterns'], lentilles: [], recette: ['responsive'], consignes: [] },
  python: { skills: ['ecc:python-patterns', 'ecc:python-testing'], lentilles: [], recette: [], consignes: [] },
  rust: { skills: ['ecc:rust-patterns', 'ecc:rust-testing'], lentilles: [], recette: [], consignes: [] },
  go: { skills: ['ecc:golang-patterns', 'ecc:golang-testing'], lentilles: [], recette: [], consignes: [] },
  roblox: {
    skills: [],
    lentilles: [],
    recette: [],
    consignes: ['Roblox : passer par les serveurs MCP robloxstudio / roblox-forge pour lire et modifier la place.'],
  },
};

if (A.__tables) return { SOCLE, PROFILS, TRAITS };

const unique = (elements) => Array.from(new Set(elements));

function construireProfil(type, traits) {
  const base = PROFILS[type] || PROFILS.software;
  const extras = traits.map((trait) => TRAITS[trait]).filter(Boolean);
  const unir = (cle, debut) => unique([].concat(debut, base[cle] || [], ...extras.map((extra) => extra[cle] || [])));
  return {
    skills: unir('skills', SOCLE),
    lentilles: unir('lentilles', []),
    recette: unir('recette', []),
    consignes: unir('consignes', []),
  };
}

const texte = { type: 'string' };
const textes = { type: 'array', items: texte };
const priorite = { type: 'string', enum: ['P0', 'P1', 'P2'] };

const LECTURE = {
  type: 'object',
  properties: { briefTrouve: { type: 'boolean' }, resume: texte, env: texte, existant: texte },
  required: ['briefTrouve', 'resume', 'env', 'existant'],
};

const EXTRACTION = {
  type: 'object',
  properties: {
    exigences: {
      type: 'array',
      items: {
        type: 'object',
        properties: { titre: texte, description: texte, priorite, criteres: textes, source: texte },
        required: ['titre', 'description', 'priorite', 'criteres'],
      },
    },
    ambiguites: {
      type: 'array',
      items: {
        type: 'object',
        properties: { question: texte, hypothese: texte, impact: texte },
        required: ['question', 'hypothese'],
      },
    },
    risques: textes,
  },
  required: ['exigences', 'ambiguites', 'risques'],
};

const SPEC = {
  type: 'object',
  properties: {
    nom: texte,
    resume: texte,
    type: { type: 'string', enum: Object.keys(PROFILS) },
    traits: { type: 'array', items: { type: 'string', enum: Object.keys(TRAITS) } },
    exigences: {
      type: 'array',
      items: { type: 'object', properties: { id: texte, titre: texte, priorite }, required: ['id', 'titre', 'priorite'] },
    },
    hypotheses: textes,
    questionsOuvertes: textes,
  },
  required: ['nom', 'resume', 'type', 'traits', 'exigences', 'hypotheses', 'questionsOuvertes'],
};

const CRITIQUE = {
  type: 'object',
  properties: {
    complet: { type: 'boolean' },
    manques: {
      type: 'array',
      items: {
        type: 'object',
        properties: { element: texte, gravite: { type: 'string', enum: ['bloquant', 'majeur', 'mineur'] }, correction: texte },
        required: ['element', 'gravite', 'correction'],
      },
    },
    inventions: textes,
    criteresVagues: textes,
  },
  required: ['complet', 'manques', 'inventions', 'criteresVagues'],
};

// --- Lecture : le brief devient un fichier que tous les agents peuvent relire.
phase('Lecture');
const sourceBrief = TEXTE_LIBRE || A.brief || A.briefPath || '';
const lecture = await lancer(
  'ouvrier',
  [
    'Mission : préparer le dossier de cadrage.',
    sourceBrief
      ? `1. Source du brief : « ${sourceBrief} ». Si c’est le chemin d’un fichier existant, lis-le (docx : skill docx ou lecture de word/document.xml ; pdf : skill pdf ; texte : tel quel). Sinon, c’est le brief lui-même.`
      : '1. Aucune source fournie : cherche dans le répertoire courant un cahier des charges ou un brief (docx, pdf, md, txt).',
    `2. Écris ${DOCS}/BRIEF.md : le contenu INTÉGRAL du brief en Markdown (titres, listes, tableaux), sans résumer, reformuler ni ajouter quoi que ce soit.`,
    '3. Relève l’environnement réel par des commandes : système, versions de node, npm, python, git ; docker utilisable ou non (docker info) ; ports déjà pris parmi 3000, 5173, 5432, 8080 ; contenu actuel du répertoire (projet vierge ou existant).',
    A.env ? `Faits d’environnement déjà établis : ${A.env}` : '',
    'briefTrouve=false si tu n’as trouvé aucun brief exploitable. « existant » décrit ce que contient déjà le répertoire.',
  ]
    .filter(Boolean)
    .join('\n'),
  { label: 'lecture du brief', phase: 'Lecture', schema: LECTURE },
);
if (!lecture || !lecture.briefTrouve) {
  throw new Error('atelier-cadrage : aucun brief exploitable (fournis le texte du brief ou le chemin du cahier des charges).');
}

// --- Extraction : quatre lectures indépendantes, chacune aveugle aux autres.
phase('Extraction');
const ANALYSTES = [
  { cle: 'fonctionnel', consigne: 'exigences fonctionnelles et parcours utilisateur : qui fait quoi, dans quel ordre, avec quel résultat observable' },
  { cle: 'donnees', consigne: 'modèle de données, règles métier et invariants (unicité, atomicité, traçabilité, cohérence des quantités)' },
  { cle: 'qualite', consigne: 'exigences non fonctionnelles : performance, sécurité, confidentialité et RGPD, accessibilité, exploitation, contraintes techniques imposées' },
  { cle: 'experience', consigne: 'expérience et interaction : écrans, états, navigation, scénario de démonstration, modes dégradés' },
];
const ANALYSTE_UNIQUE = { cle: 'complet', consigne: ANALYSTES.map((analyste) => analyste.consigne).join(' ; ') };
const extractions = (
  await parallel(
    selon([ANALYSTE_UNIQUE], ANALYSTES, ANALYSTES).map((analyste) => () =>
      lancer(
        'expert',
        [
          `Tu es l’analyste « ${analyste.cle} » du cadrage. Lis intégralement ${DOCS}/BRIEF.md.`,
          `Angle exclusif : ${analyste.consigne}.`,
          'Pour chaque exigence : un titre court, une description précise, une priorité (P0 indispensable au MVP, P1 important, P2 plus tard — reprends celles du brief quand il en donne), des critères d’acceptation observables et vérifiables par un test, et la section du brief qui la fonde.',
          'Relève aussi les ambiguïtés (avec l’hypothèse que tu recommandes et son impact) et les risques. N’invente aucune exigence absente du brief : ce qui te semble manquer va dans les ambiguïtés.',
        ].join('\n'),
        { label: `analyste:${analyste.cle}`, phase: 'Extraction', schema: EXTRACTION },
      ).then((resultat) => resultat && Object.assign({ angle: analyste.cle }, resultat)),
    ),
  )
).filter(Boolean);
if (!extractions.length) throw new Error('atelier-cadrage : aucune extraction exploitable.');

// --- Spécification : synthèse, critique de complétude, révision si nécessaire.
phase('Spécification');
const consigneSpec = [
  `Écris ${DOCS}/SPEC.md avec les sections : Résumé et objectifs · Utilisateurs et parcours · Exigences (tableau : ID, priorité, titre, description, critères d’acceptation) · Modèle de données et invariants · Exigences non fonctionnelles · Scénario de démonstration · Critères d’acceptation globaux (CA-01, CA-02…) · Hypothèses · Hors périmètre · Questions ouvertes · Risques.`,
  'Identifiants stables REQ-001, REQ-002… ; priorités du brief conservées ; chaque critère d’acceptation vérifiable par un test automatisé ou une observation précise (seuil, état attendu).',
  'Tranche chaque ambiguïté par une hypothèse explicite. Ce qui relève d’un choix du client sans bloquer la réalisation va en questions ouvertes.',
  `Classe le projet : type parmi ${Object.keys(PROFILS).join(', ')} ; traits parmi ${Object.keys(TRAITS).join(', ')} (seulement ceux réellement présents).`,
].join('\n');
const premiereSpec = await lancer(
  'expert',
  [
    `Tu rédiges la spécification du projet à partir du brief (${DOCS}/BRIEF.md) et des extractions des analystes ci-dessous : fusionne, dédoublonne, ne perds rien.`,
    consigneSpec,
    `Extractions : ${JSON.stringify(extractions)}`,
  ].join('\n'),
  { label: 'rédaction SPEC', phase: 'Spécification', schema: SPEC },
);
if (!premiereSpec) throw new Error('atelier-cadrage : la rédaction de la SPEC a échoué.');

const critiques = (
  await parallel(
    selon(['couverture du brief'], ['couverture du brief'], ['couverture du brief', 'vérifiabilité des critères']).map((angle) => () =>
      lancer(
        'expert',
        [
          `Tu es le critique de complétude (angle : ${angle}). Compare ${DOCS}/BRIEF.md et ${DOCS}/SPEC.md, phrase par phrase.`,
          'Relève : (a) les manques — éléments du brief absents, affaiblis ou déformés dans la SPEC ; (b) les inventions — exigences de la SPEC sans fondement dans le brief et non déclarées comme hypothèses ; (c) les critères d’acceptation invérifiables (vagues, subjectifs, sans seuil).',
          'Considère la SPEC incomplète tant que tu n’as pas vérifié chaque section du brief. Ne modifie aucun fichier.',
        ].join('\n'),
        { label: `critique:${angle}`, phase: 'Spécification', schema: CRITIQUE },
      ),
    ),
  )
).filter(Boolean);

const aReprendre = critiques.some(
  (critique) =>
    !critique.complet ||
    critique.manques.some((manque) => manque.gravite !== 'mineur') ||
    critique.inventions.length > 0 ||
    critique.criteresVagues.length > 0,
);
const reprise = aReprendre
  ? await lancer(
      'expert',
      [
        `Révise ${DOCS}/SPEC.md pour lever les critiques ci-dessous, en repartant du brief (${DOCS}/BRIEF.md). Garde les identifiants REQ existants.`,
        consigneSpec,
        `Critiques : ${JSON.stringify(critiques)}`,
      ].join('\n'),
      { label: 'révision SPEC', phase: 'Spécification', schema: SPEC },
    )
  : null;
const spec = reprise || premiereSpec;

const type = PROFILS[A.type] ? A.type : spec.type;
const profil = construireProfil(type, spec.traits);
const etat = {
  version: 1,
  nom: spec.nom,
  type,
  traits: spec.traits,
  resume: spec.resume,
  exigences: spec.exigences,
  skills: profil.skills,
  lentilles: profil.lentilles,
  recette: profil.recette,
  consignes: profil.consignes,
  env: [lecture.env, A.env].filter(Boolean).join(' | '),
  stack: [],
  zones: [],
  commandes: {},
  phasesFaites: ['cadrage'],
  vaguesFaites: [],
};
log(`Projet « ${etat.nom} » : type ${type} (${PROFILS[type].libelle}), traits ${liste(spec.traits)}, ${spec.exigences.length} exigence(s).`);
const cloture = await cloturer(etat, 'docs: add project brief and specification', 'Spécification');

return {
  etat,
  existant: lecture.existant,
  hypotheses: spec.hypotheses,
  questionsOuvertes: spec.questionsOuvertes,
  critiques,
  revisee: Boolean(reprise),
  cloture,
};
