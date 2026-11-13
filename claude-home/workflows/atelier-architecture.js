export const meta = {
  name: 'atelier-architecture',
  description: 'Atelier 2/6 · Architecture : veille des versions, propositions concurrentes, jury multi-angles, ARCHITECTURE.md et ADR vérifiés',
  whenToUse: 'Après atelier-cadrage, pour choisir la pile technique et l’architecture avant de planifier.',
  phases: [
    { title: 'État' },
    { title: 'Veille', detail: 'versions réelles, squelettes éprouvés, pièges connus' },
    { title: 'Propositions', detail: 'architectes indépendants, un angle chacun' },
    { title: 'Jury', detail: 'notation croisée, un angle de jugement par juré' },
    { title: 'Synthèse', detail: 'ARCHITECTURE.md et ADR, vérification adverse' },
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

const VEILLE = {
  type: 'object',
  properties: {
    constats: {
      type: 'array',
      items: { type: 'object', properties: { sujet: texte, fait: texte, source: texte }, required: ['sujet', 'fait', 'source'] },
    },
    recommandations: textes,
  },
  required: ['constats', 'recommandations'],
};

const PROPOSITION = {
  type: 'object',
  properties: {
    titre: texte,
    resume: texte,
    stack: {
      type: 'array',
      items: {
        type: 'object',
        properties: { brique: texte, choix: texte, version: texte, justification: texte },
        required: ['brique', 'choix', 'justification'],
      },
    },
    structure: texte,
    donnees: texte,
    interfaces: texte,
    tests: texte,
    securite: texte,
    risques: textes,
    compromis: textes,
  },
  required: ['titre', 'resume', 'stack', 'structure', 'donnees', 'interfaces', 'tests', 'securite', 'risques', 'compromis'],
};

const ARCHITECTURE = {
  type: 'object',
  properties: {
    resume: texte,
    stack: {
      type: 'array',
      items: { type: 'object', properties: { brique: texte, choix: texte, version: texte }, required: ['brique', 'choix', 'version'] },
    },
    modules: {
      type: 'array',
      items: { type: 'object', properties: { nom: texte, chemin: texte, responsabilite: texte }, required: ['nom', 'chemin', 'responsabilite'] },
    },
    commandes: {
      type: 'object',
      properties: { install: texte, dev: texte, test: texte, typecheck: texte, lint: texte, build: texte, e2e: texte },
      required: ['install', 'test', 'build'],
    },
    decisions: {
      type: 'array',
      items: { type: 'object', properties: { titre: texte, choix: texte, raison: texte }, required: ['titre', 'choix', 'raison'] },
    },
    risques: textes,
  },
  required: ['resume', 'stack', 'modules', 'commandes', 'decisions', 'risques'],
};

const VERIFICATION = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    problemes: {
      type: 'array',
      items: {
        type: 'object',
        properties: { gravite: { type: 'string', enum: ['bloquant', 'majeur', 'mineur'] }, description: texte, correction: texte },
        required: ['gravite', 'description', 'correction'],
      },
    },
  },
  required: ['ok', 'problemes'],
};

// Classement : moins de vetos d'abord, puis meilleure moyenne.
function classer(propositions, verdicts) {
  return propositions
    .map((proposition) => {
      const notes = verdicts.flatMap((verdict) => verdict.notes.filter((note) => note.proposition === proposition.cle));
      const moyenne = notes.length ? notes.reduce((somme, note) => somme + note.note, 0) / notes.length : 0;
      const vetos = notes.filter((note) => note.redhibitoire && note.redhibitoire.trim()).length;
      return { cle: proposition.cle, moyenne: Math.round(moyenne * 10) / 10, vetos };
    })
    .sort((a, b) => a.vetos - b.vetos || b.moyenne - a.moyenne);
}

phase('État');
const etat = await chargerEtat();
const contexte = [
  `Projet : ${etat.nom} — ${etat.resume || ''}`,
  `Type : ${etat.type} ; traits : ${liste(etat.traits)}.`,
  `Environnement réel : ${etat.env || 'non relevé'}.`,
  A.contraintes ? `Contraintes supplémentaires : ${A.contraintes}` : '',
  (etat.consignes || []).length ? `Consignes du profil : ${etat.consignes.join(' ')}` : '',
]
  .filter(Boolean)
  .join('\n');

// --- Veille : des faits vérifiés par commande, pas des souvenirs.
phase('Veille');
const VEILLES = [
  {
    cle: 'versions',
    consigne:
      'Détermine les versions stables actuelles des briques candidates (registre réel : npm view <paquet> version peerDependencies, pip index versions, cargo search…) et leurs compatibilités croisées (dépendances paires, version du runtime installé). Ne te fie pas à ta mémoire.',
  },
  {
    cle: 'squelettes',
    consigne:
      'Cherche des squelettes ou starters éprouvés pour ce projet (règle « Skeleton Projects ») ; pour chacun : pertinence, activité récente, sécurité, extensibilité. Dis s’il vaut mieux partir d’un squelette ou de l’échafaudage officiel des outils.',
  },
  {
    cle: 'pieges',
    consigne:
      'Pour les exigences les plus risquées de la SPEC, cherche les pièges connus et les pratiques recommandées récentes (documentation officielle, notes de version, problèmes notoires).',
  },
];
const veille = (
  await parallel(
    selon(VEILLES.slice(0, 1), VEILLES, VEILLES).map((sujet) => () =>
      lancer(
        'eclaireur',
        [
          `Tu es éclaireur technique (sujet : ${sujet.cle}). Lis ${DOCS}/SPEC.md, notamment les contraintes techniques imposées.`,
          contexte,
          sujet.consigne,
          'Chaque constat cite sa source (commande exécutée ou URL). Ne modifie aucun fichier.',
        ].join('\n'),
        { label: `veille:${sujet.cle}`, phase: 'Veille', schema: VEILLE },
      ).then((resultat) => resultat && Object.assign({ sujet: sujet.cle }, resultat)),
    ),
  )
).filter(Boolean);

// --- Propositions : chaque architecte défend son angle sans voir les autres.
phase('Propositions');
const ANGLES = [
  { cle: 'mvp', consigne: 'le plus court chemin vers une démonstration fiable : peu de pièces, peu de dépendances, rien de spéculatif' },
  { cle: 'robustesse', consigne: 'intégrité des données, sécurité et évolutivité : invariants garantis par construction, frontières nettes' },
  { cle: 'experience', consigne: 'expérience utilisateur et performance perçue : fluidité, temps de réponse, modes dégradés, accessibilité' },
];
const propositions = (
  await parallel(
    selon(ANGLES.slice(0, 2), ANGLES, ANGLES).map((angle) => () =>
      lancer(
        'architecte',
        [
          `Tu es architecte, angle « ${angle.cle} » : ${angle.consigne}.`,
          `Lis ${DOCS}/SPEC.md (exigences, invariants, contraintes imposées) puis propose UNE architecture complète, en respectant les contraintes techniques de la SPEC et l’environnement réel.`,
          contexte,
          `Veille (versions réelles, squelettes, pièges) : ${JSON.stringify(veille)}`,
          'Couvre : pile technique (chaque brique avec version et justification), arborescence du dépôt, modèle de données et garantie de chaque invariant, interfaces (API, contrats, modules), stratégie de tests par niveau (unitaire, intégration, bout en bout) avec leurs commandes, sécurité, risques et compromis assumés.',
          'Tu travailles seul : ne cherche pas le consensus, défends ton angle.',
        ].join('\n'),
        { label: `architecte:${angle.cle}`, phase: 'Propositions', schema: PROPOSITION },
      ).then((resultat) => resultat && Object.assign({ cle: angle.cle }, resultat)),
    ),
  )
).filter(Boolean);
if (!propositions.length) throw new Error('atelier-architecture : aucune proposition exploitable.');

// --- Jury : les angles de la règle « Multi-Perspective Analysis ».
phase('Jury');
const cles = propositions.map((proposition) => proposition.cle);
const VERDICT = {
  type: 'object',
  properties: {
    notes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          proposition: { type: 'string', enum: cles },
          note: { type: 'number' },
          forces: textes,
          faiblesses: textes,
          redhibitoire: texte,
        },
        required: ['proposition', 'note', 'forces', 'faiblesses'],
      },
    },
    aGreffer: textes,
  },
  required: ['notes', 'aGreffer'],
};
const JURES = [
  { cle: 'ingenieur-senior', agent: 'expert', consigne: 'faisabilité, simplicité, maintenabilité, coût de mise en œuvre pour une petite équipe' },
  { cle: 'coherence', agent: 'expert', consigne: 'couverture de chaque exigence P0/P1 et de chaque critère d’acceptation de la SPEC ; contradictions internes' },
  { cle: 'securite', agent: 'securite', consigne: 'surface d’attaque, authentification et droits côté serveur, données sensibles, secrets' },
  { cle: 'factuel', agent: 'ouvrier', consigne: 'exactitude des affirmations : versions, compatibilités, capacités réelles des bibliothèques — vérifie par commande ou documentation' },
  { cle: 'redondance', agent: 'expert', consigne: 'sur-ingénierie, doublons, pièces spéculatives à retirer' },
];
const verdicts =
  propositions.length > 1
    ? (
        await parallel(
          selon(JURES.slice(0, 2), JURES.slice(0, 3), JURES).map((jure) => () =>
            lancer(
              jure.agent,
              [
                `Tu es juré (angle : ${jure.cle} — ${jure.consigne}). Lis ${DOCS}/SPEC.md, puis note chaque proposition d’architecture de 0 à 10 sous ton seul angle.`,
                contexte,
                'Pour chacune : forces, faiblesses, et « redhibitoire » seulement si un défaut la disqualifie (sinon chaîne vide). Liste dans « aGreffer » les meilleures idées des propositions perdantes à reprendre. Ne modifie aucun fichier.',
                `Propositions : ${JSON.stringify(propositions)}`,
              ].join('\n'),
              { label: `juré:${jure.cle}`, phase: 'Jury', schema: VERDICT },
            ).then((resultat) => resultat && Object.assign({ jure: jure.cle }, resultat)),
          ),
        )
      ).filter(Boolean)
    : [];
const classement = classer(propositions, verdicts);
const gagnante = propositions.find((proposition) => proposition.cle === classement[0].cle);
log(`Jury : ${classement.map((ligne) => `${ligne.cle} ${ligne.moyenne}/10${ligne.vetos ? ` (${ligne.vetos} veto)` : ''}`).join(' · ')} → « ${gagnante.cle} » retenue.`);

// --- Synthèse : rédaction, puis vérification adverse et révision jusqu'à lever les problèmes graves.
phase('Synthèse');
const consigneRedaction = [
  `Écris ${DOCS}/ARCHITECTURE.md : vue d’ensemble ; pile technique versionnée ; arborescence ; modules, responsabilités et chemins ; modèle de données complet (entités, contraintes, index) et mécanisme garantissant chaque invariant de la SPEC ; contrats d’interface (points d’entrée, schémas, enveloppe de réponse, erreurs) ; sécurité ; stratégie de tests et commandes ; performance ; matrice de traçabilité exigence → module → test.`,
  'Écris aussi docs/adr/NNNN-titre.md : une ADR par décision structurante (contexte, décision, alternatives écartées, conséquences).',
  'Les commandes (install, dev, test, typecheck, lint, build, e2e) doivent s’exécuter telles quelles depuis la racine. « modules[].chemin » est le dossier de chaque module, relatif à la racine.',
].join('\n');
const premiere = await lancer(
  'expert',
  [
    `Tu es l’architecte en chef. La proposition « ${gagnante.cle} » a gagné le jury. Rédige l’architecture définitive à partir d’elle : greffe les meilleures idées des autres et corrige les faiblesses relevées par les jurés.`,
    contexte,
    consigneRedaction,
    `Proposition retenue : ${JSON.stringify(gagnante)}`,
    `Autres propositions : ${JSON.stringify(propositions.filter((proposition) => proposition.cle !== gagnante.cle))}`,
    `Verdicts du jury : ${JSON.stringify(verdicts)}`,
    `Veille : ${JSON.stringify(veille)}`,
  ].join('\n'),
  { label: 'rédaction ARCHITECTURE', phase: 'Synthèse', schema: ARCHITECTURE },
);
if (!premiere) throw new Error('atelier-architecture : la rédaction de l’architecture a échoué.');

const graves = (verification) => (verification ? verification.problemes.filter((probleme) => probleme.gravite !== 'mineur') : []);
const verifier = (tour) =>
  lancer(
    'expert',
    [
      `Tu es le vérificateur adverse de l’architecture. Lis ${DOCS}/SPEC.md et ${DOCS}/ARCHITECTURE.md et cherche à la mettre en défaut :`,
      '1. Chaque paquet et version cités existent-ils réellement, et sont-ils compatibles entre eux et avec le runtime installé (vérifie par commande) ?',
      '2. Chaque exigence P0/P1 et chaque critère d’acceptation global a-t-il un module responsable et un moyen de test ?',
      '3. Chaque invariant de données est-il garanti par un mécanisme précis (contrainte, transaction, verrou, version) plutôt que par une intention ?',
      `4. L’environnement réel est-il respecté ? ${contexte}`,
      'Classe chaque problème (bloquant, majeur, mineur) avec la correction proposée. ok=true seulement sans problème bloquant ni majeur. Ne modifie aucun fichier.',
    ].join('\n'),
    { label: `vérification adverse #${tour}`, phase: 'Synthèse', schema: VERIFICATION },
  );

let architecture = premiere;
let verification = await verifier(1);
for (let tour = 1; graves(verification).length && tour <= selon(1, 1, 2); tour += 1) {
  const reprise = await lancer(
    'expert',
    [
      `Révise ${DOCS}/ARCHITECTURE.md et les ADR pour lever les problèmes ci-dessous, sans dégrader le reste.`,
      consigneRedaction,
      `Problèmes : ${JSON.stringify(graves(verification))}`,
    ].join('\n'),
    { label: `révision ARCHITECTURE #${tour}`, phase: 'Synthèse', schema: ARCHITECTURE },
  );
  if (reprise) architecture = reprise;
  verification = PROFONDEUR === 'eco' ? null : await verifier(tour + 1);
}
if (graves(verification).length) log(`Architecture : ${graves(verification).length} problème(s) grave(s) non levé(s) — voir le résultat.`);

const suivant = marquer(etat, 'architecture', {
  stack: architecture.stack.map((brique) => `${brique.brique} : ${brique.choix} ${brique.version}`.trim()),
  zones: architecture.modules.map((module) => module.chemin),
  commandes: architecture.commandes,
});
const cloture = await cloturer(suivant, 'docs: add architecture and decision records', 'Synthèse', ['stack', 'zones', 'commandes', 'phasesFaites']);

return { etat: suivant, architecture, classement, verification, cloture };
