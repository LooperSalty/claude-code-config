export const meta = {
  name: 'atelier-recette',
  description: 'Atelier 6/6 · Recette : critères d’acceptation prouvés un par un, parcours réels dans l’application lancée, corrections, verdict SHIP / NEEDS WORK / BLOCKED',
  whenToUse: 'En fin de projet, après la revue : produit docs/atelier/RECETTE.md, le README et le verdict de livraison.',
  phases: [
    { title: 'État' },
    { title: 'Vérifications', detail: 'voie vivante (application lancée, une vérification à la fois) et voie statique en parallèle' },
    { title: 'Corrections', detail: 'défauts de recette corrigés puis revérifiés' },
    { title: 'Livraison', detail: 'README, RECETTE.md, verdict, commit' },
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
const PREUVES = `${DOCS}/preuves`;

// Vérifications de recette : les clés sont celles des profils d'atelier-cadrage.js (contrôlé par les tests).
// « vivant » = lance l'application ou un navigateur : ces vérifications passent une par une pour ne pas se disputer les ports.
const RECETTES = {
  criteres: { vivant: false, agent: 'expert', consigne: `Audit des critères d’acceptation : pour CHAQUE exigence P0/P1 et chaque critère global (CA-xx) de ${DOCS}/SPEC.md, trouve la preuve — un test automatisé nommé que tu exécutes réellement, ou une observation reproductible. Un élément par critère dans « resultats ». Sans preuve, le critère est « echec » ou « non-verifiable », jamais validé par défaut.` },
  e2e: { vivant: true, agent: 'e2e', consigne: `Tests de bout en bout : écris (s’ils manquent) et exécute des tests Playwright couvrant le scénario de démonstration et les parcours clés de ${DOCS}/SPEC.md, contre l’application réellement lancée. Captures d’écran dans ${PREUVES}/. Un élément par parcours.` },
  demo: { vivant: true, agent: 'expert', consigne: `Démonstration réelle : lance l’application, puis déroule pas à pas le scénario de démonstration de ${DOCS}/SPEC.md dans un vrai navigateur (Playwright ou MCP navigateur). À chaque étape, prends une capture dans ${PREUVES}/ et REGARDE-la (outil Read sur l’image) : l’écran montre-t-il ce que la SPEC annonce ? Relève aussi les erreurs de console. Un élément par étape.` },
  accessibilite: { vivant: true, agent: 'expert', consigne: 'Accessibilité : passe axe-core (via Playwright) sur chaque écran principal, puis vérifie au clavier seul les parcours clés (ordre de tabulation, focus visible, activation) et que l’information n’est jamais portée par la seule couleur. Un élément par écran ou parcours.' },
  lighthouse: { vivant: true, agent: 'expert', consigne: 'Lighthouse sur le build de production servi localement : performance, accessibilité, bonnes pratiques, SEO, et les Core Web Vitals. Un élément par page et par catégorie sous 90.' },
  responsive: { vivant: true, agent: 'expert', consigne: `Adaptation aux écrans : captures à 375, 768 et 1440 px de large de chaque écran principal dans ${PREUVES}/, que tu regardes ; signale débordements, textes coupés, cibles tactiles trop petites. Un élément par écran et par largeur fautive.` },
  jouabilite: { vivant: true, agent: 'expert', consigne: `Essai de jeu réel : lance le jeu, joue une partie complète par automatisation des entrées (démarrage, contrôles, progression, fin, relance), captures dans ${PREUVES}/ que tu regardes. Vérifie que la boucle de jeu tient, que les entrées répondent, que victoire et défaite sont atteignables. Un élément par phase de jeu.` },
  'budget-fps': { vivant: true, agent: 'expert', consigne: 'Budget de fluidité : dans un navigateur avec accélération matérielle (non headless, ou headless avec GPU), mesure la cadence d’images de la scène principale pendant navigation et interaction (échantillonnage requestAnimationFrame sur au moins 10 s), ainsi que les appels de dessin (renderer.info). Compare au seuil de la SPEC ; précise les conditions de mesure (machine, GPU, taille de scène).' },
  cli: { vivant: false, agent: 'ouvrier', consigne: 'Interface en ligne de commande : exécute chaque commande documentée avec des entrées valides, invalides et limites ; vérifie sorties, messages d’erreur, codes de sortie et aide. Un élément par commande.' },
  'integrite-donnees': { vivant: false, agent: 'expert', consigne: `Intégrité des données : pour chaque invariant de ${DOCS}/SPEC.md (unicité, atomicité, non-négativité, traçabilité, cohérence soldes ↔ mouvements…), écris et exécute un test qui tente de le VIOLER, y compris en concurrence. L’invariant tient-il au niveau de la base ? Un élément par invariant.` },
  'matrice-droits': { vivant: false, agent: 'securite', consigne: `Matrice des droits : pour chaque rôle de ${DOCS}/SPEC.md et chaque action sensible, vérifie par des tests d’intégration que le serveur autorise ou refuse correctement, sans se fier à l’interface (appels directs à l’API, y compris non authentifiés). Un élément par couple rôle-action fautif, ou un élément global si tout tient.` },
  api: { vivant: false, agent: 'ouvrier', consigne: `Contrat d’API : pour chaque point d’entrée de ${DOCS}/ARCHITECTURE.md, vérifie par des tests en processus (sans port réseau) les codes de statut, la validation des entrées, l’enveloppe de réponse et les messages d’erreur. Un élément par point d’entrée fautif, ou un élément global si tout tient.` },
  installation: { vivant: true, agent: 'ouvrier', consigne: 'Installation à blanc : copie le dépôt (fichiers suivis par git uniquement) dans un dossier temporaire hors projet, puis suis le README à la lettre — installation, build, tests, démarrage — sans rien ajouter de tête. Chaque étape qui échoue ou manque au README est un défaut. Nettoie le dossier temporaire à la fin.' },
};

if (A.__tables) return { RECETTES: Object.keys(RECETTES) };

const VERIFICATION = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    resultats: {
      type: 'array',
      items: {
        type: 'object',
        properties: { element: texte, statut: { type: 'string', enum: ['valide', 'partiel', 'echec', 'non-verifiable'] }, preuve: texte },
        required: ['element', 'statut', 'preuve'],
      },
    },
    echecs: {
      type: 'array',
      items: {
        type: 'object',
        properties: { titre: texte, gravite: { type: 'string', enum: ['critique', 'haute', 'moyenne', 'basse'] }, description: texte, piste: texte },
        required: ['titre', 'gravite', 'description'],
      },
    },
  },
  required: ['ok', 'resultats', 'echecs'],
};

phase('État');
const etat = await chargerEtat();
const demandees = (Array.isArray(A.verifications) && A.verifications.length ? A.verifications : etat.recette || []).filter((cle) => RECETTES[cle]);
const toutes = demandees.length ? demandees : ['criteres'];
const cles = selon(toutes.filter((cle) => cle !== 'installation'), toutes, toutes);
if (cles.length < toutes.length) log('Profondeur eco : installation à blanc non vérifiée.');

const enEchec = (verification) => !verification.ok || verification.echecs.some((echec) => echec.gravite === 'critique' || echec.gravite === 'haute');

async function verifier(cle, tour) {
  const recette = RECETTES[cle];
  const resultat = await lancer(
    recette.agent,
    [
      `Tu es recetteur (vérification « ${cle} »). ${recette.consigne}`,
      `Projet : ${etat.nom} (${etat.type}). Commandes : ${JSON.stringify(etat.commandes || {})}. Références : ${DOCS}/SPEC.md, ${DOCS}/ARCHITECTURE.md.`,
      recette.vivant
        ? 'Tu es le seul à lancer l’application en ce moment : démarre-la toi-même, et ARRÊTE tout ce que tu as lancé (serveurs, navigateurs) avant de rendre la main. Si un port est pris, choisis-en un autre plutôt que de tuer un processus qui n’est pas le tien.'
        : 'D’autres vérifications tournent en parallèle : n’ouvre aucun port réseau et ne lance pas l’application ; travaille par tests en processus.',
      'Tu ne corriges pas le produit : tu constates. Les seuls fichiers que tu peux ajouter sont des tests et des preuves. ok=true seulement si tout ce que tu as vérifié tient réellement ; chaque défaut va dans « echecs » avec sa gravité et une piste.',
      `Skills utiles : ${liste(etat.skills)}.`,
    ].join('\n'),
    { label: `recette:${cle}${tour > 1 ? ` #${tour}` : ''}`, phase: tour > 1 ? 'Corrections' : 'Vérifications', schema: VERIFICATION },
  );
  const defaut = { ok: false, resultats: [], echecs: [{ titre: `vérification « ${cle} » sans réponse`, gravite: 'haute', description: 'le recetteur n’a pas répondu', piste: 'relancer la vérification' }] };
  return Object.assign({ cle }, resultat || defaut);
}

// Une campagne : la voie vivante passe ses vérifications une à une, pendant que la voie statique tourne en parallèle.
async function campagne(selection, tour) {
  const vivantes = selection.filter((cle) => RECETTES[cle].vivant);
  const statiques = selection.filter((cle) => !RECETTES[cle].vivant);
  const voieVivante = async () => {
    let sortie = [];
    for (const cle of vivantes) sortie = sortie.concat(await verifier(cle, tour));
    return sortie;
  };
  const groupes = await parallel([voieVivante].concat(statiques.map((cle) => () => verifier(cle, tour).then((resultat) => [resultat]))));
  return groupes.filter(Boolean).flat();
}

phase('Vérifications');
let verifications = await campagne(cles, 1);

// --- Corrections : un correcteur à la fois (ils peuvent toucher les mêmes fichiers), puis seules les vérifications rouges repassent.
for (let tour = 1; tour <= selon(0, 1, 2); tour += 1) {
  const rouges = verifications.filter(enEchec);
  if (!rouges.length) break;
  phase('Corrections');
  for (const rouge of rouges) {
    await lancer(
      'expert',
      [
        `La recette « ${rouge.cle} » a relevé des défauts. Corrige-les en TDD (test de non-régression qui échoue d’abord, puis correction minimale), sans affaiblir aucun test existant. Tu es seul sur le dépôt.`,
        `Défauts : ${JSON.stringify(rouge.echecs)}`,
        `Résultats en échec : ${JSON.stringify(rouge.resultats.filter((resultat) => resultat.statut !== 'valide'))}`,
        'Si un « défaut » est en réalité une erreur de la vérification elle-même, corrige le test fautif et explique-le. Git : aucun commit.',
      ].join('\n'),
      { label: `correction recette:${rouge.cle} #${tour}`, phase: 'Corrections' },
    );
  }
  await stabiliser(etat, `recette, corrections #${tour}`, 'Corrections');
  const refaites = await campagne(rouges.map((rouge) => rouge.cle), tour + 1);
  verifications = verifications.map((verification) => refaites.find((refaite) => refaite.cle === verification.cle) || verification);
}

// --- Livraison : contrôle final, verdict mécanique, documents.
phase('Livraison');
const controle = await stabiliser(etat, 'recette finale', 'Livraison');
const rougesFinales = verifications.filter(enEchec);
const verdict = !controle.ok ? 'BLOCKED' : rougesFinales.length ? 'NEEDS WORK' : 'SHIP';
log(`Verdict : ${verdict} — ${verifications.length - rougesFinales.length}/${verifications.length} vérification(s) au vert${controle.ok ? '' : ', intégration rouge'}.`);

await lancer(
  'documentaliste',
  [
    'Rédige les documents de livraison, sans rien inventer ni embellir :',
    `- README.md à la racine : ce que fait le projet, prérequis, installation, lancement, tests, scénario de démonstration pas à pas, structure du dépôt, liens vers ${DOCS}/SPEC.md et ${DOCS}/ARCHITECTURE.md. Chaque commande citée doit exister et fonctionner.`,
    `- ${DOCS}/RECETTE.md : le verdict « ${verdict} » en tête ; la matrice des critères d’acceptation (critère, statut, preuve) ; le résultat de chaque vérification ; les défauts restants avec gravité et piste ; les limites connues ; l’état du contrôle d’intégration.`,
    `Vérifications : ${JSON.stringify(verifications)}`,
    `Contrôle d’intégration : ${JSON.stringify(controle)}`,
  ].join('\n'),
  { label: 'documents de livraison', phase: 'Livraison' },
);
const suivant = marquer(etat, 'recette');
const cloture = await cloturer(suivant, 'docs: add acceptance report and README', 'Livraison');
const criteres = (verifications.find((verification) => verification.cle === 'criteres') || { resultats: [] }).resultats;

return {
  verdict,
  etat: suivant,
  verifications: verifications.map((verification) => ({
    cle: verification.cle,
    ok: !enEchec(verification),
    echecs: verification.echecs.filter((echec) => echec.gravite === 'critique' || echec.gravite === 'haute').map((echec) => echec.titre),
  })),
  criteres,
  controle,
  cloture,
};
