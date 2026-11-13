export const meta = {
  name: 'atelier-revue',
  description: 'Atelier 5/6 · Revue : relecteurs par lentille, contre-expertise adverse de chaque constat, correction en TDD des défauts confirmés',
  whenToUse: 'Après le build (ou sur un projet existant doté de docs/atelier/etat.json), avant la recette.',
  phases: [
    { title: 'État' },
    { title: 'Constats', detail: 'un relecteur par lentille et par zone' },
    { title: 'Contre-expertise', detail: 'chaque constat doit survivre à une tentative de réfutation' },
    { title: 'Corrections', detail: 'test de non-régression d’abord, correction, contrôle, commit' },
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
const gravite = { type: 'string', enum: ['critique', 'haute', 'moyenne', 'basse'] };

// Lentilles de revue : les clés sont celles des profils d'atelier-cadrage.js (contrôlé par les tests).
const LENTILLES = {
  correction: { agent: 'relecteur', consigne: 'bugs de logique, cas limites, gestion d’erreurs, mutations d’état, valeurs en dur, fichiers de plus de 800 lignes, fonctions de plus de 50 lignes' },
  securite: { agent: 'securite', consigne: 'liste « Mandatory Security Checks » de rules/security.md : secrets, validation des entrées, injection, XSS, CSRF, authentification et droits vérifiés côté serveur, limitation de débit, fuites dans les messages d’erreur' },
  conformite: { agent: 'expert', consigne: `conformité à ${DOCS}/SPEC.md : pour chaque exigence P0/P1 et chaque critère d’acceptation, le comportement est-il réellement implémenté ET testé ? Signale tout écart, oubli ou comportement contraire` },
  tests: { agent: { agentType: 'agent-skills:test-engineer' }, consigne: 'qualité des tests : assertions qui prouvent quelque chose, cas limites et chemins d’erreur manquants, tests fragiles, couverture sous 80 %, équilibre unitaire / intégration / bout en bout' },
  'echecs-silencieux': { agent: { agentType: 'ecc:silent-failure-hunter' }, consigne: 'erreurs avalées, replis silencieux, promesses non attendues, valeurs par défaut qui masquent une panne' },
  performance: { agent: { agentType: 'ecc:performance-optimizer' }, consigne: 'goulots : requêtes en boucle, travail répété, rendus inutiles, charges utiles trop grosses, fuites mémoire' },
  'performance-3d': { agent: 'expert', consigne: 'rendu 3D (charge les skills threejs-performance et threejs-react-three-fiber) : appels de dessin, instancing, allocations dans la boucle de rendu, re-rendus React de la scène, géométries et matériaux non libérés, sélection et survol coûteux' },
  accessibilite: { agent: { agentType: 'ecc:a11y-architect' }, consigne: 'WCAG 2.2 : navigation clavier, focus visible, noms accessibles, contrastes, information jamais portée par la seule couleur, équivalent non visuel des vues graphiques' },
  donnees: { agent: { agentType: 'ecc:database-reviewer' }, consigne: 'schéma, contraintes, index, migrations, transactions : chaque invariant de la SPEC est-il garanti par la base ?' },
  concurrence: { agent: 'expert', consigne: 'accès concurrents : conditions de course, lectures-écritures non atomiques, idempotence des écritures, verrous et versions optimistes' },
  seo: { agent: { agentType: 'ecc:seo-specialist' }, consigne: 'SEO technique : balises, données structurées, sitemap, robots, liens, performances de chargement' },
  contenu: { agent: 'expert', consigne: 'textes de l’interface et du site (charge antislop-copywriting) : clarté, justesse, ton, formules creuses, fautes' },
  jouabilite: { agent: 'expert', consigne: 'boucle de jeu, pas de temps, entrées, états (menu, partie, pause, fin), équilibrage évident, retours visuels et sonores' },
  types: { agent: { agentType: 'ecc:type-design-analyzer' }, consigne: 'conception des types : invariants exprimés par les types, états impossibles représentables, any et conversions forcées' },
};

const RELECTEURS_LANGAGE = [
  { nom: 'TypeScript', motif: /typescript|javascript|node/i, agentType: 'ecc:typescript-reviewer' },
  { nom: 'React', motif: /react|next/i, agentType: 'ecc:react-reviewer' },
  { nom: 'Python', motif: /python|django|fastapi/i, agentType: 'ecc:python-reviewer' },
  { nom: 'Rust', motif: /rust|bevy/i, agentType: 'ecc:rust-reviewer' },
  { nom: 'Go', motif: /golang|\bgo\b/i, agentType: 'ecc:go-reviewer' },
  { nom: 'Vue', motif: /vue|nuxt/i, agentType: 'ecc:vue-reviewer' },
  { nom: 'Kotlin', motif: /kotlin|android/i, agentType: 'ecc:kotlin-reviewer' },
  { nom: 'Swift', motif: /swift|ios/i, agentType: 'ecc:swift-reviewer' },
  { nom: 'C++', motif: /c\+\+|cpp|unreal/i, agentType: 'ecc:cpp-reviewer' },
  { nom: 'C#', motif: /c#|csharp|\.net|unity/i, agentType: 'ecc:csharp-reviewer' },
  { nom: 'Flutter', motif: /flutter|dart/i, agentType: 'ecc:flutter-reviewer' },
];

if (A.__tables) return { LENTILLES: Object.keys(LENTILLES) };

const CONSTATS = {
  type: 'object',
  properties: {
    constats: {
      type: 'array',
      items: {
        type: 'object',
        properties: { titre: texte, gravite, fichier: texte, ligne: { type: 'number' }, preuve: texte, scenario: texte, correction: texte },
        required: ['titre', 'gravite', 'fichier', 'preuve', 'scenario'],
      },
    },
  },
  required: ['constats'],
};

const VERDICTS = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: { id: texte, reel: { type: 'boolean' }, raison: texte, gravite },
        required: ['id', 'reel', 'raison'],
      },
    },
  },
  required: ['verdicts'],
};

const CORRECTION = {
  type: 'object',
  properties: {
    corriges: textes,
    reports: { type: 'array', items: { type: 'object', properties: { id: texte, raison: texte }, required: ['id', 'raison'] } },
    fichiers: textes,
    tests: textes,
  },
  required: ['corriges', 'reports', 'fichiers', 'tests'],
};

const RANG = { critique: 0, haute: 1, moyenne: 2, basse: 3 };
const normaliser = (chemin) => String(chemin || '').replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase();
const zoneDe = (fichier) => normaliser(fichier).split('/').slice(0, 2).join('/') || '.';
const cleDe = (constat) =>
  constat.ligne ? `${normaliser(constat.fichier)}#${Math.floor(constat.ligne / 8)}` : `${normaliser(constat.fichier)}#${String(constat.titre).toLowerCase().slice(0, 24)}`;

// Fusionne les constats qui visent le même endroit : garde le plus grave, conserve les autres formulations.
function fusionner(constats) {
  const groupes = constats.reduce((acc, constat) => {
    const cle = cleDe(constat);
    return Object.assign({}, acc, { [cle]: (acc[cle] || []).concat(constat) });
  }, {});
  return Object.keys(groupes).map((cle) => {
    const tries = groupes[cle].slice().sort((a, b) => RANG[a.gravite] - RANG[b.gravite]);
    return Object.assign({}, tries[0], {
      cle,
      lentilles: Array.from(new Set(tries.map((constat) => constat.lentille))),
      aussi: tries.slice(1).map((constat) => `${constat.titre} — ${constat.scenario}`),
    });
  });
}

function lotir(constats, taille) {
  const parZone = constats.reduce((acc, constat) => {
    const zone = zoneDe(constat.fichier);
    return Object.assign({}, acc, { [zone]: (acc[zone] || []).concat(constat) });
  }, {});
  return Object.keys(parZone).flatMap((zone) =>
    Array.from({ length: Math.ceil(parZone[zone].length / taille) }, (_vide, index) => ({
      zone,
      constats: parZone[zone].slice(index * taille, (index + 1) * taille),
    })),
  );
}

phase('État');
const etat = await chargerEtat();
const demandees = (Array.isArray(A.lentilles) && A.lentilles.length ? A.lentilles : etat.lentilles || []).filter((cle) => LENTILLES[cle]);
const toutes = demandees.length ? demandees : ['correction', 'securite', 'conformite'];
const retenues = selon(toutes.slice(0, 3), toutes, toutes);
if (retenues.length < toutes.length) log(`Profondeur eco : lentilles non passées — ${toutes.slice(retenues.length).join(', ')}.`);
const zones = PROFONDEUR === 'max' && (etat.zones || []).length > 1 ? etat.zones : ['.'];
const langages = RELECTEURS_LANGAGE.filter(({ motif }) => motif.test((etat.stack || []).join(' '))).slice(0, selon(0, 2, 4));
const postesComplets = retenues
  .flatMap((cle) => zones.map((zone) => ({ cle, zone, agent: LENTILLES[cle].agent, consigne: LENTILLES[cle].consigne })))
  .concat(langages.map(({ nom, agentType }) => ({ cle: `langage:${nom}`, zone: '.', agent: { agentType }, consigne: `idiomes, pièges et sûreté propres à ${nom}` })));

async function chercher(postes, perimetre) {
  const trouvailles = await parallel(
    postes.map((poste) => () =>
      lancer(
        poste.agent,
        [
          `Tu es relecteur, lentille « ${poste.cle} » : ${poste.consigne}.`,
          `Périmètre : ${perimetre || (poste.zone === '.' ? 'tout le projet' : `le dossier ${poste.zone}`)}${A.base ? ` — changements depuis ${A.base} (git diff ${A.base}...HEAD)` : ''}. Références : ${DOCS}/SPEC.md, ${DOCS}/ARCHITECTURE.md.`,
          'Ne modifie aucun fichier. Lis réellement le code ; exécute des tests ou des commandes de lecture si cela aide à prouver un constat.',
          'Ne rapporte que des défauts réels et précis : fichier, ligne, preuve (extrait ou sortie), scénario concret (entrée ou état → mauvais résultat), correction proposée. Gravité : critique (perte ou corruption de données, faille exploitable, fonction P0 cassée), haute, moyenne, basse. Ni remarque de style pur, ni généralité.',
          `Skills utiles : ${liste(etat.skills)}.`,
        ].join('\n'),
        { label: `constats:${poste.cle}${poste.zone === '.' ? '' : `@${poste.zone}`}`, phase: 'Constats', schema: CONSTATS },
      ).then((resultat) => (resultat ? resultat.constats.map((constat) => Object.assign({ lentille: poste.cle }, constat)) : [])),
    ),
  );
  return trouvailles.filter(Boolean).flat();
}

// Contre-expertise : un constat est confirmé s'il n'est pas réfuté par la majorité des contre-experts.
async function contreExpertiser(constats) {
  const angles = selon(['contre-expertise'], ['contre-expertise'], ['contre-expertise', 'reproduction']);
  const verdicts = (
    await parallel(
      lotir(constats, 5).flatMap((lot) =>
        angles.map((angle) => () =>
          lancer(
            'expert',
            [
              `Tu es contre-expert (angle : ${angle}). Pour chaque constat ci-dessous, cherche à le RÉFUTER en lisant le code réel${angle === 'reproduction' ? ' et en tentant de le reproduire par un test ciblé ou une exécution' : ''} : garde-fou existant ailleurs, cas impossible, mauvaise lecture, test qui prouve le contraire.`,
              'Un constat n’est « réel » que si tu as toi-même constaté le défaut. Dans le doute : reel=false. Tu peux réviser la gravité. Ne modifie aucun fichier du projet (un test jetable hors dépôt est permis).',
              `Constats : ${JSON.stringify(lot.constats.map(({ id, titre, gravite: niveau, fichier, ligne, preuve, scenario, aussi }) => ({ id, titre, gravite: niveau, fichier, ligne, preuve, scenario, aussi })))}`,
            ].join('\n'),
            { label: `${angle}:${lot.zone}`, phase: 'Contre-expertise', schema: VERDICTS },
          ),
        ),
      ),
    )
  )
    .filter(Boolean)
    .flatMap((resultat) => resultat.verdicts);
  return constats.map((constat) => {
    const votes = verdicts.filter((verdict) => verdict.id === constat.id);
    const reels = votes.filter((verdict) => verdict.reel);
    const confirme = votes.length === 0 || (reels.length > 0 && reels.length >= votes.length - reels.length);
    const revisee = reels.find((verdict) => verdict.gravite);
    return Object.assign({}, constat, {
      confirme,
      nonVerifie: votes.length === 0,
      gravite: revisee ? revisee.gravite : constat.gravite,
      raisons: votes.map((verdict) => verdict.raison),
    });
  });
}

async function corrigerLots(confirmes) {
  const lots = lotir(confirmes, 6);
  const comptesRendus = (
    await parallel(
      lots.map((lot) => () => {
        const grave = lot.constats.some((constat) => RANG[constat.gravite] <= 1);
        const voisins = lots.filter((autre) => autre !== lot).map((autre) => autre.zone);
        return lancer(
          grave ? 'expert' : 'ouvrier',
          [
            `Corrige ces défauts confirmés de la zone « ${lot.zone} », en TDD : pour chacun, écris d’abord un test de non-régression qui échoue, puis la correction minimale, puis vérifie avec les tests ciblés.`,
            `D’autres correcteurs travaillent en même temps sur : ${liste(Array.from(new Set(voisins)))}. Ne touche pas à leurs fichiers ; si une correction l’exige, ou si tu ne parviens pas à reproduire un défaut, ne force pas : mets-le dans « reports » avec la raison.`,
            'Git : aucun commit. N’installe rien. Ne lance aucun serveur persistant.',
            `Défauts : ${JSON.stringify(lot.constats.map(({ id, titre, gravite: niveau, fichier, ligne, preuve, scenario, correction, aussi }) => ({ id, titre, gravite: niveau, fichier, ligne, preuve, scenario, correction, aussi })))}`,
          ].join('\n'),
          { label: `correction:${lot.zone}`, phase: 'Corrections', schema: CORRECTION },
        );
      }),
    )
  ).filter(Boolean);
  const reports = comptesRendus.flatMap((compteRendu) => compteRendu.reports);
  if (!reports.length) return comptesRendus;
  const transverse = await lancer(
    'expert',
    [
      'Tu es seul sur le dépôt. Traite les défauts que les correcteurs parallèles ont reportés (changement transverse ou reproduction difficile) : test de non-régression d’abord, correction minimale ensuite. Reporte à ton tour ce qui se révèle ne pas être un défaut, en expliquant pourquoi.',
      `Reportés : ${JSON.stringify(reports)}`,
      `Défauts d’origine : ${JSON.stringify(confirmes.filter((constat) => reports.some((report) => report.id === constat.id)).map(({ id, titre, fichier, ligne, scenario, correction }) => ({ id, titre, fichier, ligne, scenario, correction })))}`,
    ].join('\n'),
    { label: 'correction transverse', phase: 'Corrections', schema: CORRECTION },
  );
  return transverse ? comptesRendus.concat(transverse) : comptesRendus;
}

// --- Tours de revue : le premier couvre tout ; le second (profondeur max) ne relit que ce que les corrections ont touché.
const seuilVerification = selon(1, 2, 2);
const seuilCorrection = selon(1, 2, 2);
let vus = [];
let bilan = [];
let fichiersCorriges = [];
let idsCorriges = [];
for (let tour = 1; tour <= selon(1, 1, 2); tour += 1) {
  phase('Constats');
  const postes = tour === 1 ? postesComplets : postesComplets.filter((poste) => poste.zone === '.' && ['correction', 'securite'].includes(poste.cle));
  const perimetre = tour === 1 ? '' : `uniquement les fichiers modifiés par les corrections du tour précédent : ${fichiersCorriges.join(', ')}`;
  if (tour > 1 && (!postes.length || !fichiersCorriges.length)) break;
  const neufs = fusionner(await chercher(postes, perimetre)).filter((constat) => !vus.includes(constat.cle));
  if (!neufs.length) break;
  vus = vus.concat(neufs.map((constat) => constat.cle));
  const numerotes = neufs
    .slice()
    .sort((a, b) => RANG[a.gravite] - RANG[b.gravite])
    .map((constat, index) => Object.assign({}, constat, { id: `C${tour}-${index + 1}`, tour }));
  const aVerifier = numerotes.filter((constat) => RANG[constat.gravite] <= seuilVerification);
  const laisses = numerotes.filter((constat) => RANG[constat.gravite] > seuilVerification);
  if (laisses.length) log(`Tour ${tour} : ${laisses.length} constat(s) de faible gravité consignés sans contre-expertise.`);

  phase('Contre-expertise');
  const juges = aVerifier.length ? await contreExpertiser(aVerifier) : [];
  bilan = bilan.concat(juges, laisses.map((constat) => Object.assign({}, constat, { confirme: false, nonVerifie: true, raisons: [] })));
  const aCorriger = juges.filter((constat) => constat.confirme && RANG[constat.gravite] <= seuilCorrection);
  log(`Tour ${tour} : ${numerotes.length} constat(s), ${juges.filter((constat) => constat.confirme).length} confirmé(s), ${juges.filter((constat) => !constat.confirme).length} réfuté(s), ${aCorriger.length} à corriger.`);
  if (!aCorriger.length) break;

  phase('Corrections');
  const comptesRendus = await corrigerLots(aCorriger);
  idsCorriges = idsCorriges.concat(comptesRendus.flatMap((compteRendu) => compteRendu.corriges));
  fichiersCorriges = Array.from(new Set(comptesRendus.flatMap((compteRendu) => compteRendu.fichiers)));
}

const controle = await stabiliser(etat, 'après revue', 'Corrections');
const confirmes = bilan.filter((constat) => constat.confirme);
const refutes = bilan.filter((constat) => !constat.confirme && !constat.nonVerifie);
const nonVerifies = bilan.filter((constat) => constat.nonVerifie && !constat.confirme);
const corriges = confirmes.filter((constat) => idsCorriges.includes(constat.id));
const restants = confirmes.filter((constat) => !idsCorriges.includes(constat.id));
const resume = (constat) => ({ id: constat.id, gravite: constat.gravite, titre: constat.titre, fichier: constat.fichier, ligne: constat.ligne, lentilles: constat.lentilles });

await lancer(
  'documentaliste',
  [
    `Écris ${DOCS}/REVUE.md à partir du bilan ci-dessous, sans rien inventer : synthèse chiffrée ; défauts confirmés et corrigés ; défauts confirmés restants (avec gravité, fichier, scénario et correction proposée) ; constats réfutés et pourquoi ; constats de faible gravité non vérifiés ; état du contrôle d’intégration.`,
    `Bilan : ${JSON.stringify({ corriges, restants, refutes: refutes.map((constat) => Object.assign(resume(constat), { raisons: constat.raisons })), nonVerifies: nonVerifies.map(resume), controle })}`,
  ].join('\n'),
  { label: 'rapport de revue', phase: 'Corrections' },
);
const suivant = marquer(etat, 'revue');
const cloture = await cloturer(suivant, 'fix: address confirmed review findings', 'Corrections', ['phasesFaites']);

return {
  ok: controle.ok && !restants.some((constat) => RANG[constat.gravite] <= 1),
  etat: suivant,
  confirmes: confirmes.map(resume),
  refutes: refutes.map(resume),
  corriges: corriges.map(resume),
  restants: restants.map(resume),
  nonVerifies: nonVerifies.map(resume),
  controle,
  cloture,
};
