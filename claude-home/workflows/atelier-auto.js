export const meta = {
  name: 'atelier-auto',
  description: 'Atelier · Chaîne complète sans supervision : cadrage → architecture → plan → build de toutes les vagues → revue → recette',
  whenToUse: 'Pour dérouler tout le pipeline de création de projet d’une traite. Pour garder la main entre les phases, préférer la commande /atelier.',
  phases: [
    { title: 'Cadrage' },
    { title: 'Architecture' },
    { title: 'Plan' },
    { title: 'Build' },
    { title: 'Revue' },
    { title: 'Recette' },
  ],
};

// Enchaîne les six phases de l'Atelier. Chaque phase reste un workflow autonome, appelé ici par son nom
// (workflow() n'autorise qu'un niveau d'imbrication : les phases n'appellent elles-mêmes aucun workflow).
const options = args && typeof args === 'object' && !Array.isArray(args) ? args : {};
const communs = { profondeur: options.profondeur, agents: options.agents };
const arret = (etape, detail) => ({ verdict: 'BLOCKED', etape, detail });

phase('Cadrage');
const cadrage = await workflow('atelier-cadrage', typeof args === 'string' ? args : options);
if (!cadrage || !cadrage.etat) return arret('cadrage', cadrage);

phase('Architecture');
const architecture = await workflow(
  'atelier-architecture',
  Object.assign({}, communs, { etat: cadrage.etat, contraintes: options.contraintes }),
);
if (!architecture || !architecture.etat) return arret('architecture', architecture);

phase('Plan');
const planification = await workflow('atelier-plan', Object.assign({}, communs, { etat: architecture.etat }));
if (!planification || !planification.plan) return arret('plan', planification);
if (planification.erreurs && planification.erreurs.length) return arret('plan', planification.erreurs);

phase('Build');
let etat = planification.etat;
let vaguesLivrees = [];
for (let tour = 0; tour < planification.plan.vagues.length; tour += 1) {
  const build = await workflow(
    'atelier-build',
    Object.assign({}, communs, { etat, plan: planification.plan, vagues: 'suivante' }),
  );
  if (!build || !build.ok) return arret('build', build);
  etat = build.etat;
  vaguesLivrees = vaguesLivrees.concat(build.vagues.map((vague) => vague.id));
  log(`Vague(s) livrée(s) : ${vaguesLivrees.join(', ')} — restantes : ${build.restantes.join(', ') || 'aucune'}`);
  if (!build.restantes.length) break;
}

phase('Revue');
const revue = await workflow('atelier-revue', Object.assign({}, communs, { etat }));
const etatApresRevue = (revue && revue.etat) || etat;

phase('Recette');
const recette = await workflow('atelier-recette', Object.assign({}, communs, { etat: etatApresRevue }));
if (!recette) return arret('recette', null);

return {
  verdict: recette.verdict,
  etat: recette.etat || etatApresRevue,
  hypotheses: cadrage.hypotheses,
  questionsOuvertes: cadrage.questionsOuvertes,
  vaguesLivrees,
  revue: revue ? { confirmes: revue.confirmes.length, corriges: revue.corriges.length, restants: revue.restants } : null,
  recette: { verifications: recette.verifications, criteres: recette.criteres },
};
