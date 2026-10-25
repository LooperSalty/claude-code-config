# agent-link

Un mod Claude Code qui aide les agents d'une session (fil principal, subagents, teammates) à se coordonner, et qui affiche leurs échanges.

## Ce qu'il ajoute

- **`mcp__agent-link__roster`** : liste les agents de la session (id, nom pour SendMessage, type, statut, parent, tâche) et dit à l'appelant qui il est.
- **`mcp__agent-link__board`** : un tableau partagé. `action=post` (topic + texte court) et `action=read` (tout, un topic, ou `since` un horodatage). Limité aux 200 derniers posts.
- **Consignes de coordination** : chaque nouveau subagent ou teammate reçoit à la fin de sa tâche un court paragraphe qui lui explique roster, board et SendMessage. Les reprises, les forks et les agents de workflow ne la reçoivent pas. Le fil principal reçoit une courte section dans son prompt système.
- **Journal** : chaque message SendMessage (envoyé ou reçu) et chaque post du board est noté (200 lignes maximum).
- **`/agents-chat`** : ouvre le pane « Agents », avec la liste des agents et leur statut (clique pour basculer sur leur vue) puis les derniers échanges. `/agents-chat clear` vide le board et le journal.
- **Ligne de statut** : `🔗 actifs/total agents · msgs · posts`, visible seulement tant qu'un agent existe.
- **Relais de session (`/relais`)** : à la place du résumé de compaction (seuil automatique ou `/compact`), le mod archive tout l'historique **mot pour mot** dans `~/.claude/agent-link/handoffs/<session>/` (un fichier markdown, coupé en parties au-delà d'un million de caractères). Il repart ensuite d'un contexte neuf qui contient les demandes de l'utilisateur telles quelles, les fichiers modifiés, les derniers échanges (environ 20 000 caractères) et le chemin de l'archive, à consulter avec Grep au lieu de deviner. Aucun appel au modèle de résumé : la compaction est instantanée et rien n'est paraphrasé. Un relais suivant reprend les demandes et les fichiers du précédent. `/relais on` (par défaut) : tout passe par le relais. `/relais auto` : seulement le seuil automatique, `/compact` garde le résumé classique. `/relais off` : compaction classique partout. `/relais` seul : mode actuel et liste des relais. Si l'écriture de l'archive échoue, la compaction classique prend le relais.
