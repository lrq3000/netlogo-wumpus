;=== Reinforcement learning to solve the "Hunt the Wumpus" game
;=== par LARROQUE Stephen and GILBERT Hugo

;=== DECLARING VARIABLES AND CLASSES ===

extensions[matrix] ; for the neural network

; Variables globales
globals[
  count-eaten count-fell count-killed-monster count-won
  reward-bonus penalty-eaten penalty-fell penalty-fear reward-bounty reward-bounty-available? reward-bounty-exploration reward-bounty-exploration-available? penalty-backtrack penalty-step ; reward-bounty = quand monstre est tué
  tdval-start qval-start sqval-start
  qval-prev-patch-pxcor qval-prev-patch-pycor qval-prev-action
  sqval-prev-patch-pxcor sqval-prev-patch-pycor sqval-prev-action sqval-prev-iteration
  nqval-prev-max nqval-prev-action nqval-prev-X nqval-prev-A nqval-prev-Z nqval-megaX nqval-megaY nqval-megaTheta nqval-neurons_per_layer nbactions nbfeatures replay-last-ticks replay-count
  playmode playmode-direction
  pup pdown pleft pright parrowup parrowdown parrowleft parrowright
  justfinished ; détail d'implémentation: utilisé pour le mode harder et pour pouvoir appeler resetup-harder, car il faut etre en contexte observateur
  ;fogofwar-pvisible ; liste des patches visibles (deja visité) pendant fogofwar
  monster-prev-patch ; memoire de la position du monstre pour le respawn entre deux parties en mode simple (quand l'environnement reste le même), sinon l'agent ne peut pas apprendre à tirer sur le monstre s'il disparait définitivement après le premier tir qui le tue
  gen steps found-treasure ; nombre de générations (parties de jeu) et nombre de pas de l'agent pour la partie en cours
  prev-loss prev-won prev-gen ; pour le plot win-loss ratio
  chicken-path ; pour Chicken Search, memorise le chemin a prendre, sinon il va recalculer le chemin a chaque pas et va boucler car il peut y aller de plusieurs facons differentes qui sont tous d'egal distance
  ]

; Types d'agents dispo
breed[monsters monster] ; Wumpus
breed[explorers explorer] ; Agent explorateur ou joueur
breed[pits pit] ; Trou d'air
breed[treasures treasure] ; Trésor
breed[happyfaces happyface] ; juste pour le playmode, sorte d'écran "You Won!"

; Variables spécifiques à chaque type d'agent
patches-own[
  breeze stench glitter ; sens
  tdval ; td-learning
  qval-up qval-down qval-left qval-right qval-arrow-up qval-arrow-down qval-arrow-left qval-arrow-right ; q-learning (liste des actions à chaque case)
  sqval-up sqval-down sqval-left sqval-right sqval-arrow-up sqval-arrow-down sqval-arrow-left sqval-arrow-right
  reward ; récompense ou pénalité d'être sur cette case
  visited safe monster-threat-score pits-threat-score prev-unknown-pxcor prev-unknown-pycor ; heuristique simplificatrice d'une mixture de filtres particulaires permettant de définir la "probabilité" qu'un danger se trouve sur cette case
  path-tocheck path-checked
  visited-count ; pour penalty-backtrack, uniquement pour NQ-Learning
  global-score ; une aggrégation de plusieurs scores pour définir l'intéret et le danger pour l'agent sur ce patch
  ]

explorers-own[arrow-left heard-scream]

;=== INITIALIZATION PROCEDURES ===

; Initialisation du Play Mode
to setup-play-mode
  ; Initialise normalement
  setup
  ; Puis initialise l'écran "You Won" mais en caché
  setup-happyface
  ; Active le play mode
  set playmode true
  ; Affiche un message d'introduction
  user-message "Hello adventurer! You are lost in a labyrinth, searching for a fabulous treasure one can only dream of. Find it! But beware of the Wumpus fangs and try to keep your feet out of the bottomless pits!"
end

; Initialisation de l'écran "You Won", seulement en Play Mode
to setup-happyface
  create-happyfaces 1 [
    set shape "face crown"
    set color yellow
    setxy (max-pxcor / 2) (max-pycor / 2)
    set size 5
    hide-turtle
  ]
end

; Initialisation de la seconde version du problème, appelé pour la variation plus difficile
; Réinitialise aléatoirement les positions des agents à chaque fois que l'agent perd/gagne
to resetup-harder
  clear-turtles
  setup-patches
  setup-explorers
  setup-monsters
  setup-treasures
  setup-pits
  reset-threats
  setup-patches-visu
  if fogofwar [ setup-fogofwar ]
  if (playmode) [setup-happyface set playmode true]
end

; Initialisation principale
to setup
  clear-all
  ;check-options
  setup-globals
  setup-patches
  reset-learning
  setup-explorers
  setup-monsters
  setup-treasures
  setup-pits
  reset-threats
  setup-patches-visu
  if fogofwar [ setup-fogofwar ]
  reset-ticks
end

; Initialisation du fogofwar (toute la carte est cachée sauf là où est allé l'agent explorateur)
to setup-fogofwar
  ask turtles [
    hide-turtle
  ]
  ask patches [
    set pcolor black
  ]
  ask explorers [
    show-turtle
    ask patch-here [
      color-patch-visu
      ;set fogofwar-pvisible (list self)
    ]
  ]
end

;to check-options
;  if td-learning and q-learning or td-learning and seq-q-learning or q-learning and seq-q-learning [
;    user-message "Plusieurs learning sont activés, veuillez n'en activer qu'un seul à la fois (TD-Learning ou Q-Learning ou Seq-Q-Learning)"
;    error "Invalid options on GUI. Please check them."
;  ]
;end

; Initialisation des valeurs des variables globales
to setup-globals
  ; Définition des valeurs d'initialisation d'apprentissage et reward
  ; IMPORTANT: ne pas définir des valeurs trop hautes ici (eg: > 500, sinon NQ-Learning va bugguer car les nombres vont être trop grand pour être calculés par NetLogo)
  set reward-bonus 10 ; recompense quand le tresor est trouvé
  set penalty-eaten -5 ; pénalité quand l'explorateur perd en se faisant dévorer par le monstre
  set penalty-fell -2 ; pénalité quand l'explorateur tombe
  set penalty-fear 0 ; -1 ; pénalité pour représenter la peur sur les cases ou il y a un indice d'un danger (breeze, stench)
  set reward-bounty 5 ; récompense juste après avoir tué le monstre, pour favoriser l'action qui a tué le monstre
  set reward-bounty-available? false ; permet de donner la récompense uniquement à l'action qui vient de tuer le monstre, après on désactive
  set reward-bounty-exploration 1 ; récompense pour avoir exploré un patch inconnu (non visité jusqu'alors). Utilisé seulement pour le NQ-Learning
  set reward-bounty-exploration-available? false ; permet de donner la récompense uniquement juste quand on vient de visiter le patch inconnu
  set penalty-backtrack -0.1 ; -2 ; pénalité quand l'agent backtrack sur un patch déjà visité plus d'une fois (une fois c'est OK et peut être normal s'il y a un gros danger)
  set penalty-step -0.1

  set tdval-start 10 ; valeur tdval au depart
  set qval-start 10
  set sqval-start 10
  set justfinished false

  set gen 0
  set steps 0
  set found-treasure false

  set count-won 0
  set count-eaten 0
  set count-fell 0
  set count-killed-monster 0
  set prev-loss 0
  set prev-won 0
  set prev-gen 0

  set playmode false

  ; Constantes de déplacement/action (on peut mettre n'importe quelle valeur tant qu'elle est unique)
  set pup 0
  set pdown 1
  set pleft 2
  set pright 3
  set parrowup 4
  set parrowdown 5
  set parrowleft 6
  set parrowright 7

  ; SQ-Learning
  set sqval-prev-iteration 0

  ; NQ-Learning
  set nbactions 8
  ifelse nq-surroundFeatures [
    ifelse use-global-score [
      set nbfeatures 5
    ]
    [
      set nbfeatures 28
    ]
  ]
  [
    ifelse use-global-score [
      set nbfeatures 2
    ]
    [
      set nbfeatures 10
    ]
  ]
end

; Initialisation des patches (init de l'environnement)
to setup-patches
  ; Initialisation des valeurs
  ask patches [
    set plabel ""

    set stench false
    set breeze false
    set glitter false
    set reward 0
    ;if learning-mode = "Neural Q-Learning" [
    ;  set reward 30 ; Pour promouvoir l'exploration avec le neural net
    ;]
  ]
end

; Initialisation des valeurs d'apprentissage (placés sur les patches)
to reset-learning
  ask patches [
    set tdval tdval-start

    set qval-left qval-start
    set qval-right qval-start
    set qval-up qval-start
    set qval-down qval-start
    set qval-arrow-left qval-start - 1 ; -1 juste pour ne pas les afficher par défaut dans la reinf-visu
    set qval-arrow-right qval-start - 1
    set qval-arrow-down qval-start - 1
    set qval-arrow-up qval-start - 1

    ; SQ-Learning: on cree des listes vides, la valeur sqval-start sera initialisée pour chaque itération lors de l'apprentissage
    set sqval-left []
    set sqval-right []
    set sqval-up []
    set sqval-down []
    set sqval-arrow-left []
    set sqval-arrow-right []
    set sqval-arrow-down []
    set sqval-arrow-up []

    ; Adaptive epsilon-greedy exploration strategy, on commence à 0.9
    if auto-epsi [
      set epsilon 0.9
    ]
  ]

  set chicken-path []

  ; Neural Q-Learning
  reset-nnlearning
end

to reset-nnlearning

  ; Nombre d'expériences passées qu'on mémorise et qu'on donne tout de suite dans le réseau de neurones comme features (en gros le réseau recoit en entrée: features en ce moment (dans nq-make-example) + features de n expériences passées)
  if nq-memo < 1 [set nq-memo 1]

  ; Init schéma du réseau de neurones
  ; On peut faire un schéma plus spécifique en le définissant directement ici: set nqval-neurons_per_layer (list nbfeatures 1 2 3 4 .. n 1)
  set nqval-neurons_per_layer (sentence (nbfeatures * nq-memo) (n-values h-layers [h-neurons]) 1)

  ; Init weights
  set nqval-megaTheta (n-values nbactions [?]) ; Size = number of possible actions
  let i 0
  while [i < nbactions] [
    set nqval-megaTheta (replace-item i nqval-megaTheta (nnInitializeWeights nqval-neurons_per_layer false))
    set i (i + 1)
  ]

  ; Initialisation des mega matrices X et Y (qui vont contenir un ensemble d'exembles pour chaque action)
  set nqval-megaX (n-values nbactions [nobody])
  set nqval-megaY (n-values nbactions [nobody])

  ; Init prevX (l'exemple précédent)
  set nqval-prev-X matrix:from-row-list (list (n-values (nbfeatures * nq-memo) [0]))

  set replay-last-ticks 0
  set replay-count 0

  set stop-nqlearn false

  ask patches [
    set global-score 0
  ]
end

; Initialisation de l'explorateur
to setup-explorers
  create-explorers 1 [
    resetup-explorer

    ; Apparence
    set shape "explorer"
    set color pink + 3
  ]
end

to resetup-explorer
  ifelse random-startpos [
    ; Placement aléatoire (aide à la convergence normalement)
    place-on-empty-patch
  ]
  [
    ; Placement en bas à gauche
    setxy min-pxcor min-pycor
  ]

  ; Initialisation des variables propres
  reset-explorers-items
end

; Réinitialise les objets porté par l'explorateur
to reset-explorers-items
  ask explorers [
    set arrow-left 1
    set heard-scream false
  ]
end

; TODO comments
to reset-threats
  ; Reset tous les threats scores sur toutes les cases
  ask patches [
    set visited false
    set visited-count 0
    set safe false ; utilisé quand on est sûr que la case est sûre (une case voisine est bleue, sans indice de danger), et donc qu'on ne veut absolument pas rajouter de score sur cette case
    set monster-threat-score 0
    set pits-threat-score 0
    set prev-unknown-pxcor []
    set prev-unknown-pycor []
  ]
  ; Initialisation du premier pas (car l'explorateur est déjà sur une case)
  ask explorers [
    ask patch-here [
      set visited true
      ;set visited-count 1
    ]
    update-threats
    update-global-scores
  ]
end

; Initialisation du trésor
to setup-treasures
  create-treasures 1 [
    ; Placement dans une case aléatoire où il n'y a aucun autre agent
    place-on-empty-patch
    ; Apparence
    set shape "crown"
    set color yellow

    ; Initialisation des variables propres
    ask patch-here [
      ; Mise en place du sens glitter uniquement sur cette case
      set glitter true
      ; Initialisation de la valeur de la récompense quand le trésor est trouvé
      set reward reward + reward-bonus
    ]
  ]
end

; Initialisation du monstre Wumpus
to setup-monsters
  create-monsters 1 [
    ; Placement dans une case aléatoire où il n'y a aucun autre agent
    place-on-empty-patch
    ; Apparence
    set shape "monster"
    set color red

    ; Initialisation des variables propres
    ask neighbors4 [
      ; Mise en place des sens sur les cases voisines
      set stench true
      ; Pénalité d'approcher le monstre
      set reward reward + penalty-fear ; On rajoute car plusieurs pénalités peuvent s'ajouter si plusieurs dangers sont voisins
    ]
    ; Pénalité pour l'explorateur quand il se fait manger
    ask patch-here [set reward reward + penalty-eaten]
    ; On mémorise la position du monstre pour le respawn en mode simple
    set monster-prev-patch patch-here
  ]
end

; Réinitalisation du monstre à sa précédente position en mode simple (quand l'environnement reste stable entre plusieurs parties)
to respawn-monsters
  create-monsters 1 [
    ; Replace le monstre à son emplacement précédent
    move-to monster-prev-patch
    ; Apparence
    set shape "monster"
    set color red

    ; Initialisation des variables propres
    ask neighbors4 [
      ; Mise en place des sens sur les cases voisines
      set stench true
      ; Pénalité d'approcher le monstre
      set reward reward + penalty-fear
    ]
    ; Pénalité pour l'explorateur quand il se fait manger
    ask patch-here [set reward reward + penalty-eaten]
  ]
end

; Initialisation des trous d'air
to setup-pits
  let path-is-possible false
  while [not path-is-possible] [
    if any? pits [
      ask pits [
        die
      ]
    ]
    create-pits pits-count [
      ; Placement dans une case aléatoire où il n'y a aucun autre agent
      place-on-empty-patch
    ]

    set path-is-possible setup-check-path-is-possible
  ]

  ask pits [
    ; Apparence
    set shape "tile water"
    set color black

    ; Initialisation des variables propres
    ask neighbors4 [
      ; Mise en place des sens sur les cases voisines
      set breeze true
      ; Pénalité d'approcher le précipice
      set reward reward + penalty-fear
    ]
    ; Pénalité pour l'explorateur quand il tombe
    ask patch-here [set reward reward + penalty-fell]
  ]
end

; TODO comments
to-report setup-check-path-is-possible
  ask patches [
    set path-tocheck false
    set path-checked false
  ]

  let treasure-patch nobody
  ask treasures [
    set treasure-patch patch-here
  ]
  ask treasure-patch [
    ask neighbors4 [
      set path-tocheck true
    ]
  ]

  let reached-goal false
  let failed-goal false
  while [not reached-goal and not failed-goal] [
    ifelse count patches with [path-tocheck] = 0 [
      set failed-goal true
    ]
    [
      ask patches with [path-tocheck] [
        set path-tocheck false
        set path-checked true
        if not any? pits-here [
          ; Cas spécial où le trésor est juste à coté de l'explorateur dès le départ...
          if any? explorers-here [
            set reached-goal true
          ]

          ask neighbors4 with [not path-checked] [
            set path-tocheck true
            if any? explorers-here [
              set reached-goal true
            ]
          ]
        ]
      ]
    ]
  ]
  report reached-goal
end

; Trouve le chemin le plus court vers une case spécifiée en utilisant une recherche par programmation dynamique
to-report find-path-to [case]
  let l []
  ;show (word "But: " case)
  ask patch-here [
    set l find-path-to-aux (list self) case 0
  ]
  ;show l
  report l
end

to-report find-path-to-aux [path-list case level]
  ; On limite à 5 la récursion, sinon sur une grande carte et quand on a beaucoup visité de cases, l'agent mettra énormément de temps à converger vers la solution. Là il ne prend dans tous les cas que les 5 premières cases du chemin, et recalculera son itinéraire au bout de ces 5 cases.
  if (level) > 4 + (random 4) [ report (list nobody) ]

  ; Fin de récursion: Si un des voisins direct à la case en cours est la case cible, alors on la retourne et on a atteint notre but
  ifelse member? case neighbors4 [
    set path-list sentence path-list case
    report (list case)
  ]
  ; Sinon on va faire une récursion avec les voisins
  [
    ; S'il y a un moins un voisin déjà visité (donc qu'on peut emprunter de manière sûre), on y va
    ifelse count neighbors4 with [visited or safe] > 0 [
      let rlist [] ; can't report inside ask, thus we have to create a temporary variable outside. See https://github.com/NetLogo/NetLogo/issues/424 and https://github.com/NetLogo/NetLogo/issues/322
      ask neighbors4 with [visited or safe] [ ; Pour atteindre son but, l'agent a le droit de traverser soit des patches déjà visités, soit des patchs qui sont inconnus mais safe
        let l []
        let found-goal false
        if not found-goal [
          ; Si cette case n'est pas un backtrack (n'est pas dans la liste des chemins parcourus), alors on l'explore et on le rajoute à notre liste qui sera retournée
          ifelse not member? self path-list [ ; evite les backtracks
            set path-list sentence path-list self
            set l (sentence self (find-path-to-aux path-list case (level + 1)) )
          ]
          ; Si cette case n'est pas un backtrack (n'est pas dans la liste des chemins parcourus), alors on l'explore et on le rajoute à notre liste qui sera retournée
          [
            set l (list nobody)
          ]
          if empty? rlist [ set rlist l ]
          if (item ((length l) - 1) l) != nobody [ (set rlist l) (set found-goal true) ]
        ]
      ]
      report rlist
    ]
    ; Sinon si aucun voisin visité n'existe à partir de cette case (et qu'aucun voisin direct n'est la case cible), alors on arrête d'explorer ce chemin et on retourne nobody
    [
      report (list nobody) ; rien trouvé sur ce chemin
    ]
  ]
  report (list nobody)
end

; Trouve le chemin vers une case ayant un global-score aussi haut que max-gscore par programmation dynamique
to-report find-path-to-nearest-max-gscore
  let l []
  let max-gscore max [global-score + penalty-backtrack * visited-count] of patches with [count neighbors4 with [visited or safe] >= 1]
  ;show (word "But: " max-gscore)
  ask patch-here [
    set l find-path-to-nearest-max-gscore-aux (list self) max-gscore 0
  ]
  ;show l ; DEBUG
  report l
end

to-report find-path-to-nearest-max-gscore-aux [path-list max-gscore level]
  ; On limite à 5 la récursion, sinon sur une grande carte et quand on a beaucoup visité de cases, l'agent mettra énormément de temps à converger vers la solution. Là il ne prend dans tous les cas que les 5 premières cases du chemin, et recalculera son itinéraire au bout de ces 5 cases.
  if (level) > 4 + (random 4) [ report [] ]
  ; Note: on limite à un niveau aléatoire pour éviter les boucles infinies (ou l'agent est à égale distance du but et va dans un sens puis dans l'autre puis revient sur ses pas, etc..).
  ; De plus, on DOIT retourner nobody pour dire que ce chemin ne mène pas au but pour fallback sur le Chicken Heuristic (marche sur un voisin direct) car sinon si on retourne [] une liste vide, il va tenter de se rapprocher indéfiniment de la cible, mais ça peut ne pas marcher si aucun chemin n'y mène! Moyen de contourner:
  ; Un moyen de contourner: s'assurer que le max-gscore donné soit le score max d'un patch _accessible_, pas juste de n'importe quel patch!

  ;set pcolor green + 2 - 0.5 * level ; DEBUG

  ; Fin de récursion: Si un des voisins direct à la case en cours possède un score max, alors on la retourne et on a atteint notre but
  ifelse count neighbors4 with [(global-score + penalty-backtrack * visited-count) >= max-gscore] > 0 [
    let case (one-of neighbors4 with [(global-score + penalty-backtrack * visited-count) >= max-gscore])
    set path-list sentence path-list case
    ;ask case [ set pcolor pink ] ; DEBUG
    report (list case)
  ]
  ; Sinon on va faire une récursion avec les voisins
  [
    ; S'il y a un moins un voisin déjà visité (donc qu'on peut emprunter de manière sûre), on y va
    ifelse count neighbors4 with [visited or safe] > 0 [
      let rlist [] ; can't report inside ask, thus we have to create a temporary variable outside. See https://github.com/NetLogo/NetLogo/issues/424 and https://github.com/NetLogo/NetLogo/issues/322
      ask neighbors4 with [visited or safe] [ ; Pour atteindre son but, l'agent a le droit de traverser soit des patches déjà visités, soit des patchs qui sont inconnus mais safe
        let l []
        let found-goal false
        if not found-goal [
          ; Si cette case n'est pas un backtrack (n'est pas dans la liste des chemins parcourus), alors on l'explore et on le rajoute à notre liste qui sera retournée
          ifelse not member? self path-list [ ; evite les backtracks
            set path-list sentence path-list self
            set l (sentence self (find-path-to-nearest-max-gscore-aux path-list max-gscore (level + 1)) )
          ]
          ; Sinon si on a déjà exploré cette possibilité, on annule et on dit qu'ici c'est sans issue (l'algo continuera avec les autres chemins et retournera le plus long)
          [
            set l (list nobody)
          ]
          if empty? rlist [ set rlist l ]
          if (item ((length l) - 1) l) != nobody [ (set rlist l) (set found-goal true) ]
        ]
      ]
      report rlist
    ]
    ; Sinon si aucun voisin visité n'existe à partir de cette case (et qu'aucun voisin direct ne possède un max-gscore), alors on arrête d'explorer ce chemin et on retourne nobody
    [
      report (list nobody) ; rien trouvé sur ce chemin
    ]
  ]
  ; Au cas ou...
  report (list nobody)
end

; Colorie tous les patchs pour visualisation des sens
to setup-patches-visu ; Doit être appelé après les autres setup (car la couleur est définie par rapport aux dangers voisins)
  ask patches[
    color-patch-visu
  ]
end

; Couleur pour un patch pour visualisation des sens (permet de voir ce que l'agent perçoit)
to color-patch-visu

  ; Par défaut: bleue
  set pcolor blue

  ; Si à la fois breeze et stench, en violet
  ifelse breeze = true and stench = true [
    set pcolor magenta
  ]
  [
    ; Breeze: gris
    if breeze = true [
      set pcolor grey
    ]
    ; Stench: rouge
    if stench = true [
      set pcolor red
    ]
  ]
  ; Glitter (trésor): orange
  if glitter = true [
    set pcolor orange
  ]
end

; Placement aléatoire sur un patch où il n'y a aucun agent
; Permet de s'assurer par exemple que le monstre n'est pas sur la même case que l'explorateur, ni superposé avec un trou d'air
to place-on-empty-patch
  let patchx 0
  let patchy 0
  ask one-of patches with [not any? turtles-here] [
    set patchx pxcor
    set patchy pycor
  ]
  setxy patchx patchy
end

; MAJ à chaque pas des scores de danger sur les 4 cases voisines et la case actuelle de l'explorateur (pas besoin de MAJ les autres cases puisqu'elles n'auront pas changées, le score n'étant calculé qu'avec les 4 cases voisines).
; Note: seulement si on n'est pas mort, les derniers scores étant totalement à coté de la plaque puisqu'on assume qu'un patch visité est donc sans danger, ce qui n'est pas le cas si on vient de mourir
to update-threats
  ask explorers [
    if not any? monsters-here and not any? pits-here [
      ; Tout d'abord il faut marquer la case en cours comme visitée afin que les voisins puissent MAJ leur liste de cases voisines inconnues
      ask patch-here [
        set visited true
        set visited-count (visited-count + 1)
      ]
      ; On MAJ les voisins d'abord, pour eviter de mettre a 0 la case actuelle si il n'y a aucun danger dessus, et qu'ensuite une case voisine diminue le score encore en négatif en retranchant son score attribué précédemment
      ask neighbors4 with [visited = true] [
        compute-threats-scores self
      ]
      ; Enfin on MAJ la case en cours, qui n'a donc aucun danger puisqu'on n'est pas mort (mais il peut y avoir un indice de danger!)
      ask patch-here [
        compute-threats-scores self
      ]
    ]
  ]
end

; MAJ du score global. Doit être appelé uniquement après update-threats.
to update-global-scores
  ask explorers [
    ask patch-here [
      compute-global-score self
    ]
    ask neighbors4 [
      compute-global-score self
    ]
  ]
end

; Mixture Particle Filter by Vermaak, logical implementation by Stephen Larroque
; Compute the global-score (threats and exploratory interest) only on needed squares, without needing N particles (because we only update neighbors squares).
; These are simply logical rules, simplifying and reducing the originally probabilistic mixture particle filter algorithm.
; Here are the rules in pseudo prolog BDI:
; c = current square
; n = each neighbor square in 4 directions
; u = unknown neighbor square (ie, not yet visited) in the n neighbors: u <- not visited(n)
; @check-no-death: not on(monster, c) and not on(pits, c) <- visited(c).
; @check-safe-patch: not stench(c) and not breeze(c) <- +safe(c), +safe(n), monster-threat-score(c, 0), pits-threat-score(c, 0), monster-threat-score(n, 0), pits-threat-score(n, 0).
; @first-update-remove-previous-scores: stench(c) <- previous-unknown(c, n), prevscore = 1/n, monster-threat-score(monster-threat-score(u) - prevscore, n).
; @same for breeze: ...
; @second-update-rule-add-new-score: stench(c) <- score = 1/u, monster-threat-score(monster-threat-score(u) + score, u), +previous-unknown(c, u).
; @same for breeze: breeze(c) <- score = 1/u, pits-threat-score(pits-threat-score(u) + score, u), +previous-unknown(c, u).
; @check-no-stench-clue: not stench(c) <- monster-threat-score(c, 0), monster-threat-score(n, 0).
; @check-no-breeze-clue: not breeze(c) <- pits-threat-score(c, 0), pits-threat-score(n, 0).
;
; Note: A known problem is that the weights are not always correctly updated with the maximum threat score for all neighboring squares (eg, when we visit a "safe" square), but the neighbor squares to the current square are always up-to-date. This is because we update only neighbors, so n+2 neighbors might also need an update but we don't propagate the update. So to summarize: immediately neighboring squares to current always have reliable and correct scores, but further squares might underestimate the threat-score. This should not be an issue since these "remote" squares will get updated when we will reach them (because the game only allows to move from square to square, if we could jump or teleport, it would be more problematic).
;
; TODO? Mixture particles filters by Vermaak en utilisant nombre de neighbors with stench - 4 (nombre de neighbors with stench si c'est vraiment la case avec le monstre) pour la vraissemblance. Avec le resampling avec la wheel, la proba augmentera si les cases autour sont vides (pas besoin de visiter les 4 neighbors du monstre pour avoir une proba proche de 1).

to compute-threats-scores [case]
  ask case [
    if visited [

      ; Un indice de danger sur la case en cours? On est sûr qu'au moins un danger se trouve sur une ou plusieurs cases voisines
      ; MAJ procédure: si la case en cours possède un indice de danger, on attribue un threat score 1/nombre de cases voisines inconnues réparti pour chaque case voisine inconnue (non visitée et non safe)
      ; La procédure MAJ est en 2 temps: 1- on retranche tout score précédemment attribué lors d'une précédente visite à toutes les cases voisines alors inconnues (permet 2 choses: stabilité des scores car quand on visite deux fois la même cases on ne va pas rajouter artificiellement du score, et deuxièmement quand une case voisine inconnue devient connue, on réattribue son threat score aux cases voisines encore inconnues); 2- on attribue le threat score calculé par 1/nombre de cases voisines inconnues
      ifelse stench = true or breeze = true [
        let parent-stench stench
        let parent-breeze breeze

        ; 1- MAJ des threats scores des cases voisines: on enleve tout d'abord la partie du score qu'on a attribué à des cases précédemment inconnues et qu'on connait maintenant (score qu'on va pouvoir réattribuer à d'autres cases)
        ; Retranche les scores précédemment attribués aux cases voisines par la case en cours (d'autres cases ont aussi pu contribuer au score, aussi on n'enleve que le score attribué par cette case)
        if not empty? prev-unknown-pxcor [
          let prev-certainty-score 1 / (length prev-unknown-pxcor)
          (foreach prev-unknown-pxcor prev-unknown-pycor [ ; NB: les parentheses avant et apres le foreach sont necessaires pour pouvoir passer plusieurs listes en argument, sinon il n'accepte qu'une seule liste!
            ask patch ?1 ?2 [
              if parent-stench = true [
                set monster-threat-score monster-threat-score - prev-certainty-score
                ; NOTE: enlever la borne mini si vous constatez des bugs, sera plus facile à debugguer
                set monster-threat-score (max (list 0 monster-threat-score)) ; borne a 0 au minimum, ne devrait jamais arriver que cela descende en dessous mais on ne sait jamais
              ]
              if parent-breeze = true [
                set pits-threat-score pits-threat-score - prev-certainty-score
                ; NOTE: enlever la borne mini si vous constatez des bugs, sera plus facile à debugguer
                set pits-threat-score (max (list 0 pits-threat-score)) ; borne a 0 au minimum, ne devrait jamais arriver que cela descende en dessous mais on ne sait jamais
              ]
            ]
          ])
          set prev-unknown-pxcor []
          set prev-unknown-pycor []
        ]

        ; 2- MAJ assignation des threats scores aux cases voisines inconnues et non marquées comme safe
        let count-unknown-patches count neighbors4 with [visited = false and safe = false]
        let temp-pxcor []
        let temp-pycor []
        ask neighbors4 with [visited = false and safe = false] [
          let certainty-score 1 / count-unknown-patches
          if parent-stench = true [
            set monster-threat-score monster-threat-score + certainty-score
            ;set monster-threat-score (min (list 1 monster-threat-score)) ; on borne à 1 pour avoir des pseudos probas
          ]
          if parent-breeze = true [
            set pits-threat-score pits-threat-score + certainty-score
            ;set pits-threat-score (min (list 1 pits-threat-score)) ; on borne à 1 pour avoir des pseudos probas
          ]
          ; On maintient la liste des cases voisines inconnues, c'est ce qui permet de faire l'étape MAJ 1 de retranchement des scores
          set temp-pxcor lput pxcor temp-pxcor
          set temp-pycor lput pycor temp-pycor
        ]
        set prev-unknown-pxcor temp-pxcor
        set prev-unknown-pycor temp-pycor

        ; MAJ de la case en cours
        if stench = false [
          set monster-threat-score 0
        ]
        if breeze = false [
          set pits-threat-score 0
        ]
      ]

      ; Pas d'indice de danger? On est non seulement sûr qu'ici c'est sans danger, mais qu'aussi toutes les autres cases autour sont safe.
      [
        set monster-threat-score 0
        set pits-threat-score 0
        ask neighbors4 [
          set monster-threat-score 0
          set pits-threat-score 0
          set safe true ; on les considère même comme safe pour éviter qu'un autre indice de danger ne rajoute un score threat! Si la case est marquée comme safe, on signifie que la case est sûre pour de bon!
        ]
      ]
    ]
  ]
end

; Calcule un score d'aggrégation, qui en gros permet à l'agent de savoir où il vaudrait mieux aller en un seul score (on fait donc le boulot d'une couche cachée ici, normalement le neural net devrait avoir de meilleures performances en utilisant ce score directement)
to compute-global-score [case]
  ask case [
    let safe-score 0
    if safe [set safe-score 1]
    set global-score safe-score + (compute-exploratory-interest / 4) - (monster-threat-score + pits-threat-score)
  ]
end



;=== MAIN LOOP AND MECANICS ===

; Boucle principale en mode simulation (apprentissage par renforcement)
to go
  ; Déplacement de l'agent explorateur
  move-explorers
  ; MAJ des dangers sur la case
  update-threats
  ; MAJ du score global
  update-global-scores
  ; Visualisation des valeurs d'apprentissage par renforcement
  if (not playmode) [ reinf-visualisation ]
  ; Plot le win-loss ratio
  if (gen mod winloss-update-interval = 0 and gen > 0) [ do-plot-winloss-ratio ]

  ; Quelques post-traitements après chaque fin de partie
  if justfinished [
    ; Variation simple (avec environnement stable): on replace le monstre à sa place pour pouvoir apprendre à le tuer
    ifelse not harder-variation [
      if not any? monsters [ respawn-monsters ] ; On respawn le monstre s'il a été tué
      setup-patches-visu ; On MAJ la coloration des patchs
      if fogofwar [ setup-fogofwar ] ; On recache toutes les cases à chaque fin de partie, sauf celle du début
      reset-explorers-items ; On réattribue tous les items à l'explorateur
      reset-threats ; On reinit les threats
    ]
    ; Harder variation du problème: on réinitialise la position de tous les agents aléatoirement
    [
      resetup-harder
    ]

    ; Plot et reset steps
    if found-treasure [ do-plot-learning-curve ] ; ne tracer que si on a trouvé le trésor, sinon si on a perdu l'information n'a aucune valeur (si on bouge aléatoirement dans n'importe quel sens on aura un nombre bas de steps)
    set gen gen + 1
    set steps 0

    ; Reset variables globales de l'état du jeu
    set justfinished false
    set found-treasure false
  ]
end

; FONCTION PRINCIPALE POUR L'APPRENTISSAGE
; Gère les déplacements de l'agent explorateur en mode automatique (apprentissage par renforcement)
; Gère aussi la MAJ des valeurs d'apprentissage (TD-Learning, Q-Learning, etc)
; Note: la MAJ des valeurs d'apprentissage se fait à postériori (on explore le prochain état avant de mettre à jour le précédent/en cours)
to move-explorers
  ; Ajoute un pas
  tick
  set steps steps + 1 ; réinitialisé à 0 à chaque fois que l'explorateur meurt ou gagne

  ask explorers [
    ; On mémorise le patch en cours, sera utilisé pour calculer la nouvelle valeur d'apprentissage sur ce patch
    let prev-patch patch-here

    ; == TD-Learning (tout est calculé ici, pas de fonction auxiliaire)
    if learning-mode = "TD-Learning" [
      let next-reward 0
      let next-tdval 0
      ; Meurt s'il y a un danger et téléporte à la case de départ, ou sinon continue de se déplacer
      ifelse (check-dangers-and-treasure)
        [
          ; Si on trouve le tresor ou un piege, alors on adapte l'equation car il n'y a pas de case ensuite
          ask prev-patch[
            set next-reward reward
            set next-tdval tdval
          ]
        ]
        [
          ; Sinon c'est une case vide, on fait le calcul normal
          ifelse random-float 1 < epsilon [
            ; au hasard on explore de temps en temps selon la probabilité epsilon
            move-to one-of neighbors4
          ]
          [
            ; sinon on fait du hillclimbing par rapport à la variable tdval
            move-to max-one-of neighbors4 [tdval]
          ]
          ; On récupère les valeurs max-tdval et reward du prochain état s'
          ask patch-here [
            set next-reward reward
            set next-tdval tdval
          ]
        ]
      ; MAJ de la valeur TDval pour la case précédente
      ask prev-patch [
        set tdval tdval + alpha * (next-reward + gamma * next-tdval - tdval)
        ;if tdval < 0 [ set tdval 0 ] ; Minimum pour tdval = 0
      ]
    ]

    ; == Q-Learning
    if learning-mode = "Q-Learning" [
      let action 0
      let next-reward 0
      let next-qval 0
      ; Meurt s'il y a un danger et téléporte à la case de départ, ou sinon continue de se déplacer
      ifelse (check-dangers-and-treasure)
        [
          ; Si on trouve le tresor ou un piege, alors on adapte l'equation car il n'y a pas de case ensuite
          ask prev-patch [
            set qval-prev-patch-pxcor pxcor
            set qval-prev-patch-pycor pycor

            set qval-prev-action (choice-action-max-qval true)

            set next-reward reward
            set next-qval get-qval qval-prev-action
          ]
        ]
        [
          ; = Récupération des valeurs en cours pour max(Q(s,a))
          ask patch-here [
            set qval-prev-patch-pxcor pxcor
            set qval-prev-patch-pycor pycor
          ]

          set action choose-and-do-action-carefully
          set qval-prev-action action
          if verbose [ show word "action faite: " action ]

          ; = Récupération des valeurs next(max(Q(s',a'))) et reward'
          ask patch-here [
            set next-reward reward
            set next-qval get-qval choice-action
          ]
        ]
      ; MAJ de la valeur Qval pour l'action précédente + assigne next-qval dans prev-qval (change next state s' en s)
      update-qval next-reward next-qval
    ]

    ; == Sequential Q-Learning
    if learning-mode = "Sequential Q-Learning" [
      let action 0
      let next-reward 0
      let next-sqval 0
      let new-iteration 0
      ; Meurt s'il y a un danger et téléporte à la case de départ, ou sinon continue de se déplacer
      ifelse (check-dangers-and-treasure)
        [
          ; Si on trouve le tresor ou un piege, alors on adapte l'equation car il n'y a pas de case ensuite
          ask prev-patch [
            set sqval-prev-patch-pxcor pxcor
            set sqval-prev-patch-pycor pycor

            set sqval-prev-action (choice-action-max-sqval true sqval-prev-iteration)

            set next-reward reward
            set next-sqval (get-sqval sqval-prev-action sqval-prev-iteration)
          ]
        ]
        [
          ; = Récupération des valeurs en cours pour max(Q(s,a))
          ; On mémorise la case précédente (case en cours)
          ask patch-here [
            set sqval-prev-patch-pxcor pxcor
            set sqval-prev-patch-pycor pycor
          ]
          ; On se déplace + récupère l'action qu'on a fait, et qui est donc l'action avec la max-sqval (ou l'action choisie au hasard par epsilon-greedy)
          set action sq-choose-and-do-action-carefully sqval-prev-iteration
          set sqval-prev-action action

          ; = Récupération des valeurs next(max(Q(s',a'))) et reward'
          ; Coeur du SQ-Learning: si on change de patch, on revient à la 1ere itération des valeurs sqval de cette case (la prochaine case en fait, donc ici c'est la prochaine itération qu'on calcule)
          ifelse sqval-prev-iteration < 0 or patch sqval-prev-patch-pxcor sqval-prev-patch-pycor != patch-here [
            set new-iteration 0
          ]
          ; Sinon, si on est resté sur la même case (par exemple on a tiré une flèche), alors on change l'itération (ce qui crée une séquence d'actions dans le temps!)
          [
            set new-iteration sqval-prev-iteration + 1
          ]
          if verbose [ show (word "action faite: " action " it: " sqval-prev-iteration) ]

          ; Enfin, on récupère la max-sqval et reward de la prochaine case (maintenant case en cours)
          ask patch-here [
            set next-reward reward
            set next-sqval (get-sqval (sq-choice-action new-iteration) new-iteration)
          ]
        ]
      ; MAJ de la valeur SQval pour l'action et l'iteration précédente + assigne next-qval dans prev-qval (change next state s' en s)
      update-sqval next-reward next-sqval
      set sqval-prev-iteration new-iteration ; et surtout ne pas oublier de MAJ l'itération mémorisée pour la nouvelle case en cours!
    ]

    ; == Neural Q-Learning
    if learning-mode = "Neural Q-Learning" [
      let action 0
      let next-reward 0
      let next-max-nqval 0
      let next-visited-count 0
      let X []
      ; Meurt s'il y a un danger et téléporte à la case de départ, ou sinon continue de se déplacer
      if not (check-dangers-and-treasure) [
        ; Si on trouve le tresor ou un piege, alors on adapte l'equation car il n'y a pas de case ensuite
        ; set next-max-nqval nqval-prev-max
        ; ask prev-patch [
        ;   set next-reward reward
        ; ]
        ;]
        ;[

        ; = Récupération des valeurs en cours pour max(Q(s,a))
        ; Une pierre 2 coups: On calcule la valeur nqval-max pour l'état en cours + on exécute l'action associée
        let tmpa choice-action-max-nqval true
        set action (item 0 tmpa)
        set nqval-prev-X nq-update-example (nq-make-example action) nqval-prev-X ; On doit stocker avant l'action l'exemple (état) en cours, en mémorisant les Exemples précédents (on mémorise les nq-memo features précédentes pour que le réseau de neurones puisse avoir une certaine mémoire du passé)
        let tmp nq-choose-and-do-action-carefully
        set nqval-prev-action (item 0 tmp)
        set nqval-prev-max (item 1 tmp) ; IMPORTANT: On doit recalculer à chaque fois la nouvelle valeur de la case en cours, car next-max-nqval != nqval-prev-max (car entre le moment ou on a calculé la next-max-nqval et qu'on s'est déplacé sur le next patch et qu'on recalcule la max-nqval, elle peut avoir changé car on a fait une backpropagation du réseau de neurones! Donc si l'action précédente et l'action prochaine est la même et donc utilise le même réseau de neurones, rien n'assure que la prédiction sera la même! C'est donc une différence primordiale avec le Q-Learning ou les prédictions suivantes ne peuvent pas être changées par les prédictions passées)
        set nqval-prev-A (item 2 tmp)
        set nqval-prev-Z (item 3 tmp)
        if verbose [ show word "action faite: " nqval-prev-action ]

        ; = Récupération des valeurs next(max(Q(s',a'))) et reward'
        ; Ensuite on calcule la valeur next-max-nqval du nouvel état sur lequel on se trouve (donc on fait une MAJ à postériori: on se déplace puis on MAJ la valeur de l'état précédent après avoir exploré le prochain état)
        ask patch-here [
          set next-reward reward
          let temp choice-action-max-nqval true ; optimisation: remplace get-qval choice-action car sinon il faudrait calculer 2 fois la propagation avant!
          set next-max-nqval (item 1 temp)
          if not visited [ set reward-bounty-exploration-available? true ] ; favorisation de l'exploration: si la case était inconnue, l'agent a droit à une récompense supplémentaire!
          set next-visited-count visited-count
        ]

        ; MAJ de la valeur NQval pour l'action précédente
        update-nqval next-reward next-max-nqval next-visited-count
      ]
    ]

    ; == Random
    ; Utilisé juste pour comparer avec les autres algos avec le hasard
    if learning-mode = "Random" [
      if not (check-dangers-and-treasure) [
        move-to one-of neighbors4
      ]
    ]

    ; == Chicken Heuristic
    ; Simple heuristique visitant le voisin ayant le max global-score
    if learning-mode = "Chicken Heuristic" [
      if not (check-dangers-and-treasure) [
        ; Va au voisin possédant le global-score maximum, pondéré par le nombre de fois qu'on a déjà visité cette case (pour empêcher l'agent de revenir sur ses pas indéfiniment)
        move-to max-one-of neighbors4 [global-score + penalty-backtrack * visited-count]
      ]
    ]

    ; == Chicken Search
    ; Exploration en largeur (breadth-first) des patchs ayant le max global-score en premier (même s'ils sont loin, l'agent calculera le chemin adéquat, s'il en existe un)
    if learning-mode = "Chicken Search" [
      ifelse (check-dangers-and-treasure) [
        ; On vient de finir une partie, on réinitialise le chicken-path
        set chicken-path []
      ]
      ; Sinon on est encore en train de jouer
      [
        if (length chicken-path = 0 or item ((length chicken-path) - 1) chicken-path = nobody) [
          ; Cherche un chemin vers la case possédant le global-score maximum, pondéré par le nombre de fois qu'on a déjà visité cette case (pour empêcher l'agent de revenir sur ses pas indéfiniment)
          if verbose [ print "Computing new chicken path" ]
          set chicken-path find-path-to-nearest-max-gscore
          if verbose [ print (word "Found path: " chicken-path) ]
        ]
        ; Puis fait une exploration en largeur (breadth-search) des cases avec ce score maximum
        ifelse (length chicken-path > 0 and item ((length chicken-path) - 1) chicken-path != nobody) [
          if verbose [ print "Traversing chicken path" ]
          move-to (item 0 chicken-path)
          set chicken-path sublist chicken-path 1 (length chicken-path) ; On enleve le patch qu'on vient d'emprunter de la liste du chemin
        ]
        ; Sinon il n'y a aucun chemin qui mene vers un patch ayant un max-gscore, on va sur un patch voisin
        [
          if verbose [ print "No chicken-path, go to neighbor with highest global-score" ]
          move-to max-one-of neighbors4 [global-score + penalty-backtrack * visited-count]
          set chicken-path [] ; On réinitialise le chicken-path pour qu'il soit calculé la prochaine fois
        ]
      ]
    ]

    ; == Divers post-processing
    ; MAJ du statut visited pour le patch en cours
    ask patch-here [
      set visited true
    ]

    ; On affiche la case actuelle si il y a un fogofwar
    if fogofwar [
      manage-fogofwar prev-patch
    ]
  ]
end

; Verifie s'il y a des dangers ou un tresor et teleporte l'explorateur s'il y en a un
; Enfin, retourne Vrai si un danger ou un trésor était là, et Faux sinon
; C'est un peu la fonction qui vérifie si le jeu se termine
to-report check-dangers-and-treasure
  let found-danger false

  ask explorers [
    ask turtles-on patch-here [
      if is-monster? self [
        set count-eaten count-eaten + 1
        ask myself [ resetup-explorer ]
        set found-danger true
      ]
      if is-pit? self [
        set count-fell count-fell + 1
        ask myself [ resetup-explorer ]
        set found-danger true
      ]
      if is-treasure? self [
        set count-won count-won + 1
        ask myself [ resetup-explorer ]
        set found-danger true
        set found-treasure true
        ; Adaptive epsilon-greedy exploration strategy, on diminue à chaque fois qu'on trouve le trésor selon une vitesse de décroissannce définie par le paramètre auto-epsi-param
        if auto-epsi [ set epsilon (precision (epsilon ^ (auto-epsi-param * 1 + 1)) 3) ] ; on décroit epsilon automatiquement après chaque fois qu'on gagne, en faisant une exponentielle entre 1 et 2 (selon auto-epsi-param), ce qui va tendre vers 0 dans tous les cas
      ]
    ]
  ]

  if found-danger [set justfinished true]

  report found-danger
end

;=== VISUALISATION ===

; Visualisation des valeurs de l'apprentissage par renforcement avec:
; Pour TD-Learning:
; 1- la valeur écrite sur chaque case
; 2- un dégradé de bleu du plus foncé au plus clair respectivement selon les valeurs min et max apprises
; Pour Q-Learning:
; Un texte sur la case affichant la meilleure action (flèche pour direction et A pour arrow) + la valeur qval
to reinf-visualisation

  ; Vide tous les textes sur tous les patchs
  ask patches [
    set plabel ""
  ]

  ; Montre les scores des apprentissage par renforcement
  if reinf-visu [

    ; TD-Learning: On affiche le score sur chaque patch ainsi qu'un dégradé de couleur: plus clair pour le max au plus foncé pour le min entre toutes les valeurs de tous les patchs
    if learning-mode = "TD-Learning" [
      let max-tdval max [tdval] of patches ; Récupère la valeur tdval max entre tous les patchs
      let min-tdval min [tdval] of patches ; Idem pour min tdval
      if (max-tdval != min-tdval) [ ; Si on a au moins appris quelquechose (sinon tous les patchs auront la même couleur, ce n'est pas intéressant)
        ask patches with [not any? treasures-here] [ ; Et sauf sur la case du trésor (pas utile car ce sera toujours la case avec score max si l'agent l'atteint au moins une fois...)
          set pcolor (101 + ((tdval - min-tdval) / (max-tdval - min-tdval)) * 8 )
        ]
        ask patches [
          set plabel precision tdval 1
        ]
      ]
    ]

    ; Q-Learning: on affiche la meilleure action via un symbole (direction pour déplacement ou A+direction pour fleche) et le score
    if learning-mode = "Q-Learning" [
      ask patches [
        let best-action (choice-action-max-qval false)
        let best-qval (get-qval best-action)
        let best-action-text ""

        if best-action = pup [ set best-action-text "^" ]
        if best-action = pdown [ set best-action-text "v" ]
        if best-action = pleft [ set best-action-text "<" ]
        if best-action = pright [ set best-action-text ">" ]
        if best-action = parrowup [ set best-action-text "A^" ]
        if best-action = parrowdown [ set best-action-text "Av" ]
        if best-action = parrowleft [ set best-action-text "A<" ]
        if best-action = parrowright [ set best-action-text "A>" ]

        set plabel (word best-action-text " " (precision best-qval 1))
      ]
    ]

    ; SQ-Learning: comme Q-Learning mais on affiche selon l'itération choisie dans l'interface
    if learning-mode = "Sequential Q-Learning" [
      ask patches [
        if (length sqval-up) >= 1 [
          carefully [
            let best-action (choice-action-max-sqval false visu-sq-it) ; on n'affiche que pour l'itération 1, pas la place pour afficher tous les choix pour toutes les iterations
            let best-qval (get-sqval best-action visu-sq-it)
            let best-action-text ""

            if best-action = pup [ set best-action-text "^" ]
            if best-action = pdown [ set best-action-text "v" ]
            if best-action = pleft [ set best-action-text "<" ]
            if best-action = pright [ set best-action-text ">" ]
            if best-action = parrowup [ set best-action-text "A^" ]
            if best-action = parrowdown [ set best-action-text "Av" ]
            if best-action = parrowleft [ set best-action-text "A<" ]
            if best-action = parrowright [ set best-action-text "A>" ]

            set plabel (word best-action-text " " (precision best-qval 1))
          ]
          []
        ]
      ]
    ]

    ; NQ-Learning: on affiche les valeurs pour chaque action dans les cases autour de l'agent (on ne peut pas tout visualiser...)
    if learning-mode = "Neural Q-Learning" [
      ask explorers[
        let paction 0
        foreach neighbors4-ordered [
          if ? != nobody [
            ask ? [
              let parrowaction (paction + 4)
              let tmp (get-nqval paction)
              let nqval (item 0 tmp)
              let tmpa (get-nqval parrowaction)
              let nqval-arrow (item 0 tmpa)
              set plabel (word (precision nqval 1) " A" (precision nqval-arrow 1))
            ]
          ]
          set paction (paction + 1)
        ]
      ]
    ]
  ]
  ; Montre les scores de danger pour chaque patch (monstre et trous)
  if show-threats-scores [
    ask patches [
      set plabel (word plabel " M" (precision monster-threat-score 2) " P" (precision pits-threat-score 2))
    ]
  ]
  if show-global-scores [
    ask patches [
      set plabel (word plabel " G" (precision global-score 1))
    ]
  ]
end

;=== Q-LEARNING FONCTIONS AUXILIAIRES ===

; Récupère la valeur qval pour une action
to-report get-qval [action]
  let qval 0
  if action = pup [ set qval qval-up]
  if action = pdown [ set qval qval-down]
  if action = pleft [ set qval qval-left]
  if action = pright [ set qval qval-right]
  if action = parrowup [ set qval qval-arrow-up]
  if action = parrowdown [ set qval qval-arrow-down]
  if action = parrowleft [ set qval qval-arrow-left]
  if action = parrowright [ set qval qval-arrow-right]
  report qval
end

; MAJ la qval pour l'action précédente
to update-qval [next-reward next-qval]
  ask patch qval-prev-patch-pxcor qval-prev-patch-pycor [
    let qval-prev (get-qval qval-prev-action)
    let qval-new 0

    ifelse reward-bounty-available?
    [
      set qval-new (1 - alpha) * qval-prev + alpha * (next-reward + reward-bounty + gamma * next-qval)
      set reward-bounty-available? false
    ]
    [
      set qval-new (1 - alpha) * qval-prev + alpha * (next-reward + gamma * next-qval)
    ]

    if qval-prev-action = pup [ set qval-up qval-new ]
    if qval-prev-action = pdown [ set qval-down qval-new ]
    if qval-prev-action = pleft [ set qval-left qval-new ]
    if qval-prev-action = pright [ set qval-right qval-new ]
    if qval-prev-action = parrowup [ set qval-arrow-up qval-new ]
    if qval-prev-action = parrowdown [ set qval-arrow-down qval-new ]
    if qval-prev-action = parrowleft [ set qval-arrow-left qval-new ]
    if qval-prev-action = parrowright [ set qval-arrow-right qval-new ]
  ]
end

; Choisit une action (soit au hasard soit par rapport à max qval)
to-report choice-action
  let action 0
  if learning-mode = "Q-Learning" [
    ifelse random-float 1 < epsilon [
      ifelse check-arrow-left [
        set action random 8
      ]
      [
        set action random 4
      ]
    ]
    [
      set action (choice-action-max-qval true)
    ]
  ]
  report action
end

; Retourne l'action avec la max qval
; Cette fonction va aussi prendre en compte les flèches si dispo
; On peut outrepasser la vérification des flèches avec check-arrow? false, utilisé pour la reinf-visu (pour afficher l'action avec la max qval quelquesoit s'il reste des flèches ou pas)
to-report choice-action-max-qval [check-arrow?]
  let list-actions 0
  ; La liste doit être dans le même ordre que les constantes de déplacement pup pdown, etc..
  ifelse not check-arrow? or check-arrow-left [
    set list-actions (list qval-up qval-down qval-left qval-right qval-arrow-up qval-arrow-down qval-arrow-left qval-arrow-right)
  ]
  [
    set list-actions (list qval-up qval-down qval-left qval-right)
  ]
  ; On choisit l'action avec la valeur qval maximum et uniquement parmi les actions possibles (eg: si le patch n'existe pas on ne choisit pas)
  let action 0
  let i 0
  let qval-max nobody
  foreach list-actions [
    ; Si cette action a une plus grand valeur qval que les précédentes
    if qval-max = nobody or ? >= qval-max [
      ; On vérifie que le patch existe dans cette direction, sinon on y va pas c'est inutile
      if get-neighbour-patch i != nobody [
        set action i ; On mémorise cette action
        set qval-max ? ; On met à jour la valeur qval-max
      ]
    ]
    set i i + 1
  ]
  report action
end

; Fait une action du Q-Learning
to do-action [action]
  if member? action (list pup pdown pleft pright) [
    move-to get-neighbour-patch action
  ]
  if member? action (list parrowup parrowdown parrowleft parrowright) [
    if verbose [ if verbose [ show "SHOT!" ] ]
    check-and-shoot-arrow action
  ]
end

; Fait une action pour le Q-Learning et évite les exceptions (par exemple si la case n'existe pas)
to-report choose-and-do-action-carefully
  let action 0

  ; On reessaie en boucle jusqu'à trouver une action qui ne cause pas d'erreur (au hasard si necessaire)
  let no-error false
  let i 0
  let maxIter 200
  while [not no-error] [
    carefully
    [
      set action choice-action
      do-action action
      set no-error true
    ]
    [
    ]
    ; Devrait ne jamais arriver, mais si on boucle, c'est qu'il y a un probleme (soit pas d'action possible, soit agent choisit toujours la même et elle est mauvaise et epsilon-greedy est désactivé/mis à 0)
    if i > maxIter [
      error "L'agent ne peut pas choisir d'action valide! Soit il est complètement bloqué, soit il choisit toujours une action impossible et epsilon-greedy est désactivé! Essayez de mettre epsilon > 0"
    ]
    set i (i + 1)
  ]
  report action
end

;=== SEQUENTIAL Q-LEARNING FONCTIONS AUXILIAIRES ===
; SQ-Learning permet d'avoir 2 polices différentes pour un même état consécutivement exécuté n fois (ce qui n'est pas possible avec le Q-Learning, un même état ayant ayant toujours la même police)
; Q(s',a',0) = f(Q(s,a,t)) si s != s', c'est-à-dire si on change d'état après avoir exécuté l'action a
; Q(s',a',t+1) = f(Q(s,a,t)) si s = s', c'est-à-dire si on reste dans le même état après avoir exécuté l'action a
; Ceci permet effectivement de construire des séquences temporelles pour les Q-valeurs (au lieu d'une seule valeur par action, on a une séquence de valeurs par action selon le pas temporel).

; Récupère la valeur qval pour une action
to-report get-sqval [action iteration]
  let sqval 0
  ; Si cette itération n'existe pas (encore), on l'initialise
  ; Note: on fait ca en loop car la facon dont on a implémenté (on update la case précédente apres avoir bougé), on accede donc a la prochaine sqval avant la précédente, donc si on reste sur la même case on va avoir a créer l'itération 1 et 2 à la fois.
  while [(length sqval-up) < (iteration + 1)] [
    set sqval-up lput sqval-start sqval-up
    set sqval-down lput sqval-start sqval-down
    set sqval-left lput sqval-start sqval-left
    set sqval-right lput sqval-start sqval-right
    set sqval-arrow-up lput sqval-start sqval-arrow-up
    set sqval-arrow-down lput sqval-start sqval-arrow-down
    set sqval-arrow-left lput sqval-start sqval-arrow-left
    set sqval-arrow-right lput sqval-start sqval-arrow-right
  ]
  ; On recupere la valeur pour l'action et l'itération spécifiés
  if action = pup [ set sqval (item iteration sqval-up)]
  if action = pdown [ set sqval (item iteration sqval-down)]
  if action = pleft [ set sqval (item iteration sqval-left)]
  if action = pright [ set sqval (item iteration sqval-right)]
  if action = parrowup [ set sqval (item iteration sqval-arrow-up)]
  if action = parrowdown [ set sqval (item iteration sqval-arrow-down)]
  if action = parrowleft [ set sqval (item iteration sqval-arrow-left)]
  if action = parrowright [ set sqval (item iteration sqval-arrow-right)]
  report sqval
end

; MAJ la qval pour l'action précédente
to update-sqval [next-reward next-sqval]
  ask patch sqval-prev-patch-pxcor sqval-prev-patch-pycor [
    let sqval-prev (get-sqval sqval-prev-action sqval-prev-iteration)
    let sqval-new 0
    ;show (word sqval-prev " " next-sqval)

    ifelse reward-bounty-available?
    [
      set sqval-new (1 - alpha) * sqval-prev + alpha * (next-reward + reward-bounty + gamma * next-sqval)
      set reward-bounty-available? false
    ]
    [
      set sqval-new (1 - alpha) * sqval-prev + alpha * (next-reward + gamma * next-sqval)
    ]

    let iteration sqval-prev-iteration
    if sqval-prev-action = pup [ set sqval-up (replace-item iteration sqval-up sqval-new) ]
    if sqval-prev-action = pdown [ set sqval-down (replace-item iteration sqval-down sqval-new) ]
    if sqval-prev-action = pleft [ set sqval-left (replace-item iteration sqval-left sqval-new) ]
    if sqval-prev-action = pright [ set sqval-right (replace-item iteration sqval-right sqval-new) ]
    if sqval-prev-action = parrowup [ set sqval-arrow-up (replace-item iteration sqval-arrow-up sqval-new) ]
    if sqval-prev-action = parrowdown [ set sqval-arrow-down (replace-item iteration sqval-arrow-down sqval-new) ]
    if sqval-prev-action = parrowleft [ set sqval-arrow-left (replace-item iteration sqval-arrow-left sqval-new) ]
    if sqval-prev-action = parrowright [ set sqval-arrow-right (replace-item iteration sqval-arrow-right sqval-new) ]
  ]
end

; Choisit une action (soit au hasard soit par rapport à max qval)
to-report sq-choice-action [iteration]
  let action 0
  ifelse random-float 1 < epsilon [
    ifelse check-arrow-left [
      set action random 8
    ]
    [
      set action random 4
    ]
  ]
  [
    set action (choice-action-max-sqval true iteration)
  ]
  report action
end

; Retourne l'action avec la max qval
; Cette fonction va aussi prendre en compte les flèches si dispo
; On peut outrepasser la vérification des flèches avec check-arrow? false, utilisé pour la reinf-visu (pour afficher l'action avec la max qval quelquesoit s'il reste des flèches ou pas)
to-report choice-action-max-sqval [check-arrow? iteration]
  let list-actions 0
  ; Initialiser les valeurs pour cette itération si elles n'existent pas (c'est fait dans get-sqval)
  let temp get-sqval pup iteration ; temp et pup sont choisis au hasard ici
  ; La liste doit être dans le même ordre que les constantes de déplacement pup pdown, etc..
  ifelse not check-arrow? or check-arrow-left [
    set list-actions (list (item iteration sqval-up) (item iteration sqval-down) (item iteration sqval-left) (item iteration sqval-right) (item iteration sqval-arrow-up) (item iteration sqval-arrow-down) (item iteration sqval-arrow-left) (item iteration sqval-arrow-right))
  ]
  [
    set list-actions (list (item iteration sqval-up) (item iteration sqval-down) (item iteration sqval-left) (item iteration sqval-right))
  ]
  ; On choisit l'action avec la valeur qval maximum et uniquement parmi les actions possibles (eg: si le patch n'existe pas on ne choisit pas)
  let action 0
  let i 0
  let sqval-max nobody
  foreach list-actions [
    ; Si cette action a une plus grand valeur qval que les précédentes
    if sqval-max = nobody or ? >= sqval-max [
      ; On vérifie que le patch existe dans cette direction, sinon on y va pas c'est inutile
      if get-neighbour-patch i != nobody [
        set action i ; On mémorise cette action
        set sqval-max ? ; On met à jour la valeur qval-max
      ]
    ]
    set i i + 1
  ]
  report action
end

; Fait une action pour le Q-Learning et évite les exceptions (par exemple si la case n'existe pas)
to-report sq-choose-and-do-action-carefully [iteration]
  let action 0

  ; On reessaie en boucle jusqu'à trouver une action qui ne cause pas d'erreur (au hasard si necessaire)
  let no-error false
  let i 0
  let maxIter 200
  while [not no-error] [
    carefully
    [
      set action sq-choice-action iteration
      do-action action
      set no-error true
    ]
    [
    ]
    ; Devrait ne jamais arriver, mais si on boucle, c'est qu'il y a un probleme (soit pas d'action possible, soit agent choisit toujours la même et elle est mauvaise et epsilon-greedy est désactivé/mis à 0)
    if i > maxIter [
      error "L'agent ne peut pas choisir d'action valide! Soit il est complètement bloqué, soit il choisit toujours une action impossible et epsilon-greedy est désactivé! Essayez de mettre epsilon > 0"
    ]
    set i (i + 1)
  ]
  report action
end


;=== NEURAL Q-LEARNING FONCTIONS AUXILIAIRES ===
; Voir http://computing.dcu.ie/~humphrys/Notes/RL/q.neural.html et http://computing.dcu.ie/~humphrys/PhD/ch4.html#4.3.2 ou le livre de Richard S. Sutton and Andrew G. Barto : http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html
; Cette implémentation suit l'approche proposée par Lin [Lin, 1992], on utilise donc un réseau de neurone par action a, qui chacun s'occupera d'approximer une fonction (action). On peut donc considérer que chaque réseau donne en sortie une fonction Qa(s) au lieu de Q(s, a) (l'action est implicitement donnée par le réseau qu'on choisit).

; Retourne la liste des neighbors4 mais toujours dans le meme ordre (pas comme neighbors4): haut bas gauche droite
to-report neighbors4-ordered
  report (list (patch-at 0 1) (patch-at 0 -1) (patch-at -1 0) (patch-at 1 0))
  ; to be used like this: ask patch 1 1 [ let i 0 foreach neighbors4-ordered [ ask ? [ set plabel i set i (i + 1) ] ] ]
end

; Crée un nouvel example à rajouter dans X
; Sert à définir les features utilisées pour l'apprentissage du neural network
; NOTE: si vous rajoutez des features ici, n'ouliez pas de modifier nbfeatures dans setup-globals!
to-report nq-make-example [action]

  let current-qval nqval-prev-max
  let stench? 0
  let breeze? 0
  let monster-killed? 0 ; peut produire du bruit, et de toutes facons il y a la bounty pour favoriser la chasse au wumpus
  let arrow-left-count 0

  let monster-threat-neighbors []
  let pits-threat-neighbors []
  let unexplored-neighbors []
  let visited-count-neighbors []
  let exploratory-interests []
  let g-scores []
  let safe-scores []

  let monster-threat-neighbor 0
  let pits-threat-neighbor 0
  let unexplored-neighbor 0
  let visited-count-neighbor 0
  let exploratory-interest 0
  let g-score 0
  let safe-score 0

  ask explorers [

    set arrow-left-count arrow-left
    if heard-scream [ set monster-killed? 1 ]

    ask patch-here [

      if stench [ set stench? 1 ]
      if breeze [ set breeze? 1 ]

      ifelse nq-surroundFeatures [
        let max-visited-count (max [visited-count] of patches) ; Optimisation...
        foreach neighbors4-ordered [ ; On doit fixer l'ordre dans lequel on ajoute les threats, car ils doivent toujours se trouver au meme noeud pour que le neural net puisse en déduire la direction
          ifelse ? = nobody [ ; Si la case n'existe pas (on est au bord) alors on doit quand même ajouter une valeur (nulle par defaut)
            set monster-threat-neighbors (lput 0 monster-threat-neighbors)
            set pits-threat-neighbors (lput 0 pits-threat-neighbors)
            set unexplored-neighbors (lput 0 unexplored-neighbors) ; Inintéressant de visiter une case inaccessible! Donc on met 0
            set visited-count-neighbors (lput max-visited-count visited-count-neighbors)
            set exploratory-interests (lput 0 exploratory-interests)
            set g-scores (lput 0 g-scores)
            set safe-scores (lput 0 safe-scores)
          ]
          [ ; Sinon la case existe, on peut recuperer sa valeur
            ask ? [
              set monster-threat-neighbors (lput monster-threat-score monster-threat-neighbors)
              set pits-threat-neighbors (lput pits-threat-score pits-threat-neighbors)
              let un 0
              if not visited [ set un 1 ]
              set unexplored-neighbors (lput un unexplored-neighbors)
              set visited-count-neighbors (lput visited-count visited-count-neighbors)
              set exploratory-interests (lput compute-exploratory-interest exploratory-interests)
              set g-scores (lput global-score g-scores)
              let s 0
              if safe [set s 0]
              set safe-scores (lput s safe-scores)
            ]
          ]
        ]
      ]
      [
        let neighbor-patch (get-neighbour-patch action)
        if neighbor-patch != nobody [
          ask neighbor-patch [
            set monster-threat-neighbor monster-threat-score
            set pits-threat-neighbor pits-threat-score
            if not visited [ set unexplored-neighbor 1 ]
            set visited-count-neighbor visited-count
            set exploratory-interest compute-exploratory-interest
            set g-score global-score
            if safe [set safe-score 1]
          ]
        ]
      ]
    ]
  ]

  ifelse nq-surroundFeatures [
    ifelse use-global-score [
      report matrix:from-row-list (list (sentence (list current-qval ) g-scores ))
    ]
    [
      report matrix:from-row-list (list (sentence (list current-qval stench? breeze? arrow-left-count) monster-threat-neighbors pits-threat-neighbors unexplored-neighbors visited-count-neighbors exploratory-interests safe-scores))
    ]
  ]
  [
    ;show "current-qval stench? breeze? arrow-left-count monster-threat-neighbor pits-threat-neighbor unexplored-neighbor visited-count-neighbor exploratory-interest" ; DEBUG
    ;print (word action ": " (monster-threat-neighbor * -1 + pits-threat-neighbor * -1 + exploratory-interest))
    ifelse use-global-score [
      report matrix:from-row-list (list (sentence (list current-qval g-score )))
    ]
    [
      report matrix:from-row-list (list (sentence (list current-qval stench? breeze? arrow-left-count monster-threat-neighbor pits-threat-neighbor unexplored-neighbor visited-count-neighbor exploratory-interest safe-score)))
    ]
  ]
end

to-report nq-update-example [newX prevX]
  let prev-X-list (matrix:get-row prevX 0)
  let prev-X-memo (sublist prev-X-list 0 (length prev-X-list - nbfeatures)) ; Exemples précédents mémorisés (on mémorise les nq-memo features précédentes pour que le réseau de neurones puisse avoir une certaine mémoire du passé)
  let new-X (matrix:get-row newX 0) ; Nouvel exemple qu'on va préposer devant les anciens exemples
  report matrix:from-row-list (list (sentence new-X prev-X-memo )) ; On doit stocker avant l'action l'exemple (état) en cours
end

to-report compute-exploratory-interest
  let v 1
  if visited [
    set v 0
  ]
  report (count neighbors4 with [not visited]) + v
end

; Procédure de révision: on mémorise les expériences précédentes pour les rejouer (replay)
; On limite la liste a revisionHistory, sauf si revisionHistory est a 0
to-report nq-append-example [action megaX xnew]
  let X []

  ; D'abord on prétraite xnew pour que ce soit toujours une matrice
  ; Scalaire -> matrice
  ifelse is-number? xnew [
    set xnew matrix:from-row-list (list (list xnew ))
  ]
  [
    ; Liste -> matrice
    if is-list? xnew [
      set xnew matrix:from-row-list (list xnew )
    ]
    ; Sinon c'est deja une matrice, on n'a rien à faire
  ]

  ifelse (item action megaX) = nobody [ ; Si pas d'ancienne matrice (on initialize là avec la première valeur) alors on en crée une
    set X xnew ; Tout simplement on dit que xnew c'est X
  ]
  ; On récupère l'ancienne matrice seulement s'il y en a une
  [
    ; On extrait la matrice des exemples et des labels pour cette action (pour le réseau de neurones de cette action)
    set X (item action megaX)

    ; On rajoute le nouvel exemple
    ifelse is-number? X [
      set X (matrix-append-row X (list xnew))
    ]
    [
      set X (matrix-append-row X (matrix:get-row xnew 0))
    ]
  ]

  ; Pruning des anciennes entrée si on excède revisionHistory
  ;let dim matrix:dimensions X
  ;let M item 0 dim
  ;if replayHistory > 0 and M > replayHistory [
  ;  let excedent (M - replayHistory)
  ;  set X (matrix-slice X (word excedent ":end") ":") ; On enleve les plus anciennes entree
  ;]

  ; Enfin on replace les matrices mises à jour dans les mega matrices
  set megaX (replace-item action megaX X)

  report megaX
end

; Récupère la valeur qval pour une action
to-report get-nqval [action]
  let X nq-make-example action ; Ne pas ajouter l'unité de biais ici, c'est fait automatiquement par le réseau de neurones

  let tmp nobody
  set tmp nnForwardProp 0 nqval-neurons_per_layer (item action nqval-megaTheta) (nq-update-example X nqval-prev-X)
  let nqval (matrix:get (item 0 tmp) 0 0) ; On extrait l'unique nombre qui est retourné dans la matrice
  let A (item 1 tmp)
  let Z (item 2 tmp)

  report (list nqval A Z)
end

; MAJ la qval pour l'action précédente
to update-nqval [next-reward next-nqval next-visited-count] ; les deux arguments sont la recompense et la q-valeur de la prochaine case à la case en cours (enfin la case précédente puisqu'on fait la MAJ à postériori)
  ; Quand le monstre est tué, on a droit a une récompense supplémentaire
  if reward-bounty-available? [
    set next-reward next-reward + reward-bounty
    set reward-bounty-available? false
  ]
  ; Idem quand on explore une case inconnue, favorise l'exploration
  if reward-bounty-exploration-available? [
    set next-reward next-reward + reward-bounty-exploration
    set reward-bounty-exploration-available? false
  ]
  ; Penalité quand on backtrack (on revisite une case déjà visitée, pas juste la case précédente)
  set next-reward next-reward + (penalty-backtrack * 0.1 * (min (list (max (list 0 (next-visited-count - 1))) 3)) ) ; visité 0 ou 1 fois OK, mais plus non. On limite max a 3 pour eviter trop grand nombres (constantes magiques!)
  ; Pénalité pour chaque pas (devrait favoriser la recherche du trésor et du chemin le plus court)
  set next-reward next-reward + penalty-step * (min (list 50 steps)) ; limité à 50 pas max sinon ca va tendre vers l'infini

  ; On calcule la valeur Y pour l'action qu'on vient d'effectuer
  let new-nqval (1 - alpha) * nqval-prev-max + alpha * (next-reward + gamma * next-nqval)

  ; == Mode stochastique simple: on fait la backpropagation tout de suite avec un seul exemple
  ifelse not replay [
    let M 1
    let Y new-nqval
    let Ypred nqval-prev-max
    let Theta (item nqval-prev-action nqval-megaTheta)
    ; Au cas où, si on n'a pas tous les éléments nécessaires pour la précédente action, on recalcule la propagation avant
    if nqval-prev-A = nobody [
      let tmp get-nqval nqval-prev-action
      set nqval-prev-max (item 0 tmp)
      set nqval-prev-A (item 1 tmp)
      set nqval-prev-Z (item 2 tmp)
    ]

    ; Calcul du coût actuel (avant propagation avant)
    let J (nnComputeCost 0 nqval-neurons_per_layer nnlambda M Theta nqval-prev-A Y Ypred)
    plot-learning-curve "learn" nqval-prev-action J

    ; Propagation arrière (propagation de l'erreur et apprentissage)
    let Theta_grad (nnBackwardProp 0 nqval-neurons_per_layer nnlambda M Theta nqval-prev-A nqval-prev-Z Y )

    ; Modification des paramètres en utilisant le gradient
    let Layers (length nqval-neurons_per_layer)
    let i 0
    while [i < (Layers - 1)] [
      let newval (matrix:plus (item i Theta) (matrix:times-scalar (matrix:times-scalar (item i Theta_grad) nnStep) -1))
      set Theta (replace-item i Theta newval)
      set i (i + 1)
    ]

    ; Enfin, on replace le nouveau Theta à sa place dans megaTheta (qui contient les Thetas pour toutes les actions)
    set nqval-megaTheta (replace-item nqval-prev-action nqval-megaTheta Theta)
  ]

  ; == Mode replay: on stocke les exemples et on les apprend plus tard, puis on recommence. Cela assure une certaine convergence d'après [Lin, 1992]
  [
    ;  On est obligé de limiter car sinon la valeur ne va jamais converger puisque les valeurs se basent sur leurs propres estimations!
    if nbReplays = 0 or replay-count < nbReplays [ ; On peut désactiver la vérification avec nbReplays 0

      ; D'abord on ajoute le nouvel exemple
      let rep 1
      if next-reward > 0 [ set rep 5 ]
      let i 0
      while [i < rep] [
        set nqval-megaX (nq-append-example nqval-prev-action nqval-megaX nqval-prev-X)
        set nqval-megaY (nq-append-example nqval-prev-action nqval-megaY new-nqval)
        set i (i + 1)
      ]

      let stoch 0
      if nq-stochastic [
        set stoch 2
      ]
      ; Et si on dépasse le nombre de générations pour un replay, on effectue le replay
      if ticks > (ticksPerReplay + replay-last-ticks) [
        let action 0
        while [action < nbactions] [
          if verbose [ print (word "NQ-Learning replay for action " action) ]
          let X (item action nqval-megaX)
          let Y (item action nqval-megaY)
          let Theta (item action nqval-megaTheta)

          if X != nobody and Y != nobody [ ; On apprend seulement si on a au moins un exemple pour cette action dans ce replay
            let tmp (nnLearn action 0 stoch Theta nqval-neurons_per_layer nnlambda X Y [] [] nnStep nqIterMax nqSeuilDiffGrad)
            set Theta (item 0 tmp)
            set nqval-megaTheta (replace-item action nqval-megaTheta Theta) ; On remplace le nouveau Theta appris dans la mega matrice Theta
          ]

          set action (action + 1)
        ]

        ; Enfin on vide les mega matrices d'exemples pour le prochain replay
        set nqval-megaX (n-values nbactions [nobody])
        set nqval-megaY (n-values nbactions [nobody])

        ; Et on mémorise le nombre de ticks où on s'est arrété la dernière fois qu'on a fait un replay
        set replay-last-ticks ticks

        ; Et on incrémente le compte des replays. On est obligé de limiter car sinon la valeur ne va jamais converger puisque les valeurs se basent sur leurs propres estimations!
        set replay-count (replay-count + 1)
      ]
    ]
  ]
end

; Retourne l'action avec la max qval
; Cette fonction va aussi prendre en compte les flèches si dispo
; On peut outrepasser la vérification des flèches avec check-arrow? false, utilisé pour la reinf-visu (pour afficher l'action avec la max qval quelquesoit s'il reste des flèches ou pas)
to-report choice-action-max-nqval [check-arrow?]
  let list-actions 0
  ; La liste doit être dans le même ordre que les constantes de déplacement pup pdown, etc..
  ifelse not check-arrow? or check-arrow-left [ ; On peut outrepasser la verification pour la visualisation par exemple afin d'afficher l'action maximum meme s'il n'y a plus de fleches
    set list-actions (list pup pdown pleft pright parrowup parrowdown parrowleft parrowright)
  ]
  [
    set list-actions (list pup pdown pleft pright)
  ]
  ; On choisit l'action avec la valeur qval maximum et uniquement parmi les actions possibles (eg: si le patch n'existe pas on ne choisit pas)
  let action nobody
  let nqval-max nobody
  let A nobody
  let Z nobody
  ; Pour chaque action possible
  foreach list-actions [
    ; On vérifie que cette action est possible (soit que la case existe, soit que c'est un tir de flèche donc pas un déplacement)
    if ? >= 4 or get-neighbour-patch ? != nobody [
      ; Si cette action a une plus grand valeur qval que les précédentes
      let tmp (get-nqval ?)
      let nqval (item 0 tmp) ; On récupère la valeur nqval pour cette action
      if nqval-max = nobody or nqval > nqval-max or (nqval = nqval-max and 0.5 > random-float 1) [ ; Si l'action a la même nqval qu'une autre, on choisit l'une des deux au hasard (évite les boucles ou l'agent revient sans arret sur ses pas, et aussi au démarrage où il va toujours dans la même direction puisque par défaut toutes les actions ont la même qval)
        ; On vérifie que le patch existe dans cette direction, sinon on y va pas c'est inutile
        if get-neighbour-patch ? != nobody [
          set action ? ; On mémorise cette action
          set nqval-max nqval ; On met à jour la valeur qval-max
          set A (item 1 tmp) ; Et on mémorise les paramètres pour la propagation arrière
          set Z (item 2 tmp)
        ]
      ]
    ]
  ]
  report (list action nqval-max A Z)
end

; Choisit une action (soit au hasard soit par rapport à max qval)
to-report nq-choice-action
  let action 0
  let nqval-max 0
  let A nobody
  let Z nobody
  ifelse random-float 1 < epsilon [
    ifelse check-arrow-left [
      set action random 8
    ]
    [
      set action random 4
    ]
    ; Même au hasard, on doit quand même calculer les valeurs nécessaires pour la backpropagation plus tard
    let tmp get-nqval action
    set nqval-max (item 0 tmp)
    set A (item 1 tmp)
    set Z (item 2 tmp)
  ]
  [
    let tmp (choice-action-max-nqval true)
    set action (item 0 tmp)
    set nqval-max (item 1 tmp)
    set A (item 2 tmp)
    set Z (item 3 tmp)
  ]
  report (list action nqval-max A Z)
end

; Fait une action pour le NQ-Learning et évite les exceptions (par exemple si la case n'existe pas)
to-report nq-choose-and-do-action-carefully
  let action 0
  let nqval-max 0
  let A nobody
  let Z nobody

  ; On reessaie en boucle jusqu'à trouver une action qui ne cause pas d'erreur (au hasard si necessaire)
  let no-error false
  let i 0
  let maxIter 200
  while [not no-error] [
;    carefully
;    [
      let tmp nq-choice-action
      set action (item 0 tmp)
      set nqval-max (item 1 tmp)
      set A (item 2 tmp)
      set Z (item 3 tmp)
      carefully [
        do-action action
        set no-error true
      ]
      [
      ]
;    ]
;    [
;    ]
    ; Devrait ne jamais arriver, mais si on boucle, c'est qu'il y a un probleme (soit pas d'action possible, soit agent choisit toujours la même et elle est mauvaise et epsilon-greedy est désactivé/mis à 0)
    if i > maxIter [
      error "L'agent ne peut pas choisir d'action valide! Soit il est complètement bloqué, soit il choisit toujours une action impossible et epsilon-greedy est désactivé! Essayez de mettre epsilon > 0"
    ]
    set i (i + 1)
  ]
  report (list action nqval-max A Z)
end

;=== GESTION DU PLAY MODE ===

; Déplacement haut/bas/gauche/droite
to play-mode-go-up
  if playmode [
    set playmode-direction pup
    play-mode-go
  ]
end

to play-mode-go-down
  if playmode [
    set playmode-direction pdown
    play-mode-go
  ]
end

to play-mode-go-left
  if playmode [
    set playmode-direction pleft
    play-mode-go
  ]
end

to play-mode-go-right
  if playmode [
    set playmode-direction pright
    play-mode-go
  ]
end

; TODO comments
to manage-fogofwar [ prev-patch ]
  ; En fogofwar-harder, on ne voit _uniquement_ que la case actuelle, pas les cases précédemment visitées
  if fogofwar-harder [
    ask prev-patch [ set pcolor black ]
  ]
  ask patch-here [
    ; on affiche la case actuelle avec la bonne couleur (sens)
    color-patch-visu
    ; on la rajoute dans la liste (inutile pour le moment)
    ;set fogofwar-pvisible lput self fogofwar-pvisible
    ; et on affiche les agents présents sur la case
    if any? turtles-here[
      ask turtles-here [
        if not is-happyface? self [ ; au cas ou on est sur la case ou il y a Happyface qui est caché...
          show-turtle
        ]
      ]
    ]
  ]
end

; Boucle principale en mode Play
; Gestion de la mécanique de déplacement du joueur et de l'état du jeu (fogofwar, fin de partie)
; Remplace la boucle go en jeu automatique (donc si de nouvelles mécaniques sont rajoutées dans go, les rajouter aussi ici)
to play-mode-go
  ; Ajoute un pas
  tick
  set steps steps + 1

  ask explorer 0 [
    ; Mémorise la case précédente (pour fogofwar-advanced)
    let prev-patch patch-here
    ; Déplacement du joueur selon les touches pressées
    carefully[move-to get-neighbour-patch playmode-direction][ print error-message ] ; carefully pour eviter erreur si la case n'existe pas (l'agent va juste rester sur place)

    ; On affiche la case actuelle si il y a un fogofwar
    if fogofwar [
      manage-fogofwar prev-patch
    ]

    ; Vérification de l'état du jeu (gagné/perdu ou rien et on peut continuer?)
    let curr-patch patch-here
    if check-dangers-and-treasure [
      ask turtles-on curr-patch [
        if is-treasure? self [
          ask happyfaces [
            show-turtle
            user-message "You found the fabulous treasure, congratulations!!!"
            hide-turtle
          ]
        ]
        if is-monster? self [
          let prevpatch patch-here
          setxy (max-pxcor / 2) (min-pycor / 2)
          set size 9
          ask treasures [ show-turtle ]
          user-message "YUMMY! The Wumpus devoured you wholly! It is now well fed, thanks to you..."
          set size 1
          move-to prevpatch
        ]
        if is-pit? self [
          set size 10
          ask treasures [ show-turtle ]
          user-message "Aaaaaah! You disappear in a bottomless pit, never to be seen again..."
          set size 1
        ]
      ]
    ]
  ]

  ; MAJ des dangers sur la case
  update-threats
  ; MAJ du score global
  update-global-scores
  ; Visualisation des valeurs d'apprentissage par renforcement
  reinf-visualisation

  ; Quelques post-traitements après chaque fin de partie
  if justfinished [
    ; Variation simple (avec environnement stable): on replace le monstre à sa place pour pouvoir apprendre à le tuer
    ifelse not harder-variation [
      reset-explorers-items ; On réattribue tous les items à l'explorateur
      if not any? monsters [ respawn-monsters ] ; On respawn le monstre s'il a été tué
      reset-threats ; On reinit les threats
      setup-patches-visu ; On MAJ la coloration des patchs
      if fogofwar [ setup-fogofwar ] ; On recache toutes les cases à chaque fin de partie, sauf celle du début
    ]
    ; Harder variation du problème: on réinitialise la position de tous les agents aléatoirement
    [
      resetup-harder
    ]

    ; Plot et reset steps
    if found-treasure [ do-plot-learning-curve ] ; ne tracer que si on a trouvé le trésor, sinon si on a perdu l'information n'a aucune valeur (si on bouge aléatoirement dans n'importe quel sens on aura un nombre bas de steps)
    set gen gen + 1
    set steps 0

    ; Reset variables globales de l'état du jeu
    set justfinished false
    set found-treasure false
  ]
end

; Récupère le patch voisin selon la direction indiquée
to-report get-neighbour-patch [ direction ]
  if (direction = pup or direction = parrowup) [report patch-at 0 1]
  if (direction = pdown or direction = parrowdown) [report patch-at 0 -1]
  if (direction = pleft or direction = parrowleft) [report patch-at -1 0]
  if (direction = pright or direction = parrowright) [report patch-at 1 0]
  error (word "get-neighbour-patch called with unvalid direction: " direction)
end

;=== GESTION ACTION FLÈCHES ET MORT DU MONSTRE ===

; Tir haut/bas/gauche/droite
to shoot-arrow-up
  check-and-shoot-arrow pup
end

to shoot-arrow-down
  check-and-shoot-arrow pdown
end

to shoot-arrow-left
  check-and-shoot-arrow pleft
end

to shoot-arrow-right
  check-and-shoot-arrow pright
end

; Vérifie qu'on peut tirer et initialise le tir
to check-and-shoot-arrow [direction]
  if check-arrow-left [
    ; On tire dans la direction souhaitée
    shoot-arrow direction
  ]
end

; Vérifie le nombre de flèches disponibles
to-report check-arrow-left
  let can-shoot-arrow true
  ask explorer 0 [
    ifelse arrow-left > 0 [
      set can-shoot-arrow true
    ]
    [
      set can-shoot-arrow false
    ]
  ]
  if playmode and not can-shoot-arrow [
    user-message "Vous n'avez plus de flèche, vous ne pouvez pas tirer!"
  ]
  report can-shoot-arrow
end

to shoot-arrow [direction]
  shoot-arrow-aux direction 0 0 0 0 0
end

; Gère la mécanique de mouvement de la flèche, en la déplaçant en ligne droite jusqu'à ce qu'elle rencontre un obstacle, les limites du monde ou le monstre
to shoot-arrow-aux [direction step currpxcor currpycor startpxcor startpycor]

  ; Initialisation et on décrémente le nombre de flèches dispo si c'est la première itération
  if step = 0 [
    ask explorer 0 [
      set arrow-left arrow-left - 1
      set currpxcor pxcor
      set currpycor pycor
      set startpxcor pxcor
      set startpycor pycor
    ]
  ]

  ; Mécanique de déplacement de la flèche
  let missed false

  ask patch currpxcor currpycor [
    ; Si on a atteint les limites du monde, on arrete
    ifelse get-neighbour-patch direction = nobody [
      set missed true
    ]
    [
      ask get-neighbour-patch direction [
        ifelse any? monsters-here [
          kill-monster
        ]
        [
          ; Si la flèche tombe dans un trou ou dans le trésor ou qu'on est revenu au point de départ, on la perd
          ifelse any? pits-here or any? treasures-here or (currpxcor = startpxcor and currpycor = startpycor) [
            set missed true
          ]
          [
            shoot-arrow-aux direction (step + 1) pxcor pycor startpxcor startpycor
          ]
        ]
      ]
    ]
  ]
  if playmode and missed [
    user-message "Flèche perdue..."
  ]
end

; Quand on tue le monstre, supprimer toutes les traces (sens stench et pénalités), les threats scores et émettre un cri
to kill-monster
  if verbose [ show "Monstre tué!" ]
  set count-killed-monster count-killed-monster + 1
  ask monsters [
    ; On enleve les sens et pénalités sur les cases voisines (important pour l'apprentissage par renforcement, sinon l'agent continuera d'avoir peur)
    ask neighbors4 [
      set stench false
      set reward reward - penalty-fear
      if (not fogofwar or pcolor != black) [ color-patch-visu ] ; MAJ de la couleur de cette case (seulement si pas de fogofwar ou sinon si on a déjà visité cette case)
    ]
    ; Enlève la super pénalité sur cette case où il y avait le monstre avant
    ask patch-here [
      set reward reward - penalty-eaten
      if (not fogofwar) [ color-patch-visu ] ; MAJ de la couleur de cette case
    ]
    ; L'explorateur entend le cri du monstre (permet à l'agent de savoir que ce danger est écarté, très important pour apprentissage)
    ask explorers [
      set heard-scream true
    ]
    ; La récompense pour avoir tué le monstre devient disponible
    set reward-bounty-available? true
    ; Affiche un message en playmode (pour que le joueur sache aussi comme l'agent que le monstre a été tué)
    if playmode [
      user-message "Un cri retentit depuis les entrailles de ce sombre endroit. Le Wumpus n'est plus!"
    ]
    die
  ]
  ask patches with [visited = false] [
    set monster-threat-score 0
  ]
end

;=== MAP EDITOR/DESIGNER ===

; Permet la création de trous par click
; Note: on est obligé de procéder ainsi car il faut une boucle qui tourne pour qu'on puisse détecter un click, en NetLogo 5 on ne peut pas encore appeler une procédure suite à un click
to design
  let create-pit? false
  ; S'il y a un click
  if mouse-down? [
    ; On regarde le patch à l'endroit du click
    ask patch mouse-xcor mouse-ycor [
      ; S'il y a déjà un trou, on le supprime ainsi que les sens et pénalités associées
      ifelse any? pits-here [
        ask pits-here [
          ask neighbors4 [
            set breeze false
            set reward reward - penalty-fear ; on enleve seulement la part de pénalité, car il peut y avoir d'autres pénalité avec d'autres dangers autour
          ]
          die
        ]
        set reward reward - penalty-fell
      ]
      ; Sinon, pas de trou, on en crée un (on doit sortir du contexte agent)
      [
        if not any? turtles-here [
          set create-pit? true
        ]
      ]
    ]
    ; Crée le trou en contexte observer
    if create-pit? [
      create-pits 1 [
        move-to patch mouse-xcor mouse-ycor
        ; Apparence
        set shape "tile water"
        set color black

        ; Initialisation des variables propres
        ask neighbors4 [
          ; Mise en place des sens sur les cases voisines
          set breeze true
          ; Pénalité d'approcher le précipice
          set reward reward + penalty-fear
        ]
        ; Pénalité pour l'explorateur quand il tombe
        ask patch-here [set reward reward + penalty-fell]
      ]
    ]

    ; Maj visualisation
    setup-patches-visu
  ]
end

;=== Plot ===

to do-plot-learning-curve
  set-current-plot "Learning curve"
  set-current-plot-pen "steps"
  plot steps
end

to do-plot-winloss-ratio
  if prev-gen != gen [
    let new-won (count-won - prev-won)
    let new-loss (count-eaten + count-fell - prev-loss)
    let full (new-won + new-loss)

    set-current-plot "Win/Loss Ratio"
    set-current-plot-pen "win"
    plot new-won * 100 / full
    set prev-won count-won
    set-current-plot-pen "loss"
    plot new-loss * 100 / full
    set prev-loss (count-eaten + count-fell)

    set prev-gen gen
  ]
end



;==============================================================
;=================  NEURAL NETWORK FUNCTIONS  =================
;==============================================================

to-report addBias [X]
  let dim matrix:dimensions X
  let N item 0 dim
  let M item 1 dim
  let X2 matrix:make-constant N (M + 1) 1

  let i 0
  let j 0
  while [i < N] [
    while [j < M] [
      let a (matrix:get X i j)
      matrix:set X2 i (j + 1) a ; we don't touch on the first column
      set j (j + 1)
    ]
    set i (i + 1)
    set j 0
  ]
  report X2
end

to-report sigmoidGradient [X]
  let sigX (sigmoid X)
  report matrix:times-element-wise sigX (matrix:plus-scalar (matrix:times-scalar sigX -1) 1)
end

to-report sigmoid [X]
  let dim matrix:dimensions X
  let M item 0 dim
  let N item 1 dim
  let X2 matrix:copy X

  let i 0
  let j 0
  while [i < M] [
    while [j < N] [
      let a (matrix:get X i j)
      matrix:set X2 i j (sig_aux a) ; WARNING: it's possible to edit X directly, but this will produce a side-effect, as modifying X in this procedure will also modify X in the parent procedure and anywhere else where the same matrix X is used! So if we call let X2 sigmoid (sigmoid X), X will also be modified, and twice! That's why we must use a copy of X here.
      set j (j + 1)
    ]
    set i (i + 1)
    set j 0
  ]
  report X2
end

to-report sig_aux [x]
  let ret 0
  carefully [
    set ret 1 / (1 + e ^ (- x)) ; use e ^ (-x) so as to show to user that some values are too big and may produce weird results in the learning
  ]
  [
    if verbose [ show (word "Number too big for NetLogo, produces infinity in sig_aux with e ^ (- x): -x = " (- x)) ]
    set ret 1 / (1 + exp (min (list (- x) 709.7827128))) ; prefer using (exp x) rather than (e ^ x) because the latter can produce an error (number is too big) while exp will produce Infinity without error
    ; limit to a magic constant 709... to avoid Infinity. FIXME this is not a clean way to do it
  ]

  report ret
end

;ln(sigmoid(x) * (1+e^x)) = ln(1)
;ln(sigmoid(x)) + ln(1+e^x) = 0

;ln(s) + x * e = 0
;ln(s) = -(x * e)
;s = exp

to-report nnInitializeWeights [neurons_per_layer debug?]
  let Layers (length neurons_per_layer)
  let Theta n-values Layers [0]
  let i 1
  while [i < Layers] [
    ifelse debug? [
      set Theta replace-item (i - 1) Theta (nnDebugInitializeWeights (item (i - 1) neurons_per_layer) (item i neurons_per_layer))
    ]
    [
      set Theta replace-item (i - 1) Theta (nnRandInitializeWeights (item (i - 1) neurons_per_layer) (item i neurons_per_layer))
    ]
    set i (i + 1)
  ]
  report Theta
end

to-report nnDebugInitializeWeights [L_in L_out]
  let N (1 + L_in) ; 1+L_in because the first row of W handles the "bias" term
  let M L_out
  let W matrix:make-constant M N 0

  let ind 1
  let i 0
  let j 0
  while [j < N] [
    while [i < M] [
      matrix:set W i j (sin(rad-to-angle(ind)) / 10)
      set i (i + 1)
      set ind (ind + 1)
    ]
    set j (j + 1)
    set i 0
  ]

  report W
end

to-report rad-to-angle [rad]
  report 180 * rad / pi
end

to-report nnRandInitializeWeights [L_in L_out]
  let N (1 + L_in) ; 1+L_in because the first row of W handles the "bias" term
  let M L_out
  let W matrix:make-constant M N 0

  let epsilon_init 1 ; or 0.12, both are magic values anyway, so you can change that to anything you want

  let ind 1
  let i 0
  let j 0
  while [j < N] [
    while [i < M] [
      matrix:set W i j ((random-float 1) * 2 * epsilon_init - epsilon_init)
      set i (i + 1)
      set ind (ind + 1)
    ]
    set j (j + 1)
    set i 0
  ]

  report W
end

to-report crossval [X y learnPart]
  let dim matrix:dimensions X
  let M item 0 dim
  let N item 1 dim

  ; Generate list of indexes and shuffle it
  let indlist (gen-range 0 1 (M - 1))
  let indrand (shuffle indlist)

  ; Get the index where we must split the two datasets
  let numlearn (round (learnPart * M)) ; Be careful with round: round a * b = (round a) * b != (round (a * b))

  let indrandlearn sort (sublist indrand 0 numlearn)
  let indrandtest sort (sublist indrand numlearn M)


  ; Split the dataset in two
  let Xlearn (matrix-slice X indrandlearn ":")
  let Ylearn (matrix-slice y indrandlearn ":")
  let Xtest (matrix-slice X indrandtest ":")
  let Ytest (matrix-slice y indrandtest ":")

  report (list Xlearn Ylearn Xtest Ytest)
end

to-report nnLearn [action fmode stochOrBatch
    Theta neurons_per_layer lambda
    Xlearn Ylearn Xtest Ytest
    nnepsilon nIterMax diffThreshold]

  let dim matrix:dimensions Xlearn
  let M item 0 dim
  let Mtest []
  if not is-list? Xtest [ ; if it's not a matrix, it's an empty list
    let dimtest matrix:dimensions Xtest
    set Mtest item 0 dimtest
  ]

  let Layers (length neurons_per_layer)

  let orig_Xlearn []
  let orig_Ylearn []
  let orig_M M
  if stochOrBatch = 1 or stochOrBatch = 2 [
    set orig_Xlearn Xlearn
    set orig_Ylearn Ylearn
    set M 1
  ]

  let t 1
  let err []
  let errtest []

  let converged false
  let temp []
  let lastind (orig_M - 1)
  while [not converged] [
    ; == Preparing for stochastic gradient (pick only one example for each iteration instead of all examples at once)
    let randind -1
    if stochOrBatch = 1 [
      set randind (floor ((random-float 1) * orig_M))
      set Xlearn (matrix-slice orig_Xlearn randind ":")
      set Ylearn (matrix-slice orig_Ylearn randind ":")
    ]
    if stochOrBatch = 2 [
      set Xlearn (matrix-slice orig_Xlearn lastind ":")
      set Ylearn (matrix-slice orig_Ylearn lastind ":")
      set lastind (lastind - 1)
      if lastind < 0 [
        set lastind (orig_M - 1)
      ]
    ]

    ; == Forward propagating + Computing cost
    ; = Learning dataset
    set temp (nnForwardProp fmode neurons_per_layer Theta Xlearn)
    let Ypred (item 0 temp)
    let A (item 1 temp)
    let Z (item 2 temp)

    let er (nnComputeCost fmode neurons_per_layer lambda M Theta A Ylearn Ypred)
    ;set err (lput er err) ; Works OK but commented out to save memory space since it's plotted anyway
    plot-learning-curve "learn" action er
    if not is-list? Xtest [ ; if it's not a matrix, it's an empty list
      set temp (nnForwardProp fmode neurons_per_layer Theta Xtest)
      let Ypredtest (item 0 temp)
      let Atest (item 1 temp)
      let Ztest (item 2 temp)

      let ert (nnComputeCost fmode neurons_per_layer lambda Mtest Theta Atest Ytest Ypredtest)
      ;set errtest (lput ert errtest) ; Works OK but commented out to save memory space since it's plotted anyway
      plot-learning-curve "test" action ert
    ]

    ; == Back propagating
    ; Computing gradient
    let Theta_grad (nnBackwardProp fmode neurons_per_layer lambda M Theta A Z Ylearn)

    ; Comitting gradient to change our parameters Theta
    let Theta2 (n-values Layers [0])
    let i 0
    while [i < (Layers - 1)] [
      let newval (matrix:plus (item i Theta) (matrix:times-scalar (matrix:times-scalar (item i Theta_grad) (nnepsilon / (1 + (t - 1) * nnDecay)) ) -1))
      set Theta2 (replace-item i Theta2 newval)
      set i (i + 1)
    ]

    ; Computing gradient diff (to stop if we are below diffThreshold, meaning there's no change anymore in the gradient, which may not be true if we are in a plateau!)
    let wdiff 999
    if diffThreshold > 0 [
      let mdiff (matrix:plus (item 0 Theta2) (matrix:times-scalar (item 0 Theta) -1))
      set wdiff (matrix-sum (matrix-sqrt (matrix-sum (matrix:times-element-wise mdiff mdiff) 2) ) 1) ; First we must sum over the features and square root to compute the euclidian distance, then we can sum over all examples errors

      ; 2nd stop criterion: not enough change in the latest gradient, we suppose we converged (warning: way simply may be in a plateau, happens frequently in neural networks and other concave problems!)
      if wdiff <= diffThreshold [ set converged true ]
      ; NB: no need to recompute the cost/error, since the gradient didn't change so much, the error is about stable
    ]
    ; Update Theta now (we delayed and used Theta2 only to compute the gradient diff)
    set Theta Theta2

    ; Increment t (iteration counter)
    set t (t + 1)

    ; 1st stop criterion: we reached the maximum number of iterations allowed or user want to stop
    if t >= nIterMax or stop-nqlearn [
      set converged true

      ; Recompute the forward propagation and cost with the latest gradient... Below is just a copy-paste of what is above (FIXME)
      set er (nnComputeCost fmode neurons_per_layer lambda M Theta A Ylearn Ypred)
      ;set err (lput er err)
      plot-learning-curve "learn" action er
      if not is-list? Xtest [ ; if it's not a matrix, it's an empty list
        set temp (nnForwardProp fmode neurons_per_layer Theta Xtest)
        let Ypredtest (item 0 temp)
        let Atest (item 1 temp)
        let Ztest (item 2 temp)

        let ert (nnComputeCost fmode neurons_per_layer lambda Mtest Theta Atest Ytest Ypredtest)
        ;set errtest (lput ert errtest)
        plot-learning-curve "test" action ert
      ]
    ]

    ; Force refreshing of the plots
    display
  ]

  report (list Theta err errtest t)
end

to-report nnForwardProp [fmode neurons_per_layer Theta X]
  let dim matrix:dimensions X
  let M item 0 dim
  let Layers (length neurons_per_layer)

  ; == Forward propagation
  let Ypred []
  let A (n-values Layers [0]) ; Stores the untampered scores on each layer (mainly hidden layers)
  let Z (n-values Layers [0]) ; Stores the scores on each layer passed through a sigmoid function (you can edit here if you want to change the function of the hidden layers. For the output layer it's in nnComputeCost and nnBackwardProp)

  ; Init the propagation with X (with added bias unit)
  set A (replace-item 0 A addBias(X))

  ; Forward propagating the prediction scores
  let L 1
  while [L < Layers] [
    let Zval matrix:times (item (L - 1) A) (matrix:transpose (item (L - 1) Theta))
    set Z (replace-item L Z Zval)
    ifelse fmode = 0 and L = (Layers - 1) [
      set A (replace-item L A (item L Z))
    ]
    [
      set A (replace-item L A (sigmoid (item L Z)))
    ]
    if L < (Layers - 1) [
      set A (replace-item L A (addBias (item L A))) ; adding bias unit (except for the last layer)
    ]

    set L (L + 1)
  ]

  ; Post-processing the scores into a prediction (of class or of value)
  ifelse fmode = 0 [
    set Ypred (item (Layers - 1) A)
  ]
  [
    set Ypred (matrix-max (item (Layers - 1) A) 2)
  ]

  report (list Ypred A Z)
end

to-report nnComputeCost [fmode neurons_per_layer lambda M Theta A Y Ypred]
  let Layers (length neurons_per_layer)

  let J -1 ; -1 so that it will produce an error later if J is not assigned since a cost cannot be negative
  if fmode = 0 [
    if is-number? Y [ set Y matrix:from-row-list (list (list Y)) ] ; Convert to a matrix if it's only a number
    if is-number? Ypred [ set Ypred matrix:from-row-list (list (list Ypred)) ] ; Convert to a matrix if it's only a number
    let ydiff (matrix:plus Y (matrix:times-scalar Ypred -1))
    set J (matrix-sum (matrix-sum (matrix:times-element-wise ydiff ydiff) 2) 1) ; First we must sum over the features to compute some kind of euclidian distance, then we can sum over all examples errors (don't try to put a square root here, it degrades a lot the performances! This is a correct implementation after checking with nnCheckGradient)
    set J (1 / (2 * M) * J) ; scale cost relatively to the size of the dataset
  ]

  let sum_thetas 0
  let L 0
  while [L < (Layers - 1)] [
    let th (matrix-slice (item L Theta) ":" "1:end")
    set sum_thetas (sum_thetas + (matrix-sum (matrix-sum (matrix:times-element-wise th th) 1) 2) )
    set L (L + 1)
  ]
  set J (J + lambda / (2 * M) * sum_thetas)

  report J
end

to-report nnBackwardProp [fmode neurons_per_layer lambda M Theta A Z Y]
  let Layers (length neurons_per_layer)

  let delta (n-values Layers [0])
  let Ypredscores (item (Layers - 1) A)

  if is-number? Y [ set Y matrix:from-row-list (list (list Y)) ] ; Convert to a matrix if it's only a number

  ; == Initializing first value for back propagation
  ; We initialize the last delta for the last layer, and we backpropagate to the first layer (kind of the opposite of the forward propagation)
  if fmode = 0 [
    set delta (replace-item (Layers - 1) delta (matrix:plus Ypredscores (matrix:times-scalar Y -1)))
  ]

  ; == Backpropagating errors (for nodes) from last layer towards first layer
  let L (Layers - 2)
  while [L >= 1] [
    let tmp (matrix:times (item (L + 1) delta) (matrix-slice (item L Theta) ":" "1:end") )
    set tmp (matrix:times-element-wise tmp (sigmoidGradient (item L Z))) ; Change here if you want to use another function for the hidden layers than sigmoid! Sigmoid was used for its non-linearity and because it's pretty easy to use and understand.
    set delta (replace-item L delta tmp)
    set L (L - 1)
  ]

  ; == Backpropagating error gradient (for the weights) from last layer towards first layer
  let D (n-values Layers [0]) ; D = Theta_grad
  set L (Layers - 2)
  while [L >= 0] [
    let tmp (matrix:times (matrix:transpose (item (L + 1) delta)) (item L A))
    set tmp (matrix:times-scalar tmp (1 / M)) ; scale cost relatively to the size of the dataset
    set D (replace-item L D tmp)
    set L (L - 1)
  ]

  ; == Regularize the gradient
  set L (Layers - 2)
  while [L >= 0] [
    let tg (matrix-slice (item L D) ":" "1:end")
    let t (matrix-slice (item L Theta) ":" "1:end")
    let tmp matrix:plus tg (matrix:times-scalar t (lambda / M))
    let t1 (matrix:get-column (item L D) 0)
    set D (replace-item L D (matrix-prepend-column tmp t1))

    set L (L - 1)
  ]


  let Theta_grad D

  report Theta_grad
end

; Plot la courbe d'apprentissage (convergence) du reseau de neurones (pas de la réussite de l'agent!)
to plot-learning-curve [pen action cost]
  if curve-nqlearn [
    set-current-plot "Learning curve (neural net)"
    set-current-plot-pen (word pen action)
    plot cost
  ]
end



;==============================================================
;================= MATRIX AUXILIARY FUNCTIONS =================
;==============================================================

to-report matrix-slice [X indx indy]
  let X2 matrix:copy X

  let dim matrix:dimensions X2
  let M item 0 dim
  let N item 1 dim

  ifelse is-number? indx and is-number? indy [
    report matrix:get X2 indx indy
  ]
  [
    let indx1 0
    let indx2 M
    let indy1 0
    let indy2 N

    ; ROW VECTORIZED SLICING
    ; indx is a range of indexes (given as a string, format: "start:end")
    ifelse is-string? indx [
      ; Special case: we want all rows, in this case there's no processing necessary here
      ifelse indx = ":" or indx = "0:end" or indx = (word "0:" (M - 1)) [
        set indx1 0
        set indx2 M
      ]
      [
        ; Special case: contains only "end", shorthand for the maximum index
        ifelse indx = "end" [
          set indx (end-to-index M indx)
        ]
        ; Any other case we have a range
        [
          ; Extract the starting and ending indexes
          let indxsplit (split indx ":")
          set indx1 (item 0 indxsplit)
          set indx2 (item 1 indxsplit)

          ; Check if one of them contains "end" and replace with the correct value
          set indx1 (end-to-index M indx1)
          set indx2 (end-to-index M indx2)
          set indx2 min (list M (indx2 + 1)) ; add 1 because that's how submatrix works for the end column: it's exclusive, so indx1=1 indx2=2 will only select the 2nd row (not the 2nd and 3rd)
        ]
      ]
    ]
    [
      ; indx is just one index (a number)
      ; Note: indx as a list of indexes is processed later (since we can't vectorize it, there's no built-in function)
      if (is-list? indx) = false [
        set indx1 indx
        set indx2 (indx + 1)
        ;set X2 matrix:submatrix X2 indx 0 (indx + 1) N ; submatrix is more consistent than get-row or get-column since it always returns a matrix, and not a list or a number
      ]
    ]

    ; COLUMN VECTORIZED SLICING
    ; indx is a range of indexes (given as a string, format: "start:end")
    ifelse is-string? indy [
      ; Special case: we want all rows, in this case there's no processing necessary here
      ifelse indy = ":" or indy = "0:end" or indy = (word "0:" (N - 1)) [
        set indy1 0
        set indy2 N
      ]
      [
        ; Special case: contains only "end", shorthand for the maximum index
        ifelse indy = "end" [
          set indy (end-to-index N indy)
        ]
        ; Any other case we have a range
        [
          ; Extract the starting and ending indexes
          let indysplit (split indy ":")
          set indy1 (item 0 indysplit)
          set indy2 (item 1 indysplit)

          ; Check if one of them contains "end" and replace with the correct value
          set indy1 (end-to-index N indy1)
          set indy2 (end-to-index N indy2)
          set indy2 min (list N (indy2 + 1)) ; add 1 because that's how submatrix works for the end column: it's exclusive, so indy1=1 indy2=2 will only select the 2nd column (not the 2nd and 3rd)
        ]
      ]
    ]
    [
      ; indy is just one index (a number)
      ; Note: indy as a list of indexes is processed later (since we can't vectorize it, there's no built-in function)
      if (is-list? indy) = false [
        set indy1 indy
        set indy2 (indy + 1)
        ;set X2 matrix:submatrix X2 0 indy M (indy + 1) ; submatrix is more consistent than get-row or get-column since it always returns a matrix, and not a list or a number
      ]
    ]

    ; PROCESSING VECTORIZED SLICING
    ; Show precise error if out of bounds
    ;show (word indx1 " " indx2 " " indy1 " " indy2)
    if indx1 < 0 or indx2 > M or indy1 < 0 or indy2 > N [
      let errmsg ""
      if indx1 < 0 [ set errmsg (word errmsg "x (" indx1 ") can't be negative. ") ]
      if indx2 > M [ set errmsg (word errmsg "x (" indx2 ") exceeds matrix dimensions (" M "). ") ]
      if indy1 < 0 [ set errmsg (word errmsg "y (" indy1 ") can't be negative. ") ]
      if indy2 > N [ set errmsg (word errmsg "y (" indy2 ") exceeds matrix dimensions (" N "). ") ]
      error errmsg
    ]
    ; Doing the slicing using submatrix
    set X2 matrix:submatrix X2 indx1 indy1 indx2 indy2


    ; UNVECTORIZED SLICING
    ; if indx and/or indy is a list of indexes (not a range nor number), thus it can be a non-contiguous range of indexes (eg: 1, 3, 4), we have to do it by ourselves in a for loop
    ; We suppose that if indx is a list, then necessarily all rows from X were not touched and are intact in X2 (with same indexes), and thus we can extract from X2 with same indexes as in X. Same for indy and columns.

    ; indx is a list of indexes
    if is-list? indx [
      let Xlist []
      let i 0
      ; For each index in the list
      while [i < (length indx)] [
        ; Extract the index ind and then the row from X2 at this index
        let ind (item i indx)
        let row (matrix:get-row X2 ind)
        ; Append this row to our list of rows
        set Xlist lput row Xlist
        ; Increment...
        set i (i + 1)
      ]
      ; Finally, convert back to a matrix!
      set X2 matrix:from-row-list Xlist
    ]

    ; indy is a list of indexes
    if is-list? indy [
      let Xlist []
      let j 0
      ; For each index in the list
      while [j < (length indy)] [
        ; Extract the index ind and then the column from X2 at this index
        let ind (item j indy)
        let column (matrix:get-column X2 ind)
        ; Append this column to our list of columns
        set Xlist lput column Xlist
        ; Increment...
        set j (j + 1)
      ]
      ; Finally, convert back to a matrix!
      set X2 matrix:from-column-list Xlist
    ]
  ]

  ; RETURN THE SLICED MATRIX
  report X2
end

; row is a list
to-report matrix-append-row [X row]
  if not is-list? row [
    error "matrix-append-row: specified row is not a list!"
  ]
  report matrix:from-row-list lput row (matrix:to-row-list X)
end

; column is a list
to-report matrix-append-column [X column]
  if not is-list? column [
    error "matrix-append-column: specified column is not a list!"
  ]
  report matrix:from-column-list lput column (matrix:to-column-list X)
end

; row is a list
to-report matrix-prepend-row [X row]
  if not is-list? row [
    error "matrix-prepend-row: specified row is not a list!"
  ]
  report matrix:from-row-list fput row (matrix:to-row-list X)
end

; column is a list
to-report matrix-prepend-column [X column]
  if not is-list? column [
    error "matrix-prepend-column: specified column is not a list!"
  ]
  report matrix:from-column-list fput column (matrix:to-column-list X)
end

to-report end-to-index [maxind itm]
  let ret itm
  ; If it's a special string, we process it
  ifelse is-string? itm [
    if itm = "end" [
      set ret maxind
    ]
  ]
  ; Else if it's not a string (number, matrix, etc.), we return it as-is
  [
    set ret itm
  ]
  report ret
end

to-report split [string delimiter]
  let splitted []
  let i 0
  let previ 0
  ; Append any item place before where we meet the delimiter
  while [i < (length string)][
    if (item i string) = delimiter [
      let substr (substring string previ i)
      ; Try to convert to a number if possible
      carefully [
        set substr (read-from-string substr)
      ][]
      set splitted (lput substr splitted)
      set previ (i + 1)
    ]
    set i (i + 1)
  ]
  ; Appending the remaining substring after the last delimiter found
  if (length string) > previ [
    let substr (substring string previ (length string))
    carefully [
      set substr (read-from-string substr)
    ][]
    set splitted (lput substr splitted)
  ]
  report splitted
end

to-report gen-range [start step endd]
  report n-values (((endd - start) + step) / step) [(? * step) + start]
end

; Square root element wise for matrixes
; NB: does not support imaginary values (so you need to make sure there's no negative number in the matrix!)
to-report matrix-sqrt [X]
  let X2 matrix:copy X

  let dim matrix:dimensions X2
  let M item 0 dim
  let N item 1 dim

  let i 0
  let j 0
  while [j < N] [
    while [i < M] [
      matrix:set X2 i j (sqrt (matrix:get X2 i j))
      set i (i + 1)
    ]
    set j (j + 1)
    set i 0
  ]

  report X2
end

; Semi-vectorized procedure to sum a matrix over rows or columns
; If the matrix is not summable or there is only one element, it will return the same matrix
to-report matrix-sum [X columnsOrRows?] ; columnsOrRows? 2 = over columns
  if is-number? X [report X] ; if it's already a number, we've got nothing to do, just return it
  if is-list? X [report sum X] ; if it's a list we use the built-in function

  let dim matrix:dimensions X
  let M item 0 dim
  let N item 1 dim

  let Xret []
  ; Sum over columns
  ifelse columnsOrRows? = 2 [
    let i 0
    while [i < M] [
      set Xret lput (sum matrix:get-row X i) Xret
      set i (i + 1)
    ]
    ; Convert to a number if the list contains only one number
    ifelse (length Xret) = 1 [
      set Xret (item 0 Xret)
    ]
    ; Else convert back to a matrix (a vector in fact) to ease computations later
    [
      set Xret (matrix:from-column-list (list Xret))
    ]
  ]
  ; Else sum over rows
  [
    let j 0
    while [j < N] [
      set Xret lput (sum matrix:get-column X j) Xret
      set j (j + 1)
    ]
    ; Convert to a number if the list contains only one number
    ifelse (length Xret) = 1 [
      set Xret (item 0 Xret)
    ]
    ; Else convert back to a matrix (a vector in fact) to ease computations later
    [
      set Xret (matrix:from-row-list  (list Xret))
    ]
  ]

  report Xret
end

; Semi-vectorized procedure to return the max value of a matrix over rows or columns
; If the matrix is not summable or there is only one element, it will return the same matrix
; NB: it's a copy-cat of matrix-sum
to-report matrix-max [X columnsOrRows?] ; columnsOrRows? 2 = over columns
  if is-number? X [report X] ; if it's already a number, we've got nothing to do, just return it
  if is-list? X [report max X] ; if it's a list we use the built-in function

  let dim matrix:dimensions X
  let M item 0 dim
  let N item 1 dim

  let Xret []
  ; Over columns
  ifelse columnsOrRows? = 2 [
    let i 0
    while [i < M] [
      set Xret lput (max matrix:get-row X i) Xret
      set i (i + 1)
    ]
    ; Convert to a number if the list contains only one number
    ifelse (length Xret) = 1 [
      set Xret (item 0 Xret)
    ]
    ; Else convert back to a matrix (a vector in fact) to ease computations later
    [
      set Xret (matrix:from-column-list (list Xret))
    ]
  ]
  ; Else Over rows
  [
    let j 0
    while [j < N] [
      set Xret lput (max matrix:get-column X j) Xret
      set j (j + 1)
    ]
    ; Convert to a number if the list contains only one number
    ifelse (length Xret) = 1 [
      set Xret (item 0 Xret)
    ]
    ; Else convert back to a matrix (a vector in fact) to ease computations later
    [
      set Xret (matrix:from-row-list  (list Xret))
    ]
  ]

  report Xret
end

; Semi-vectorized procedure to return the min value of a matrix over rows or columns
; If the matrix is not summable or there is only one element, it will return the same matrix
; NB: it's a copy-cat of matrix-sum
to-report matrix-min [X columnsOrRows?] ; columnsOrRows? 2 = over columns
  if is-number? X [report X] ; if it's already a number, we've got nothing to do, just return it
  if is-list? X [report min X] ; if it's a list we use the built-in function

  let dim matrix:dimensions X
  let M item 0 dim
  let N item 1 dim

  let Xret []
  ; Over columns
  ifelse columnsOrRows? = 2 [
    let i 0
    while [i < M] [
      set Xret lput (min matrix:get-row X i) Xret
      set i (i + 1)
    ]
    ; Convert to a number if the list contains only one number
    ifelse (length Xret) = 1 [
      set Xret (item 0 Xret)
    ]
    ; Else convert back to a matrix (a vector in fact) to ease computations later
    [
      set Xret (matrix:from-column-list (list Xret))
    ]
  ]
  ; Else Over rows
  [
    let j 0
    while [j < N] [
      set Xret lput (min matrix:get-column X j) Xret
      set j (j + 1)
    ]
    ; Convert to a number if the list contains only one number
    ifelse (length Xret) = 1 [
      set Xret (item 0 Xret)
    ]
    ; Else convert back to a matrix (a vector in fact) to ease computations later
    [
      set Xret (matrix:from-row-list  (list Xret))
    ]
  ]

  report Xret
end

to-report pretty-print-megamatrix [megamatrix] ; megamatrix is a list of matrixes
  ifelse is-list? megamatrix [
    let n (length megamatrix)
    let i 0
    let text ""
    while [i < n] [
      set text (word text "\n" (word i ": " pretty-print-megamatrix (item i megamatrix)))
      set i (i + 1)
    ]
    report text
  ]
  [
    if megamatrix = nobody [
      report "nobody"
    ]
    ifelse is-number? megamatrix or is-string? megamatrix [
      report megamatrix
    ]
    [
      report (word "\n" matrix:pretty-print-text megamatrix)
    ]
  ]
end

; Deep recursive copy of a megamatrix (list of matrix)
to-report megamatrix-copy [megamatrix]
  ; Entry point: if it's a list, we will recursively copy everything inside (deep copy)
  ifelse is-list? megamatrix [
    let n (length megamatrix)
    let m2 (n-values n [0])

    let i 0
    while [i < n] [
      set m2 (replace-item i m2 (megamatrix-copy (item i megamatrix)))
      set i (i + 1)
    ]
    report m2
  ]
  ; Recursively called from here
  [
    ; If it's a number or string we just report it
    ifelse is-number? megamatrix or is-string? megamatrix [
      report  megamatrix
    ]
    ; Else if it's a matrix (we have no other way to check it but by default), then we make a copy and report it
    [
      report matrix:copy megamatrix
    ]
  ]
end
@#$#@#$#@
GRAPHICS-WINDOW
245
110
510
396
-1
-1
42.6
1
10
1
1
1
0
1
1
1
0
5
0
5
0
0
1
ticks
30.0

BUTTON
245
75
300
108
Init
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
355
75
410
108
Go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
300
75
355
108
1-step
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

INPUTBOX
0
210
60
270
pits-count
3
1
0
Number

SLIDER
0
140
150
173
epsilon
epsilon
0
1
0.9
0.025
1
(random walk)
HORIZONTAL

SLIDER
0
105
150
138
gamma
gamma
0
1
0.575
0.025
1
NIL
HORIZONTAL

SLIDER
0
70
150
103
alpha
alpha
0
1
0.41
0.01
1
NIL
HORIZONTAL

MONITOR
0
355
50
400
Eaten
count-eaten
0
1
11

MONITOR
50
355
100
400
Fell
count-fell
0
1
11

MONITOR
0
400
50
445
Killed W
count-killed-monster
0
1
11

MONITOR
50
400
100
445
Won
count-won
0
1
11

SWITCH
0
295
95
328
reinf-visu
reinf-visu
0
1
-1000

BUTTON
565
120
697
153
Launch Play Mode!
setup-play-mode
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
570
225
625
258
<-
play-mode-go-left
NIL
1
T
OBSERVER
NIL
Q
NIL
NIL
0

BUTTON
625
225
680
258
->
play-mode-go-right
NIL
1
T
OBSERVER
NIL
D
NIL
NIL
0

BUTTON
600
190
655
223
^
play-mode-go-up
NIL
1
T
OBSERVER
NIL
Z
NIL
NIL
0

BUTTON
600
260
655
293
v
play-mode-go-down
NIL
1
T
OBSERVER
NIL
S
NIL
NIL
0

SWITCH
80
240
210
273
harder-variation
harder-variation
1
1
-1000

TEXTBOX
580
90
730
111
PLAY MODE
20
0.0
1

TEXTBOX
5
10
225
56
WUMPUS REINF LEARNING
17
0.0
1

SWITCH
655
155
757
188
fogofwar
fogofwar
0
1
-1000

BUTTON
680
225
735
258
Arrow
shoot-arrow-right
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
600
295
655
328
Arrow
shoot-arrow-down
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
515
225
570
258
Arrow
shoot-arrow-left
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

BUTTON
600
155
655
188
Arrow
shoot-arrow-up
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

SWITCH
655
190
785
223
fogofwar-harder
fogofwar-harder
1
1
-1000

BUTTON
410
75
470
108
Design
design
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

MONITOR
100
400
150
445
Gen
gen
17
1
11

MONITOR
100
355
150
400
Step
steps
17
1
11

PLOT
0
445
200
595
Learning curve
gen
steps
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"steps" 1.0 0 -2674135 true "" ""

SWITCH
95
295
240
328
show-threats-scores
show-threats-scores
1
1
-1000

INPUTBOX
150
355
205
415
visu-sq-it
0
1
0
Number

SWITCH
150
415
240
448
verbose
verbose
1
1
-1000

SWITCH
0
175
107
208
auto-epsi
auto-epsi
0
1
-1000

SLIDER
105
175
200
208
auto-epsi-param
auto-epsi-param
0
1
0.2
0.01
1
NIL
HORIZONTAL

TEXTBOX
40
50
190
68
Simulation Parameters
15
0.0
1

TEXTBOX
300
10
450
28
Learning modes
15
0.0
1

TEXTBOX
70
275
150
293
Visualizations
15
0.0
1

PLOT
790
240
1115
440
Learning curve (neural net)
Iteration
Cost
0.0
10.0
0.0
0.0
true
true
"" ""
PENS
"learn0" 1.0 0 -13345367 true "" ""
"learn1" 1.0 0 -7500403 true "" ""
"learn2" 1.0 0 -2674135 true "" ""
"learn3" 1.0 0 -955883 true "" ""
"learn4" 1.0 0 -6459832 true "" ""
"learn5" 1.0 0 -1184463 true "" ""
"learn6" 1.0 0 -10899396 true "" ""
"learn7" 1.0 0 -5825686 true "" ""

INPUTBOX
790
75
850
135
h-neurons
500
1
0
Number

INPUTBOX
850
75
900
135
h-layers
1
1
0
Number

SLIDER
790
135
920
168
nnlambda
nnlambda
0
20
0
0.01
1
NIL
HORIZONTAL

INPUTBOX
900
75
950
135
nnStep
0.005
1
0
Number

CHOOSER
245
30
470
75
learning-mode
learning-mode
"Random" "TD-Learning" "Q-Learning" "Sequential Q-Learning" "Neural Q-Learning" "Chicken Heuristic" "Chicken Search"
6

INPUTBOX
1135
105
1225
165
ticksPerReplay
1000
1
0
Number

SWITCH
1075
75
1165
108
replay
replay
1
1
-1000

INPUTBOX
1075
165
1135
225
nqIterMax
200
1
0
Number

INPUTBOX
1135
165
1225
225
nqSeuilDiffGrad
0.001
1
0
Number

TEXTBOX
935
15
1120
51
Neural Network parameters
15
0.0
1

TEXTBOX
1105
55
1255
73
Replay parameters (if enabled)
11
0.0
1

TEXTBOX
860
55
1010
73
General parameters
11
0.0
1

SWITCH
940
170
1030
203
stop-nqlearn
stop-nqlearn
1
1
-1000

SWITCH
790
440
905
473
curve-nqlearn
curve-nqlearn
0
1
-1000

INPUTBOX
1075
105
1135
165
nbReplays
4
1
0
Number

SWITCH
790
170
940
203
nq-surroundFeatures
nq-surroundFeatures
1
1
-1000

SWITCH
1165
75
1275
108
nq-stochastic
nq-stochastic
0
1
-1000

SWITCH
80
210
210
243
random-startpos
random-startpos
0
1
-1000

INPUTBOX
950
75
1005
135
nnDecay
0.05
1
0
Number

SWITCH
95
325
240
358
show-global-scores
show-global-scores
1
1
-1000

SWITCH
790
205
915
238
use-global-score
use-global-score
0
1
-1000

PLOT
200
445
400
595
Win/Loss Ratio
NIL
Ratio
0.0
0.0
0.0
100.0
true
true
"" ""
PENS
"loss" 1.0 1 -2674135 true "" ""
"win" 1.0 1 -13840069 true "" ""

INPUTBOX
410
520
530
580
winloss-update-interval
100
1
0
Number

INPUTBOX
1005
75
1065
135
nq-memo
5
1
0
Number

BUTTON
0
325
95
358
Reset pcolors
ask patches [ color-patch-visu if visited = false and fogofwar [ set pcolor black ] ]
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

@#$#@#$#@
## WHAT IS IT?

This model is an example of heuristics and reinforcement learning to solve the "Hunt the Wumpus" game: an adventurer walks through a labyrinth, he has to find a fabulous treasure but numerous treacherous traps await him. Indeed, multiple bottomless pits and a terrible monster, the Wumpus, lie in the labyrinth.

To help himself, the adventurer only has a light torch (that does not light up a lot...), a bow with **one arrow** and his **senses** to help him. Indeed, getting close enough to a bit will make him feel a soft breeze of wind (grey squares), whereas a close encounter with the Wumpus will have disagreable relents (red squares). And if he gets really unlucky, he will get really close to both (purple squares).

## HOW IT WORKS

There are several variants of the game, and thus, several ways to win it.

The easiest variant is with a fixed labyrinth, so that when the adventurer dies, he will respawn in the same labyrinth and reuse what he already learnt.

The hardest variant is with a dynamic labyrinth that gets randomly regenerated everytime the adventurer dies or wins.

Several reinforcement learning algorithms and heuristics are available to solve both of these variants, from TD-learning or Q-learning with epsilon-greedy strategy, to neural networks. The agent will use these algorithms to learn how to maximize the chances to get the treasure while avoiding the labyrinth's dangers.

A few heuristics are offered, like the "chicken search" strategy, which is not a general purpose learning algorithm but are the best strategies for this game (at least known to the author). These heuristics set the ceiling for the maximum performance a reinforcement learning algorithm can achieve in this game, but whereas learning algorithms are generic and can be applied to any game, the provided heuristics only work for this specific game.

## HOW TO USE IT

Two modes are available:

* Simulation mode using reinforcement learning or heuristics, where the agent tries to autonomously learn and solve the game.
* Game mode where the adventurer is guided by the player. This is very useful to intuitively understand the complexity and difficulty of this game while spending a fun moment!

### Simulation mode

Alpha, gamma and epsilon are parameters to tweak the learning algorithms.

- Alpha is the learning rate.
- Gamma represents the propension of the agent to consider future rewards (instead of instant rewards).
- Epsilon is the exploration probability: with a probability of (1-epsilon) the agent makes the best move that he learned for the current situation, and with a probability epsilon he does a random action.
- init is the initialization button.
- 1-step allows to go forward 1 iteration = 1 action.
- Go launches the simulation (ie, all the iterations).
- reinf-visu allows to visualize all the informations pertaining to the learning phase.
- pits-count is the number of bottomless pits.
- Gen is the total count of past games (how many games won + lost).
- Step is the number of steps done during the current game up to now.
- the selector "learning-mode" allows to choose the agent's strategy (random, TD-learning, Q-learning, Sequential Q-Learning aka SQ-Learning, Neural Q-Learning aka NQ-Learning, Chicken Heuristic, Chicken Search). Only one of these modes can be selected at a time (ie, they are mutually exclusive).
- Eaten is the number of times the adventurer lost by being eaten by the Wumpus.
- Fell is the number of times the adventurer lost by falling into a bottomless pit.
- Killed W is the number of times the adventurer killed the Wumpus (with an arrow).
- Won is the number of times the agent found the treasure (and won!).
- the "learning curve" represents the evolution of the agent's learning for each game.
- the button "design" allow to modify the labyrinth by clicking on the map where you want to place a bottomless pit.
- the switch "random-startpos" enables an harder variant of the game, where the agent is randomly spawned in the labyrinth, but the labyrinth is always fixed (so the agent does not know where it is, it has to safely explore a bit to recognize the landscape before knowing where it is and thus apply its learnt strategies).
- the switch "harder-variable" enables the hardest variant of the Wumpus game, where the labyrinth is dynamically regenerated at each game.

There are

### Play Mode

Under the Play Mode, the player can move the adventurer in 4 directions (left, right, up, down) or shoot an arrow in these 4 directions.

A few advices:
* Before beginning a game, if you disable the switch "fogofwar", you will be able to see through the whole labyrinth. If it is the first time you play this game, this can help you understand how the game works.
* The common version of the game as found on Atari and old gaming consoles is when "fogofwar" is enabled: you see only the squares you already visited, and the colors of the squares (representing your senses) are your only clues to stay alive!
* The switch "fogofwar-harder" enables an even harder game mode, where you can only visualize the current square. Indeed, in this mode, already visited squares will be blacked out as soon as you move to another square!

Note that the fogofwar does not impact the simulation (where the agent learns to play the game autonomously). Indeed, the agent is always using a fogofwar and must rely solely on its memory.

## THINGS TO NOTICE

In play mode, notice the different square colors:
* A blue square has no danger nearby.
* A grey square as a nearby bottomless pit.
* A red square is close to the Wumpus!
* A purple square is close to both the Wumpus and a bottomless pit!!!

In simulation mode, look at the Win/Loss Ratio curve, this is the most important statistic: you want to maximize the green bars (ratio of wins) vs minimizing the red bars (ratio of losses). Try to play with the parameters and the different "learning-mode" to see how this impacts the win/loss ratio.

You can also watch the other statistics and the learning curves to see how the agent's internal learning algorithm is progressing over the games.

## THINGS TO TRY

Try various learning-mode algorithms and modify the parameters, and watch the influence on the Win/Loss Ratio curve and on other statistics and curves.

## EXTENDING THE MODEL

The neural network model does not work well with the dynamic mode, whereas it should be very efficient theoretically!

Currently the neural network is implemented this way: TD-learning or Q-learning (or SQ-Learning for temporal actions with arrows) is used to assign a value for each square/action, then the value is backpropagated through the neural network to define the synapses weights.

Another approach would be to use TD-Learning/Q-Learning directly on the neural network, to define the values of the weights. So instead of using reinforcement learning on the labyrinth's squares (ie, on the environment), it would be used on the neural network (ie, on the internal model of the environment). The easiest would be to have a neural network with as many neurons as there are possible actions from start to end, but it would quickly get too huge. Another way would be to just have 3 layers of n neurons, where n is the number of possible actions and senses and the layers a modelisation of the temporality: last layer L would be the current situation, L-1 would be the situation one step before, L-2 would be two steps before, etc.

The advantage with this approach is that TD-Learning/Q-Learning should be able to do the famous "blocking", which is that useless inputs will be filtered out, so the network's weights should be sparse (only the useful synapses will remain non-zero).

## PARTICLE FILTERS AND CHICKEN STRATEGIES

The chicken strategies were devised by Stephen Larroque, they gave the best results among all possible strategies (reinforcement learning and neural networks included). The chicken strategies are based on the "global-score" which is a score that is assigned to each square and which is computed as an aggregation of several parameters to represent the danger of this square. So obviously, when the square is visited, it gets a score of 0 (because there is no danger anymore), as only unknown squares have a "global-score" (which is really a "danger score"). The relevant code is this:

to compute-global-score [case]
  ask case [
    let safe-score 0
    if safe [set safe-score 1]
    set global-score safe-score + (compute-exploratory-interest / 4) - (monster-threat-score + pits-threat-score)
  ]
end

compute-exploratory-interest sums 1 for each neighbor square (including current square) if not visited. Eg, if a square is unknown and all its neighbors too, the exploratory-interest will be 5.

monster-threat-score and pits-threat-score are probabilities for each square to contain the related danger. They are simply calculated given the neighbors: more neighbor squares with a "sense" clue, the higher the probability. Eg, if for an unknown square, there is a neighbor grey square, then pits-threat-score will be 0.25 ; if the unknown square has 2 neighbors grey squares, then pits-threat-score will be 0.5 ; if the unknown square has 4 neighbors grey squares, then probability will be 1. Note that in the current implementation, the calculation is done the other way around for optimization reasons, but this is suboptimal: when an unknown square is visited, the current square's clue updates neighbors threats scores, but this is a wrong bayesian inference, it should be the other way around.

The safe score is a binary flag that is set when an unknown square has a "blue" neighbor square, which means that all neighbors to this square are safe.

In the end, the global score is an aggregation of these values (safe score and exploratory-interest are added, while the threats-scores are subtracted), which gives a scalar that we can easily use in heuristics (or in the neural network).

To be a bit more precise, the global-score was in fact initially devised as a logical (instead of probabilistic) implementation of Vermaak's Mixture Particle Filter. This is an evolution of the common Particle Filter which allows to track several objects at once instead of just one. The goal was to assign a danger-vs-exploratoryinterest score for each square as a way to provide a "memory map" for the agent: indeed, players remember what squares they visited and what clues they had, but not the agent unless we provide it a way to make such a map - the MPF, mixture particle filter, was done to fill this need.

Why is this useful? For the monster, there is no issue as there is only one, so no need to make a mental map of the labyrinth just for that, but there are multiple bottomless pits, so the agent needs to memorize their positions to avoid falling into them, but the common particle filter algorithm can only track one object per particle filter. See the references for more information.

Here is the algorithm in pseudo-code for the logical mixture particle filter algorithm (with compute-global-score being the NetLogo implementation):

; Mixture Particle Filter by Vermaak, logical implementation by Stephen Larroque
; Compute the global-score (threats and exploratory interest) only on needed squares, without needing N particles (because we only update neighbors squares).
; These are simply logical rules, simplifying and reducing the originally probabilistic mixture particle filter algorithm.
; Here are the rules in pseudo prolog BDI:
; c = current square
; n = each neighbor square in 4 directions
; u = unknown neighbor square (ie, not yet visited) in the n neighbors: u <- not visited(n)
; @check-no-death: not on(monster, c) and not on(pits, c) <- visited(c).
; @check-safe-patch: not stench(c) and not breeze(c) <- +safe(c), +safe(n), monster-threat-score(c, 0), pits-threat-score(c, 0), monster-threat-score(n, 0), pits-threat-score(n, 0).
; @first-update-remove-previous-scores: stench(c) <- previous-unknown(c, n), prevscore = 1/n, monster-threat-score(monster-threat-score(u) - prevscore, n).
; @same for breeze: ...
; @second-update-rule-add-new-score: stench(c) <- score = 1/u, monster-threat-score(monster-threat-score(u) + score, u), +previous-unknown(c, u).
; @same for breeze: breeze(c) <- score = 1/u, pits-threat-score(pits-threat-score(u) + score, u), +previous-unknown(c, u).
; @check-no-stench-clue: not stench(c) <- monster-threat-score(c, 0), monster-threat-score(n, 0).
; @check-no-breeze-clue: not breeze(c) <- pits-threat-score(c, 0), pits-threat-score(n, 0).
;
; Note: A known problem is that the weights are not always correctly updated with the maximum threat score for all neighboring squares (eg, when we visit a "safe" square), but the neighbor squares to the current square are always up-to-date. This is because we update only neighbors, so n+2 neighbors might also need an update but we don't propagate the update. So to summarize: immediately neighboring squares to current always have reliable and correct scores, but further squares might underestimate the threat-score. This should not be an issue since these "remote" squares will get updated when we will reach them (because the game only allows to move from square to square, if we could jump or teleport, it would be more problematic).

The current (logical) implementation is not perfect, but it has several advantages:

- Update scores only when necessary and on a subset of patches (ie, scales well), there is no update if the agent discovers nothing new (no rules are triggered).

- Memory and CPU complexity correlated only with the number of neighbors considered (here 4 because we can only move in 4 directions), whereas a probabilistic particle filter needs to work on N particules, which generally needs to be increased depending on the environment's size. Here, the logical implementation will consume the same resources independently from any environment's size.

There are two chicken strategies, and they both use the global score, but with different exploration strategies:

* Chicken Heuristic just tries to reach the square maximizing the global-score by walking the shortest route (and can thus stumble upon a trap).

* Chicken Search is very conservative as it will backtrack on already visited squares to reach the square maximizing the global-score. So if you use this heuristic, you will notice that the agent will often take very long and stupid routes, but this will allow it to reach a very high Win/Loss Ratio (about 85%/15%).

## NETLOGO FEATURES

NA.

## RELATED MODELS

NA.

## CREDITS AND REFERENCES

* Original Wumpus game by Ahl, David H. (1979), MORE BASIC Computer Games. New York: Workman Publishing. (ISBN 0-89480-137-6).
* Formalization of the Wumpus as a classical Artificial Intelligence problem by Russel and Norvig, An Introduction to Artificial Intelligence: A Modern Approach, Prentice Hall International, 1996.
* Mixture Particle Filter by: Vermaak, J., Doucet, A., & Pérez, P. (2003, October). Maintaining multimodality through mixture tracking. In Computer Vision, 2003. Proceedings. Ninth IEEE International Conference on (pp. 1110-1116). IEEE.

But the name "Mixture Particle Filter" was only coined by another article proposing a further evolution called "Boosted Particle Filter": Okuma, K., Taleghani, A., De Freitas, N., Little, J. J., & Lowe, D. G. (2004). A boosted particle filter: Multitarget detection and tracking. In Computer Vision-ECCV 2004 (pp. 28-39). Springer Berlin Heidelberg.

## AUTHORS AND LICENSE

Stephen Larroque and Hugo Gilbert for a Master project at University Pierre-And-Marie-Curie Paris 6.
Opensource licensed under MIT.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

bird
false
0
Polygon -7500403 true true 135 165 90 270 120 300 180 300 210 270 165 165
Rectangle -7500403 true true 120 105 180 237
Polygon -7500403 true true 135 105 120 75 105 45 121 6 167 8 207 25 257 46 180 75 165 105
Circle -16777216 true false 128 21 42
Polygon -7500403 true true 163 116 194 92 212 86 230 86 250 90 265 98 279 111 290 126 296 143 298 158 298 166 296 183 286 204 272 219 259 227 235 240 241 223 250 207 251 192 245 180 232 168 216 162 200 162 186 166 175 173 171 180
Polygon -7500403 true true 137 116 106 92 88 86 70 86 50 90 35 98 21 111 10 126 4 143 2 158 2 166 4 183 14 204 28 219 41 227 65 240 59 223 50 207 49 192 55 180 68 168 84 162 100 162 114 166 125 173 129 180

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

checker piece 2
false
0
Circle -7500403 true true 60 60 180
Circle -16777216 false false 60 60 180
Circle -7500403 true true 75 45 180
Circle -16777216 false false 83 36 180
Circle -7500403 true true 105 15 180
Circle -16777216 false false 105 15 180

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

crown
false
0
Rectangle -7500403 true true 45 165 255 240
Polygon -7500403 true true 45 165 30 60 90 165 90 60 132 166 150 60 169 166 210 60 210 165 270 60 255 165
Circle -2674135 true false 222 192 22
Circle -2674135 true false 56 192 22
Circle -2674135 true false 99 192 22
Circle -2674135 true false 180 192 22
Circle -2674135 true false 139 192 22

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

explorer
false
0
Circle -7500403 true true 115 34 70
Polygon -7500403 true true 105 105 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 105
Rectangle -7500403 true true 127 94 172 109
Polygon -7500403 true true 195 105 240 165 225 195 165 120
Polygon -7500403 true true 105 105 60 165 75 195 135 120
Polygon -6459832 true false 225 195 240 195 270 135 255 135 255 120 225 180 225 195
Polygon -1184463 true false 255 120 255 105 255 90 270 120 270 105 285 120 285 135 270 135 255 135 255 120
Polygon -955883 true false 270 120 255 105 270 90 270 105 285 90 270 75 300 90 285 105 300 105 285 120 270 120
Polygon -6459832 true false 116 29 116 39 121 47 165 72 192 80 202 71 191 62 183 53 183 47 188 27 165 15 145 17 138 27 125 24 120 27
Polygon -16777216 true false 137 34 174 50 183 54 184 53 183 47 147 30 140 26 138 27 134 29

face crown
false
0
Circle -7500403 true true 15 45 270
Circle -16777216 true false 60 90 60
Circle -16777216 true false 180 90 60
Polygon -16777216 true false 150 270 90 254 62 228 45 210 75 210 105 210 120 210 150 210 195 210 210 210 225 210 255 210 236 232 212 255
Polygon -1184463 true false 0 0 0 45 300 45 300 0 255 30 210 0 165 30 120 0 75 30 0 0
Rectangle -1184463 true false 0 45 315 90
Circle -2674135 true false 240 45 30
Circle -13791810 true false 60 45 30
Circle -13840069 true false 150 45 30

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

monster
false
0
Polygon -7500403 true true 75 150 90 195 210 195 225 150 255 120 255 45 180 0 120 0 45 45 45 120
Circle -16777216 true false 165 60 60
Circle -16777216 true false 75 60 60
Polygon -7500403 true true 225 150 285 195 285 285 255 300 255 210 180 165
Polygon -7500403 true true 75 150 15 195 15 285 45 300 45 210 120 165
Polygon -7500403 true true 210 210 225 285 195 285 165 165
Polygon -7500403 true true 90 210 75 285 105 285 135 165
Rectangle -7500403 true true 135 165 165 270

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tile water
false
0
Rectangle -7500403 true true -1 0 299 300
Polygon -1 true false 105 259 180 290 212 299 168 271 103 255 32 221 1 216 35 234
Polygon -1 true false 300 161 248 127 195 107 245 141 300 167
Polygon -1 true false 0 157 45 181 79 194 45 166 0 151
Polygon -1 true false 179 42 105 12 60 0 120 30 180 45 254 77 299 93 254 63
Polygon -1 true false 99 91 50 71 0 57 51 81 165 135
Polygon -1 true false 194 224 258 254 295 261 211 221 144 199

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270

@#$#@#$#@
NetLogo 5.3.1
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180

@#$#@#$#@
1
@#$#@#$#@
