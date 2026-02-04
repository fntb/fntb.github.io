---
title: UNet - Un banc de filtres appris ?
description: Pourquoi le U-Net fonctionne-t-il si bien ? Exploration de l'architecture à travers la théorie de l'approximation multirésolution.
date: 2026-02-04
---

Aujourd'hui je vais parler UNet.

J'ai procédé de manière assez différente, et moins confortable, que dans les articles précédents. D'habitude je pars d'un cadre mathématique élégant _et confortable_ duquel je dérive une implémentation canonique. Tandis qu'aujourd'hui je pars de l'architecture unet, telle que proposée dans la publication originale [[c.f. U-Net: Convolutional Networks for Biomedical Image Segmentation, O. Ronneberger, P. Fischer, T. Brox]](https://arxiv.org/pdf/1505.04597), et essaye de proposer un cadre mathématique pour la justifier.

Mon idée est de proposer un problème d'apprentissage statistique restreint par quelques hypothèses de modélisation formalisées par la théorie de l'approximation multirésolution pour essayer de justifier pourquoi le choix d'un unet en tant que famille paramétrique d'approximateurs fait sens.

On va pouvoir en tirer l'idée de quelques architectures de réseaux (i.e. familles paramétriques) alternatives, et on pourra les comparées sur un problème simple pour valider (ou non) l'intuition.

#### Commentaire

C'est par le formalisme accessible de l'approximation multirésolution (MRA) qu'on va réussir à poser un cadre de manière agréable.

Etant issu d'une formation plutôt de probabiliste je ne suis pas intimement familier avec cette théorie qui trouve plus sa place en traitement du signal, une introduction rigoureuse m'a donc été utile et j'essaie d'en extraire et retranscrire les éléments qui seront immédiatement utiles. La présentation de que je fais se base sur le livre [A Wavelet Tour of Signal Processing](https://doi.org/10.1016/B978-0-12-374370-1.X0001-8) de Mallat (particulièrement le chapitre 7, et en moindre mesure les chapitres 4 et 5).

Essentiellement trois points nous seront utiles :

1. l'introduction des termes ;
2. l'équivalence, sous quelques hypthèses, entre une MRA, une base d'ondelette orthogonale, et un filtre mirroir conjugué ;
3. et l'algorithme de transformée en ondelettes rapide.

#### Introduction rapide à l'approximation multirésolution

L'approximation multirésolution constitue l'approximation des fonctions de $L^2(\R)$ à diverses résolutions en les projetants orthogonalement sur une suite de sous espaces emboités $(V_j)$ (dits d'approximation à résolution $j$).

Une **approximation multirésolution (MRA)** de $L^2(\R)$ est la donnée d'une suite de sous espaces emboités $(V_j)$, $j \in \Z$, vérifiants :

1. $V_j \subset V_{j+1}$
2. $f \in V_j  \iff f(2 \cdot) \in V_{j+1}$
3. $f \in V_0  \iff f(\cdot - k) \in V_0 ,\ \forall k \in \Z$
4. $\underset{j \in \Z}{\bigcap} V_j = \set{0}$ et $\underset{j \in \Z}{\bigcup} V_j$ dense dans $L^2(\R)$
5. Il existe une $\phi \in V_0$ telle que $\set{\phi_k := \phi( \cdot - k) }_{k \in \Z}$ est une base orthonormale de $V_0$. $\phi$ est appelée **fonction échelle** (ou éventuellement ondelette père).

Remarquons tout de suite que le sous espace $V_j$ est engendré par une base orthonormale consituée de $\set{\phi_{j, k} := 2^{j / 2}\phi(2^j \cdot - k) }_{k \in \Z}$. Une approximation multirésolution est donc entièrement caractérisée par la fonction d'échelle $\phi$. Par ailleurs il existe une suite $h$ telle que 

$$\phi(x) = \sqrt{2} \underset{k \in \Z}{\sum} h[k] \phi(2 x - k)$$

Il suffit de voir que $\phi \in V_0 \subset V_1$ et que $\set{\phi_{1, k}}$ est base de $V_1$.

Cette suite $h$, interprétée comme un filtre discret (étant noyau d'une convolution discrète), est appelé **filtre miroir conjugué**. On peut montrer que la fonction échelle et donc la MRA est entièrement caractérisée par $h$, c'est dû à un théorème de Mallat et Meyer dont la preuve, un peu longue, peut être trouvée [page 310].

Le role des ondelette est de porter les détails nécessaire à l'augmentation de résolution d'un approximation de $f$. 

Plus formellement notons $W_j$ le supplémentaire de $V_j$ dans $V_{j+1}$ : $V_{j+1} = V_j \oplus W_j$ puis décomposons la projection de $f$ sur $V_{j+1}$ : $P_{V_{j+1}} f = P_{V_j} f + P_{W_j} f$. On interprète $P_{W_j} f$ comme les détails de $f$ qui apparaissent à échelle $j+1$ mais disparaissent à échelle $j$. On peut prouver qu'il est possible de construire une base orthonormale de $W_j$ en dilatant et translatant une ondelette $\psi$. La preuve peut être trouvée [page 320].

#### Formulation du problème


Considérons $(x, y) \in L^2([0, 1]^d)^2$ et intéressons nous au problème de régression

$$\underset{\theta}{\min}\ \mathbb{E} \left[ l(f_\theta(x), y) \right]$$

pour une fonction de perte $l$ donnée et une famille paramétrique $\set{f_\theta}$.

Faisons alors l'hypothèse de l'existence et de la connaissance d'une MRA $V_0 \subset V_1 \subset \dots \subset L^2([0, 1]^d)$ dans lequel ce problème se décompose "bien", c'est à dire que la convergence suivante est "très rapide" lorsque $j$ augmente : 

$$\mathbb{E} \left \lVert f_\theta(P_{V_j} x) - y \right \rVert ^2 \to 0$$

Remarquons bien que c'est une hypothèse différente de la convergence rapide de $\mathbb{E} \left \lVert P_{V_j} y - y \right \rVert ^2 \to 0$ qui est une résultat bien connu pour les bases d'ondelettes sous des hypothèses assez faible de régularité de $y$. En revanche si on considère le problème de reconstruction, obtenu dans le cas du modèle de bruit blanc gaussien $y = x + \epsilon$ et de la perte quadratique, cette hypothèse est bien vérifiée.

#### Idée de famille paramétrique approriée


On pourrait très bien utiliser une famille paramétrique quelconque $\set{f_\theta}$, après tout le théorème d'approximation universel en version originale de Cybenko nous dit bien qu'il suffit d'un simple MLP à une couche cachée pour approximer toute fonction réelle continue sur un compact de $\R^n$. 

Le problème est que l'existence d'un solution au problème d'optimisation n'implique pas qu'il soit faisable de la trouver en pratique. Un choix irrefléchi de famille parametrique n'a à priori aucune raison d'avoir un bon biais inductif, et on ne parviendra jamais à l'entrainer de manière satisfaisante.

Ici, en s'inspirant du boosting on peut envisager de construire un apprenant de manière récursive.

1. Estimer $y \vert P_{V_0} x$ avec $f_{\theta, 0}(P_{V_0}x)$
2. Estimer $y \vert (P_{W_1} x, y \vert P_{V_0} x)$ avec $f_{\theta, 1}(P_{W_1}x, f_{\theta, 0}(P_{V_0}x))$
3. Etc.

En notant $W_{j+1}$ le complémentaire de $V_j$ dans $V_{j+1}$ de sorte que $V_{j+1} = W_{j+1} \oplus V_j$.

Cette idée mène à la structure suivante :

<img src="/unet/wavelet_cnn.svg" width=400 style="padding: inherit; margin:auto;  display: block;"/>

_Remarquons que plutôt que de simplement justifier une structure de réseau il serait intéressant d'essayer d'expliciter une solution à ce problème dans le cadre d'apprenants très simple, comme pour le boosting. Bien que je n'ai pas poursuivit cette idée, une recherche dans cette direction mène à la méthode de Viola et Jones qui semble suivre exactement cette route :_

> _Viola–Jones is essentially a boosted feature learning algorithm, trained by running a modified AdaBoost algorithm on Haar feature classifiers to find a sequence of classifiers $f_{1},f_{2},...,f_{k}$. Haar feature classifiers are crude, but allows very fast computation, and the modified AdaBoost constructs a strong classifier out of many weak ones._
>
> _[[méthode de Viola et Jones | Wikipédia]](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)_

#### Algorithme de transformée en ondelettes rapide

Dans l'optique d'implémenter cette architecture il nous faut une procédure pour décomposer la fonction $x$ dans la base d'ondelette associée à la MRA connue. Il existe un tel algorithme, c'est la **transformée en ondelettes rapide**. Cet algorithme de [banc de filtres](https://en.wikipedia.org/wiki/Filter_bank) calcul les coefficients d'ondelettes d'une fonction échantillonée à résolution finie $j$ en décomposant récursivement chaque approximation $P_{V_{j + 1}} x$ en une approximation plus grossière $P_{V_j} x$ et les coefficients d'ondelettes portés par $P_{W_{j + 1}} x$.

Notons $\phi$ la fonction d'échelle et $\psi$ l'ondelette de la MRA et $h$ et $g$ les filtres correspondants. Notons $a_j[k] = \langle x, \phi_{j, k} \rangle$ et $d_j[k] = \langle x, \psi_{j, k} \rangle$ les coefficients des projections de $x$. On peut montrer que ces coefficient sont calculable par une récursion de convolutions discrètes et de sous-échantillonages, c'est à dire que

$$
\begin{align*}
a_j[k] &= \underset{k' \in \Z}{\sum} h[k' - 2k] a_{j+1}[k] \\
d_j[k] &= \underset{k' \in \Z}{\sum} g[k' - 2k] d_{j+1}[k]
\end{align*}
$$

_Schéma de preuve : décomposer $\phi_{j, k}$ dans la base, écrire le produit scalaire comme intégral, changer de variable, repasser en porduit scalaire pour identifier._

_Les termes de $h$ et $g$ non nuls correspondent au support de la fonction d'échelle, ils sont en nombre fini dès lors que celle-ci est a support compact, ce qui sera toujours le cas pour nous._


_Par ailleurs il est possible de dériver un algorithme de transformée inverse du résultat complémentaire_

$$
a_{j+1}[k] = 
\underset{k' \in \Z}{\sum} h[k - 2k'] a_j[k']
+ \underset{k' \in \Z}{\sum} g[k - 2k'] d_j[k']
$$


Ce résultat donne assez directement l'algorithme récursif de transformée en ondelette.


![By User:Johnteslade, Public Domain, https://commons.wikimedia.org/w/index.php?curid=225721](https://upload.wikimedia.org/wikipedia/commons/2/22/Wavelets_-_Filter_Bank.png)

_Schéma de l'étape de décomposition : récursivement, on applique les filtres $h$ et $g$, puis on sous-échantillone par 2._

Remarquons alors qu'en prenant les $f_{\theta, j}$ paramétrées par des convolution, relu, upsample on retrouve la structure d'un réseau UNet dont l'encodeur est non appris mais fixé par le choix de base d'ondelette.

#### Cas de l'approximation multirésolution inconnue

Le choix de la MRA est donc un à priori de modélisation. En s'inspirant des résultats de décennies de recherche en traitement du signal ou vision, il semble envisageable de faire des choix raisonnablement éclairés. Cependant, on peut affaiblir notre hypothèse et ne pas considérer une telle base connue, on est naturellement menés à penser à chercher à l'estimer.

Or on sait qu'une MRA est caractérisée par le filtre $h$, par conséquent simplement paramétrer la convolution dans l'algorithme de transformée en ondelette permet d'approximer n'importe quelle MRA. L'apprentissage de la base se fera alors en même temps que celui des $f_{\theta, n}$, par optimisation du critère $\mathbb{E} \left [ l(f_\theta(x), y) \right ]$.

Et on abouti à une fonction paramétrique dont le schéma est le suivant, très très proche de celui du UNet original

<img src="/unet/unet.svg" width=400 style="padding: inherit; margin:auto;  display: block;"/>

#### Conclusion : Interprétation de l'architecture UNet

On peut voir un UNet (son décodeur) comme une version apprise d'une décomposition en ondelettes. A contrario d'un décomposition en ondelette fixée, le UNet apprend la base optimale pour résoudre le problème de régression.

L'encodeur agit comme une récursion de projections dans des espaces d'approximation de plus en plus grossier en extrayant les basses fréquences (approximations) des hautes fréquences (détails).

Les basses fréquences, parcimonieuses, sont envoyées au bottleneck $f_{\theta, 0}$ pour être traitées "sémantiquement" (i.e. en basse résolution, les convolutions du bottleneck peuvent traiter les structures globales).

Les hautes fréquences, sont transmises au décodeur par skip-connexion pour préconditionner la reconstruction à plus haute résolution (un prior spatial régularisant l'upsampling du décodeur).

## Experiments

Je propose de comparer les résultats de quatre architectures sur une tache de segmentation.

1. UNet classique, tel que défini dans [U-Net: Convolutional Networks for Biomedical Image Segmentation, O. Ronneberger, P. Fischer, T. Brox](https://arxiv.org/pdf/1505.04597)
2. Wavelet-CNN, c'est à dire un UNet où l'encodeur a été remplacé par une décomposition en base de Haar comme on l'a présenté ici
3. Adaptative-Wavelet-CNN, un UNet où les paramètres des différentes échelles de l'encodeur sont partagées entre niveau, comme on l'a présenté dans la section MRA inconnue
4. Un simple FCN pour servir de baseline

_WIP_

## Références

- [A Unified Framework for U-Net Design and Analysis, C. Williams, F. Falck](https://arxiv.org/pdf/2305.19638)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation, O. Ronneberger, P. Fischer, T. Brox](https://arxiv.org/pdf/1505.04597)
- [A Wavelet Tour of Signal Processing, Mallat](https://doi.org/10.1016/B978-0-12-374370-1.X0001-8)