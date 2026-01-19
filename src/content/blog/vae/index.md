---
title: "Variational Autoencoder"
description: "Comment dériver l'implémentation de VAE la plus répendue à partir du modèle mathématique."
date: "Jan 17 2025"
---

#### Modèle

L'objet est d'apprendre à échantilloner sous une loi complexe $\pi$. Pour ça on dispose d'observations $X_1, X_2, .., X_n$ i.i.d. sous cette même loi.

Le modèle de VAE est un modèle à variable latente, c'est à dire que l'on fait l'hypothèse de l'existence d'une variable, latente, $z$ qu'on interprète ici comme un représentation encodée de $x$.

On veut alors encoder $x$ de l'espace d'observation en une loi conditionelle $q_\varphi(z|x)$, appelé encodeur, sur l'espace latent. Puis de décoder $z$, de l'espace latent en une densité conditionelle $p_\theta(x|z=z)$, appelé décodeur, sur l'espace d'observation.

_L'approche est donc variationelle au sens où l'on associe à chaque point $x$ une fonctionelle._

On muni alors l'espace latent d'un prior $p(z)$, connu (et on fait l'hypothèse bayésienne classique $p_\theta(x, z) = p_\theta(x|z)p(z)$).

#### Echantillonage sous $p_\theta(x)$ 

Admettons avoir correctement estimé $\pi$ via $p_\theta(x, z)$ alors on pourra approximtativement échantilloner sous $\pi$ très simplement :

_En supposant un prior gaussien, et avec l'implémentation de `model` présentée plus bas, on a :_

```python
# Echantillone z ~ p(z) = N(0, 1)
z = torch.randn(0, 1, size=(1, 45))
# Echantillone x ~ p_\theta(x|z=z)
x, _, _ = model(z)
```

#### Optimisation de $p_\theta(x)$ 

L'objectif étant l'appoximation de $\pi$, l'approche naturelle du maximum de vraisemblance nous invite à maximiser

$$
l_n(\theta) = \frac{1}{n} \sum_{i = 1}^{n} \log p_\theta(X_i) = \frac{1}{n} \sum_{i = 1}^{n} \log \int p_\theta(X_i | z) p(z) dz
$$

Dans le plus simple des cas cette intégrale peut éventuellement s'expliciter. Mais puisque l'on souhaite paramétrer $p_\theta(x|z)$ par un réseau neuronal on ne peut absolument pas compter dessus. On pourrait tenter une approche d'estimation de l'espérance par montecarlo couplée à une méthode d'optimisation par gradient mais ça va couter cher.

Alternativement on peut considérer un estimateur $q_\varphi(z|x)$ de $p_\theta(z|x)$.

Remarquons alors que

$$
\begin{aligned}
    \log p_\theta(x) 

    &= \mathbb{E}_{z \sim q_\varphi(z|x)} \log p_\theta(x) \\

    &= \mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{p_\theta(x, z)}{p_\theta(z | x)} \\

    &= \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{p_\theta(x, z)}{q_\varphi(z|x)}}_{=: \mathcal{L}(\theta, \varphi, x)} + \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{q_\varphi(z|x)}{p_\theta(z | x)}}_{D_{KL}(q_\varphi(z|x) \parallel p_\theta(z|x))}
\end{aligned}
$$

Le terme $\mathcal{L(\theta, \varphi, x)}$, appelé Evidence Lower Bound (ELBO), est donc un minorant de la vraisemblance (La distance KL étant positive). Et donc à vraisemblance fixée, maximiser l'ELBO revient à minimiser la divergence KL ; et réciproquement à divergence fixée, maximiser l'ELBO revient à maximiser la vraisemblance.

On peut donc considérer l'ELBO comme critère d'optimisation de substitution.

Aussi, par un argument symétrique on obtient 

$$
\mathcal{L(\theta, \varphi, x)}
= \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{q_\varphi(z|x)}{p(z)}}_{D_{KL}(q_\varphi(z|x) \parallel p(z))} - \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log p_\theta(x | z)}_{(*)}
$$

Et on voit qu'on décompose cet objectif en deux termes. Le premier, la divergence de KL, est un terme de pénalisation de l'encodeur $q_\varphi(z|x)$, le forçant à rester similaire à notre prior $q$. Le second, $(*)$, un terme de reconstruction que l'on étudiera plus loin.

#### Optimisation par descente de gradient

En toute généralité on a donc besoin du gradient des deux termes d'ELBO, pour chacun des deux paramètres $\theta$ et $\varphi$. Le problème est l'obtention du gradient par rapport à $\varphi$ d'une espérance sous une loi dépendante de $\varphi$.  _On ne peut, à priori, absolument pas faire $\nabla_\varphi \mathbb{E}_{z \sim q_\varphi(z|x)} f(\theta, \varphi, x, z) = \mathbb{E}_{z \sim q_\varphi(z|x)} \nabla_\varphi f(\theta, \varphi, x, z)$._

On peut alors 

1. soit chercher à expliciter analytiquement ce gradient ;
2. soit faire l'hypothèse d'existence d'une fonction deterministe $\varphi \mapsto g(\varphi, x, \xi)$ telle que pour $\xi \sim q$ où $q$ est une loi donnée, on ait $\mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{q_\varphi(z|x)}{p(z)} = \mathbb{E}_{\xi \sim q} \log \frac{q_\varphi(g(\varphi, x, \xi)|x)}{p(g(\varphi, x, \xi))}$.

Cette seconde méthode est celle dite de reparamétrisation, et alors en posant 

$$
\mathcal{L}(\theta, \varphi, x, \xi)
:= \log \frac{q_\varphi(g(\varphi, x, \xi)|x)}{p(g(\varphi, x, \xi))} - \log p_\theta(x | g(\varphi, x, \xi))
$$

on a

$$
\nabla_{\theta, \varphi} \mathbb{E}_{\xi \sim q} \mathcal{L}(\theta, \varphi, x, \xi) = \mathcal{L}(\theta, \varphi, x)
$$

et on peut tout à fait optimiser $\mathcal{L}(\theta, \varphi) := \mathbb{E}_{\xi \sim q,\ x \sim \pi} \mathcal{L}(\theta, \varphi, x, \xi)$ par méthode de descente de gradient stochastique.

#### Hypothèses de Gaussianité pour l'obtention d'un critère simple explicite

Fixons nous donc un prior simple $\mathcal{N}(0, I)$ et supposons que le posterior $q_\varphi(z|x)$ soit également gaussien $\mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x))$.

Il est alors aisé d'explicité le terme de divergence KL. Le résultat est bien connu :

$$
D_{KL}(\mathcal{N}(\mu_1, \Sigma_1) \parallel \mathcal{N}(\mu_1, \Sigma_1))
= \frac{1}{2} \left[ \log \frac{\det{\Sigma_2}}{\det{\Sigma_1}} - d + (\mu_1 - \mu_2)^{T}\Sigma_2^{-1}(\mu_1 - \mu_2) + tr(\Sigma_2^{-1}\Sigma_1) \right]
$$

_Qu'on obtient en séparant l'espérance en deux termes d'espérances de $||.||^{2}$, en faisant un changement de variable sur $\Sigma_1^{-\frac{1}{2}}(z - \mu_1)$, et enfin en appliquant l'astuce de la trace $\mathbb{E}X^TX = tr(\mathbb{E}XX^T) = tr(\mathbb{V}(X)) + ||\mathbb{E}X||^2$._

qui nous permet d'obtenir 

$$
D_{KL}(\mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x)) \parallel \mathcal{N}(0, I))
= \frac{1}{2} \left[ - tr(\log{\Sigma_\varphi(x)}) - d + || \mu_\varphi(x) ||^2 + tr(\Sigma_\varphi(x)) \right]
$$

_car $\log \det A = tr(\log(A))$ où $\log(A)$ est une matrice dont les vp sont exactement le $\log$ de celles de $A$._

Tandis que pour retravailler le terme $(*)$, en supposant de plus la gaussianité de $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \Sigma_\theta(z))$ alos son $\log$ se simplifie en

$$
\log p_\theta(x|z) 
= - \log{\sqrt{2\pi}^{d} \sqrt{\det{\Sigma_\theta(z)}}} - \frac{1}{2} (x - \mu_\theta(z))^T\Sigma_\theta(z)^{-1}(x - \mu_\theta(z))
$$

En donc en fixant, par hypothèse, $\Sigma_\theta(z) = I$ on obtient alors

$$
\log p_\theta(x|z) 
= - \frac{1}{2} MSE(x, \mu_\theta(z)) + c
$$

ou $c$ est une constante relativement à $\theta$.

On cherche donc à optimiser 

$$
\mathcal{L}(\theta, \varphi, x, \xi)
:= 
{
    D_{KL}(\mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x)) \parallel \mathcal{N}(0, I))
} + {
    \frac{1}{2} 
    \mathbb{E}_{z \sim \mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x))} [ MSE(x, \mu_\theta(z)) ]
}
$$

Il ne reste qu'à reparamétrer pour se débarasser de l'espérance.

Prenons $\xi \sim \mathcal{N}(0, I)$ on a alors $g(\varphi, x, \xi) = \Sigma_\varphi(x)^{\frac{1}{2}} \xi + \mu_\varphi(x) \sim \mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x))$ et on obtient le critère simple, explicite, 

$$
\mathcal{L}(\theta, \varphi, x, \xi)
:= 
{
    \frac{1}{2} \left[ - tr(\log{\Sigma_\varphi(x)}) - d + || \mu_\varphi(x) ||^2 + tr(\Sigma_\varphi(x)) \right]
} + {
    \frac{1}{2} MSE(x, \mu_\theta(\Sigma_\varphi(x)^{\frac{1}{2}} \xi + \mu_\varphi(x)))
}
$$

qui nous permet d'optimiser par descente de gradient stochastique pour la fonction

$$
\mathcal{L}(\theta, \varphi)
:= 
\mathbb{E}_{x \sim \pi}
\mathbb{E}_{\xi \sim q}
\mathcal{L}(\theta, \varphi, x, \xi)
$$


Donc on obtient la version explicite la plus répendue de ce critère, facile à implémenter :

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAE(nn.Module):
    """Simple Gaussian VAE with linear encoder and decoder.
    """
    def __init__(self, input_dim=18*18, hidden_dim=10*10, latent_dim=45):
        super().__init__()

        # Encodeur 
        #
        # Sous hypothèse gaussienne il nous faut juste les paramètres 
        # de q_phi. 
        #
        # Ici on paramètre q_phi par un reseau quelconque 
        # (de préférence bien expressif pour le domaine d'application),
        # et on ajoute deux tête qui vont correspondre à nos paramètres
        self.q_phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_phi = nn.Linear(hidden_dim, latent_dim)
        # Remarquons qu'on fait une hypothèse supplémentaire ici :
        #
        # On suppose la matrice Sigma_phi diagonale
        #
        # Cette hypothèse va limiter (encore) l'expressivité du 
        # modèle mais permet l'utilisation de torch.exp() pour 
        # retrouve la matrice exponentielle plutôt que d'utiliser
        # torch.linalg.matrix_exp(). 
        # 
        # Celle ci est bien différentiable, mais a une complexité 
        # en O(latent_dim^3) tandis que  torch.exp() est en 0(latent_dim).
        self.logsigma_phi = nn.Linear(hidden_dim, latent_dim)

        # Decodeur
        #
        # Même principe que l'encodeur.
        self.mu_theta = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        # Encode
        hid = self.q_phi(x.view(-1, 784))
        mu_phi, logsigma_phi = self.mu_phi(hid), self.logsigma_phi(hid)

        # Echantillone sous N(0, I) et reparamètre
        #
        # Remarquons que la modèlisation par SGD abstraite 
        # que l'on a fait dans la section précédente suppose 
        # avoir (x, \xi) iid selon \pi x N(0, I) c'est ce qu'on
        # fait ici.
        #
        # Une autre façon de faire, plus intuitive mathématiquement
        # mais moins du point de vue de l'implémentation serait de 
        # créer un nouveau DataLoader qui ne retourne pas juste x
        # mais un couple (x , \xi)
        z = mu_phi + torch.randn_like(mu_phi) * torch.exp(0.5 * logsigma_phi)
        
        # Décode
        x_hat = self.p_theta(z)

        return x_hat, mu_phi, logsigma_phi

def elbo(x_hat, x, mu, logvar):
    # On double vérifie bien pour une éventuelle erreur de signe
    BCE = F.mse_loss(x_hat, x, reduction="sum")
    KLD = 0.5 * torch.sum(- 1 - logvar + mu ** 2 + logvar.exp())

    return BCE + KLD
```
