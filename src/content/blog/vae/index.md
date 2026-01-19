---
title: "Variational Autoencoder"
description: "Comment dériver l'implémentation de VAE la plus répendue à partir du modèle mathématique."
date: "Jan 17 2025"
---

## Modèle

L'objet est d'apprendre à échantilloner sous une loi complexe $\pi$. Pour ça on dispose d'observations $X_1, X_2, .. X_n$ i.i.d. sous cette même loi.

Le modèle de VAE est un modèle à variable latente, c'est à dire que l'on fait l'hypothèse de l'existence d'une variable, latente, $z$ qu'on interprète ici comme un représentation encodée de $x$.

On veut donc encoder $x$, point de l'espace d'observation, vers une loi conditionelle $q_\varphi(z|x=x)$, appelée encodeur, sur l'espace latent. Puis décoder $z$, point de l'espace latent, vers une densité conditionelle $p_\theta(x|z=z)$, appelée décodeur, sur l'espace d'observation.

_L'approche est donc variationelle au sens où l'on associe à chaque point $x$ une fonctionelle._

L'objectif étant l'appoximation de $\pi$, l'approche naturelle du maximum de vraisemblance nous invite à maximiser

$$
l_n(\theta) = \frac{1}{n} \sum_{i = 1}^{n} \log p_\theta(X_i)
$$

Remarquons alors que

$$
\begin{aligned}
    \log p_\theta(x) 

    &= \mathbb{E}_{z \sim q_\varphi(z|x)} \log p_\theta(x) \\

    &= \mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{p_\theta(x, z)}{p_\theta(z | x)} \\

    &= \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{p_\theta(x, z)}{q_\varphi(z|x)}}_{=: \mathcal{L}(\theta, \varphi, x)} + \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{q_\varphi(z|x)}{p_\theta(z | x)}}_{D_{KL}(q_\varphi(z|x) \parallel p_\theta(z|x))}
\end{aligned}
$$

Le terme $\mathcal{L(\theta, \varphi, x)}$, appelé Evidence Lower Bound (ELBO), est donc un minorant de la vraisemblance (la divergence KL étant positive). Et donc à vraisemblance fixée maximiser l'ELBO revient à minimiser la divergence KL ; et réciproquement à divergence fixée maximiser l'ELBO revient à maximiser la vraisemblance.

On peut donc raisonablement considérer l'ELBO comme critère d'optimisation de substitution.

On muni alors l'espace latent d'un prior $p(z)$, connu. Par un argument symétrique on obtient une nouvelle décomposition

$$
\mathcal{L(\theta, \varphi, x)}
= \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log p_\theta(x | z)}_{(*)} - \underbrace{\mathbb{E}_{z \sim q_\varphi(z|x)} \log \frac{q_\varphi(z|x)}{p(z)}}_{D_{KL}(q_\varphi(z|x) \parallel p(z))}
$$

Le premier terme $(*)$, qui peut s'interpréter comme une vraisemblance, est un terme de reconstruction. Le second est une divergence KL, c'est un terme de pénalisation de l'encodeur $q_\varphi(z|x)$, le forçant à rester similaire à notre prior $p(z)$. 

### Reparamétrisation pour l'optimisation par descente de gradient stochastique

En toute généralité on a donc besoin du gradient des deux termes d'ELBO, pour chacun des deux paramètres $\theta$ et $\varphi$. Le problème est l'obtention du gradient par rapport à $\varphi$ d'une espérance sous une loi dépendante de $\varphi$. Note $\mathcal{L}(\theta, \varphi, x, z) = \log p_\theta(x | z) - \log \frac{q_\varphi(z|x)}{p(z)}$ les termes sous l'espance dans l'ELBO. _On ne peut, à priori, absolument pas faire $\nabla_\varphi \mathbb{E}_{z \sim q_\varphi(z|x)} \mathcal{L}(\theta, \varphi, x, z) = \mathbb{E}_{z \sim q_\varphi(z|x)} \nabla_\varphi \mathcal{L}(\theta, \varphi, x, z)$._

On peut alors 

1. soit dérouler l'espérance dans l'espoir de trouver une parade, _et c'est possible_ ;
2. soit faire l'hypothèse d'existence d'une fonction deterministe $\varphi \mapsto f(\theta, \varphi, x, \xi)$ telle que pour $\xi \sim q$ où $q$ est une loi donnée, on ait 
   $$
   \mathbb{E}_{z \sim q_\varphi(z|x)} \mathcal{L}(\theta, \varphi, x, z) = \mathbb{E}_{\xi \sim q} \log \mathcal{L}(\theta, \varphi, x, f(\theta, \varphi, x, \xi))
   $$

Cette seconde méthode est celle dite de reparamétrisation, et alors en notant $\mathcal{L}(\theta, \varphi, x, \xi) := \mathcal{L}(\theta, \varphi, x, f(\theta, \varphi, x, \xi))$ on a ce qu'on voulait :

$$
\nabla_{\theta, \varphi} \mathcal{L}(\theta, \varphi, x) := \mathbb{E}_{\xi \sim q} \nabla_{\theta, \varphi} \mathcal{L}(\theta, \varphi, x, \xi)
$$

et on peut tout à fait optimiser 

- soit $\mathcal{L}(\theta, \varphi) := \mathbb{E}_{\xi \sim q,\ x \sim \pi} \mathcal{L}(\theta, \varphi, x, \xi)$ par descente de gradient stochastique single-pass puisqu'on peut échantilloner $n$ fois sous $\pi \otimes p$, puisque $p$ est connue et $X_1 .. X_n$ sont i.i.d. sous $\pi$.

- soit $\mathcal{L_{empirique}}(\theta, \varphi) := \frac{1}{n} \sum_{i = 1}^{n} \mathbb{E}_{\xi \sim q} \mathcal{L}(\theta, \varphi, X_i, \xi)$ par descente de gradient stochastique multi-pass en tirant $i \sim \mathcal{U}(1 .. n)$ et $\xi \sim p$.

### Hypothèses de Gaussianité pour l'obtention d'un critère simple explicite

Ces résultats sont intéressants, et suffisants en eux même pour des implémentations en utilisant des méthodes de Monte Carlo par exemple. 
Mais ce n'est pas ce qu'on trouve lorsque l'on recherche des implémentations concrète de VAE. Pour ça on va fixer notre prior et faire quelques
hypothèses qui vont permettre d'obtenir un critère d'optimisation bien plus facile.

#### Divergence KL - Forme explicite dans le cas de prior latent et encodeur gaussiens

Fixons déjà un prior simple $\mathcal{N}(0, I)$ et supposons que le posterior $q_\varphi(z|x)$ soit également gaussien $\mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x))$.

Il est alors aisé d'explicité le terme de divergence KL. Le résultat est bien connu :

$$
D_{KL}(\mathcal{N}(\mu_1, \Sigma_1) \parallel \mathcal{N}(\mu_2, \Sigma_2))
= \frac{1}{2} \left[ \log \frac{\det{\Sigma_2}}{\det{\Sigma_1}} - d + (\mu_1 - \mu_2)^{T}\Sigma_2^{-1}(\mu_1 - \mu_2) + tr(\Sigma_2^{-1}\Sigma_1) \right]
$$

_Qu'on obtient en séparant l'espérance en deux termes d'espérances de $||.||^{2}$, en faisant un changement de variable sur $\Sigma_1^{-\frac{1}{2}}(z - \mu_1)$, et enfin en appliquant l'astuce de la trace $\mathbb{E}X^TX = tr(\mathbb{E}XX^T) = tr(\mathbb{V}(X)) + ||\mathbb{E}X||^2$._

qui nous permet d'obtenir 

$$
D_{KL}(\mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x)) \parallel \mathcal{N}(0, I))
= \frac{1}{2} \left[ - tr(\log{\Sigma_\varphi(x)}) - d + || \mu_\varphi(x) ||^2 + tr(\Sigma_\varphi(x)) \right]
$$

_car $\log \det A = tr(\log(A))$ où $\log(A)$ est une matrice dont les vp sont exactement le $\log$ de celles de $A$._

#### Terme de reconstruction - Forme quasi-explicite dans le cas de décodeur gaussien

Supposons donc de plus la gaussianité de $p_\theta(x|z) = \mathcal{N}(\mu_\theta(z), \Sigma_\theta(z))$, alors son $\log$ se simplifie en

$$
\log p_\theta(x|z) 
= - \log{\sqrt{2\pi}^{d} \sqrt{\det{\Sigma_\theta(z)}}} - \frac{1}{2} (x - \mu_\theta(z))^T\Sigma_\theta(z)^{-1}(x - \mu_\theta(z))
$$

Si on fixe de plus $\Sigma_\theta(z) = I$ on obtient alors

$$
\log p_\theta(x|z) 
= - \frac{1}{2} MSE(x, \mu_\theta(z)) + c
$$

ou $c$ est une constante relativement à $\theta$ (et $\varphi$).

#### Terme de reconstruction - Forme explicite après reparamétrisation

Il ne reste qu'à reparamétrer pour se débarasser de l'espérance du terme de reconstruction. Sous l'hypothèse de gaussianité précédente on a juste à reparamétrer une loi $\mathcal{N}(\mu_\varphi(x), \Sigma_\varphi(x))$, pour ça il suffit de prendre $\xi \sim \mathcal{N}(0, I)$ et $f(\theta, \varphi, x, \xi) = \Sigma_\varphi(x)^{\frac{1}{2}} \xi + \mu_\varphi(x)$ et on obtient le critère simple, explicite, 

$$
\mathcal{L}(\theta, \varphi, x, \xi)
:= 
\underbrace{
    \frac{1}{2} \left[ tr(\log{\Sigma_\varphi(x)}) + d - || \mu_\varphi(x) ||^2 - tr(\Sigma_\varphi(x)) \right]
}_{D_{KL}} - \underbrace{
    \frac{1}{2} MSE(x, \mu_\theta(\Sigma_\varphi(x)^{\frac{1}{2}} \xi + \mu_\varphi(x)))
}_{MSE}
$$


### Implémentation

Avec `torch` on peut alors implémenter :


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAE(nn.Module):
    """Simple Gaussian VAE with linear encoder and decoder.
    """
    def __init__(self, input_dim=18*18, hidden_dim=10*10, latent_dim=45):
        super().__init__()
        self.latent_dim = latent_dim

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

    def sample(self):
        # Echantillone z ~ p(z) = N(0, 1)
        z = torch.randn(0, 1, size=(1, self.latent_dim))
        # Echantillone x ~ p_\theta(x|z=z)
        x_hat = self.p_theta(z)

        return x_hat

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
    # puisqu'on a dérivé un critère à maximiser, tandis que torch
    # en attend un à minimiser
    BCE = F.mse_loss(x_hat, x, reduction="sum")
    KLD = 0.5 * torch.sum(- 1 - logvar + mu ** 2 + logvar.exp())

    return BCE + KLD
```