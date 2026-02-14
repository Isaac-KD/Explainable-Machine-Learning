"""
feynman_tasks.py — Générateurs de tâches issus du Feynman Symbolic Regression Dataset
=======================================================================================

Chaque fonction `_make_*` génère (X, y) à partir de la vraie formule physique.
Les docstrings indiquent :
  - La formule analytique exacte
  - Les variables d'entrée et leurs gammes physiques typiques
  - La difficulté estimée pour un réseau de neurones

Source originale : Udrescu & Tegmark (2020) — "AI Feynman: A Physics-Inspired Method
for Symbolic Regression", Science Advances 6(16), eaay2631.
Feynman Symbolic Regression Database : https://space.mit.edu/home/tegmark/aifeynman.html

Toutes les formules supposent les constantes physiques suivantes (intégrées comme scalaires) :
  G  = 6.674e-11   (constante gravitationnelle, m³ kg⁻¹ s⁻²)
  k  = 1.381e-23   (constante de Boltzmann, J K⁻¹)
  h  = 6.626e-34   (constante de Planck, J s)
  c  = 3.0e8       (vitesse de la lumière, m s⁻¹)
  eps0 = 8.854e-12 (permittivité du vide, F m⁻¹)
  q  = 1.602e-19   (charge élémentaire, C)
"""

import torch
import torch.nn as nn
import math

# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME I — MÉCANIQUE & ÉLECTROSTATIQUE  (réf. Feynman Lectures Vol. I)
# ══════════════════════════════════════════════════════════════════════════════

def _make_I_6_2a(n: int):
    """
    Feynman I.6.2a — Distribution de probabilité gaussienne (sans normalisation)
    ─────────────────────────────────────────────────────────────────────────────
    Formule  : f(x) = exp(-x² / (2 σ²))
    Variables :
      x  ∈ [-5, 5]    — valeur de la variable aléatoire
      σ  ∈ [0.5, 3]   — écart-type
    Sortie   : f ∈ (0, 1]
    Difficulté : facile (1 entrée latente, forme gaussienne lisse)
    """
    x = torch.FloatTensor(n).uniform_(-5, 5)
    sigma = torch.FloatTensor(n).uniform_(0.5, 3.0)
    y = torch.exp(-x**2 / (2 * sigma**2))
    X = torch.stack([x, sigma], dim=1)
    return X, y.unsqueeze(1)


def _make_I_9_18(n: int):
    """
    Feynman I.9.18 — Force gravitationnelle (réduction à 2D)
    ─────────────────────────────────────────────────────────
    Formule  : F = G·m₁·m₂ / ((x₂-x₁)² + (y₂-y₁)²)
    Variables :
      m₁ ∈ [1e10, 1e30]  (kg)   — masse du corps 1
      m₂ ∈ [1e10, 1e30]  (kg)   — masse du corps 2
      x₁ ∈ [0, 1e11]     (m)    — position x du corps 1
      x₂ ∈ [0, 1e11]     (m)    — position x du corps 2
      y₁ ∈ [0, 1e11]     (m)    — position y du corps 1
      y₂ ∈ [0, 1e11]     (m)    — position y du corps 2
    Sortie   : F (N)
    Difficulté : moyenne (quotient + carrés, singularités si corps confondus)

    Note : on normalise log10 les entrées et sortie pour stabilité numérique.
    """
    G = 6.674e-11
    m1 = 10 ** torch.FloatTensor(n).uniform_(10, 30)
    m2 = 10 ** torch.FloatTensor(n).uniform_(10, 30)
    x1 = torch.FloatTensor(n).uniform_(0, 1e11)
    x2 = torch.FloatTensor(n).uniform_(0, 1e11)
    y1 = torch.FloatTensor(n).uniform_(0, 1e11)
    y2 = torch.FloatTensor(n).uniform_(0, 1e11)
    r2 = (x2 - x1)**2 + (y2 - y1)**2 + 1e10  # évite division par 0
    F = G * m1 * m2 / r2
    # Normalisation log
    X = torch.stack([
        torch.log10(m1), torch.log10(m2),
        x1 / 1e11, x2 / 1e11, y1 / 1e11, y2 / 1e11
    ], dim=1)
    y = torch.log10(F + 1e-30).unsqueeze(1)
    return X, y


def _make_I_12_1(n: int):
    """
    Feynman I.12.1 — Loi de Coulomb (force électrostatique)
    ─────────────────────────────────────────────────────────
    Formule  : F = q₁·q₂ / (4π·ε₀·r²)
    Variables :
      q₁ ∈ [1e-9, 1e-5]   (C)  — charge 1
      q₂ ∈ [1e-9, 1e-5]   (C)  — charge 2
      r  ∈ [0.01, 10]     (m)  — distance entre les charges
    Sortie   : F (N)
    Difficulté : facile (quotient simple, 3 variables)
    """
    eps0 = 8.854e-12
    q1 = 10 ** torch.FloatTensor(n).uniform_(-9, -5)
    q2 = 10 ** torch.FloatTensor(n).uniform_(-9, -5)
    r  = torch.FloatTensor(n).uniform_(0.01, 10)
    F  = q1 * q2 / (4 * math.pi * eps0 * r**2)
    X  = torch.stack([torch.log10(q1), torch.log10(q2), r], dim=1)
    y  = torch.log10(F.abs() + 1e-30).unsqueeze(1)
    return X, y


def _make_I_12_2(n: int):
    """
    Feynman I.12.2 — Champ électrique d'une charge ponctuelle
    ─────────────────────────────────────────────────────────────
    Formule  : E = q₁ / (4π·ε₀·r²)
    Variables :
      q₁ ∈ [1e-9, 1e-5]  (C)   — charge source
      r  ∈ [0.01, 10]    (m)   — distance
    Sortie   : E (V/m)
    Difficulté : très facile (2 variables, décroissance en 1/r²)
    """
    eps0 = 8.854e-12
    q1 = 10 ** torch.FloatTensor(n).uniform_(-9, -5)
    r  = torch.FloatTensor(n).uniform_(0.01, 10)
    E  = q1 / (4 * math.pi * eps0 * r**2)
    X  = torch.stack([torch.log10(q1), r], dim=1)
    y  = torch.log10(E.abs() + 1e-30).unsqueeze(1)
    return X, y


def _make_I_15_1(n: int):
    """
    Feynman I.15.1 — Contraction de Lorentz (longueur)
    ─────────────────────────────────────────────────────
    Formule  : x' = (x - v·t) / sqrt(1 - v²/c²)
    Variables :
      x  ∈ [-1e6, 1e6]  (m)    — position dans le référentiel d'origine
      v  ∈ [0, 0.9c]    (m/s)  — vitesse du référentiel
      t  ∈ [0, 10]      (s)    — temps
    Sortie   : x' (m) — position dans le référentiel en mouvement
    Difficulté : moyenne (interaction multiplicative, singularité à v=c)
    """
    c = 3e8
    x = torch.FloatTensor(n).uniform_(-1e6, 1e6)
    v = torch.FloatTensor(n).uniform_(0, 0.9 * c)
    t = torch.FloatTensor(n).uniform_(0, 10)
    gamma_inv = torch.sqrt(1 - v**2 / c**2)
    xp = (x - v * t) / gamma_inv
    X  = torch.stack([x / 1e6, v / c, t], dim=1)
    y  = (xp / 1e7).unsqueeze(1)
    return X, y


def _make_I_15_10(n: int):
    """
    Feynman I.15.10 — Transformation de Lorentz (temps)
    ───────────────────────────────────────────────────────
    Formule  : t' = (t - v·x/c²) / sqrt(1 - v²/c²)
    Variables :
      t  ∈ [0, 10]      (s)     — temps dans le référentiel d'origine
      v  ∈ [0, 0.9c]    (m/s)   — vitesse relative
      x  ∈ [-1e6, 1e6]  (m)     — position
    Sortie   : t' (s)
    Difficulté : moyenne (similaire à I.15.1 mais en temps)
    """
    c = 3e8
    t = torch.FloatTensor(n).uniform_(0, 10)
    v = torch.FloatTensor(n).uniform_(0, 0.9 * c)
    x = torch.FloatTensor(n).uniform_(-1e6, 1e6)
    gamma_inv = torch.sqrt(1 - v**2 / c**2)
    tp = (t - v * x / c**2) / gamma_inv
    X  = torch.stack([t, v / c, x / 1e6], dim=1)
    y  = tp.unsqueeze(1)
    return X, y


def _make_I_18_4(n: int):
    """
    Feynman I.18.4 — Centre de masse (deux corps)
    ───────────────────────────────────────────────
    Formule  : r = (m₁·r₁ + m₂·r₂) / (m₁ + m₂)
    Variables :
      m₁ ∈ [1, 100]   (kg)  — masse du corps 1
      m₂ ∈ [1, 100]   (kg)  — masse du corps 2
      r₁ ∈ [0, 10]    (m)   — position du corps 1
      r₂ ∈ [0, 10]    (m)   — position du corps 2
    Sortie   : r (m) — position du centre de masse
    Difficulté : très facile (linéaire, 4 variables)
    """
    m1 = torch.FloatTensor(n).uniform_(1, 100)
    m2 = torch.FloatTensor(n).uniform_(1, 100)
    r1 = torch.FloatTensor(n).uniform_(0, 10)
    r2 = torch.FloatTensor(n).uniform_(0, 10)
    r  = (m1 * r1 + m2 * r2) / (m1 + m2)
    X  = torch.stack([m1, m2, r1, r2], dim=1)
    y  = r.unsqueeze(1)
    return X, y


def _make_I_26_2(n: int):
    """
    Feynman I.26.2 — Loi de Snell (angle de réfraction)
    ───────────────────────────────────────────────────────
    Formule  : θ₂ = arcsin(n₁ · sin(θ₁) / n₂)
    Variables :
      θ₁ ∈ [0, π/2)   (rad)  — angle d'incidence
      n₁ ∈ [1, 2.5]          — indice du milieu 1
      n₂ ∈ [1, 2.5]          — indice du milieu 2
    Sortie   : θ₂ (rad) — angle de réfraction
    Difficulté : facile (transcendant, mais monotone)
    """
    theta1 = torch.FloatTensor(n).uniform_(0, math.pi / 2 - 0.05)
    n1 = torch.FloatTensor(n).uniform_(1.0, 2.5)
    n2 = torch.FloatTensor(n).uniform_(1.0, 2.5)
    sin_t2 = (n1 * torch.sin(theta1) / n2).clamp(-1 + 1e-6, 1 - 1e-6)
    theta2 = torch.asin(sin_t2)
    X  = torch.stack([theta1, n1, n2], dim=1)
    y  = theta2.unsqueeze(1)
    return X, y


def _make_I_27_18(n: int):
    """
    Feynman I.27.18 — Focale d'une lentille (formule des lunetiers)
    ─────────────────────────────────────────────────────────────────
    Formule  : 1/f = (n-1) · (1/R₁ - 1/R₂)
    Variables :
      n  ∈ [1.2, 2.0]     — indice de réfraction du matériau
      R₁ ∈ [0.05, 1.0]    (m)  — rayon de courbure face 1
      R₂ ∈ [-1.0, -0.05]  (m)  — rayon de courbure face 2 (négatif = concave)
    Sortie   : f (m) — distance focale
    Difficulté : facile (3 variables, formule linéarisée)
    """
    n_idx = torch.FloatTensor(n).uniform_(1.2, 2.0)
    R1 = torch.FloatTensor(n).uniform_(0.05, 1.0)
    R2 = torch.FloatTensor(n).uniform_(-1.0, -0.05)
    f_inv = (n_idx - 1) * (1.0 / R1 - 1.0 / R2)
    f = 1.0 / f_inv
    X  = torch.stack([n_idx, R1, R2], dim=1)
    y  = f.unsqueeze(1)
    return X, y


def _make_I_34_8(n: int):
    """
    Feynman I.34.8 — Effet Doppler relativiste
    ─────────────────────────────────────────────
    Formule  : ω₁ = ω₀ · sqrt((1 - v/c) / (1 + v/c))
    Variables :
      ω₀ ∈ [1e9, 1e12]   (rad/s)  — fréquence source (radio → UV)
      v  ∈ [0, 0.9c]     (m/s)    — vitesse de la source (vers l'observateur : négatif)
    Sortie   : ω₁ (rad/s) — fréquence observée
    Difficulté : facile (2 variables, forme analytique simple)
    """
    c = 3e8
    omega0 = 10 ** torch.FloatTensor(n).uniform_(9, 12)
    v = torch.FloatTensor(n).uniform_(-0.9 * c, 0.9 * c)
    ratio = ((1 - v / c) / (1 + v / c)).clamp(min=1e-6)
    omega1 = omega0 * torch.sqrt(ratio)
    X  = torch.stack([torch.log10(omega0), v / c], dim=1)
    y  = torch.log10(omega1).unsqueeze(1)
    return X, y


def _make_I_37_4(n: int):
    """
    Feynman I.37.4 — Intensité des interférences à deux fentes
    ─────────────────────────────────────────────────────────────
    Formule  : I = I₁ + I₂ + 2·sqrt(I₁·I₂)·cos(δ)
    Variables :
      I₁ ∈ [0, 1]      (W/m²)  — intensité fente 1
      I₂ ∈ [0, 1]      (W/m²)  — intensité fente 2
      δ  ∈ [0, 2π]     (rad)   — différence de phase
    Sortie   : I (W/m²) — intensité résultante
    Difficulté : facile (3 variables, cosinus + racine carrée)
    """
    I1 = torch.FloatTensor(n).uniform_(0, 1)
    I2 = torch.FloatTensor(n).uniform_(0, 1)
    delta = torch.FloatTensor(n).uniform_(0, 2 * math.pi)
    I  = I1 + I2 + 2 * torch.sqrt(I1 * I2) * torch.cos(delta)
    X  = torch.stack([I1, I2, delta], dim=1)
    y  = I.unsqueeze(1)
    return X, y


def _make_I_41_16(n: int):
    """
    Feynman I.41.16 — Loi de Planck (densité spectrale d'énergie)
    ───────────────────────────────────────────────────────────────
    Formule  : u = (h·ω³) / (4π²·c³) · 1/(exp(h·ω/(k·T)) - 1)
    Variables :
      ω ∈ [1e10, 1e14]  (rad/s)  — pulsation
      T ∈ [100, 10000]  (K)      — température
    Sortie   : u (J·s/m³) — densité spectrale d'énergie
    Difficulté : difficile (exponentielles, produit de termes disparates)
    """
    h = 6.626e-34
    k = 1.381e-23
    c = 3e8
    omega = 10 ** torch.FloatTensor(n).uniform_(10, 14)
    T = torch.FloatTensor(n).uniform_(100, 10000)
    exponent = (h * omega / (k * T)).clamp(max=50)
    u = (h * omega**3) / (4 * math.pi**2 * c**3) / (torch.exp(exponent) - 1 + 1e-10)
    X  = torch.stack([torch.log10(omega), torch.log10(T)], dim=1)
    y  = torch.log10(u.abs() + 1e-300).unsqueeze(1)
    return X, y


def _make_I_43_31(n: int):
    """
    Feynman I.43.31 — Coefficient de diffusion (gaz)
    ──────────────────────────────────────────────────
    Formule  : D = μ·k·T / q
    Variables :
      μ ∈ [1e-4, 1e-1]   (m²/V/s)  — mobilité des porteurs
      T ∈ [200, 1000]    (K)        — température
    Sortie   : D (m²/s) — coefficient de diffusion
    Difficulté : très facile (2 variables, produit linéaire)
    """
    k = 1.381e-23
    q = 1.602e-19
    mu = 10 ** torch.FloatTensor(n).uniform_(-4, -1)
    T  = torch.FloatTensor(n).uniform_(200, 1000)
    D  = mu * k * T / q
    X  = torch.stack([torch.log10(mu), T], dim=1)
    y  = torch.log10(D).unsqueeze(1)
    return X, y


def _make_I_50_26(n: int):
    """
    Feynman I.50.26 — Oscillateur harmonique amorti (position)
    ─────────────────────────────────────────────────────────────
    Formule  : x(t) = x₁ · cos(ω·t) + v₁/ω · sin(ω·t)
    Variables :
      x₁ ∈ [-1, 1]     (m)     — position initiale
      v₁ ∈ [-5, 5]     (m/s)  — vitesse initiale
      ω  ∈ [0.1, 10]   (rad/s) — pulsation propre
      t  ∈ [0, 10]     (s)    — temps
    Sortie   : x(t) (m)
    Difficulté : facile (4 variables, somme de sinusoïdes)
    """
    x1 = torch.FloatTensor(n).uniform_(-1, 1)
    v1 = torch.FloatTensor(n).uniform_(-5, 5)
    omega = torch.FloatTensor(n).uniform_(0.1, 10)
    t = torch.FloatTensor(n).uniform_(0, 10)
    x = x1 * torch.cos(omega * t) + (v1 / omega) * torch.sin(omega * t)
    X  = torch.stack([x1, v1, omega, t], dim=1)
    y  = x.unsqueeze(1)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME II — ÉLECTROMAGNÉTISME & THERMODYNAMIQUE
# ══════════════════════════════════════════════════════════════════════════════

def _make_II_11_27(n: int):
    """
    Feynman II.11.27 — Polarisation électronique (modèle de Clausius-Mossotti)
    ─────────────────────────────────────────────────────────────────────────────
    Formule  : Pol = n·α / (1 - n·α/3) · ε₀
    Variables :
      n_mol ∈ [1e20, 1e28]   (m⁻³)  — densité volumique de molécules
      α     ∈ [1e-40, 1e-30] (C²·s²/kg) — polarisabilité
    Sortie   : Pol (C/m²) — polarisation
    Difficulté : difficile (quotient avec saturation, gammes d'ordres de grandeur)
    """
    eps0 = 8.854e-12
    n_mol = 10 ** torch.FloatTensor(n).uniform_(20, 27)
    alpha = 10 ** torch.FloatTensor(n).uniform_(-40, -30)
    denom = (1 - n_mol * alpha / 3).clamp(min=1e-6)
    Pol = n_mol * alpha / denom * eps0
    X  = torch.stack([torch.log10(n_mol), torch.log10(alpha)], dim=1)
    y  = torch.log10(Pol.abs() + 1e-300).unsqueeze(1)
    return X, y


def _make_II_11_28(n: int):
    """
    Feynman II.11.28 — Constante diélectrique (susceptibilité)
    ─────────────────────────────────────────────────────────────
    Formule  : 1 + n·α / (1 - n·α/3)
    Variables :
      n_mol ∈ [1e20, 1e26]   (m⁻³)   — densité moléculaire
      α     ∈ [1e-40, 1e-32] (C²s²/kg) — polarisabilité
    Sortie   : ε_r (sans unité) — permittivité relative
    Difficulté : moyenne (quotient, 2 variables)
    """
    n_mol = 10 ** torch.FloatTensor(n).uniform_(20, 26)
    alpha = 10 ** torch.FloatTensor(n).uniform_(-40, -32)
    denom = (1 - n_mol * alpha / 3).clamp(min=1e-6)
    eps_r = 1 + n_mol * alpha / denom
    X  = torch.stack([torch.log10(n_mol), torch.log10(alpha)], dim=1)
    y  = eps_r.clamp(1, 1e6).log10().unsqueeze(1)
    return X, y


def _make_II_21_32(n: int):
    """
    Feynman II.21.32 — Potentiel de Debye-Hückel
    ──────────────────────────────────────────────
    Formule  : V = q / (4π·ε₀·r) · exp(-r / λ_D)
    Variables :
      q    ∈ [1e-19, 1e-16]   (C)  — charge de l'ion central
      r    ∈ [1e-10, 1e-7]    (m)  — distance radiale
      λ_D  ∈ [1e-10, 1e-6]    (m)  — longueur de Debye
    Sortie   : V (V) — potentiel écranté
    Difficulté : difficile (exponentielles + quotient, 3 ordres de grandeur)
    """
    eps0 = 8.854e-12
    q  = 10 ** torch.FloatTensor(n).uniform_(-19, -16)
    r  = 10 ** torch.FloatTensor(n).uniform_(-10, -7)
    ld = 10 ** torch.FloatTensor(n).uniform_(-10, -6)
    V  = q / (4 * math.pi * eps0 * r) * torch.exp(-r / ld)
    X  = torch.stack([torch.log10(q), torch.log10(r), torch.log10(ld)], dim=1)
    y  = torch.log10(V.abs() + 1e-300).unsqueeze(1)
    return X, y


def _make_II_34_2a(n: int):
    """
    Feynman II.34.2a — Magnéton de Bohr
    ──────────────────────────────────────
    Formule  : μ_B = q·h / (4π·m_e)
    Variables : aucune variable continue — équation aux constantes
    Note : on génère des variantes en faisant varier q et m comme
           si c'était des particules généralisées (charge / masse).
      q ∈ [1e-20, 1e-18]  (C)   — charge de la particule
      m ∈ [1e-31, 1e-29]  (kg)  — masse de la particule
    Sortie   : μ (J/T)
    Difficulté : très facile (produit de 2 variables)
    """
    h = 6.626e-34
    q_val = 10 ** torch.FloatTensor(n).uniform_(-20, -18)
    m_val = 10 ** torch.FloatTensor(n).uniform_(-31, -29)
    mu = q_val * h / (4 * math.pi * m_val)
    X  = torch.stack([torch.log10(q_val), torch.log10(m_val)], dim=1)
    y  = torch.log10(mu).unsqueeze(1)
    return X, y


def _make_II_38_3(n: int):
    """
    Feynman II.38.3 — Force de Lorentz (composante électromagnétique)
    ──────────────────────────────────────────────────────────────────
    Formule  : F = q·(E + v·B)    (cas 1D)
    Variables :
      q ∈ [1e-19, 1e-16]  (C)      — charge
      E ∈ [-1e6, 1e6]     (V/m)    — champ électrique
      v ∈ [-1e5, 1e5]     (m/s)    — vitesse de la particule
      B ∈ [-1, 1]         (T)      — champ magnétique
    Sortie   : F (N)
    Difficulté : très facile (somme linéaire, 4 variables)
    """
    q = 10 ** torch.FloatTensor(n).uniform_(-19, -16)
    E = torch.FloatTensor(n).uniform_(-1e6, 1e6)
    v = torch.FloatTensor(n).uniform_(-1e5, 1e5)
    B = torch.FloatTensor(n).uniform_(-1, 1)
    F = q * (E + v * B)
    X  = torch.stack([torch.log10(q), E / 1e6, v / 1e5, B], dim=1)
    y  = (F / q.detach().clamp(min=1e-20)).unsqueeze(1)  # normaliser par q
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME III — MÉCANIQUE QUANTIQUE
# ══════════════════════════════════════════════════════════════════════════════

def _make_III_4_32(n: int):
    """
    Feynman III.4.32 — Population thermique de Bose-Einstein
    ──────────────────────────────────────────────────────────
    Formule  : n = 1 / (exp(h·ω / (k·T)) - 1)
    Variables :
      ω ∈ [1e9, 1e13]   (rad/s)  — fréquence angulaire du mode
      T ∈ [10, 10000]   (K)      — température
    Sortie   : n̄ — nombre moyen d'occupation
    Difficulté : difficile (exponentielle, singularité à ω/T → 0)
    """
    h = 6.626e-34
    k = 1.381e-23
    omega = 10 ** torch.FloatTensor(n).uniform_(9, 13)
    T = torch.FloatTensor(n).uniform_(10, 10000)
    exponent = (h * omega / (k * T)).clamp(1e-4, 50)
    n_bose = 1.0 / (torch.exp(exponent) - 1 + 1e-10)
    X  = torch.stack([torch.log10(omega), torch.log10(T)], dim=1)
    y  = torch.log10(n_bose + 1e-10).unsqueeze(1)
    return X, y


def _make_III_7_38(n: int):
    """
    Feynman III.7.38 — Fréquence de précession de Rabi
    ─────────────────────────────────────────────────────
    Formule  : Ω = μ_B · B / ℏ     (ici ℏ = h/2π)
    Variables :
      μ_B ∈ [1e-24, 1e-22]  (J/T)  — magnéton (variant)
      B   ∈ [1e-3, 10]      (T)    — champ magnétique
    Sortie   : Ω (rad/s) — pulsation de Rabi
    Difficulté : très facile (produit de 2 variables)
    """
    h = 6.626e-34
    hbar = h / (2 * math.pi)
    mu = 10 ** torch.FloatTensor(n).uniform_(-24, -22)
    B  = torch.FloatTensor(n).uniform_(1e-3, 10)
    Omega = mu * B / hbar
    X  = torch.stack([torch.log10(mu), torch.log10(B + 1e-10)], dim=1)
    y  = torch.log10(Omega).unsqueeze(1)
    return X, y


def _make_III_9_52(n: int):
    """
    Feynman III.9.52 — Probabilité de transition (formule de Rabi)
    ────────────────────────────────────────────────────────────────
    Formule  : P = (μ_B·B/ℏ)² / ((ω-ω₀)²/4 + (μ_B·B/ℏ)²) · sin²(t · sqrt((ω-ω₀)²/4 + (μ_B·B/ℏ)²))
    Variables :
      ω   ∈ [0.1, 100]   (rad/s)  — fréquence de conduite
      ω₀  ∈ [0.1, 100]   (rad/s)  — fréquence de résonance
      μ_B ∈ [0.1, 10]    (J/T / ℏ normalisé) — couplage normalisé
      B   ∈ [0.1, 10]    (T normalisé)
      t   ∈ [0, 5]       (s)      — durée du pulse
    Sortie   : P ∈ [0, 1] — probabilité de transition
    Difficulté : très difficile (5 variables, oscillations, singularités)
    """
    omega  = torch.FloatTensor(n).uniform_(0.1, 100)
    omega0 = torch.FloatTensor(n).uniform_(0.1, 100)
    mu_B   = torch.FloatTensor(n).uniform_(0.1, 10)
    B      = torch.FloatTensor(n).uniform_(0.1, 10)
    t      = torch.FloatTensor(n).uniform_(0, 5)
    Omega_R  = mu_B * B
    detuning = (omega - omega0) / 2
    generalized = torch.sqrt(detuning**2 + Omega_R**2) + 1e-8
    P = Omega_R**2 / (detuning**2 + Omega_R**2) * torch.sin(t * generalized)**2
    X  = torch.stack([omega, omega0, mu_B, B, t], dim=1)
    y  = P.unsqueeze(1)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  FORMULES THERMODYNAMIQUES
# ══════════════════════════════════════════════════════════════════════════════

def _make_thermo_entropy(n: int):
    """
    Entropie d'un gaz idéal — formule de Sackur-Tetrode (simplifiée)
    ──────────────────────────────────────────────────────────────────
    Formule  : S = n_mol · k · ln(T^(Cv/R) · V / n_mol)
    Variables :
      T     ∈ [100, 10000]  (K)    — température
      V     ∈ [1e-3, 1]    (m³)   — volume
      n_mol ∈ [0.1, 100]   (mol)  — quantité de matière
    Sortie   : S (J/K) — entropie
    Difficulté : facile (logarithme, 3 variables)
    """
    k_B  = 1.381e-23
    N_A  = 6.022e23
    k    = k_B * N_A  # ≈ R
    T     = torch.FloatTensor(n).uniform_(100, 10000)
    V     = torch.FloatTensor(n).uniform_(1e-3, 1)
    n_mol = torch.FloatTensor(n).uniform_(0.1, 100)
    # Cv/R = 3/2 pour gaz monoatomique
    S = n_mol * k * torch.log((T**1.5) * V / n_mol + 1e-8)
    X  = torch.stack([torch.log10(T), torch.log10(V), torch.log10(n_mol)], dim=1)
    y  = S.unsqueeze(1)
    return X, y


def _make_thermo_carnot(n: int):
    """
    Rendement d'un cycle de Carnot
    ───────────────────────────────
    Formule  : η = 1 - T_cold / T_hot
    Variables :
      T_hot  ∈ [400, 2000]  (K)  — température de la source chaude
      T_cold ∈ [200, 400]   (K)  — température de la source froide
    Contrainte : T_cold < T_hot  (appliquée par construction)
    Sortie   : η ∈ [0, 1) — rendement thermodynamique
    Difficulté : très facile (2 variables, linéaire)
    """
    T_cold = torch.FloatTensor(n).uniform_(200, 400)
    T_hot  = T_cold + torch.FloatTensor(n).uniform_(100, 1600)
    eta    = 1 - T_cold / T_hot
    X  = torch.stack([T_hot, T_cold], dim=1)
    y  = eta.unsqueeze(1)
    return X, y


def _make_thermo_kinetic_energy(n: int):
    """
    Énergie cinétique thermique moyenne (théorème d'équipartition)
    ───────────────────────────────────────────────────────────────
    Formule  : <E_k> = (3/2) · k · T  (pour un gaz monoatomique)
    Variables :
      T ∈ [1, 10000]  (K)  — température absolue
    Sortie   : <E_k> (J) — énergie cinétique moyenne par particule
    Difficulté : triviale (1 variable, linéaire)
    """
    k = 1.381e-23
    T = torch.FloatTensor(n).uniform_(1, 10000)
    Ek = 1.5 * k * T
    X  = T.unsqueeze(1)
    y  = Ek.unsqueeze(1)
    return X, y


def _make_thermo_maxwell_speed(n: int):
    """
    Distribution de Maxwell-Boltzmann — densité de probabilité de vitesse
    ───────────────────────────────────────────────────────────────────────
    Formule  : f(v) = 4π · (m/(2π·k·T))^(3/2) · v² · exp(-m·v²/(2·k·T))
    Variables :
      v ∈ [1, 3000]     (m/s)  — vitesse moléculaire
      T ∈ [100, 10000]  (K)    — température
      m ∈ [1e-27, 1e-25] (kg)  — masse moléculaire
    Sortie   : f(v) (s/m) — densité de probabilité
    Difficulté : difficile (Gaussienne 3D, produit de 3 variables)
    """
    k = 1.381e-23
    v = torch.FloatTensor(n).uniform_(1, 3000)
    T = torch.FloatTensor(n).uniform_(100, 10000)
    m = 10 ** torch.FloatTensor(n).uniform_(-27, -25)
    coeff = 4 * math.pi * (m / (2 * math.pi * k * T))**1.5
    exponent = (-m * v**2 / (2 * k * T)).clamp(min=-100)
    f = coeff * v**2 * torch.exp(exponent)
    X  = torch.stack([v / 1000, torch.log10(T), torch.log10(m)], dim=1)
    y  = torch.log10(f.abs() + 1e-300).unsqueeze(1)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  FORMULES MÉCANIQUES SUPPLÉMENTAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _make_kepler_period(n: int):
    """
    3ème loi de Kepler — période orbitale
    ──────────────────────────────────────
    Formule  : T = 2π · sqrt(a³ / (G·M))
    Variables :
      a ∈ [1e9, 1e12]    (m)   — demi-grand axe de l'orbite
      M ∈ [1e24, 1e32]   (kg)  — masse de l'astre central
    Sortie   : T (s) — période orbitale
    Difficulté : facile (2 variables, puissances et racine)
    """
    G = 6.674e-11
    a = 10 ** torch.FloatTensor(n).uniform_(9, 12)
    M = 10 ** torch.FloatTensor(n).uniform_(24, 32)
    T_period = 2 * math.pi * torch.sqrt(a**3 / (G * M))
    X  = torch.stack([torch.log10(a), torch.log10(M)], dim=1)
    y  = torch.log10(T_period).unsqueeze(1)
    return X, y


def _make_pendulum_period(n: int):
    """
    Pendule simple — période (approximation angle faible)
    ───────────────────────────────────────────────────────
    Formule  : T = 2π · sqrt(L / g)
    Variables :
      L ∈ [0.01, 10]  (m)    — longueur du pendule
      g ∈ [1, 30]     (m/s²) — accélération de la pesanteur (planètes différentes)
    Sortie   : T (s) — période du pendule
    Difficulté : très facile (2 variables, racine carrée)
    """
    L = torch.FloatTensor(n).uniform_(0.01, 10)
    g = torch.FloatTensor(n).uniform_(1, 30)
    T_period = 2 * math.pi * torch.sqrt(L / g)
    X  = torch.stack([L, g], dim=1)
    y  = T_period.unsqueeze(1)
    return X, y


def _make_wave_power(n: int):
    """
    Puissance rayonnée par une onde électromagnétique (formule de Larmor)
    ────────────────────────────────────────────────────────────────────────
    Formule  : P = q²·a² / (6π·ε₀·c³)
    Variables :
      q ∈ [1e-19, 1e-16]  (C)     — charge
      a ∈ [1e5, 1e15]     (m/s²)  — accélération de la charge
    Sortie   : P (W) — puissance rayonnée
    Difficulté : facile (2 variables, produit de puissances)
    """
    eps0 = 8.854e-12
    c    = 3e8
    q = 10 ** torch.FloatTensor(n).uniform_(-19, -16)
    a = 10 ** torch.FloatTensor(n).uniform_(5, 15)
    P = q**2 * a**2 / (6 * math.pi * eps0 * c**3)
    X  = torch.stack([torch.log10(q), torch.log10(a)], dim=1)
    y  = torch.log10(P + 1e-300).unsqueeze(1)
    return X, y


def _make_hydrogen_energy(n: int):
    """
    Niveaux d'énergie de l'atome d'hydrogène
    ──────────────────────────────────────────
    Formule  : E_n = -m_e · e⁴ / (8 · ε₀² · h² · n²)
    Variables :
      n (entier) ∈ {1, 2, ..., 10}  — nombre quantique principal
      (les constantes sont fixes, on les incorpore)
    Sortie   : E_n (J) — énergie du niveau n
    Difficulté : triviale (1 variable discrète)
    Note : on traite n comme continu pour la régression.
    """
    m_e  = 9.109e-31
    e    = 1.602e-19
    eps0 = 8.854e-12
    h    = 6.626e-34
    n_qn = torch.FloatTensor(n).uniform_(1, 10)
    E = -m_e * e**4 / (8 * eps0**2 * h**2 * n_qn**2)
    X  = n_qn.unsqueeze(1)
    y  = torch.log10(E.abs()).unsqueeze(1)
    return X, y


def _make_relativistic_momentum(n: int):
    """
    Moment relativiste
    ─────────────────────
    Formule  : p = m·v / sqrt(1 - v²/c²) = γ·m·v
    Variables :
      m ∈ [1e-31, 1e-27]  (kg)   — masse au repos
      v ∈ [0, 0.99c]      (m/s)  — vitesse
    Sortie   : p (kg·m/s) — quantité de mouvement relativiste
    Difficulté : facile (2 variables, singularité à v→c)
    """
    c = 3e8
    m = 10 ** torch.FloatTensor(n).uniform_(-31, -27)
    v = torch.FloatTensor(n).uniform_(0, 0.99 * c)
    gamma = 1.0 / torch.sqrt(1 - v**2 / c**2)
    p = m * gamma * v
    X  = torch.stack([torch.log10(m), v / c], dim=1)
    y  = torch.log10(p.abs() + 1e-300).unsqueeze(1)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  CATALOGUE COMPLET — format compatible avec benchmark.py
# ══════════════════════════════════════════════════════════════════════════════

FEYNMAN_TASKS = [
    # ── Mécanique / Ondes ──────────────────────────────────────────────────
    {
        "name"       : "I.6.2a — Gaussienne",
        "difficulty" : "easy",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_I_6_2a,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "f = exp(-x² / (2σ²))",
        "variables"  : "x∈[-5,5], σ∈[0.5,3]",
    },
    {
        "name"       : "I.18.4 — Centre de masse",
        "difficulty" : "easy",
        "input_dim"  : 4,
        "output_dim" : 1,
        "generator"  : _make_I_18_4,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "r = (m₁r₁ + m₂r₂)/(m₁+m₂)",
        "variables"  : "m₁,m₂∈[1,100]kg, r₁,r₂∈[0,10]m",
    },
    {
        "name"       : "I.26.2 — Loi de Snell",
        "difficulty" : "easy",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_I_26_2,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "θ₂ = arcsin(n₁·sin(θ₁)/n₂)",
        "variables"  : "θ₁∈[0,π/2)rad, n₁,n₂∈[1,2.5]",
    },
    {
        "name"       : "I.27.18 — Lentille (lunetiers)",
        "difficulty" : "easy",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_I_27_18,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "1/f = (n-1)(1/R₁ - 1/R₂)",
        "variables"  : "n∈[1.2,2], R₁∈[0.05,1]m, R₂∈[-1,-0.05]m",
    },
    {
        "name"       : "I.37.4 — Interférences",
        "difficulty" : "easy",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_I_37_4,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "I = I₁+I₂+2√(I₁I₂)·cos(δ)",
        "variables"  : "I₁,I₂∈[0,1]W/m², δ∈[0,2π]rad",
    },
    {
        "name"       : "I.50.26 — Oscillateur harmonique",
        "difficulty" : "easy",
        "input_dim"  : 4,
        "output_dim" : 1,
        "generator"  : _make_I_50_26,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "x(t) = x₁cos(ωt) + (v₁/ω)sin(ωt)",
        "variables"  : "x₁∈[-1,1]m, v₁∈[-5,5]m/s, ω∈[0.1,10]rad/s, t∈[0,10]s",
    },
    {
        "name"       : "Pendule — Période",
        "difficulty" : "easy",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_pendulum_period,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "T = 2π√(L/g)",
        "variables"  : "L∈[0.01,10]m, g∈[1,30]m/s²",
    },
    {
        "name"       : "Carnot — Rendement",
        "difficulty" : "easy",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_thermo_carnot,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "η = 1 - T_cold/T_hot",
        "variables"  : "T_hot∈[400,2000]K, T_cold∈[200,400]K",
    },
    # ── Niveau intermédiaire ────────────────────────────────────────────────
    {
        "name"       : "I.12.1 — Loi de Coulomb",
        "difficulty" : "medium",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_I_12_1,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "F = q₁q₂/(4πε₀r²)   [log10 I/O]",
        "variables"  : "q₁,q₂∈[1e-9,1e-5]C (log), r∈[0.01,10]m",
    },
    {
        "name"       : "I.15.1 — Contraction de Lorentz",
        "difficulty" : "medium",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_I_15_1,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "x' = (x-v·t)/√(1-v²/c²)",
        "variables"  : "x∈[-1e6,1e6]m, v∈[0,0.9c], t∈[0,10]s",
    },
    {
        "name"       : "I.15.10 — Lorentz (temps)",
        "difficulty" : "medium",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_I_15_10,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "t' = (t-v·x/c²)/√(1-v²/c²)",
        "variables"  : "t∈[0,10]s, v∈[0,0.9c], x∈[-1e6,1e6]m",
    },
    {
        "name"       : "I.34.8 — Doppler relativiste",
        "difficulty" : "medium",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_I_34_8,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "ω₁ = ω₀·√((1-v/c)/(1+v/c))   [log10 I/O]",
        "variables"  : "ω₀∈[1e9,1e12]rad/s (log), v∈[-0.9c,0.9c]",
    },
    {
        "name"       : "Entropie gaz idéal (Sackur-Tetrode)",
        "difficulty" : "medium",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_thermo_entropy,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "S = n·k·ln(T^(3/2)·V/n)",
        "variables"  : "T∈[100,10000]K (log), V∈[1e-3,1]m³ (log), n∈[0.1,100]mol (log)",
    },
    {
        "name"       : "Kepler — 3ème loi",
        "difficulty" : "medium",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_kepler_period,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "T = 2π√(a³/(G·M))   [log10 I/O]",
        "variables"  : "a∈[1e9,1e12]m (log), M∈[1e24,1e32]kg (log)",
    },
    {
        "name"       : "II.11.28 — Permittivité relative",
        "difficulty" : "medium",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_II_11_28,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "ε_r = 1 + nα/(1-nα/3)   [log10 I/O]",
        "variables"  : "n∈[1e20,1e26]m⁻³ (log), α∈[1e-40,1e-32]C²s²/kg (log)",
    },
    {
        "name"       : "Moment relativiste (γmv)",
        "difficulty" : "medium",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_relativistic_momentum,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "p = m·v/√(1-v²/c²)   [log10 I/O]",
        "variables"  : "m∈[1e-31,1e-27]kg (log), v∈[0,0.99c]",
    },
    # ── Niveau difficile ────────────────────────────────────────────────────
    {
        "name"       : "I.41.16 — Loi de Planck",
        "difficulty" : "hard",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_I_41_16,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "u = (hω³)/(4π²c³) / (exp(hω/kT)-1)   [log10 I/O]",
        "variables"  : "ω∈[1e10,1e14]rad/s (log), T∈[100,10000]K (log)",
    },
    {
        "name"       : "III.4.32 — Bose-Einstein",
        "difficulty" : "hard",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_III_4_32,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "n̄ = 1/(exp(hω/kT)-1)   [log10 I/O]",
        "variables"  : "ω∈[1e9,1e13]rad/s (log), T∈[10,10000]K (log)",
    },
    {
        "name"       : "II.21.32 — Debye-Hückel",
        "difficulty" : "hard",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_II_21_32,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "V = q/(4πε₀r)·exp(-r/λ_D)   [log10 I/O]",
        "variables"  : "q∈[1e-19,1e-16]C (log), r∈[1e-10,1e-7]m (log), λ_D∈[1e-10,1e-6]m (log)",
    },
    {
        "name"       : "Maxwell — Distribution de vitesses",
        "difficulty" : "hard",
        "input_dim"  : 3,
        "output_dim" : 1,
        "generator"  : _make_thermo_maxwell_speed,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "f(v) = 4π(m/2πkT)^(3/2)·v²·exp(-mv²/2kT)   [log10 I/O]",
        "variables"  : "v∈[1,3000]m/s, T∈[100,10000]K (log), m∈[1e-27,1e-25]kg (log)",
    },
    {
        "name"       : "Larmor — Puissance rayonnée",
        "difficulty" : "hard",
        "input_dim"  : 2,
        "output_dim" : 1,
        "generator"  : _make_wave_power,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "P = q²a²/(6πε₀c³)   [log10 I/O]",
        "variables"  : "q∈[1e-19,1e-16]C (log), a∈[1e5,1e15]m/s² (log)",
    },
    # ── Très difficile ─────────────────────────────────────────────────────
    {
        "name"       : "III.9.52 — Rabi (probabilité de transition)",
        "difficulty" : "very_hard",
        "input_dim"  : 5,
        "output_dim" : 1,
        "generator"  : _make_III_9_52,
        "loss_fn"    : nn.MSELoss(),
        "formula"    : "P = Ω_R²/(Δ²+Ω_R²)·sin²(t√(Δ²+Ω_R²))  avec Δ=(ω-ω₀)/2",
        "variables"  : "ω,ω₀∈[0.1,100]rad/s, μ_B,B∈[0.1,10], t∈[0,5]s",
    },
]
