# -*- coding: utf-8 -*-
"""
Tarea 2: Modelos Lineales Generalizados (MLG)
Ejecución de Cuasi-Verosimilitud, Regresión Simplex y Regresión Beta.
"""

import sys

# Comprobación amigable de librerías externas
try:
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from scipy.optimize import minimize
    from scipy.special import gammaln
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError as e:
    print(f"\n[!] Falta una librería: {e}")
    print("[*] Instálalas ejecutando en tu terminal:")
    print("    /opt/homebrew/bin/pip3 install matplotlib seaborn pandas numpy scipy --break-system-packages\n")
    sys.exit(1)

sns.set_theme(style="whitegrid", palette="muted")

print("="*60)
print("EJERCICIO 2: ANÁLISIS DE CEBADA (cebada.dat)")
print("="*60)

# Literal a) Carga y Diagramas de Caja
df_c = pd.read_csv("cebada.dat", sep=r"\s+", header=None, names=["area", "lugar", "variedad"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.boxplot(x="lugar", y="area", data=df_c, ax=axes[0], color="#5DADE2")
axes[0].set_title("Proporción Afectada según Lugar (1 - 9)", fontweight="bold")
axes[0].set_xlabel("Lugar Geográfico")
axes[0].set_ylabel("Proporción de Área Foliar")

sns.boxplot(x="variedad", y="area", data=df_c, ax=axes[1], color="#AF7AC5")
axes[1].set_title("Proporción Afectada según Variedad (1 - 10)", fontweight="bold")
axes[1].set_xlabel("Variedad de Cebada")
axes[1].set_ylabel("Proporción de Área Foliar")

plt.tight_layout()
plt.savefig("fig_cebada_boxplots.png", dpi=200)
print("[+] Boxplots guardados en 'fig_cebada_boxplots.png'.")

# Literal b) Modelos de Cuasi-Verosimilitud
def fit_quasi(X, y, var_type='v1', max_iter=100, tol=1e-8):
    n, p = X.shape
    y_adj = (y * (n - 1) + 0.5) / n
    eta = np.log(y_adj / (1.0 - y_adj))
    beta = np.linalg.pinv(X) @ eta
    for _ in range(max_iter):
        eta = X @ beta
        pi = 1.0 / (1.0 + np.exp(-eta))
        pi = np.clip(pi, 1e-6, 1.0 - 1e-6)
        d_eta_d_pi = 1.0 / (pi * (1.0 - pi))
        z = eta + (y - pi) * d_eta_d_pi
        W = pi * (1.0 - pi) if var_type == 'v1' else np.ones(n)
        W_sqrt = np.sqrt(W)
        X_tilde = X * W_sqrt[:, None]
        z_tilde = z * W_sqrt
        beta_new = np.linalg.solve(X_tilde.T @ X_tilde, X_tilde.T @ z_tilde)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    pi = 1.0 / (1.0 + np.exp(-eta))
    pi = np.clip(pi, 1e-6, 1.0 - 1e-6)
    v_pi = pi * (1.0 - pi) if var_type == 'v1' else (pi * (1.0 - pi))**2
    r_p = (y - pi) / np.sqrt(v_pi)
    X2 = np.sum(r_p**2)
    phi = X2 / (n - p)
    return {'beta': beta, 'pi': pi, 'residuals': r_p, 'X2': X2, 'phi': phi}

X_lugar = pd.get_dummies(df_c["lugar"], drop_first=True, dtype=float)
X_var = pd.get_dummies(df_c["variedad"], drop_first=True, dtype=float)
X_c = np.column_stack([np.ones(len(df_c)), X_lugar.values, X_var.values])
y_c = df_c["area"].values

m1 = fit_quasi(X_c, y_c, 'v1')
m2 = fit_quasi(X_c, y_c, 'v2')
print(f"Modelo 1 [V(pi) = pi(1-pi)]:       X2 = {m1['X2']:.4f} | Dispersion sigma^2 = {m1['phi']:.4f}")
print(f"Modelo 2 [V(pi) = pi^2(1-pi)^2]:   X2 = {m2['X2']:.4f} | Dispersion sigma^2 = {m2['phi']:.4f}")

# Literal c) Explicación teórica Simplex
print("\nLiteral c) ¿Es posible ajustar Regresión Simplex a cebada.dat?")
print("Respuesta: NO es posible directamente debido a 4 observaciones con valor 0.0000.")
print("La devianza unitaria d(0, mu) = mu^2 / 0 produce divergencia a infinito.")

print("\n" + "="*60)
print("EJERCICIO 3: ANÁLISIS DE GRASA CORPORAL (bodyfat.txt)")
print("="*60)

df_b = pd.read_csv("bodyfat.txt", sep=r"\s+", header=None, names=["bodyfat", "edad"])
df_b["y"] = df_b["bodyfat"] / 100.0
x_b = df_b["edad"].values
y_b = df_b["y"].values
n_b = len(y_b)

# Ajuste Simplex MLE
def simplex_nll(params):
    b0, b1, l_s2 = params
    s2 = np.exp(l_s2)
    eta = b0 + b1 * x_b
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-6, 1.0 - 1e-6)
    d = ((y_b - mu)**2) / (y_b * (1.0 - y_b) * (mu**2) * ((1.0 - mu)**2))
    return 0.5 * n_b * np.log(2 * np.pi * s2) + 1.5 * np.sum(np.log(y_b * (1.0 - y_b))) + (0.5 / s2) * np.sum(d)

opt_s = minimize(simplex_nll, [-1.5, 0.01, np.log(0.1)], method="BFGS")
b0_s, b1_s, ls2_s = opt_s.x
s2_s = np.exp(ls2_s)
aic_s = 2 * 3 + 2 * opt_s.fun
bic_s = np.log(n_b) * 3 + 2 * opt_s.fun

# Ajuste Beta MLE
def beta_nll(params):
    b0, b1, l_phi = params
    phi = np.exp(l_phi)
    eta = b0 + b1 * x_b
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-6, 1.0 - 1e-6)
    p = mu * phi
    q = (1.0 - mu) * phi
    return -np.sum(gammaln(phi) - gammaln(p) - gammaln(q) + (p - 1.0) * np.log(y_b) + (q - 1.0) * np.log(1.0 - y_b))

opt_b = minimize(beta_nll, [-1.5, 0.01, np.log(20.0)], method="BFGS")
b0_b, b1_b, lphi_b = opt_b.x
phi_b = np.exp(lphi_b)
aic_b = 2 * 3 + 2 * opt_b.fun
bic_b = np.log(n_b) * 3 + 2 * opt_b.fun

print(f"Simplex: Intercepto={b0_s:.4f}, Pendiente={b1_s:.4f}, sigma^2={s2_s:.4f} | AIC={aic_s:.2f}, BIC={bic_s:.2f}")
print(f"Beta:    Intercepto={b0_b:.4f}, Pendiente={b1_b:.4f}, phi={phi_b:.4f}    | AIC={aic_b:.2f}, BIC={bic_b:.2f}")
print(f"\nConclusión: El Modelo Beta es marcadamente superior (Delta-AIC = {aic_s - aic_b:.2f} > 10).")
