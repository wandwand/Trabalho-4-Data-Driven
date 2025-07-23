

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# MATRIZES A e B reais (da Q1)
# ============================================
A_real = np.array([
    [0.99004983, 0.,         0.,         0.,         0.],
    [0.,         0.97034751, -0.01970232, 0.,         0.],
    [0.,         0.00985116, 0.99990099, 0.,         0.],
    [0.,         0.,         0.,         0.97044553, 0.],
    [0.,         0.,         0.,         0.,         1.]
])

B_real = np.array([
    [9.95016625e-03, 0.00000000e+00],
    [0.00000000e+00, 9.85116044e-03],
    [0.00000000e+00, 4.95029042e-05],
    [9.85148882e-03, 0.00000000e+00],
    [0.00000000e+00, 1.00000000e-02]
])

# Ganho K fornecido - corrigindo a dimensão
K = np.array([
    [-2.552157, -4.382169, -16.892563, 6.600941, 5.663411],
    [-0.136022, 2.910439, -7.57638, 0.217211, 0.193825]
])
K= K*0.25

# ============================================
# Simulação em malha fechada
# ============================================
Nsim = 50
x0 = np.array([1, 0.5, -0.5, 0.3, 0.0])  # estado inicial

X_closed = np.zeros((5, Nsim))
U_closed = np.zeros((2, Nsim))
X_closed[:, 0] = x0

for k in range(1, Nsim):
    u = -K @ X_closed[:, k-1]
    X_closed[:, k] = A_real @ X_closed[:, k-1] + B_real @ u
    U_closed[:, k-1] = u  # Armazenar o controle aplicado no passo anterior

# Armazenar o último controle
U_closed[:, -1] = -K @ X_closed[:, -1]

# ============================================
# Plots melhorados
# ============================================
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(X_closed[i, :], linewidth=2, label=f'Estado x{i+1}')
plt.title("Evolução dos Estados em Malha Fechada (u = -Kx)", fontsize=14)
plt.xlabel("Passos k", fontsize=12)
plt.ylabel("Valor dos Estados", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.figure(figsize=(12, 4))
for j in range(2):
    plt.plot(U_closed[j, :], linewidth=2, label=f'Sinal de Controle u{j+1}')
plt.title("Sinais de Controle com K fornecido", fontsize=14)
plt.xlabel("Passos k", fontsize=12)
plt.ylabel("Valor do Controle", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

# ============================================
# Análise de Estabilidade
# ============================================
Acl = A_real - B_real @ K
eig_Acl = np.linalg.eigvals(Acl)
magnitude_eig = np.abs(eig_Acl)

print("\nAnálise de Estabilidade:")
print("Autovalores do sistema em malha fechada:")
for i, eig in enumerate(eig_Acl):
    print(f"λ{i+1} = {eig:.4f} (magnitude: {magnitude_eig[i]:.4f})")

if all(magnitude_eig < 1):
    print("\nO sistema em malha fechada é estável (todos |λ| < 1)")
else:
    print("\nAtenção: O sistema pode ser instável (algum |λ| ≥ 1)")

plt.figure(figsize=(8, 6))
angles = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(angles), np.sin(angles), 'k--')  # Círculo unitário
plt.plot(eig_Acl.real, eig_Acl.imag, 'ro', markersize=10)
plt.title("Diagrama de Autovalores no Plano Complexo", fontsize=14)
plt.xlabel("Parte Real", fontsize=12)
plt.ylabel("Parte Imaginária", fontsize=12)
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

plt.show()

