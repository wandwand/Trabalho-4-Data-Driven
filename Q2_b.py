import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# ============================================
# Matrsizes do sistema real
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

# Ganho do controlador inicial (fornecido)
K_initial = np.array([
    [-2.552157, -4.382169, -16.892563, 6.600941, 5.663411],
    [-0.136022, 2.910439, -7.57638, 0.217211, 0.193825]
])

# ============================================
# Design do controlador LQR
# ============================================
def design_lqr_controller(A, B, Q=None, R=None):
    """Design de controlador LQR com seleção de ponderação automática se não fornecido"""
    n = A.shape[0]
    m = B.shape[1]

    # Pesos padrão se não fornecido
    if Q is None:
        Q = np.eye(n)  # Começar com matriz identidade
        # Aumentar pesos para estados que precisam de mais atenção
        Q[0,0] = 10.0  # Peso maior para x1
        Q[1,1] = 5.0   # x2
        Q[2,2] = 5.0   # x3
        Q[3,3] = 5.0    # x4
        Q[4,4] = 1.0    # x5
    
    if R is None:
        R = np.eye(m) * 0.1  # Começar com pequena penalidade de controle

    # Resolve a Equação de Riccati Algébrica Discreta
    P = solve_discrete_are(A, B, Q, R)

    # Computa a matriz de ganho ótima
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    return K, Q, R

# Design do controlador LQR
K_lqr, Q, R = design_lqr_controller(A_real, B_real)

print("Matriz de ponderação Q do LQR:\n", Q)
print("\nMatriz de penalidade de controle R do LQR:\n", R)
print("\nMatriz de ganho ótima do LQR K:\n", K_lqr)

# ============================================
# Funçao de simulação do sistema
# ============================================
def simulate_system(A, B, K, x0, Nsim=50):
    """Simula malha fechada do sistema com controlador K"""
    n = A.shape[0]
    m = B.shape[1]
    
    X = np.zeros((n, Nsim))
    U = np.zeros((m, Nsim))
    X[:, 0] = x0
    
    for k in range(1, Nsim):
        u = -K @ X[:, k-1]
        X[:, k] = A @ X[:, k-1] + B @ u
        U[:, k-1] = u
    
    # Guarda o último controle
    U[:, -1] = -K @ X[:, -1]
    
    return X, U

# Estado inicial
x0 = np.array([1, 0.5, -0.5, 0.3, 0.0])

# Simula ambos os controladores
X_initial, U_initial = simulate_system(A_real, B_real, K_initial, x0)
X_lqr, U_lqr = simulate_system(A_real, B_real, K_lqr, x0)

# ============================================
# Analise de Estabilidade
# ============================================
def analyze_stability(A, B, K):
    """Analisa a estabilidade do sistema com o controlador K"""
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)
    max_magnitude = np.max(np.abs(eigvals))
    
    print("\nMalha fechada eigenvalues:")
    for i, eig in enumerate(eigvals):
        print(f"λ{i+1} = {eig:.4f} (|λ| = {np.abs(eig):.4f})")
    
    print(f"\nMagnitude máximade eigenvalues: {max_magnitude:.4f}")
    if max_magnitude < 1:
        print("Sistema é estável (se |λ| < 1)")
    else:
        print("Sistema pode ser instável (se |λ| ≥ 1)")

    return A_cl, eigvals

print("\nAnálise de Estabilidade do Controlador Inicial:")
Acl_initial, eig_initial = analyze_stability(A_real, B_real, K_initial)

print("\nLQR Controller Stability Analysis:")
Acl_lqr, eig_lqr = analyze_stability(A_real, B_real, K_lqr)

# ============================================
# Plotando Resultados
# ============================================
def plot_states(X, title, labels=None):
    """Plotar trajetórias de estado"""
    plt.figure(figsize=(12, 6))
    n = X.shape[0]
    
    if labels is None:
        labels = [f'Estado x{i+1}' for i in range(n)]
    
    for i in range(n):
        plt.plot(X[i, :], linewidth=2, label=labels[i])
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time step k", fontsize=12)
    plt.ylabel("Valor de estado", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def plot_controls(U, title):
    """Plotar sinais de controle"""
    plt.figure(figsize=(12, 4))
    m = U.shape[0]
    
    for j in range(m):
        plt.plot(U[j, :], linewidth=2, label=f'Control u{j+1}')
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time step k", fontsize=12)
    plt.ylabel("Valor de controle", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

def plot_eigenvalues(eigvals, title):
    """Plotar eigenvalues no plano complexo"""
    plt.figure(figsize=(8, 6))
    angles = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(angles), np.sin(angles), 'k--')  # Unit circle
    plt.plot(eigvals.real, eigvals.imag, 'ro', markersize=10)
    
    # Anotar eigenvalues
    for i, eig in enumerate(eigvals):
        plt.annotate(f'λ{i+1}', (eig.real, eig.imag), 
                     textcoords="offset points", xytext=(5,5), ha='center')
    
    plt.title(title, fontsize=14)
    plt.xlabel("Parte real", fontsize=12)
    plt.ylabel("Parte imaginária", fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

# Plot dos resultados para o controlador inicial
plot_states(X_initial, "Trajetórias dos Estados com Controlador Inicial")
plot_controls(U_initial, "Sinais de Controle com Controlador Inicial")
plot_eigenvalues(eig_initial, "Autovalores com Controlador Inicial")

# Plot dos resultados para o controlador LQR
plot_states(X_lqr, "Trajetórias dos Estados com Controlador LQR")
plot_controls(U_lqr, "Sinais de Controle com Controlador LQR")
plot_eigenvalues(eig_lqr, "Autovalores com Controlador LQR")

plt.show()

# ============================================
# Comparação de performance
# ============================================
def calculate_cost(X, U, Q, R):
    """Calcule o custo LQR para a trajetórria"""
    cost = 0
    for k in range(X.shape[1]-1):
        x = X[:, k]
        u = U[:, k]
        cost += x.T @ Q @ x + u.T @ R @ u
    return cost

# Calcule custos (usando pesos LQR para uma comparação justa)
cost_initial = calculate_cost(X_initial, U_initial, Q, R)
cost_lqr = calculate_cost(X_lqr, U_lqr, Q, R)

print("\nComparação de performance:")
print(f"Custu inicial do controlador: {cost_initial:.4f}")
print(f"Custo do controlador LQR: {cost_lqr:.4f}")
print(f"Aprimoramentos: {100*(cost_initial-cost_lqr)/cost_initial:.2f}% reduction")