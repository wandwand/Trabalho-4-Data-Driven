import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2ss, cont2discrete

# --- Funções de transferência ---
num11, den11 = [1], [1, 1]           # G11(s)
num12, den12 = [1, 2], [1, 3, 2]     # G12(s)
num21, den21 = [1], [1, 3]           # G21(s)
num22, den22 = [1], [1, 0]           # G22(s)

# --- Espaço de estados contínuo ---
A11, B11, C11, D11 = tf2ss(num11, den11)
A12, B12, C12, D12 = tf2ss(num12, den12)
A21, B21, C21, D21 = tf2ss(num21, den21)
A22, B22, C22, D22 = tf2ss(num22, den22)

# --- Discretização ---
Ts = 0.01
Ad11, Bd11, Cd11, Dd11, _ = cont2discrete((A11,B11,C11,D11), Ts)
Ad12, Bd12, Cd12, Dd12, _ = cont2discrete((A12,B12,C12,D12), Ts)
Ad21, Bd21, Cd21, Dd21, _ = cont2discrete((A21,B21,C21,D21), Ts)
Ad22, Bd22, Cd22, Dd22, _ = cont2discrete((A22,B22,C22,D22), Ts)

# --- Montagem do sistema MIMO ---
n11, n12, n21, n22 = Ad11.shape[0], Ad12.shape[0], Ad21.shape[0], Ad22.shape[0]
n = n11 + n12 + n21 + n22
Ad = np.block([
    [Ad11, np.zeros((n11,n12+n21+n22))],
    [np.zeros((n12,n11)), Ad12, np.zeros((n12,n21+n22))],
    [np.zeros((n21,n11+n12)), Ad21, np.zeros((n21,n22))],
    [np.zeros((n22,n11+n12+n21)), Ad22]
])
Bd = np.block([
    [Bd11, np.zeros((n11,1))],
    [np.zeros((n12,1)), Bd12],
    [Bd21, np.zeros((n21,1))],
    [np.zeros((n22,1)), Bd22]
])
Cd = np.block([
    [Cd11, Cd12, np.zeros((1,n21+n22))],  # y1 depende de G11 e G12
    [np.zeros((1,n11+n12)), Cd21, Cd22]  # y2 depende de G21 e G22
])
Dd = np.array([[Dd11[0,0], Dd12[0,0]],
               [Dd21[0,0], Dd22[0,0]]])

# --- Simulação ---
Tsim = 10
t = np.arange(0,Tsim,Ts)
T = len(t)
u1 = np.sin(2*t)
u2 = np.cos(3*t)
u = np.vstack((u1,u2))

X = np.zeros((n,T))
Y = np.zeros((2,T))

for k in range(T-1):
    X[:,k+1] = Ad @ X[:,k] + Bd @ u[:,k]
    Y[:,k] = Cd @ X[:,k] + Dd @ u[:,k]

# Última saída
Y[:, -1] = Cd @ X[:, -1] + Dd @ u[:, -1]

# --- Matrizes de dados ---
X0 = X[:, :-1]
X1 = X[:, 1:]
U0 = u[:, :-1]

print("\n=== Parte (a): Matrizes ===")
print("X0 shape:", X0.shape)
print("X1 shape:", X1.shape)
print("U0 shape:", U0.shape)

# Parte (b): persistência
rank = np.linalg.matrix_rank(U0)
print("\n=== Parte (b): Persistência ===")
print("Rank de U0:", rank)

# Parte (c): Theta e extração de A e B
Z = np.vstack((X0, U0))
Theta = X1 @ np.linalg.pinv(Z)

n_states = X0.shape[0]
n_inputs = U0.shape[0]

A_est = Theta[:, :n_states]
B_est = Theta[:, n_states:]

print("\n=== Parte (c): Estimação ===")
print("Theta (A|B):\n", Theta)
print("\nMatriz A estimada:\n", A_est)
print("\nMatriz B estimada:\n", B_est)
print("\n=== Matrizes Reais (Discretizadas) ===")
print("Matriz A real (Ad):\n", Ad)
print("\nMatriz B real (Bd):\n", Bd)

# Ganho K (no contexto do enunciado): é [A B]
K = Theta
print("\nGanho K = [A B]:\n", K)


# --- Gráfico das saídas ---
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, Y[0,:])
plt.title('Saída y1')
plt.grid(True)
plt.subplot(2,1,2)
plt.plot(t, Y[1,:])
plt.title('Saída y2')
plt.grid(True)
plt.tight_layout()
plt.show()
