#####################################################################################################
#                           CONTROLE ADAPTATIVO - 2025.1
#               Author: Wilkley Bezerra Correia, Gabriel Costa Leite
#####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import cvxpy as cp

#####################################################################
# SIMULACAO DO SISTEMA PARA OBTENCAO DO CONJUNTO DE DADOS U0, X0, X1
#####################################################################
As = np.array([[2, -1],
               [1,  0]])
Bs = np.array([[1],
               [0]])
Cs = np.array([[0.005, 0.005]])
Ds = np.array([[0]])

m = Bs.shape[1]     # tamanho de u
n = As.shape[1]     # tamanho de x
p = Cs.shape[0]     # tamanho de y

tsim = 10           # tempo de simulacao
T = tsim - 1        # numero de amostras

U = np.hstack([np.ones((m, 1)), np.ones((m, tsim - 1))]) # entrada: impulso unitÃ¡rio
print(np.shape(U))
X = np.zeros((n, tsim))
X[:, 0] = np.array([1, 0])  # condiÃ§Ã£o inicial

for i in range(1, tsim):
    X[:, i] = As @ X[:, i - 1] + Bs @ U[:, i - 1]

# Extrai x1 e x2
Y1 = X[0, :]
Y2 = X[1, :]

X = np.array([Y1, Y2])
X1 = np.array(X[:,1:])
X0 = np.array(X[:,:-1])
U0 = U[:,:-1]

Q = cp.Variable((T, n), symmetric=False)

constraints = [
    X0 @ Q >> 0,
    cp.bmat([
        [X0 @ Q, X1 @ Q],
        [(X1 @ Q).T, X0 @ Q]
    ]) << 0
]
print(Q.value)

#J = cp.norm2(X1 - (As + Bs @ K))
J = 0
obj = cp.Minimize(J)
prob = cp.Problem(obj, constraints)
prob.solve(cp.CLARABEL)

K = (U0 @ Q.value) @ (np.linalg.inv(X0 @ Q.value))

print("K =", K)
print("Autovalores de (A + B*K):", np.abs(np.linalg.eigvals(As + Bs @ K)))