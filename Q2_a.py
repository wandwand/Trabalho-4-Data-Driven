import numpy as np
import cvxpy as cp
import Q1_main as Q1

# ===================================
# Dados da Q1
# ===================================
X0 = Q1.X0
X1 = Q1.X1
U0 = Q1.U0

n = X0.shape[0]  # número de estados
T = X0.shape[1]  # número de amostras

# ===================================
# Redução de dimensionalidade
# ===================================
# Reduzir para 100 amostras para melhor condicionamento
if T > 100:
    step = T // 100
    X0 = X0[:, ::step]
    X1 = X1[:, ::step]
    U0 = U0[:, ::step]
    T = X0.shape[1]

# ===================================
# Verificação de persistência de excitação
# ===================================
rank_UX0 = np.linalg.matrix_rank(np.vstack([U0, X0]))
print(f"Rank da matriz [U0; X0]: {rank_UX0} (deveria ser >= n+m = {n+U0.shape[0]})")

# ===================================
# Variáveis de otimização
# ===================================
Q = cp.Variable((T, n))
P = cp.Variable((n, n), symmetric=True)  # Matriz de Lyapunov

# ===================================
# Construção da LMI (Teorema 3 do artigo)
# ===================================
X0Q = X0 @ Q
X1Q = X1 @ Q

constraints = [
    P >> 1e-6 * np.eye(n),  # P > 0
    X0Q == P,               # Relação entre Q e P
    cp.bmat([
        [P, X1Q.T],
        [X1Q, P]
    ]) >> 1e-6 * np.eye(2*n)  # LMI de estabilidade
]

# ===================================
# Problema de otimização
# ===================================
prob = cp.Problem(cp.Minimize(cp.norm(Q, 'fro')), constraints)

# ===================================
# Resolver com hierarquia de solvers
# ===================================
solvers = [
    {'name': 'MOSEK', 'solver': cp.MOSEK},
    {'name': 'SCS', 'solver': cp.SCS, 'kwargs': {'max_iters': 10000, 'eps': 1e-6}},
    {'name': 'CVXOPT', 'solver': cp.CVXOPT}
]

solver_success = False
for solver_info in solvers:
    try:
        print(f"Tentando solver: {solver_info['name']}")
        kwargs = solver_info.get('kwargs', {})
        prob.solve(solver=solver_info['solver'], verbose=True, **kwargs)
        
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            solver_success = True
            break
    except Exception as e:
        print(f"Erro com {solver_info['name']}: {str(e)}")

# ===================================
# Cálculo robusto do ganho K
# ===================================
if solver_success:
    Q_val = Q.value
    X0Q_val = X0 @ Q_val
    
    # Verificar condicionamento
    cond_number = np.linalg.cond(X0Q_val)
    print(f"Número de condicionamento de X0Q: {cond_number:.2e}")
    
    if cond_number > 1e10:
        # Usar pseudo-inversa se mal condicionado
        K = U0 @ Q_val @ np.linalg.pinv(X0Q_val)
        print("Aviso: Matriz mal condicionada, usando pseudo-inversa")
    else:
        K = U0 @ Q_val @ np.linalg.inv(X0Q_val)
    
    # Verificar autovalores do sistema em malha fechada
    A_cl = X1 @ Q_val @ np.linalg.inv(X0Q_val)
    eigvals = np.linalg.eigvals(A_cl)
    print("Autovalores do sistema em malha fechada:", np.abs(eigvals))
    
    print("\nGanho K calculado (Data-Driven):\n", np.round(K, 6))
    print("Norma de Frobenius de Q:", np.linalg.norm(Q_val, 'fro'))
else:
    print("\nNão foi possível resolver a LMI. Status:", prob.status)
    print("\nRecomendações:")
    print("1. Verifique a persistência de excitação (rank atual =", rank_UX0, ")")
    print("2. Aumente o número de amostras (atual =", T, ")")
    print("3. Reduza ainda mais a dimensionalidade")
    print("4. Verifique se os dados estão bem escalados")