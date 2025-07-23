import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are

# ============================================
# System Matrices
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

# Initial controller gain (provided)
K_initial = np.array([
    [-2.552157, -4.382169, -16.892563, 6.600941, 5.663411],
    [-0.136022, 2.910439, -7.57638, 0.217211, 0.193825]
])

# ============================================
# LQR Controller Design
# ============================================
def design_lqr_controller(A, B, Q=None, R=None):
    """Design LQR controller with automatic weight selection if not provided"""
    n = A.shape[0]
    m = B.shape[1]
    
    # Default weights if not provided
    if Q is None:
        Q = np.eye(n)  # Start with identity matrix
        # Increase weights for states that need more attention
        Q[0,0] = 10.0  # Higher weight for x1
        Q[1,1] = 5.0   # x2
        Q[2,2] = 5.0   # x3
        Q[3,3] = 5.0    # x4
        Q[4,4] = 1.0    # x5
    
    if R is None:
        R = np.eye(m) * 0.1  # Start with small control penalty
    
    # Solve Discrete Algebraic Riccati Equation
    P = solve_discrete_are(A, B, Q, R)
    
    # Compute optimal gain matrix
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    return K, Q, R

# Design LQR controller
K_lqr, Q, R = design_lqr_controller(A_real, B_real)

print("LQR Weighting Matrix Q:\n", Q)
print("\nLQR Control Penalty Matrix R:\n", R)
print("\nOptimal LQR Gain Matrix K:\n", K_lqr)

# ============================================
# Simulation Function
# ============================================
def simulate_system(A, B, K, x0, Nsim=50):
    """Simulate closed-loop system"""
    n = A.shape[0]
    m = B.shape[1]
    
    X = np.zeros((n, Nsim))
    U = np.zeros((m, Nsim))
    X[:, 0] = x0
    
    for k in range(1, Nsim):
        u = -K @ X[:, k-1]
        X[:, k] = A @ X[:, k-1] + B @ u
        U[:, k-1] = u
    
    # Store final control action
    U[:, -1] = -K @ X[:, -1]
    
    return X, U

# Initial state
x0 = np.array([1, 0.5, -0.5, 0.3, 0.0])

# Simulate both controllers
X_initial, U_initial = simulate_system(A_real, B_real, K_initial, x0)
X_lqr, U_lqr = simulate_system(A_real, B_real, K_lqr, x0)

# ============================================
# Stability Analysis
# ============================================
def analyze_stability(A, B, K):
    """Analyze closed-loop stability"""
    A_cl = A - B @ K
    eigvals = np.linalg.eigvals(A_cl)
    max_magnitude = np.max(np.abs(eigvals))
    
    print("\nClosed-loop eigenvalues:")
    for i, eig in enumerate(eigvals):
        print(f"λ{i+1} = {eig:.4f} (|λ| = {np.abs(eig):.4f})")
    
    print(f"\nMaximum eigenvalue magnitude: {max_magnitude:.4f}")
    if max_magnitude < 1:
        print("System is stable (all |λ| < 1)")
    else:
        print("System may be unstable (some |λ| ≥ 1)")
    
    return A_cl, eigvals

print("\nInitial Controller Stability Analysis:")
Acl_initial, eig_initial = analyze_stability(A_real, B_real, K_initial)

print("\nLQR Controller Stability Analysis:")
Acl_lqr, eig_lqr = analyze_stability(A_real, B_real, K_lqr)

# ============================================
# Plotting Functions
# ============================================
def plot_states(X, title, labels=None):
    """Plot state trajectories"""
    plt.figure(figsize=(12, 6))
    n = X.shape[0]
    
    if labels is None:
        labels = [f'State x{i+1}' for i in range(n)]
    
    for i in range(n):
        plt.plot(X[i, :], linewidth=2, label=labels[i])
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time step k", fontsize=12)
    plt.ylabel("State value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

def plot_controls(U, title):
    """Plot control signals"""
    plt.figure(figsize=(12, 4))
    m = U.shape[0]
    
    for j in range(m):
        plt.plot(U[j, :], linewidth=2, label=f'Control u{j+1}')
    
    plt.title(title, fontsize=14)
    plt.xlabel("Time step k", fontsize=12)
    plt.ylabel("Control value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

def plot_eigenvalues(eigvals, title):
    """Plot eigenvalues in complex plane"""
    plt.figure(figsize=(8, 6))
    angles = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(angles), np.sin(angles), 'k--')  # Unit circle
    plt.plot(eigvals.real, eigvals.imag, 'ro', markersize=10)
    
    # Annotate eigenvalues
    for i, eig in enumerate(eigvals):
        plt.annotate(f'λ{i+1}', (eig.real, eig.imag), 
                     textcoords="offset points", xytext=(5,5), ha='center')
    
    plt.title(title, fontsize=14)
    plt.xlabel("Real part", fontsize=12)
    plt.ylabel("Imaginary part", fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

# Plot results for initial controller
plot_states(X_initial, "State Trajectories with Initial Controller")
plot_controls(U_initial, "Control Signals with Initial Controller")
plot_eigenvalues(eig_initial, "Eigenvalues with Initial Controller")

# Plot results for LQR controller
plot_states(X_lqr, "State Trajectories with LQR Controller")
plot_controls(U_lqr, "Control Signals with LQR Controller")
plot_eigenvalues(eig_lqr, "Eigenvalues with LQR Controller")

plt.show()

# ============================================
# Performance Comparison
# ============================================
def calculate_cost(X, U, Q, R):
    """Calculate LQR cost for trajectory"""
    cost = 0
    for k in range(X.shape[1]-1):
        x = X[:, k]
        u = U[:, k]
        cost += x.T @ Q @ x + u.T @ R @ u
    return cost

# Calculate costs (using LQR weights for both for fair comparison)
cost_initial = calculate_cost(X_initial, U_initial, Q, R)
cost_lqr = calculate_cost(X_lqr, U_lqr, Q, R)

print("\nPerformance Comparison:")
print(f"Initial controller cost: {cost_initial:.4f}")
print(f"LQR controller cost: {cost_lqr:.4f}")
print(f"Improvement: {100*(cost_initial-cost_lqr)/cost_initial:.2f}% reduction")