import numpy as np

def print_tableau(T, B, N, C_B, C_N, step):
    print(f"Iteracao {step}:")
    print("Tableau:")
    print(np.round(T, 2))
    print("Base (B):", [bi + 1 for bi in B])  # Ajuste para indexação baseada em 1
    print("Nao base (N):", [ni + 1 for ni in N])  # Ajuste para indexação baseada em 1
    print("Custo da Base C_B^T:", np.round(C_B, 2))
    print("Custo da Nao Base C_N^T:", np.round(C_N, 2))
    
    # Imprime as matrizes da base (B) e da não base (N)
    B_matrix = T[:, B]
    N_matrix = T[:, N]
    print("\nMatriz da Base (B):")
    print(np.round(B_matrix, 2))
    print("\nMatriz da Nao Base (N):")
    print(np.round(N_matrix, 2), "\n")

def pivot(T, pivot_row, pivot_col, B, N):
    # Realiza a operação de pivô
    T[pivot_row, :] /= T[pivot_row, pivot_col]
    for i in range(len(T)):
        if i != pivot_row:
            T[i, :] -= T[i, pivot_col] * T[pivot_row, :]
    B[pivot_row] = N[pivot_col]  # Atualiza o índice da variável básica
    return T, B

def simplex(c, A, b):
    num_constraints = A.shape[0]
    num_variables = A.shape[1]
    T = np.hstack((A, np.eye(num_constraints), b.reshape((-1, 1))))
    T = np.vstack((T, np.hstack(([-ci for ci in c], np.zeros(num_constraints + 1)))))
    B = list(range(num_variables, num_variables + num_constraints))  # Índices das variáveis de folga
    N = list(range(num_variables))  # Índices das variáveis originais
    C_B = np.zeros(num_constraints)  # Custos das variáveis básicas iniciais são 0
    C_N = [-ci for ci in c]  # Custos das variáveis não básicas iniciais (negativos)

    step = 0
    print_tableau(T, B, N, C_B, C_N, step)

    while True:
        print(f"\nPasso 2: Determinacao da coluna de entrada.")
        if np.all(T[-1, :-num_constraints] >= 0):
            print("Todos os custos reduzidos sao >= 0. Solucao otima encontrada.")
            break

        pivot_col = np.argmin(T[-1, :-num_constraints])
        print(f"Variavel N[{pivot_col + 1}] entra na base porque tem o menor custo reduzido, C_N[{pivot_col + 1}] = {T[-1, pivot_col]}.")

        print("\nPasso 4: Determinacao da linha de saida.")
        ratios = np.divide(T[:-1, -1], T[:-1, pivot_col], out=np.full_like(T[:-1, -1], np.inf), where=T[:-1, pivot_col] > 0)
        pivot_row = np.argmin(ratios)
        if np.all(T[:-1, pivot_col] <= 0):
            print("Todos os elementos na coluna de pivo são <= 0. Solução ilimitada.")
            return None
        print(f"Variavel B[{pivot_row + 1}] sai da base porque tem o menor ratio, E = {ratios[pivot_row]}.")

        T, B = pivot(T, pivot_row, pivot_col, B, N)
        print(f"\nPasso 5: Operacao de pivo concluida. Variavel B[{pivot_row + 1}] foi substituida por N[{pivot_col + 1}].")

        # Atualiza C_B e C_N após cada pivô
        C_B = [0 if bi >= num_variables else -c[bi] for bi in B]  # Use o sinal correto para C_B
        C_N = [-c[ni] if ni < num_variables and ni not in B else 0 for ni in range(num_variables)]  # Mantenha o sinal correto para C_N
        step += 1
        print_tableau(T, B, N, C_B, C_N, step)

    solution = np.zeros(num_variables + num_constraints)
    for i, bi in enumerate(B):
        solution[bi] = T[i, -1]
    objective_value = -T[-1, -1]
    return solution, objective_value, T, B, N

# Coeficientes da função objetivo e restrições
c = np.array([3, 5])
A = np.array([
    [3, 2],
    [1, 0],
    [0, 2],
])
b = np.array([18, 4, 12])

# Executa o algoritmo Simplex
solution, objective_value, final_tableau, B, N = simplex(c, A, b)

# Imprime a solução final e o valor da função objetivo
print("Solucao final:", solution)
print("Valor da funcao objetivo:", objective_value)
