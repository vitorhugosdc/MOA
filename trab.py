import numpy as np

def print_passo(T, T_original, B, N, C_B, C_N, passo):
    print("\n\n-------------------------------------------------------------------------------------------")
    print(f"\nIteracao {passo+1}:")
    print("T:")
    print(np.round(T, 2))
    print("\nBase (B):", [bi + 1 for bi in B])  # Ajuste para indexação baseada em 1
    print("Nao base (N):", [ni + 1 for ni in N])  # Ajuste para indexação baseada em 1
    print("Custo da Base C_B^T:", np.round(C_B, 2))
    print("Custo da Nao Base C_N^T:", np.round(C_N, 2))
    print("\nMatriz da Base (B):")
    print(np.round(T_original[:, B], 2))
    print("\nMatriz da Nao Base (N):")
    print(np.round(T_original[:, N], 2), "\n")

def leitura(caminho):    
    with open(caminho, 'r') as file:
        tipo_problema = file.readline().strip()
        c = np.array(list(map(float, file.readline().strip().split())))
        if tipo_problema == 'min':
            c = -c
        num_restricoes = int(file.readline().strip())
        A = []
        b = []
        for _ in range(num_restricoes):
            linha = list(map(float, file.readline().strip().split()))
            A.append(linha[:-1])
            b.append(linha[-1])
    return c, np.array(A), np.array(b), tipo_problema

def pivo(T, linha_pivo, coluna_pivo, B, N):
    T[linha_pivo, :] /= T[linha_pivo, coluna_pivo]
    for i in range(len(T)):
        if i != linha_pivo:
            T[i, :] -= T[i, coluna_pivo] * T[linha_pivo, :]

    saida = B[linha_pivo]  # Sai da base
    entrada = N[coluna_pivo]  # Entra na base
    B[linha_pivo] = entrada
    N[coluna_pivo] = saida

    return T, B, N

def simplex(c, A, b):
    num_restricoes = A.shape[0]
    num_variaveis = A.shape[1]
    T = np.hstack((A, np.eye(num_restricoes), b.reshape((-1, 1))))
    T_original = np.copy(T)
    T = np.vstack((T, np.hstack(([-ci for ci in c], np.zeros(num_restricoes + 1)))))
    B = list(range(num_variaveis, num_variaveis + num_restricoes))  # Índices das variáveis de folga
    N = list(range(num_variaveis))  # Índices das variáveis originais
    C_B = np.zeros(num_restricoes)  # Custos das variáveis básicas iniciais são 0
    C_N = [-ci for ci in c]  # Custos das variáveis não básicas iniciais (negativos)

    passo = 0
    print_passo(T, T_original, B, N, C_B, C_N, passo)

    while True:
        if np.all(T[-1, :-num_restricoes] >= 0):
            print("Todos os custos reduzidos sao >= 0. Solucao otima encontrada.")
            break
        
        print(f"\nPasso 1: X_B = B^-1 * b")
        print(f"\nX_B = {np.linalg.inv(T_original[:, B])} * {T[:-1, -1]}\n")   
        
        print(f"\nPasso 2: Determinacao da coluna de entrada.\n")
        lambda_ = np.dot(C_B, np.linalg.inv(T_original[:, B]))        
        print(f'i) lambda^T = C_B^T * B^-1 = \n\n{C_B} * {np.linalg.inv(T_original[:, B])} = \n{lambda_}\n')

        print(f'ii) C_Nj = C_Nj - lambda^T a_Nj\n')
        i = 0
        for nj in N:
            print(f'C_N{i+1} = C_N{i+1} - lambda^T a_N{i+1} = {C_N[i]} - {lambda_} * {T_original[:, nj]}')
            i+=1
        
        print('\n\niii) C_Nk = min {C_N1},{C_N2}...\n')                    
        valores_coluna_pivo = T[-1, :-num_restricoes]
        for i, cv in enumerate(valores_coluna_pivo, start=1):
            print(f"\nC_N[{i}] = {cv}")
        coluna_pivo = np.argmin(valores_coluna_pivo)
        print(f"\nVariavel N[{coluna_pivo + 1}] entra na base C_N[{coluna_pivo + 1}] = {T[-1, coluna_pivo]}.")

        print(f'\nPasso 3: C_Nk = {T[-1, coluna_pivo]} >= 0? Nao, entao continuamos\n')


        print('Passo 4: y = B^-1 A_Nk')
        print(f'y = {np.linalg.inv(T_original[:, B])} {T_original[:, N[coluna_pivo]]}')
        
        # Determinação da linha de saída
        print("\nPasso 5: Determinacao da linha de saida.")
        razao = np.divide(T[:-1, -1], T[:-1, coluna_pivo], out=np.full_like(T[:-1, -1], np.inf), where=T[:-1, coluna_pivo] > 0)
        for i, ratio in enumerate(razao, start=1):
            if ratio != np.inf:
                print(f"Razao para B[{i}]: {ratio}")
            else:
                print(f"Razao para B[{i}]: Infinito (variavel fora da consideracao)")
        linha_pivo = np.argmin(razao)
        print(f"\nVariavel B[{linha_pivo + 1}] sai da base porque tem o menor ratio, E = {razao[linha_pivo]}.")

        T, B, N = pivo(T, linha_pivo, coluna_pivo, B, N)
        print(f"\nIteracao concluida. Variavel B[{linha_pivo + 1}] foi substituida por N[{coluna_pivo + 1}].")

        # Atualiza C_B e C_N após cada pivô
        C_B = [0 if bi >= num_variaveis else -c[bi] for bi in B]
        C_N = [-c[ni] if ni < num_variaveis and ni not in B else 0 for ni in range(num_variaveis)]
        passo += 1
        print_passo(T, T_original, B, N, C_B, C_N, passo)

    solucao = np.zeros(num_variaveis + num_restricoes)
    for i, bi in enumerate(B):
        solucao[bi] = T[i, -1]
    valor_objetivo = -T[-1, -1]
    return solucao, valor_objetivo, T, B, N

c, A, b, tipo_problema = leitura('max.txt')
solucao, valor_objetivo, final_tableau, B, N = simplex(c, A, b)

if tipo_problema != 'min':
    valor_objetivo = -valor_objetivo

print(np.round(final_tableau, 2))
print("\nBase (B):", [bi + 1 for bi in B])
print("Nao base (N):", [ni + 1 for ni in N])
print("Solucao final:", solucao)
print("Valor da funcao objetivo:", valor_objetivo)
