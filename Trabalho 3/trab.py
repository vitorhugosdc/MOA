import numpy as np

def calcular_distancia_euclidiana(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def construir_matriz_distancia_euc_2d(cidades):
    n = len(cidades)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distancia = calcular_distancia_euclidiana(*cidades[i], *cidades[j])
            matriz[i][j] = matriz[j][i] = distancia
    return matriz

def vizinho_mais_proximo(matriz_distancia):
    n = matriz_distancia.shape[0]
    visitado = [False] * n
    percurso = [0]
    visitado[0] = True
    custo_total = 0

    for _ in range(1, n):
        ultimo = percurso[-1]
        proximo = np.argmin([matriz_distancia[ultimo][j] if not visitado[j] else np.inf for j in range(n)])
        percurso.append(proximo)
        custo_total += matriz_distancia[ultimo][proximo]
        visitado[proximo] = True

    custo_total += matriz_distancia[percurso[-1]][percurso[0]]
    percurso.append(0)
    return percurso, custo_total

def mochileiro(arquivo):
    with open(arquivo, 'r') as file:
        linhas = file.read().splitlines()
        if 'EDGE_WEIGHT_TYPE: EXPLICIT' in linhas:
            index = linhas.index('EDGE_WEIGHT_SECTION') + 1
            for linha in linhas:
                if linha.startswith('DIMENSION'):
                    n = int(linha.split()[1])
                    break
            matriz_distancia = np.array([list(map(int, linhas[i].split())) for i in range(index, index + n)])
        elif 'EDGE_WEIGHT_TYPE: EUC_2D' in linhas:
            index = linhas.index('NODE_COORD_SECTION') + 1
            n = int(linhas[linhas.index('DIMENSION: ') + 1].split()[1])
            cidades = [list(map(float, linhas[i].split()[1:])) for i in range(index, index + n)]
            matriz_distancia = construir_matriz_distancia_euc_2d(cidades)
        else:
            raise ValueError('Tipo de dados de distância não suportado.')

    percurso, custo_total = vizinho_mais_proximo(matriz_distancia)
    return percurso, custo_total

percurso, custo_total = mochileiro('bays29.tsp')
print(f'PERCURSO: {percurso}\n')
print(f'DISTANCIA TOTAL: {custo_total}\n')
