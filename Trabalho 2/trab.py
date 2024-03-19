def ler_dados(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        linhas = arquivo.read().splitlines()
        indice_n = next(i for i, linha in enumerate(linhas) if linha.startswith('n ='))
        
        linhas_pesos = ' '.join(linha.split('peso: ')[-1] for linha in linhas[:indice_n] if linha.startswith('peso:'))
        linhas_utilidades = ' '.join(linha.split('utilidade: ')[-1] for linha in linhas[:indice_n] if linha.startswith('utilidade:'))
        
        pesos = list(map(int, linhas_pesos.split()))
        utilidades = list(map(int, linhas_utilidades.split()))

        n = int(linhas[indice_n].split('n = ')[1])

    return pesos, utilidades, n

def borsa(caminho_arquivo):
    peso, utilidade, n = ler_dados(caminho_arquivo)
    itens = list(range(len(peso)))
    densidade_valor = [(utilidade[i] / peso[i], i) for i in itens]
    
    densidade_valor.sort(reverse=True)

    peso_total = 0
    utilidade_total = 0
    itens_selecionados = []

    for densidade, i in densidade_valor:
        if peso_total + peso[i] <= n:
            itens_selecionados.append(i)
            peso_total += peso[i]
            utilidade_total += utilidade[i]

    return itens_selecionados, peso_total, utilidade_total

itens_selecionados_data_1, peso_total_data_1, utilidade_total_data_1 = borsa("data_1.txt")
print("Data 1:")
print("Itens selecionados:", itens_selecionados_data_1)
print("Peso total:", peso_total_data_1)
print("Utilidade total:", utilidade_total_data_1)

itens_selecionados_data_2, peso_total_data_2, utilidade_total_data_2 = borsa("data_2.txt")
print("Data 2:")
print("Itens selecionados:", itens_selecionados_data_2)
print("Peso total:", peso_total_data_2)
print("Utilidade total:", utilidade_total_data_2)
