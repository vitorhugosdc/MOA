def ler_dados(arquivo):
    dados = {
        "data_1": {
            "peso": [382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150, 823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111, 323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684],
            "utilidade": [825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457, 1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538, 675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261],
        },
        "data_2": {
            "peso": [23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
            "utilidade": [92, 57, 49, 68, 60, 43, 67, 84, 87, 72],
        }
    }

    return dados[arquivo]["peso"], dados[arquivo]["utilidade"]

def borsa(arquivo, n):
    peso, utilidade = ler_dados(arquivo)
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

itens_selecionados_data_1, peso_total_data_1, utilidade_total_data_1 = borsa("data_1", 6404180)
print("Data 1:")
print("Itens selecionados:", itens_selecionados_data_1)
print("Peso total:", peso_total_data_1)
print("Utilidade total:", utilidade_total_data_1)
print()

itens_selecionados_data_2, peso_total_data_2, utilidade_total_data_2 = borsa("data_2", 165)
print("Data 2:")
print("Itens selecionados:", itens_selecionados_data_2)
print("Peso total:", peso_total_data_2)
print("Utilidade total:", utilidade_total_data_2)
