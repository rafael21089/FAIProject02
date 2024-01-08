import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt


def edaPaises(wine_data):
    # Agrupa os dados por país e encontra os 10 melhores vinhos com base em pontos
    topCountry = wine_data.groupby('country').apply(lambda x: x.nlargest(10, 'points')).reset_index(drop=True)

    sns.set(style="whitegrid")

    # Grafico
    plt.figure(figsize=(15, 8))
    sns.barplot(x='country', y='points', data=topCountry, errorbar=None)
    plt.title('Média de pontos por país')
    plt.xlabel('País')
    plt.ylabel('Média de pontos')
    plt.show()

def aprioriAssociation(wine_data):
    # Agrupar por country and points, counta as ocurrencias dos id das reviews (o numero das reviews)
    wineGrouped = wine_data.groupby(['country', 'points'], as_index=False).agg({'id': 'count'})
    print(wineGrouped)

    # Cria uma Pivot table
    pivot = pd.pivot_table(wineGrouped, index='points', columns=['country'], values='id', fill_value=0)
    pivot = pivot.map(lambda x: True if x > 0 else False)
    print(pivot)

    # Aplica Apriori algorithm
    freqItems = apriori(pivot, min_support=0.3, use_colnames=True)
    print(freqItems)

    # Gera regras de associação
    rules = association_rules(freqItems, metric="support", min_threshold=0.5)
    print(rules.head())

    # Exiba as regras
    print("Association Rules:")
    print(rules)

    return rules
 

def melhorPaisAssociaciado(rules):
    # Filtra as regras de associação positiva
    posRules = rules[rules['lift'] > 1]

    # Calcula o support total para cada antecedente
    paisSupp = posRules.groupby('antecedents')['support'].sum()

    # Identifica o país com maior apoio total 
    melhorPais = paisSupp.idxmax()

    return melhorPais


def melhorPais90Pontos(wine_data):
  
    # Filtra vinhos com pelo menos 90 pontos
    vinhosHighScored = wine_data[wine_data['points'] >= 90]

    # Obtenha países unicos do dataset 
    paises90 = vinhosHighScored['country'].unique()
    print("Países com pelo menos um vinho com pontuação de 90 pontos ou mais:")
    print(paises90)


    return vinhosHighScored

# Recomendacao de paises com pontucaoes similares
def recomendacaoPaisSimilhar(rules, pais, wineData, rec_count):
    paisPontosMedia = wineData[wineData['country'] == pais]['points'].mean()

    sort = rules.sort_values('lift', ascending=False)
    paisRecomendados = []

    for i, rule in sort['antecedents'].items():
        for j in list(rule):
            if j == pais:
                paisRecomendados.append((list(sort.iloc[i]['consequents']), sort.iloc[i]['lift']))

    paisRecomendados = [(country, lift) for country_list, lift in paisRecomendados for country in country_list]
    paisRecomendados = list(set(paisRecomendados))
    
    # Filtra os países recomendados com base em pontos semelhantes e pontos médios mais altos do que o target country
    paisesSimilhar = [
        country for country, _ in sorted(paisRecomendados, key=lambda x: x[1], reverse=True)
        if wineData[wineData['country'] == country]['points'].mean() > paisPontosMedia
    ]
    
    return paisesSimilhar[:rec_count]

def aprioriTastersFavoritos(wineData, pais):
    # Filtra o dados para o pais especifico
    paisData = wineData[wineData['country'] == pais]

    # Agrupa por variety e taster, e conta as occurencias de cada id da review
    variedadesTasterGrouped = paisData.groupby(['variety', 'taster_name'], as_index=False).agg({'id': 'count'})

    # Cria uma Pivot table
    pivot = pd.pivot_table(variedadesTasterGrouped, index='taster_name', columns=['variety'], values='id', fill_value=0)
    pivot = pivot.map(lambda x: True if x > 0 else False)

    # Aplica Apriori algorithm
    freqItems = apriori(pivot, min_support=0.3, use_colnames=True)

    # Gera regras de associação
    rules = association_rules(freqItems, metric="support", min_threshold=0.5)

    # Filtra as regras de associação positiva
    posRules = rules[rules['lift'] > 1]

    # Calcula o support total para cada antecedente
    varietySupp = posRules.groupby('antecedents')['support'].sum()

    sortVariety = varietySupp.sort_values(ascending=False)

    plt.figure(figsize=(15, 8))
    sns.lineplot(x=range(len(sortVariety)), y=sortVariety.values, color='skyblue')
    plt.title(f'Quandtidade da Variedade de vinhos de {pais}')
    plt.xlabel('Variedade Index')
    plt.ylabel('Número médio')
    plt.show()

    # Identifique o vinho mais escolhido
    maisEscolhido = sortVariety.max()
    maisEscolhidoLista = sortVariety[sortVariety == maisEscolhido].index.tolist()

    return maisEscolhidoLista

# File path
file_path = "wine_database\winemag-data.csv"

# Load do dataset
wine_data = pd.read_csv(file_path)

# Seleciona as colunas relevantes
wine_data = wine_data[['id', 'country', 'winery', 'variety', 'province', 'points', 'taster_name']]

# Remove linhas com valores nulos
wine_data = wine_data.dropna()

# Performa Análise exploratória de dados para cada pais
edaPaises(wine_data)

# Apriori Association Analysis
rules = aprioriAssociation(wine_data)

# Obtenha o melhor país com base nas association rules
paisMaisAssociado = melhorPaisAssociaciado(rules)
print(f"O país com mais vinhos nesta magazine com base nas regras da associação é: {paisMaisAssociado}")

# Obtenha países com pelo menos um vinho com pontuação de 90 pontos ou mais
paises90 = melhorPais90Pontos(wine_data)
rules = aprioriAssociation(paises90)

melhorPais = melhorPaisAssociaciado(rules)
print(f"O país com os melhores vinhos é: {melhorPais}")


# Recomendacao de Paises similhares a France
paisSelec = 'France'
paisRecomendado = recomendacaoPaisSimilhar(rules, paisSelec, wine_data, rec_count=3)
print(f'\nPais: {paisSelec}')
print(f'Países recomendados: {paisRecomendado}')

# Escolhe o vinho mais escolhido por pais (France neste exemplo) pelos taste reviewers 
paisSelec = 'France'
vinhosMaisEscolhidos = aprioriTastersFavoritos(wine_data, paisSelec)
print(f"Os vinhos mais escolhidos na {paisSelec} por taste reviewers são: {vinhosMaisEscolhidos}")














