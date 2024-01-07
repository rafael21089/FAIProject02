import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt

def read_and_preprocess_data(file_path):
    # Load do dataset
    wine_data = pd.read_csv(file_path)

    # Seleciona as colunas relevantes
    wine_data = wine_data[['id', 'country', 'winery', 'variety', 'province', 'points', 'taster_name']]

    # Remove linhas com valores nulos
    wine_data = wine_data.dropna()

    return wine_data

def eda_top_wines_per_country(wine_data):
    # Agrupa os dados por país e encontra os 10 melhores vinhos com base em pontos
    top_wines_per_country = wine_data.groupby('country').apply(lambda x: x.nlargest(10, 'points')).reset_index(drop=True)

    sns.set(style="whitegrid")

    # Grafico
    plt.figure(figsize=(15, 8))
    sns.barplot(x='country', y='points', data=top_wines_per_country, errorbar=None)
    plt.title('Média de pontos por país')
    plt.xlabel('País')
    plt.ylabel('Média de pontos')
    plt.show()

def apriori_association_analysis(wine_data):
    # Agrupar por country and points, counta as ocurrencias de taster_name
    variety_grouped = wine_data.groupby(['country', 'points'], as_index=False).agg({'taster_name': 'count'})
    print(variety_grouped)

    # Cria uma Pivot table
    item_count_pivot = pd.pivot_table(variety_grouped, index='points', columns=['country'], values='taster_name', fill_value=0)
    item_count_pivot = item_count_pivot.map(lambda x: True if x > 0 else False)
    print(item_count_pivot)

    # Aplica Apriori algorithm
    freq_itemsets = apriori(item_count_pivot, min_support=0.3, use_colnames=True)
    print(freq_itemsets)

    # Gera regras de associação
    rules = association_rules(freq_itemsets, metric="support", min_threshold=0.5)
    print(rules.head())

    # Exiba as regras
    print("Association Rules:")
    print(rules)

    return rules
 

def best_country_based_on_association_rules(rules):
    # Filtra as regras de associação positiva
    positive_rules = rules[rules['lift'] > 1]

    # Calcula o support total para cada antecedente
    country_support = positive_rules.groupby('antecedents')['support'].sum()

    # Identifica o país com maior apoio total 
    best_country = country_support.idxmax()

    return best_country


def get_countries_with_90_points(wine_data):
  
    # Filtra vinhos com pelo menos 90 pontos
    high_scored_wines = wine_data[wine_data['points'] >= 90]

    # Obtenha países unicos do dataset 
    countries_with_90_points = high_scored_wines['country'].unique()
    print("Países com pelo menos um vinho com pontuação de 90 pontos ou mais:")
    print(countries_with_90_points)


    return high_scored_wines


# File path
file_path = "winemag-data1.csv"

# Wine data
wine_data = read_and_preprocess_data(file_path)

# Performa Análise exploratória de dados para cada pais
eda_top_wines_per_country(wine_data)

# Apriori Association Analysis
rules = apriori_association_analysis(wine_data)

# Obtenha o melhor país com base nas association rules
best_country = best_country_based_on_association_rules(rules)
print(f"O país com mais vinhos nesta magazine com base nas regras da associação é: {best_country}")

# Obtenha países com pelo menos um vinho com pontuação de 90 pontos ou mais
countries_90_points = get_countries_with_90_points(wine_data)
rules = apriori_association_analysis(countries_90_points)

best_country_with_90_points_up = best_country_based_on_association_rules(rules)
print(f"O país com os melhores vinhos é: {best_country_with_90_points_up}")

# Recomendacao de paises com pontucaoes similares
def recommend_similar_point_countries(rules_df, target_country, wine_data, rec_count):
    target_country_average_points = wine_data[wine_data['country'] == target_country]['points'].mean()

    sorted_rules = rules_df.sort_values('lift', ascending=False)
    recommended_countries = []

    for i, rule in sorted_rules['antecedents'].items():
        for j in list(rule):
            if j == target_country:
                recommended_countries.append((list(sorted_rules.iloc[i]['consequents']), sorted_rules.iloc[i]['lift']))

    recommended_countries = [(country, lift) for country_list, lift in recommended_countries for country in country_list]
    recommended_countries = list(set(recommended_countries))
    
    # Filtra os países recomendados com base em pontos semelhantes e pontos médios mais altos do que o target country
    similar_point_countries = [
        country for country, _ in sorted(recommended_countries, key=lambda x: x[1], reverse=True)
        if wine_data[wine_data['country'] == country]['points'].mean() > target_country_average_points
    ]
    
    return similar_point_countries[:rec_count]

def get_recommended_similar_point_countries(target_country, rules_df, wine_data, rec_count):
    recommended_countries = recommend_similar_point_countries(rules_df, target_country, wine_data, rec_count)
    return recommended_countries


# Recomendacao de Paises similhares a France
target_country = 'France'
recommended_countries = get_recommended_similar_point_countries(target_country, rules, wine_data, rec_count=3)
print(f'\nPais: {target_country}')
print(f'Países recomendados: {recommended_countries}')


def apriori_most_picked_wines_by_tasters(wine_data, target_country):
    # Filtra o dados para o pais especifico
    country_data = wine_data[wine_data['country'] == target_country]

    # Agrupa por variety e taster, e conta as occurencias de cada id da review
    wine_taster_grouped = country_data.groupby(['variety', 'taster_name'], as_index=False).agg({'id': 'count'})

    # Cria uma Pivot table
    item_count_pivot = pd.pivot_table(wine_taster_grouped, index='taster_name', columns=['variety'], values='id', fill_value=0)
    item_count_pivot = item_count_pivot.map(lambda x: 1 if x > 0 else 0)

    # Aplica Apriori algorithm
    freq_itemsets = apriori(item_count_pivot, min_support=0.3, use_colnames=True)

    # Gera regras de associação
    rules = association_rules(freq_itemsets, metric="support", min_threshold=0.5)

    # Filtra as regras de associação positiva
    positive_rules = rules[rules['lift'] > 1]

    # Calcula o support total para cada antecedente
    variety_support = positive_rules.groupby('antecedents')['support'].sum()

    sorted_variety_support = variety_support.sort_values(ascending=False)

    plt.figure(figsize=(15, 8))
    sns.lineplot(x=range(len(sorted_variety_support)), y=sorted_variety_support.values, color='skyblue')
    plt.title(f'Quandtidade da Variedade de vinhos de {target_country}')
    plt.xlabel('Variedade Index')
    plt.ylabel('Número médio')
    plt.show()

    # Identifique o vinho mais escolhido
    max_support = sorted_variety_support.max()
    most_picked_wineries = sorted_variety_support[sorted_variety_support == max_support].index.tolist()

    return most_picked_wineries

# Escolhe o vinho mais escolhido por pais (Germany neste exemplo) pelos taste reviewers 
target_country = 'Germany'
most_picked_wine = apriori_most_picked_wines_by_tasters(wine_data, target_country)
print(f"Os vinhos mais escolhidos na {target_country} por taste reviewers são: {most_picked_wine}")














