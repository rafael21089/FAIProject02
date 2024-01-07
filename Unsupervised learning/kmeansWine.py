import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Load do dataset winequality-red.csv
file_path = 'wine_database/winequality-red.csv'
df = pd.read_csv(file_path)

# Remove linhas com valores nulos
df_cleaned = df.dropna()

# Usa o Elbow Method para encontrar o numero ideal de Clusters
inertia_values = []
possible_clusters = range(1, 11)  
for num_clusters in possible_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(df)
    inertia_values.append(kmeans.inertia_)

# Grafico para o Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(possible_clusters, inertia_values, marker='o', linestyle='-', color='b')
plt.title('Elbow Method Para Numero Ideal Clusters')
plt.xlabel('Numero of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# O Numero ideal foi 3
optimal_clusters = 3

# Aplica k-means clustering com 3 clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df)

# Mostra as atribuições do cluster
print("Cluster Assignments:")
print(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates',
         'alcohol', 'quality', 'cluster']])

# Calcule os valores médios para cada cluster
cluster_means = df.groupby('cluster').mean()

# Grafico que mostra os valores médios para cada cluster
cluster_means.transpose().plot(kind='bar', figsize=(12, 6))
plt.title('Valores médios para cada cluster')
plt.xlabel('Quimicos')
plt.ylabel('Valores médios')
plt.show()


# Categorização da 'quality' para 'Low Quality,' 'Medium Quality,' e 'High Quality'
# 0-5 , é Low Quality 
# 6-7 , é Medium Quality 
# 8-10 , é High Quality 

df['quality_category'] = pd.cut(df['quality'], bins=[0, 5, 7, 10], labels=['Low Quality', 'Medium Quality', 'High Quality'])

# Grafico a mostrar a distribuição de cada Catergoria de Qualidade
quality_distribution = df['quality_category'].value_counts().sort_index()
quality_distribution.plot(kind='barh', color='skyblue')
plt.title('Distribuicao de Qualidade')
plt.ylabel('Qualidade')
plt.xlabel('Numero')
plt.show()

quality_distribution = df['quality_category'].value_counts().sort_index()
print("Distribuicao de Qualidade:")
for category, count in zip(quality_distribution.index, quality_distribution):
    print(f"{category:<15}: {count}")

# Cria um Pie Chart para a relação produtos químicos e qualidade (em percentagem)
chemical_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']

# Calcula os valores médios para cada produto químico em cada categoria de qualidade
chemical_means = df.groupby('quality_category')[chemical_columns].mean()

fig, axes = plt.subplots(nrows=1, ncols=len(chemical_means), figsize=(15, 5))

for ax, quality_category in zip(axes, chemical_means.index):
    wedges, texts, autotexts = ax.pie(
        chemical_means.loc[quality_category],
        labels=None,
        autopct='',
        startangle=90,
        colors=plt.cm.tab10.colors
    )
    ax.set_title(f'Composição Química para {quality_category}')
    ax.legend(wedges, chemical_columns, title="Chemicals", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.axis('equal')

# Mostra uma tabela com as percentagens de cada quimico
plt.figure()
table_data = chemical_means.T
table_data.index.name = 'Chemicals'
table_data.reset_index(inplace=True)
table = plt.table(cellText=table_data.values,
                  colLabels=table_data.columns,
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.show()

plt.figure(figsize=(15, 8))

#Grafico que mostra a importancia de cada quimico por cada Categoria de Qualidade
for i, quality_category in enumerate(chemical_means.index):
    # Calcule a percentagem de cada produto químico
    total = chemical_means.loc[quality_category].sum()
    percentages = (chemical_means.loc[quality_category] / total) * 100

    # Grafico de Barras para a importancia
    bars = plt.bar(
        x=[chem + f" ({quality_category})" for chem in chemical_means.columns],
        height=percentages,
        color=plt.cm.tab10.colors[i],
        label=quality_category
    )

    # Adiciona as percentagens acima das Barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

plt.title('Importancia de cada quimico por cada Categoria de Qualidade')
plt.xlabel('Quimico')
plt.ylabel('Percentagem')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Quality Category', loc='upper right', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()


# Scatter plot para a correlação entre pH e Quality
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pH', y='quality', data=df, hue='quality_category', palette='viridis')
plt.title('Correlação entre pH e quality')
plt.xlabel('pH')
plt.ylabel('Quality')
plt.legend(title='Cluster', loc='upper right')

# Scatter plot para a correlação entre alcohol e Quality
plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='alcohol', data=df, hue='quality_category', palette='viridis')
plt.title('Correlação entre alcohol e quality')
plt.xlabel('Quality')
plt.ylabel('Alcohol Content')
plt.legend(title='Cluster', loc='upper right')
plt.show()




