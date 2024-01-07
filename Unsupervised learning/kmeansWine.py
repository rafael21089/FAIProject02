import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Load do dataset winequality-red.csv
file_path = 'wine_database/winequality-red.csv'
wine = pd.read_csv(file_path)

# Remove linhas com valores nulos
df_cleaned = wine.dropna()

# Usa o Elbow Method para encontrar o numero ideal de Clusters
inertia = []
for num_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(wine)
    inertia.append(kmeans.inertia_)

# Grafico para o Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method Para Numero Ideal Clusters')
plt.xlabel('Numero of Clusters')
plt.ylabel('Inertia')

plt.show()

# O Numero ideal foi 3
clusters = 3

# Aplica k-means clustering com 3 clusters
kmeans = KMeans(n_clusters=clusters, random_state=42)
wine['cluster'] = kmeans.fit_predict(wine)

# Mostra as atribuições do cluster
print("Cluster Assignments:")
print(wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates',
         'alcohol', 'quality', 'cluster']])

# Calcule os valores médios para cada cluster
clusterMedia = wine.groupby('cluster').mean()

# Grafico que mostra os valores médios para cada cluster
clusterMedia.transpose().plot(kind='bar', figsize=(12, 6))
plt.title('Valores médios para cada cluster')
plt.xlabel('Quimicos')
plt.ylabel('Valores médios')
plt.show()


# Categorização da 'quality' para 'Low Quality,' 'Medium Quality,' e 'High Quality'
# 0-5 , é Low Quality 
# 6-7 , é Medium Quality 
# 8-10 , é High Quality 

wine['qualitylabel'] = pd.cut(wine['quality'], bins=[0, 5, 7, 10], labels=['Low Quality', 'Medium Quality', 'High Quality'])

# Grafico a mostrar a distribuição de cada Catergoria de Qualidade
qualidade = wine['qualitylabel'].value_counts().sort_index()
qualidade.plot(kind='barh', color='skyblue')
plt.title('Distribuicao de Qualidade')
plt.ylabel('Qualidade')
plt.xlabel('Numero')
plt.show()

qualidade = wine['qualitylabel'].value_counts().sort_index()
print("Distribuicao de Qualidade:")
for category, count in zip(qualidade.index, qualidade):
    print(f"{category:<15}: {count}")

# Cria um Pie Chart para a relação produtos químicos e qualidade (em percentagem)
quimicosColunas = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']

# Calcula os valores médios para cada produto químico em cada categoria de qualidade
quimicosMedia = wine.groupby('qualitylabel')[quimicosColunas].mean()

fig, axes = plt.subplots(nrows=1, ncols=len(quimicosMedia), figsize=(15, 5))

for ax, quality_category in zip(axes, quimicosMedia.index):
    wedges, texts, autotexts = ax.pie(
        quimicosMedia.loc[quality_category],
        labels=None,
        autopct='',
        startangle=90,
        colors=plt.cm.tab10.colors
    )
    ax.set_title(f'Composição Química para {quality_category}')
    ax.legend(wedges, quimicosColunas, title="Quimicos", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.axis('equal')

# Mostra uma tabela com as percentagens de cada quimico
plt.figure()
table_data = quimicosMedia.T
table_data.index.name = 'Quimicos'
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
for i, quality_category in enumerate(quimicosMedia.index):
    # Calcule a percentagem de cada produto químico
    total = quimicosMedia.loc[quality_category].sum()
    percent = (quimicosMedia.loc[quality_category] / total) * 100

    # Grafico de Barras para a importancia
    bars = plt.bar(
        x=[chem + f" ({quality_category})" for chem in quimicosMedia.columns],
        height=percent,
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
sns.scatterplot(x='pH', y='quality', data=wine, hue='qualitylabel', palette='viridis')
plt.title('Correlação entre pH e quality')
plt.xlabel('pH')
plt.ylabel('Quality')
plt.legend(title='Cluster', loc='upper right')

# Scatter plot para a correlação entre alcohol e Quality
plt.figure(figsize=(10, 6))
sns.barplot(x='quality', y='alcohol', data=wine, hue='qualitylabel', palette='viridis')
plt.title('Correlação entre alcohol e quality')
plt.xlabel('Quality')
plt.ylabel('Alcohol Content')
plt.legend(title='Cluster', loc='upper right')
plt.show()




