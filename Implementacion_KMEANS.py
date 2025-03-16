import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.express as scatter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer
import warnings

warnings.filterwarnings("ignore")

#Configuración para mostrar gráficos en el navegador.
pio.renderers.default = 'browser'

df = pd.read_csv("numeros_aleatorios.csv")
print(df)

#Normalización de los datos
X = df.copy()
X1 = preprocessing.normalize(X)

#Visualización del codo para elegir el número de clusters
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 10))
visualizer.fit(X1)  # Entrenamos los datos
visualizer.show()  # Renderizamos la imagen del codo

#Algoritmo KMeans para crear los clusters
algorithm = KMeans(n_clusters=5, init='k-means++', n_init=10,
                   max_iter=300, tol=0.0001, random_state=111)
algorithm.fit(X1)
labels = algorithm.labels_
centroids = algorithm.cluster_centers_

#Añadir las etiquetas al DataFrame
df['label'] = labels

#Obtiene el grafico en 2d
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X1)

#Asignar las componentes principales a las columnas X1, X2
df['X1'] = pca_result[:, 0]
df['X2'] = pca_result[:, 1]

#Se creará un dataframe para los centroides en 2d
centroids_pca = pca.transform(centroids)

#hace el gráfico 2d con Plotly
fig = px.scatter(df, x='X1', y='X2', color='label')

#Agrega centroides al grafico como puntos rojos
fig.add_scatter(x= centroids_pca[:,0], y=centroids_pca[:,1],
                mode ='markers', marker=dict(color='red', size=8), name='Centroides')
fig.show()

fig.write_html("file.html")

#Paso para calcular las distancias de cada INSTANCIA y los centroides

#Matriz de distancias
distancias = np.zeros((X1.shape[0], centroids.shape[0]))

for i in range(X1.shape[0]):
    for j in range(centroids.shape[0]):
        distancias[i, j] = np.linalg.norm(X1[i] - centroids[j])
print("La matriz de distancias entre las instancias y los centroides son las siguientes: ")
print(distancias)