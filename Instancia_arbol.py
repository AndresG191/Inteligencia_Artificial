import random
import math
import numpy as np
import pandas as pd

file_Made = True #archivo creado variable

velocidad_Internet= ["Alta", "Media", "Baja"]
velocidad_Descarga= ["Poco", "Mucho", "Nada"]
calidad_Servicio = ["Satisfecho", "Neutral", "Insatisfecho"]
pagarx_mas_Velocidad= ["Si", "No"]


datos = [
    {
        "velocidad_Internet": random.choice(velocidad_Internet),
        "velocidad_Descarga": random.choice(velocidad_Descarga),
        "calidad_Servicio": random.choice(calidad_Servicio),
        "pagarx_mas_Velocidad": random.choice(pagarx_mas_Velocidad)

    }
    for i in range(1000)
]

df= pd.DataFrame(datos)

def divide_Dataset(df, test_size=0.3):
    indice = df.index.tolist()
    random.shuffle(indice) #mezcla el dataset
    tam_prueba = int(len(df)* test_size)
    entrenamiento = df.iloc[indice[tam_prueba:]] #Tamaño divisible por 70%
    prueba = df.iloc[indice[:tam_prueba]] #Tamaño divisible 30 %
    return entrenamiento, prueba

#Funcion de la  entropía
def entropia_Arbol(y):
    valor_conteo = y.value_counts()
    total = len(y)
    entropia = 0.0
    for contar in valor_conteo:
        prob = contar / total
        entropia -= prob * np.log2(prob)
    return entropia if not math.isnan(entropia) else 0.0


#Metodo de obtencion de ganancia
def ganancia(entrenamiento, columna):
    valores = entrenamiento[columna].unique()
    tabla = []
    for valor in valores:
        subset = entrenamiento[entrenamiento[columna] == valor]
        si = len(subset[subset['pagarx_mas_Velocidad'] == 'Si'])
        no = len(subset[subset['pagarx_mas_Velocidad'] == 'No'])
        casos = len(subset)
        entropia_atributo = entropia_Arbol(subset['pagarx_mas_Velocidad'])
        tabla.append([valor, si, no, casos, entropia_atributo])

    df_tabla = pd.DataFrame(tabla, columns=['Atributo', 'Exito', 'Fracasos', 'Casos', 'Entropía'])
    return df_tabla

entrenamiento, prueba = divide_Dataset(df)

for columna in ['velocidad_Internet', 'velocidad_Descarga', 'calidad_Servicio', 'pagarx_mas_Velocidad']:
    tabla_resultados = ganancia(entrenamiento, columna)
    print(f"Variable: {columna}")
    print(tabla_resultados)
    print("-" * 50)

#Guarda las divisiones del dataset en archivo .csv
entrenamiento.to_csv('velocidad_Internet_70.csv', index = False)
prueba.to_csv('velocidad_Internet_30.csv', index = False)

print("Ya has creado este archivo" if file_Made else "Aun no haces este archivo")
print("Archivos CSV creados con éxito.")