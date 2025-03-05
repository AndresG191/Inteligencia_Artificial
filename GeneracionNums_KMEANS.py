import random
import csv

# Función para generar números aleatorios
def generate_random_numbers():
    return [random.randint(0, 10) for _ in range(4)]

# Crear el archivo CSV
with open('numeros_aleatorios.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'X1', 'X2', 'X3', 'X4', 'R'])

    # Generar 1000 iteraciones de números aleatorios
    for i in range(1, 1001):
        random_numbers = generate_random_numbers()
        random_r = random.randint(1, 3)  # Generar el número aleatorio para la columna 'R'
        writer.writerow([i] + random_numbers + [random_r])


print("Archivo CSV creado con éxito.")
