import os
import json
from sklearn.model_selection import StratifiedKFold

def generar_cross_validation(ruta_directorio, ruta_final):
    # Cargar lista de casos de la clase 1
    casos_clase1 = [7,9,10,11,22,23,24,25, 43, 51,53, 69, 70, 71,78,79,80,81,82,83,87,88,89,90,91,92,93,94,95,96,97,99,101,102,103,104,105,106,107,108,109,110,111,112,114,115,116,117,118,119,120,121]

    # Obtener la lista de archivos en el directorio
    casos_totales = os.listdir(ruta_directorio)

    # Obtener etiquetas para cada caso
    etiquetas = [1 if caso in casos_clase1 else 0 for caso in casos_totales]

    # Crear objeto StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Lista para almacenar los resultados
    resultado_cross_validation = []

    # Iterar sobre los folds
    for fold, (train_index, test_index) in enumerate(skf.split(casos_totales, etiquetas), 1):
        # Obtener conjuntos de entrenamiento y prueba
        train_set = [casos_totales[i] for i in train_index]
        test_set = [casos_totales[i] for i in test_index]

        # Almacenar resultados en la lista
        fold_resultado = {"train": train_set, "val": test_set}
        resultado_cross_validation.append(fold_resultado)

    # Imprimir el resultado como JSON
    with open(ruta_final, 'w') as json_file:
        json.dump(resultado_cross_validation, json_file, indent=2)
    print(len(resultado_cross_validation))

# Rutas de directorios y archivos
ruta_directorio = '/mnt/workspace/jfacuse/prostate/input/UC-DATASET/images'
ruta_final = '/home/jfacuse/piccai_challenge/picai_baseline/src/picai_baseline/splits/uc/splits.json'


# Generar cross-validation y obtener resultado como JSON
generar_cross_validation(ruta_directorio, ruta_final)