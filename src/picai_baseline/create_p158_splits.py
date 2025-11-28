import os
import json
from sklearn.model_selection import StratifiedKFold
import SimpleITK as sitk
import numpy as np

def get_cspca_cases(ruta):
    #Ruta es la ruta de los labels
    casos_clase_1 = []
    for case in os.listdir(ruta): 
        if 'prostate158' in case:
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(ruta + os.path.sep + case))
            clase = float(np.max(lbl))
            if clase > 0:
                casos_clase_1.append(case[:15])
            print('Label caso ', case, clase)
    return casos_clase_1



def generar_cross_validation(ruta_directorio, ruta_final, casos_clase1):
    # Cargar lista de casos de la clase 1
    # Obtener la lista de archivos en el directorio
    casos_totales = list(filter(lambda x:'prostate158' in x, os.listdir(ruta_directorio)))
    print(casos_totales)
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
    print(resultado_cross_validation)

# Rutas de directorios y archivos
ruta_directorio = '/mnt/workspace/jfacuse/prostate/input/images'
ruta_final = '/home/jfacuse/piccai_challenge/picai_baseline/src/picai_baseline/splits/prostate158/splits.json'


# Generar cross-validation y obtener resultado como JSON
#generar_cross_validation(ruta_directorio, ruta_final)
casos = get_cspca_cases('/mnt/workspace/jfacuse/prostate/input/picai_labels/csPCa_lesion_delineations/human_expert/resampled')
print(casos)
generar_cross_validation(ruta_directorio, ruta_final, casos)