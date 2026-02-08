import os
import sys
import subprocess
import argparse
import re

def run_training(task_full_name, fold, workdir, repo_path, trainer_name, continue_training):
    """
    Script conductor flexible corregido para soportar Resume.
    """
    
    # 1. Configurar PYTHONPATH
    env = os.environ.copy()
    src_path = os.path.join(repo_path, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    
    # 2. Extraer ID de la tarea
    match = re.search(r"Task(\d+)_", task_full_name)
    if match:
        task_id = match.group(1)
    else:
        task_id = task_full_name

    print("--------------------------------------------------")
    print(f"Iniciando flujo manual para Fold {fold}")
    print(f"Task ID: {task_id}")
    print(f"Trainer usado: {trainer_name}")
    print(f"Modo Resume: {'ACTIVADO' if continue_training else 'DESACTIVADO'}")
    print("--------------------------------------------------")

    # 3. Preprocesamiento (SOLO SI NO ESTAMOS RESUMIENDO)
    # Si estamos resumiendo, asumimos que los datos ya existen.
    if not continue_training:
        cmd_plan = [
            "nnUNet_plan_and_preprocess",
            "-t", task_id,
            "--verify_dataset_integrity"
        ]
        
        print(f"EJECUTANDO PLANIFICACIÓN...")
        try:
            subprocess.check_call(cmd_plan, env=env)
        except subprocess.CalledProcessError:
            print(">>> AVISO: El preprocesamiento retornó error o ya existía.")
    else:
        print(">>> SALTANDO PREPROCESAMIENTO (Modo Resume activo)")

    # 4. Entrenamiento
    cmd_train = [
        "nnUNet_train",
        "3d_fullres",
        trainer_name,
        task_id,
        str(fold)
    ]

    # --- AQUÍ ESTÁ EL TRUCO ---
    # Si recibimos la bandera del bash, le pasamos la bandera '-c' a nnU-Net
    if continue_training:
        cmd_train.append("-c")

    print(f"EJECUTANDO ENTRENAMIENTO: {' '.join(cmd_train)}")
    
    # Ejecutamos y permitimos que el error suba si falla
    subprocess.check_call(cmd_train, env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--repo_path", type=str, required=True)
    parser.add_argument("--trainer", type=str, required=True)
    
    # NUEVO ARGUMENTO: store_true significa que si pones la bandera es True, si no, False
    parser.add_argument("--continue_training", action="store_true", help="Si se activa, añade -c a nnU-Net y salta el preprocesamiento")
    
    args = parser.parse_args()
    
    run_training(
        args.task, 
        args.fold, 
        args.workdir, 
        args.repo_path, 
        args.trainer,
        args.continue_training # Pasamos el nuevo argumento
    )