import os
import sys
import subprocess
import argparse

def run_training(task, fold, workdir, repo_path):
    """
    Lanza el entrenamiento usando el script nnunet_baseline.py del repo.
    """
    # Construimos la ruta al script interno del repo
    script_path = os.path.join(repo_path, "src", "picai_baseline", "nnunet", "nnunet_baseline.py")
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"CRÍTICO: No se encontró el script en: {script_path}")

    # Configurar PYTHONPATH para que encuentre el Custom Trainer y módulos del repo
    env = os.environ.copy()
    src_path = os.path.join(repo_path, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    # Comando equivalente a: python nnunet_baseline.py plan_train Task... /workdir ...
    # Nota: workdir aquí es la base donde nnU-Net buscará 'nnUNet_raw_data'
    cmd = [
        sys.executable, script_path,
        "plan_train",
        task,
        workdir,
        "--trainer", "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
        "--fold", str(fold)
    ]

    print(f"--- Iniciando Fold {fold} ---")
    print(f"Repo: {repo_path}")
    print(f"Task: {task}")
    print(f"Comando: {' '.join(cmd)}")
    print("-----------------------------")

    # Ejecutar
    subprocess.check_call(cmd, env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--repo_path", type=str, required=True)
    
    args = parser.parse_args()
    
    run_training(args.task, args.fold, args.workdir, args.repo_path)