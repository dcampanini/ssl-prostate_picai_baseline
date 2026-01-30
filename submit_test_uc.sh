#!/bin/bash
#SBATCH --job-name nnunet_uc_test_f0          # Nombre descriptivo
#SBATCH -t 3-00:00:00                         # 3 días (seguro para nnU-Net)
#SBATCH --cpus-per-task 16                    # 16 CPUs como usas habitualmente
#SBATCH --mem=64G                             # 64GB RAM
#SBATCH --gpus=1                              # 1 GPU

# Cargar entorno
module load conda
conda activate /home/aetamayo/miniconda3/envs/env_sslprostate

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Ruta donde está tu clon del repo picai_baseline
REPO_PATH="/home/aetamayo/ssl-prostate_picai_baseline"

# Ruta BASE de los datos (padre de nnUNet_raw_data)
DATA_WORKDIR="/workspace1/project_jxfdv/ssl_prostate_data"

# Ruta donde quieres que se guarden los modelos y preprocesamiento
OUTPUT_DIR="/home/aetamayo/aetamayo_outputs"

# Nombre exacto de la Task para UC
TASK_NAME="Task2303_uc_prostate_nnunet"

# Fold a probar (0)
FOLD_ID=0

echo "Iniciando prueba UC (Fold $FOLD_ID)"
echo "Repo: $REPO_PATH"
echo "Outputs en: $OUTPUT_DIR"

# Crear carpetas de salida si no existen (importante para evitar errores)
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/nnUNet_preprocessed"
mkdir -p "$OUTPUT_DIR/nnUNet_results"

# --- 2. VARIABLES DE ENTORNO CRÍTICAS DE NNUNET ---
# A. Donde busca los datos crudos
export nnUNet_raw_data_base="$DATA_WORKDIR"

# B. Donde guarda los datos preprocesados (escritura intensiva)
export nnUNet_preprocessed="$OUTPUT_DIR/nnUNet_preprocessed"

# C. Donde guarda los pesos finales del modelo
export RESULTS_FOLDER="$OUTPUT_DIR/nnUNet_results"

# --- 3. EJECUCIÓN ---
# Llamamos al script python conductor que creamos antes.
# Nota: No usamos '&' ni 'wait' porque es solo un fold de prueba.

python run_nnunet.py \
    --task "$TASK_NAME" \
    --fold $FOLD_ID \
    --workdir "$DATA_WORKDIR" \
    --repo_path "$REPO_PATH"

echo "Job finalizado."