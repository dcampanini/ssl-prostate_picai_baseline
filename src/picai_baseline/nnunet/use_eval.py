#%%
from picai_baseline.nnunet.eval import evaluate

dataset = "uc"
for f in [3,4]:
    if dataset == "p158":
        evaluate(
            task=f"Task2403_p158_prostate_nnunet",  # Task2403_p158_prostate_nnunet, Task2303_uc_prostate_nnunet
            trainer="nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
            workdir="/workspace1/project_jxfdv/ssl_prostate_data", # root path where the prediccions folder and results folder are located
            checkpoints=["model_best"], # ["model_best", "model_ep_200"]
            folds=[f], # [0,1,2,3,4]
            splits="p158",
            predictions_folder=f"output_p158_fold{f}_model_best", # predictions_p158_f3_2403_nnunet, output_p158_fold0, detection_maps_f0_p158
            threshold="dynamic",
        )
    elif dataset == "uc":
        evaluate(
            task="Task2303_uc_prostate_nnunet",  # Task2403_p158_prostate_nnunet, Task2303_uc_prostate_nnunet
            trainer="nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
            workdir="/workspace1/project_jxfdv/ssl_prostate_data", # root path where the prediccions folde and results folder is located
            checkpoints=["model_best"],
            folds=[f], # [0,1,2,3,4]
            splits="uc",
            # folder where the predicctions are stored
            predictions_folder=f"output_uc_fold{f}_model_best", # predictions_p158_f3_2403_nnunet, output_p158_fold0, detection_maps_f0_p158
            threshold="dynamic",
        )
