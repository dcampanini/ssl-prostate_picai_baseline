import os 

path = '/mnt/workspace/jfacuse/prostate/input/UC-DATASET/images'

for folder in os.listdir(path):
    os.rename(os.path.join(path, folder), os.path.join(path, 'UC-'+folder))