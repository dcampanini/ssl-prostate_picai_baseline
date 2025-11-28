import json
from pathlib import Path

valid_splits = {}
for fold in range(5):
    with open(Path(__file__).parent / f"ds-config-valid-fold-{fold}.json") as fp:
        ds_config = json.load(fp)
        valid_splits[fold] = ds_config
# read dataset configurations
with open(Path(__file__).parent / "splits.json") as fp:
    nnunet_splits = json.load(fp)

# expose dataset configurations
__all__ = [
    "nnunet_splits",
    "valid_splits"
]