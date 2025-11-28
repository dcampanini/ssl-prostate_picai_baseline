import json
from pathlib import Path


with open(Path(__file__).parent / "splits.json") as fp:
    nnunet_splits = json.load(fp)

# expose dataset configurations
__all__ = [
    "nnunet_splits"
]