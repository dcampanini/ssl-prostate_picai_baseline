from picai_baseline.splits import export_splits
from picai_baseline.splits.uc_test import nnunet_splits

if __name__ == "__main__":
    export_splits(
        nnunet_splits=nnunet_splits,
    )
