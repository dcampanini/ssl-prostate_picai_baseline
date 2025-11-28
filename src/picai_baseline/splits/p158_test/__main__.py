from picai_baseline.splits import export_splits
from picai_baseline.splits.p158_test import nnunet_splits

if __name__ == "__main__":
    export_splits(
        nnunet_splits=nnunet_splits,
    )
