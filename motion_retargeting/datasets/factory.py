from motion_retargeting.datasets.ai4animation import AI4AnimationDataset
from motion_retargeting.datasets.base import RetargetingDataset
from motion_retargeting.datasets.horse import HorseDataset
from motion_retargeting.datasets.solo8 import Solo8Dataset

_DATASET_REGISTRY = {
    "ai4animation": AI4AnimationDataset,
    "horse": HorseDataset,
    "solo8": Solo8Dataset,
}


def create_dataset(dataset_name: str) -> RetargetingDataset:
    """Create a dataset instance from a dataset name.

    Args:
        dataset_name: Name of the dataset to create. Must be one of the registered datasets.

    Returns:
        An instance of the requested dataset.

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    if dataset_name not in _DATASET_REGISTRY:
        available = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}",
        )
    return _DATASET_REGISTRY[dataset_name]()
