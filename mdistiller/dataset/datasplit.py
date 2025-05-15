from torch.utils.data import Dataset, Subset


class DatasetTrainValSplit(Dataset):
    """
    PyTorch Dataset wrapper to split any dataset with `data` and `targets` attributes into
    train/validation subsets by directly modifying the dataset's internal arrays,
    ensuring each class has equal representation based on fraction k, without randomization.

    Args:
        dataset (Dataset): A PyTorch Dataset instance with .data and .targets attributes.
        val_ratio (float): Fraction of each class to use as training data (0 < k < 1).
        train (bool): If True, dataset is split into the training portion; else validation.
    """

    def __init__(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        train: bool = True,
    ):
        assert hasattr(dataset, "data") and hasattr(
            dataset, "targets"
        ), "Dataset must have 'data' and 'targets' attributes for in-place split"
        assert 0.0 <= train_ratio <= 1.0, "k must be a float between 0 and 1"

        self.dataset = dataset
        self.train_ratio = train_ratio
        self._is_trainset = train

        # Original data and targets
        data = dataset.data
        targets = dataset.targets
        # Determine number of classes
        num_classes = len(set(targets))

        train_idx, val_idx = [], []
        # Deterministic class-wise split
        for c in range(num_classes):
            # Collect indices of current class
            cls_indices = [i for i, t in enumerate(targets) if t == c]
            split_pt = int(train_ratio * len(cls_indices))
            train_idx.extend(cls_indices[:split_pt])
            val_idx.extend(cls_indices[split_pt:])

        # Choose indices based on train flag
        chosen = train_idx if train else val_idx

        dataset.data = data[chosen]
        dataset.targets = [targets[i] for i in chosen]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
