import os
import numpy as np
from SchNet4AIM.data import AtomsDataError, AtomsDataSubset

def train_test_split(
    data,
    num_train=None,
    num_val=None,
    split_file=None,
    stratify_partitions=False,
    num_per_partition=False,
):
    """
    Splits the dataset into train/validation/test splits, writes split to
    an npz file and returns subsets. Either the sizes of training and
    validation split or an existing split file with split indices have to
    be supplied. The remaining data will be used in the test dataset.

    Args:
        data (s4aim.data.AtomsData): full atomistic dataset
        num_train (int): number of training examples
        num_val (int): number of validation examples
        split_file (str): Path to split file. If file exists, splits will
                          be loaded. Otherwise, a new file will be created
                          where the generated split is stored.

    Returns:
        s4aim.data.AtomsDataSubset: subset with training data
        s4aim.data.AtomsDataSubset: subset with validation data
        s4aim.data.AtomsDataSubset: subset with test data

    """
    # Retrieve information train-test-val split information

    # From the split file (if present)
    if split_file is not None and os.path.exists(split_file):
        print(" # Reading split information from file : ", split_file) 
        S = np.load(split_file)
        # Get the identifiers of the training, validation and testing subsets.
        train_idx = S["train_idx"].tolist()
        val_idx = S["val_idx"].tolist()
        test_idx = S["test_idx"].tolist()
    # Divide the dataset on the fly if no split file was found
    else:
        print(" # Performing dataset split ")
        if num_train is None or num_val is None:
            raise ValueError(
                "You have to supply either split sizes (num_train /"
                + " num_val) or an npz file with splits."
            )

        num_train = num_train if num_train > 1 else num_train * len(data)
        num_val = num_val if num_val > 1 else num_val * len(data)
        num_train = int(num_train)
        num_val = int(num_val)
        if stratify_partitions:
            partitions = data.get_metadata("partitions")
            n_partitions = len(partitions)
            if num_per_partition:
                num_train_part = num_train
                num_val_part = num_val
            else:
                num_train_part = num_train // n_partitions
                num_val_part = num_val // n_partitions

            train_idx = []
            val_idx = []
            test_idx = []
            for start, stop in partitions.values():
                idx = np.random.permutation(np.arange(start, stop))
                train_idx += idx[:num_train_part].tolist()
                val_idx += idx[num_train_part : num_train_part + num_val_part].tolist()
                test_idx += idx[num_train_part + num_val_part :].tolist()

        else:
            idx = np.random.permutation(len(data))
            train_idx = idx[:num_train].tolist()
            val_idx = idx[num_train : num_train + num_val].tolist()
            test_idx = idx[num_train + num_val :].tolist()

        if split_file is not None:
            print(" # Split file saved to :", split_file)
            np.savez(
                split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
            )

    # Print some details about the subsets.
    print("   - train IDs (",len(train_idx),") :", train_idx[0],".....",train_idx[-1]) 
    print("   - test  IDs (",len(test_idx),") :", test_idx[0],".....",test_idx[-1])   
    print("   - val   IDs (",len(val_idx),") :", val_idx[0],".....",val_idx[-1])       
    print("   - Number of molecular datapoints :",len(train_idx)+len(test_idx)+len(val_idx))
    print("   - Dataset size                   :",len(data))
    train = create_subset(data, train_idx)
    val = create_subset(data, val_idx)
    test = create_subset(data, test_idx)


    # Check the consistency of the dataset split:
    if (len(train) + len(val)) > len(data):      
       raise AssertionError("Total number of selected train + validation points exceeds the number of molecules in the database")
    return train, val, test


def create_subset(dataset, indices):
    r"""
    Create a subset of atomistic datasets.

    Args:
        dataset (torch.utils.data.Dataset): dataset
        indices (sequence): indices of the subset; no np.ndarrays, because the ase database can not handle np.int values

    Returns:
        s4aim.data.AtomsDataSubset: subset of input dataset

    """
    max_id = 0 if len(indices) == 0 else max(indices)
    if len(dataset) <= max_id:
        raise AtomsDataError(
            "The subset indices do not match the total length of the dataset!"
        )
    return AtomsDataSubset(dataset, indices)
