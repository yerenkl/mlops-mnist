import os.path

import pytest
import torch

from mlops_mnist.data import corrupt_mnist


@pytest.mark.skipif(not os.path.exists("./data"), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Train dataset did not have the correct number of samples"
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "Train dataset did not have all classes from 0 to 9"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "Test dataset did not have all classes from 0 to 9"