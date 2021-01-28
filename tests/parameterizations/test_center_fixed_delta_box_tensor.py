from box_embeddings.parameterizations import (
    BoxTensor,
    CenterFixedDeltaBoxTensor,
)
import torch
import numpy as np
import pytest
import warnings
from hypothesis.extra.numpy import arrays
import hypothesis
from hypothesis.strategies import (
    floats,
    integers,
    sampled_from,
    fixed_dictionaries,
    just,
)


def test_simple_creation() -> None:
    tensor = torch.tensor(np.random.rand(3, 3))
    delta = 0.1
    box_tensor = CenterFixedDeltaBoxTensor(tensor, delta=0.1)
    assert torch.allclose(tensor + delta / 2.0, box_tensor.Z)
    assert torch.allclose(tensor - delta / 2.0, box_tensor.z)
    assert torch.allclose(tensor, box_tensor.centre)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape))
    box2 = CenterFixedDeltaBoxTensor.from_zZ(z, z + 0.1)


@hypothesis.given(
    delta=floats(1.0, 50.0),
)
def test_creation_from_vector(delta):
    shape = (3, 1, 5)
    c = torch.tensor(np.random.rand(*shape))
    box = CenterFixedDeltaBoxTensor.from_vector(c, delta=delta)
    assert box.Z.shape == (3, 1, 5)
    assert torch.allclose(box.z, c - delta / 2.0)
    assert torch.allclose(box.Z, c + delta / 2.0)


@hypothesis.given(
    sample=sampled_from(
        [
            ((-1, 10), (5, 10), (5, 10), (5, 10)),
            ((-1, 10), (5, 4, 10), (5, 4, 10), (20, 10)),
            ((10, 2, 10), (20, 10), (20, 10), (10, 2, 10)),
            ((-1, 10), (5,), (5,), RuntimeError),
            ((2, 10), (5, 10), (5, 10), RuntimeError),
        ]
    )
)
def test_reshape(sample):
    delta = 0.5
    target_shape, input_data_shape, self_shape, expected = sample
    box = CenterFixedDeltaBoxTensor(
        torch.tensor(np.random.rand(*input_data_shape)), delta=delta
    )
    assert box.box_shape == self_shape

    if expected == RuntimeError:
        with pytest.raises(expected):
            box.box_reshape(target_shape)
    else:
        new = box.box_reshape(target_shape)
        assert new.box_shape == expected


@hypothesis.given(
    sample=sampled_from(
        [
            ((4, 5, 10), (10,), (10,), (1, 1, 10)),
            ((4, 5, 10), (3,), (3,), ValueError),
            (
                (4, 5, 10),
                (4, 2, 3),
                (
                    4,
                    2,
                    3,
                ),
                ValueError,
            ),
            (
                (4, 5, 10),
                (4, 10),
                (
                    4,
                    10,
                ),
                (4, 1, 10),
            ),
            (
                (4, 5, 10),
                (5, 10),
                (
                    5,
                    10,
                ),
                (1, 5, 10),
            ),
            (
                (4, 5, 10),
                (4, 2, 2, 3),
                (
                    4,
                    2,
                    2,
                    3,
                ),
                ValueError,
            ),
            ((1, 5, 10), (5, 1, 10), (5, 1, 10), (5, 1, 10)),
            ((5, 1, 10), (1, 5, 10), (1, 5, 10), (1, 5, 10)),
            ((5, 1, 10), (5, 5, 10), (5, 5, 10), (5, 5, 10)),
            ((5, 5, 10), (5, 5, 10), (5, 5, 10), (5, 5, 10)),
        ]
    )
)
def test_broadcasting(sample):
    target_shape, input_data_shape, self_shape, expected = sample
    box = CenterFixedDeltaBoxTensor(
        torch.tensor(np.random.rand(*input_data_shape)), delta=0.5
    )
    assert box.box_shape == self_shape

    if isinstance(expected, tuple):
        box.broadcast(target_shape)
        assert box.box_shape == expected
    else:
        with pytest.raises(expected):
            box.broadcast(target_shape)


# def test_warning_in_creation_from_zZ():
#    shape = (3, 1, 5)
#    z = torch.tensor(np.random.rand(*shape))
#    Z = z + torch.tensor(np.random.rand(*shape))
#    with pytest.warns(UserWarning):
#
