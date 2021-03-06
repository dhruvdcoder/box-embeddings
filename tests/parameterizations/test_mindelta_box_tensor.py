from box_embeddings.parameterizations import BoxTensor, MinDeltaBoxTensor
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
    tensor = torch.tensor(np.random.rand(3, 2, 3))
    box_tensor = MinDeltaBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(2, 10))
    box_tensor = MinDeltaBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()  # type: ignore
    assert isinstance(box_tensor, BoxTensor)


def test_shape_validation_during_creation():
    tensor = torch.tensor(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = MinDeltaBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = MinDeltaBoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = MinDeltaBoxTensor(tensor)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    minimum_delta = 1.0
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape)) + minimum_delta
    box = MinDeltaBoxTensor.from_zZ(z, Z, minimum_delta=minimum_delta)
    assert box.z.shape == (3, 1, 5)
    assert torch.allclose(box.z, z)
    assert torch.allclose(box.Z, Z)
    assert ((box.Z - box.z) > minimum_delta).all()


def test_round_trip():
    shape = (3, 1, 5)
    minimum_delta = 1.0
    z = torch.tensor(np.random.rand(*shape))
    Z = z + torch.tensor(np.random.rand(*shape)) + minimum_delta
    W = MinDeltaBoxTensor.W(z, Z, minimum_delta=minimum_delta)
    box = MinDeltaBoxTensor(W, minimum_delta=minimum_delta)
    assert box.z.shape == (3, 1, 5)
    assert torch.allclose(box.z, z)
    assert torch.allclose(box.Z, Z)
    assert ((box.Z - box.z) > minimum_delta).all()


@hypothesis.given(
    beta=floats(1.0, 50.0),
    threshold=integers(20, 50),
)
def test_creation_from_vector(beta, threshold):
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    w_delta = torch.tensor(np.random.rand(*shape))
    v = torch.cat((z, w_delta), dim=-1)
    minimum_delta = 1.0
    box = MinDeltaBoxTensor.from_vector(
        v, beta=beta, threshold=threshold, minimum_delta=minimum_delta
    )
    assert box.Z.shape == (3, 1, 5)
    assert torch.allclose(box.z, z)
    assert torch.allclose(
        box.Z,
        z
        + minimum_delta
        + torch.nn.functional.softplus(
            w_delta, beta=beta, threshold=threshold
        ),
    )
    assert ((box.Z - box.z) > minimum_delta).all()


@hypothesis.given(
    beta=floats(1.0, 50.0),
    threshold=integers(20, 50),
    delta=floats(1e-13, 1.0),
)
def test_creation_from_center(beta, threshold, delta):
    shape = (3, 5)
    c = torch.tensor(np.random.rand(*shape))
    box = MinDeltaBoxTensor.from_center_vector(
        c, delta=delta, beta=beta, threshold=threshold
    )
    assert torch.allclose(box.Z, c + delta / 2.0)
    assert torch.allclose(box.z, c - delta / 2.0)


@hypothesis.given(
    sample=sampled_from(
        [
            ((-1, 10), (5, 2, 10), (5, 10), (5, 10)),
            ((-1, 10), (5, 4, 2, 10), (5, 4, 10), (20, 10)),
            ((10, 2, 10), (20, 2, 10), (20, 10), (10, 2, 10)),
            ((-1, 10), (2, 5), (5,), RuntimeError),
            ((2, 10), (5, 2, 10), (5, 10), RuntimeError),
        ]
    )
)
def test_reshape(sample):
    target_shape, input_data_shape, self_shape, expected = sample
    box = BoxTensor(torch.tensor(np.random.rand(*input_data_shape)))
    assert box.box_shape == self_shape

    if expected == RuntimeError:
        with pytest.raises(expected):
            box.box_reshape(target_shape)
    else:
        new = box.box_reshape(target_shape)
        assert new.box_shape == expected


# def test_warning_in_creation_from_zZ():
#    shape = (3, 1, 5)
#    z = torch.tensor(np.random.rand(*shape))
#    Z = z + torch.tensor(np.random.rand(*shape))
#    with pytest.warns(UserWarning):
#        box = MinDeltaBoxTensor.from_zZ(z, Z)
