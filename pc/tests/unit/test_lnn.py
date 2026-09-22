import pytest
import torch

from epr.recognition.lnn import CfcCell, CfcConfig, LiquidNetwork

BATCH, STEPS, FEATURES, HIDDEN = 3, 6, 12, 16


@pytest.fixture
def config():
    return CfcConfig(
        input_size=FEATURES, hidden_size=HIDDEN, backbone_units=24, backbone_layers=2, dropout=0.0
    )


@pytest.fixture
def network(config):
    torch.manual_seed(0)
    return LiquidNetwork(config).eval()


@pytest.fixture
def sequence():
    torch.manual_seed(1)
    return torch.randn(BATCH, STEPS, FEATURES)


def test_output_shapes(network, sequence):
    output, states = network(sequence, torch.ones(BATCH, STEPS))
    assert output.shape == (BATCH, STEPS, HIDDEN)
    assert len(states) == 1
    assert states[0].shape == (BATCH, HIDDEN)


def test_elapsed_time_changes_the_state(network, sequence):
    """The point of a liquid network: the same inputs at a different rate behave differently."""
    fast, _ = network(sequence, torch.full((BATCH, STEPS), 0.1))
    slow, _ = network(sequence, torch.full((BATCH, STEPS), 30.0))
    assert not torch.allclose(fast, slow, atol=1e-4)


def test_state_can_be_carried_between_calls(network, sequence):
    """A live inspection feeds frame sets as they arrive, not as one block."""
    whole, _ = network(sequence, torch.ones(BATCH, STEPS))

    first, states = network(sequence[:, :2], torch.ones(BATCH, 2))
    second, _ = network(sequence[:, 2:], torch.ones(BATCH, STEPS - 2), states)

    torch.testing.assert_close(torch.cat([first, second], dim=1), whole)


def test_stacked_layers_return_one_state_each(config):
    network = LiquidNetwork(
        CfcConfig(
            input_size=FEATURES,
            hidden_size=HIDDEN,
            backbone_units=24,
            backbone_layers=1,
            layers=3,
            dropout=0.0,
        )
    ).eval()
    output, states = network(torch.randn(BATCH, STEPS, FEATURES), torch.ones(BATCH, STEPS))
    assert output.shape == (BATCH, STEPS, HIDDEN)
    assert len(states) == 3


def test_cell_step_shape(config):
    cell = CfcCell(config)
    hidden = cell.initial_hidden(BATCH)
    assert cell(torch.randn(BATCH, FEATURES), hidden, torch.ones(BATCH)).shape == (BATCH, HIDDEN)


def test_gradients_reach_the_backbone(network, sequence):
    output, _ = network(sequence, torch.ones(BATCH, STEPS))
    output.sum().backward()
    assert network.cells[0].time_scale.weight.grad is not None
    assert network.cells[0].time_scale.weight.grad.abs().sum() > 0


def test_wrong_shapes_are_rejected(network, sequence):
    with pytest.raises(ValueError, match="batch, time, features"):
        network(torch.randn(BATCH, FEATURES), torch.ones(BATCH, STEPS))
    with pytest.raises(ValueError, match="dt must be"):
        network(sequence, torch.ones(BATCH, STEPS + 1))


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="backbone_layers"):
        CfcConfig(input_size=FEATURES, backbone_layers=0)
    with pytest.raises(ValueError, match="layers"):
        CfcConfig(input_size=FEATURES, layers=0)
