import torch

from epr.recognition.unknown_detection import UnknownDetector, UnknownDetectorConfig, UnknownReason

CONFIDENT = torch.tensor([8.0, 0.0, 0.0, 0.0])
FLAT = torch.zeros(4)
TWO_WAY = torch.tensor([4.0, 4.0, -6.0, -6.0])
ACCEPT = torch.tensor(-8.0)


def test_a_confident_prediction_is_accepted():
    decision = UnknownDetector().decide(CONFIDENT, ACCEPT)
    assert not decision.is_unknown
    assert decision.reason is UnknownReason.ACCEPTED
    assert decision.predicted_index == 0
    assert decision.confidence > 0.99


def test_the_learned_unknown_head_overrides_a_confident_class():
    decision = UnknownDetector().decide(CONFIDENT, torch.tensor(5.0))
    assert decision.is_unknown
    assert decision.reason is UnknownReason.UNKNOWN_HEAD


def test_a_flat_distribution_is_low_confidence():
    decision = UnknownDetector().decide(FLAT, ACCEPT)
    assert decision.is_unknown
    assert decision.reason is UnknownReason.LOW_CONFIDENCE
    assert decision.entropy_ratio == 1.0


def test_two_classes_splitting_the_mass_are_ambiguous():
    """Confidence alone would pass this at 0.5; entropy is what catches it."""
    detector = UnknownDetector(UnknownDetectorConfig(min_confidence=0.4, max_entropy_ratio=0.4))
    decision = detector.decide(TWO_WAY, ACCEPT)
    assert decision.is_unknown
    assert decision.reason is UnknownReason.AMBIGUOUS


def test_temperature_scaling_softens_confidence():
    hot = UnknownDetector(UnknownDetectorConfig(temperature=4.0)).decide(CONFIDENT, ACCEPT)
    cold = UnknownDetector(UnknownDetectorConfig(temperature=1.0)).decide(CONFIDENT, ACCEPT)
    assert hot.confidence < cold.confidence


def test_entropy_ratio_stays_in_range():
    for logits in (CONFIDENT, FLAT, TWO_WAY):
        assert 0.0 <= UnknownDetector().decide(logits, ACCEPT).entropy_ratio <= 1.0
