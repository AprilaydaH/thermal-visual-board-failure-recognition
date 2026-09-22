import numpy as np
import pytest

from epr.recognition.ocr import OCR_FEATURE_DIM, MarkingReading, encode_marking

CONFIDENCE_INDEX = OCR_FEATURE_DIM - 2
PRESENT_INDEX = OCR_FEATURE_DIM - 1


def test_no_reading_is_all_zeros():
    for empty in (None, "", MarkingReading(), MarkingReading(text="   ")):
        np.testing.assert_array_equal(encode_marking(empty), np.zeros(OCR_FEATURE_DIM))


def test_reading_sets_the_present_flag_and_confidence():
    features = encode_marking(MarkingReading("LM317T", confidence=0.8))
    assert features[PRESENT_INDEX] == 1.0
    assert features[CONFIDENCE_INDEX] == np.float32(0.8)


def test_encoding_is_deterministic():
    first = encode_marking("STM32N657")
    second = encode_marking("STM32N657")
    np.testing.assert_array_equal(first, second)


def test_hash_block_is_unit_length():
    features = encode_marking("NE555")
    assert np.linalg.norm(features[:CONFIDENCE_INDEX]) == pytest.approx(1.0)


def test_separators_and_case_are_ignored():
    np.testing.assert_allclose(encode_marking("LM-317 T"), encode_marking("lm317t"))


def test_different_markings_differ():
    assert not np.allclose(encode_marking("LM317"), encode_marking("NE555"))


def test_confidence_is_clamped():
    assert encode_marking(MarkingReading("A", confidence=5.0))[CONFIDENCE_INDEX] == 1.0
