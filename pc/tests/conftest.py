import importlib.util

# The acquisition stack installs without the recognition requirements, so the model tests are
# skipped rather than failed when PyTorch is absent.
if importlib.util.find_spec("torch") is None:
    collect_ignore_glob = [
        "*/test_cnn.py",
        "*/test_lnn.py",
        "*/test_fusion_features.py",
        "*/test_unknown_detection.py",
        "*/test_recognition_pipeline.py",
        "*/test_training.py",
        "*/test_onnx_export.py",
        "*/test_cnn_pretraining.py",
        "*/test_detector.py",
    ]
