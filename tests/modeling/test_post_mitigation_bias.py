import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


def _load_assign():
    """Load the helper without modeling/__init__.py, which imports H2O."""
    inserted: list[str] = []
    modeling_name = "edvise.modeling"
    if modeling_name not in sys.modules:
        modeling = types.ModuleType(modeling_name)
        modeling.__path__ = [
            str(Path(__file__).resolve().parents[2] / "src" / "edvise" / "modeling")
        ]
        modeling.__package__ = modeling_name
        sys.modules[modeling_name] = modeling
        inserted.append(modeling_name)
    bias_name = "edvise.modeling.bias_detection"
    if bias_name not in sys.modules:
        sys.modules[bias_name] = MagicMock()
        inserted.append(bias_name)
    module_name = "edvise.modeling.post_mitigation_bias"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[2]
        / "src"
        / "edvise"
        / "modeling"
        / "post_mitigation_bias.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module.assign_dual_threshold_predictions
    finally:
        sys.modules.pop(module_name, None)
        for name in inserted:
            sys.modules.pop(name, None)


assign_dual_threshold_predictions = _load_assign()


def test_dual_threshold_flags_only_the_target_group_between_thresholds():
    scored = pd.DataFrame(
        {
            "age_bias_group": ["18-24", "18-24", "25+", "25+"],
            "gender": ["Female", "Male", "Female", "Male"],
            "pred_prob": [0.4, 0.2, 0.4, 0.6],
        }
    )

    labeled = assign_dual_threshold_predictions(
        scored,
        group_col="age_bias_group",
        target_group="18-24",
        group_threshold=0.3,
        default_threshold=0.5,
        pred_col="pred",
        pred_prob_col="pred_prob",
    )

    assert labeled["pred"].tolist() == [True, False, False, True]
    assert scored["pred_prob"].tolist() == [0.4, 0.2, 0.4, 0.6]
