import inspect
import os
import typing as t
from pathlib import Path

import pydantic
import tomlkit
from pydantic import BaseModel, Field

from edvise.config_validation import (
    TargetGraduationConfig,
    TargetRetentionConfig,
    TargetCreditsEarnedConfig,
    CheckpointNthConfig,
    CheckpointFirstConfig,
    CheckpointLastConfig,
    CheckpointFirstAtNumCreditsEarnedConfig,
    CheckpointFirstWithinCohortConfig,
    CheckpointLastInEnrollmentYearConfig,
)

def generate_template_dict(model_cls: t.Type[BaseModel]) -> dict:
    result = {}
    for name, field in model_cls.model_fields.items():
        if field.default is not pydantic.Undefined:
            value = field.default
        elif field.default_factory is not None:
            value = field.default_factory()
        else:
            value = f"<{field.annotation.__name__ if hasattr(field.annotation, '__name__') else 'value'}>"

        result[name] = value
    return result


def save_toml_file(model_name: str, content: dict, output_dir: Path):
    doc = tomlkit.document()
    for k, v in content.items():
        doc[k] = v
    output_path = output_dir / f"{model_name}_template.toml"
    with open(output_path, "w") as f:
        f.write(tomlkit.dumps(doc))
    print(f"✅ Generated: {output_path}")


def main():
    output_dir = Path("configs/templates")
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, t.Type[BaseModel]] = {
        "target_graduation": TargetGraduationConfig,
        "target_retention": TargetRetentionConfig,
        "target_credits_earned": TargetCreditsEarnedConfig,
        "checkpoint_nth": CheckpointNthConfig,
        "checkpoint_first": CheckpointFirstConfig,
        "checkpoint_last": CheckpointLastConfig,
        "checkpoint_first_at_num_credits": CheckpointFirstAtNumCreditsEarnedConfig,
        "checkpoint_first_within_cohort": CheckpointFirstWithinCohortConfig,
        "checkpoint_last_in_enrollment_year": CheckpointLastInEnrollmentYearConfig,
    }

    for model_name, model_cls in models.items():
        template_data = generate_template_dict(model_cls)
        save_toml_file(model_name, template_data, output_dir)


if __name__ == "__main__":
    main()
