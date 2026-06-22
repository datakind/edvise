import pytest

from edvise.scripts.legacy_preprocessing import (
    SSI_PIPELINES_SUBPATH,
    resolve_ssi_pipelines_workspace_root,
)


def test_resolve_ssi_pipelines_workspace_root_from_ds_run_as():
    run_as = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert resolve_ssi_pipelines_workspace_root(ds_run_as=run_as) == (
        f"/Workspace/Users/{run_as}/{SSI_PIPELINES_SUBPATH}"
    )


def test_resolve_ssi_pipelines_workspace_root_explicit_override():
    override = "/Workspace/Users/other/student-success-intervention/pipelines"
    assert (
        resolve_ssi_pipelines_workspace_root(
            ds_run_as="ignored-when-override-set",
            workspace_root=override,
        )
        == override
    )


def test_resolve_ssi_pipelines_workspace_root_requires_identity():
    with pytest.raises(ValueError, match="--ds_run_as"):
        resolve_ssi_pipelines_workspace_root()
