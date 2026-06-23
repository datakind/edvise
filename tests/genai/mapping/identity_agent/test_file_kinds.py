"""Tests for FileKind taxonomy and prompt documentation."""

from edvise.genai.mapping.identity_agent.column_roles.file_kinds import (
    FILE_KIND_EXPECTED_CONTENTS,
    FileKind,
    file_kind_prompt_section,
)


def test_file_kind_values():
    assert [k.value for k in FileKind] == [
        "student",
        "course",
        "semester",
        "degree",
        "other",
    ]


def test_file_kind_expected_contents_covers_all_kinds():
    for kind in FileKind:
        assert kind in FILE_KIND_EXPECTED_CONTENTS
        assert len(FILE_KIND_EXPECTED_CONTENTS[kind]) > 20


def test_file_kind_prompt_section_lists_all_kinds():
    section = file_kind_prompt_section()
    for kind in FileKind:
        assert f"`{kind.value}`" in section
