import pytest

from edvise import utils


@pytest.mark.parametrize(
    ["val", "exp"],
    [
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        ({"a": 1, "b": 2}, ["a", "b"]),
        ("abc", ["abc"]),
        (1, [1]),
        (None, [None]),
    ],
)
def test_to_list(val, exp):
    obs = utils.types.to_list(val)
    assert obs == exp


@pytest.mark.parametrize(
    ["value", "exp"],
    [
        ([1, 2, 3], True),
        ({"a", "b", "c"}, True),
        ((True, False, True), True),
        ("string", False),
        (b"bytes", False),
    ],
)
def test_is_collection_but_not_string(value, exp):
    obs = utils.types.is_collection_but_not_string(value)
    assert isinstance(obs, bool)
    assert obs == exp
