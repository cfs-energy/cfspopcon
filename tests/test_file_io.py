"""Tests for the JSON point-file formatting."""

import json

from cfspopcon.file_io import _ModifyJSONFloatRepr


def test_rounded_floats_remain_valid_json():
    """A float whose 6-figure form would end in a bare trailing point gains a digit instead."""
    values = [732029.47, 100000.0, 1.0, 0.123456789, 1234567.0, -732029.47]
    with _ModifyJSONFloatRepr():
        text = json.dumps(values)
    assert json.loads(text) == [732029.5, 100000.0, 1.0, 0.123457, 1.23457e06, -732029.5]
