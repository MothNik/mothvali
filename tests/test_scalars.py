"""
This test suite implements all tests for the module :mod:`mothvali.scalars`.

"""

# === Imports ===

from math import isclose as py_isclose
from typing import Any, NamedTuple, Optional, Union

import numpy as np
import pytest

from mothvali import (
    convert_to_validated_python_float,
    convert_to_validated_python_integer,
)

# === Models ===


class ScalarTestCaseSpecs(NamedTuple):
    """
    A convenience class for defining test case specifications for scalar validation
    tests.

    """

    description: str
    value: Any
    min_value: Optional[Union[int, float]]
    max_value: Optional[Union[int, float]]
    clip: bool
    expected: Union[int, float, Exception]


# === Tests ===


@pytest.mark.parametrize(
    "description, value, min_value, max_value, clip, expected",
    [
        ScalarTestCaseSpecs(
            description="0) a Python integer without any constraints",
            value=1,
            min_value=None,
            max_value=None,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description="1) a NumPy integer without any constraints",
            value=np.int64(1),
            min_value=None,
            max_value=None,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description="2) a float value without any constraints",
            value=1.0,
            min_value=None,
            max_value=None,
            clip=False,
            expected=TypeError(
                "Expected 'value' to be of type 'int', but got 'float'."
            ),
        ),
        ScalarTestCaseSpecs(
            description=(
                "3) a Python integer with a minimum constraint that is satisfied"
            ),
            value=1,
            min_value=0,
            max_value=None,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "4) a Python integer with a minimum constraint that leads to clipping"
            ),
            value=1,
            min_value=2,
            max_value=None,
            clip=True,
            expected=2,
        ),
        ScalarTestCaseSpecs(
            description=(
                "5) a Python integer with a minimum constraint that leads to an "
                "exception"
            ),
            value=1,
            min_value=2,
            max_value=None,
            clip=False,
            expected=ValueError("Expected 'value' to be >= 2, but got 1."),
        ),
        ScalarTestCaseSpecs(
            description=(
                "6) a Python integer with a maximum constraint that is satisfied"
            ),
            value=1,
            min_value=None,
            max_value=2,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "7) a Python integer with a maximum constraint that leads to clipping"
            ),
            value=3,
            min_value=None,
            max_value=2,
            clip=True,
            expected=2,
        ),
        ScalarTestCaseSpecs(
            description=(
                "8) a Python integer with a maximum constraint that leads to an "
                "exception"
            ),
            value=3,
            min_value=None,
            max_value=2,
            clip=False,
            expected=ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        ScalarTestCaseSpecs(
            description="9) a Python integer with both constraints that are satisfied",
            value=1,
            min_value=0,
            max_value=2,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "10) a Python integer with both constraints that lead to clipping"
            ),
            value=0,
            min_value=1,
            max_value=2,
            clip=True,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "11) a Python integer with both constraints that lead to clipping"
            ),
            value=3,
            min_value=0,
            max_value=2,
            clip=True,
            expected=2,
        ),
        ScalarTestCaseSpecs(
            description=(
                "12) a Python integer with both constraints that lead to an "
                "exception"
            ),
            value=0,
            min_value=1,
            max_value=2,
            clip=False,
            expected=ValueError("Expected 'value' to be >= 1, but got 0."),
        ),
        ScalarTestCaseSpecs(
            description=(
                "13) a Python integer with both constraints that lead to an "
                "exception"
            ),
            value=3,
            min_value=0,
            max_value=2,
            clip=False,
            expected=ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        ScalarTestCaseSpecs(
            description="14) a Python integer with flipped constraints",
            value=1,
            min_value=2,
            max_value=0,
            clip=False,
            expected=ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2 and max = 0."
            ),
        ),
        ScalarTestCaseSpecs(
            description=(
                "15) a NumPy integer with a minimum constraint that is satisfied"
            ),
            value=np.int64(1),
            min_value=0,
            max_value=None,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "16) a NumPy integer with a minimum constraint that leads to clipping"
            ),
            value=np.int64(1),
            min_value=2,
            max_value=None,
            clip=True,
            expected=2,
        ),
        ScalarTestCaseSpecs(
            description=(
                "17) a NumPy integer with a minimum constraint that leads to an "
                "exception"
            ),
            value=np.int64(1),
            min_value=2,
            max_value=None,
            clip=False,
            expected=ValueError("Expected 'value' to be >= 2, but got 1."),
        ),
        ScalarTestCaseSpecs(
            description=(
                "18) a NumPy integer with a maximum constraint that is satisfied"
            ),
            value=np.int64(1),
            min_value=None,
            max_value=2,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "19) a NumPy integer with a maximum constraint that leads to clipping"
            ),
            value=np.int64(3),
            min_value=None,
            max_value=2,
            clip=True,
            expected=2,
        ),
        ScalarTestCaseSpecs(
            description=(
                "20) a NumPy integer with a maximum constraint that leads to an "
                "exception"
            ),
            value=np.int64(3),
            min_value=None,
            max_value=2,
            clip=False,
            expected=ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        ScalarTestCaseSpecs(
            description="21) a NumPy integer with both constraints that are satisfied",
            value=np.int64(1),
            min_value=0,
            max_value=2,
            clip=False,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "22) a NumPy integer with both constraints that lead to clipping"
            ),
            value=np.int64(0),
            min_value=1,
            max_value=2,
            clip=True,
            expected=1,
        ),
        ScalarTestCaseSpecs(
            description=(
                "23) a NumPy integer with both constraints that lead to clipping"
            ),
            value=np.int64(3),
            min_value=0,
            max_value=2,
            clip=True,
            expected=2,
        ),
        ScalarTestCaseSpecs(
            description=(
                "24) a NumPy integer with both constraints that lead to an exception"
            ),
            value=np.int64(0),
            min_value=1,
            max_value=2,
            clip=False,
            expected=ValueError("Expected 'value' to be >= 1, but got 0."),
        ),
        ScalarTestCaseSpecs(
            description=(
                "25) a NumPy integer with both constraints that lead to an exception"
            ),
            value=np.int64(3),
            min_value=0,
            max_value=2,
            clip=False,
            expected=ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        ScalarTestCaseSpecs(
            description="26) a NumPy integer with flipped constraints",
            value=np.int64(1),
            min_value=2,
            max_value=0,
            clip=False,
            expected=ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2 and max = 0."
            ),
        ),
        ScalarTestCaseSpecs(
            description=(
                "27) a Python float with a minimum constraint that is satisfied"
            ),
            value=1.0,
            min_value=0,
            max_value=None,
            clip=False,
            expected=TypeError(
                "Expected 'value' to be of type 'int', but got 'float'."
            ),
        ),
        ScalarTestCaseSpecs(
            description=(
                "28) a Python float with a maximum constraint that is satisfied"
            ),
            value=1.0,
            min_value=None,
            max_value=2,
            clip=False,
            expected=TypeError(
                "Expected 'value' to be of type 'int', but got 'float'."
            ),
        ),
        ScalarTestCaseSpecs(
            description="29) a Python float with both constraints that are satisfied",
            value=1.0,
            min_value=0,
            max_value=2,
            clip=False,
            expected=TypeError(
                "Expected 'value' to be of type 'int', but got 'float'."
            ),
        ),
        ScalarTestCaseSpecs(
            description="30) a Python float with flipped constraints",
            value=1.0,
            min_value=2,
            max_value=0,
            clip=False,
            expected=TypeError(
                "Expected 'value' to be of type 'int', but got 'float'."
            ),
        ),
    ],
)
def test_integer_validation(
    description: str,
    value: Any,
    min_value: Optional[int],
    max_value: Optional[int],
    clip: bool,
    expected: Union[int, Exception],
) -> None:
    """
    Tests the function :func:`convert_to_validated_integer` for various input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            checked_value = convert_to_validated_python_integer(
                value=value,
                name="value",
                min_value=min_value,
                max_value=max_value,
                clip=clip,
            )

        return

    # if no exception should be raised, the function is called and the output is checked
    checked_value = convert_to_validated_python_integer(
        value=value,
        name="value",
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )

    assert (
        checked_value == expected
    ), f"'convert_to_validated_integer' failed for test case '{description}'."

    return


@pytest.mark.parametrize(
    "value, min_value, max_value, clip, expected",
    [
        (  # 0) a Python float without any constraints
            1.0,
            None,
            None,
            False,
            1.0,
        ),
        (  # 1) a NumPy float without any constraints
            np.float64(1.0),
            None,
            None,
            False,
            1.0,
        ),
        (  # 2) a Python integer without any constraints
            1,
            None,
            None,
            False,
            1.0,
        ),
        (  # 3) a NumPy integer without any constraints
            np.int64(1),
            None,
            None,
            False,
            1.0,
        ),
        (  # 4) a complex value without any constraints
            complex(1.0, 1.0),
            None,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'int', but got 'complex'."
            ),
        ),
        (  # 5) a Python float with a minimum constraint that is satisfied
            1.0,
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 6) a Python float with a minimum constraint that leads to clipping
            1.0,
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 7) a Python float with a minimum constraint that leads to an exception
            1.0,
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1.0."),
        ),
        (  # 8) a Python integer with a minimum constraint that is satisfied
            1,
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 9) a Python integer with a minimum constraint that leads to clipping
            1,
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 10) a Python integer with a minimum constraint that leads to an exception
            1,
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1."),
        ),
        (  # 11) a Python float with a maximum constraint that is satisfied
            1.0,
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 12) a Python float with a maximum constraint that leads to clipping
            3.0,
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 13) a Python float with a maximum constraint that leads to an exception
            3.0,
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 14) a Python integer with a maximum constraint that is satisfied
            1,
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 15) a Python integer with a maximum constraint that leads to clipping
            3,
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 16) a Python integer with a maximum constraint that leads to an exception
            3,
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3."),
        ),
        (  # 17) a Python float with both constraints that are satisfied
            1.0,
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 18) a Python float with both constraints that lead to clipping
            0.0,
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 19) a Python float with both constraints that lead to clipping
            3.0,
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 20) a Python float with both constraints that lead to an exception
            0.0,
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0.0."),
        ),
        (  # 21) a Python float with both constraints that lead to an exception
            3.0,
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 22) a Python integer with both constraints that are satisfied
            1,
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 23) a Python integer with both constraints that lead to clipping
            0,
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 24) a Python integer with both constraints that lead to clipping
            3,
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 25) a Python integer with both constraints that lead to an exception
            0,
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0."),
        ),
        (  # 26) a Python integer with both constraints that lead to an exception
            3,
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3."),
        ),
        (  # 27) a Python float with flipped constraints
            1.0,
            2.0,
            0.0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2.0 and max = 0.0."
            ),
        ),
        (  # 28) a Python integer with flipped constraints
            1,
            2.0,
            0.0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2.0 and max = 0.0."
            ),
        ),
        (  # 29) a NumPy float with a minimum constraint that is satisfied
            np.float64(1.0),
            0.0,
            None,
            False,
            1.0,
        ),
        (  # 30) a NumPy float with a minimum constraint that leads to clipping
            np.float64(1.0),
            2.0,
            None,
            True,
            2.0,
        ),
        (  # 31) a NumPy float with a minimum constraint that leads to an exception
            np.float64(1.0),
            2.0,
            None,
            False,
            ValueError("Expected 'value' to be >= 2.0, but got 1.0."),
        ),
        (  # 32) a NumPy float with a maximum constraint that is satisfied
            np.float64(1.0),
            None,
            2.0,
            False,
            1.0,
        ),
        (  # 33) a NumPy float with a maximum constraint that leads to clipping
            np.float64(3.0),
            None,
            2.0,
            True,
            2.0,
        ),
        (  # 34) a NumPy float with a maximum constraint that leads to an exception
            np.float64(3.0),
            None,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 35) a NumPy float with both constraints that are satisfied
            np.float64(1.0),
            0.0,
            2.0,
            False,
            1.0,
        ),
        (  # 36) a NumPy float with both constraints that lead to clipping
            np.float64(0.0),
            1.0,
            2.0,
            True,
            1.0,
        ),
        (  # 37) a NumPy float with both constraints that lead to clipping
            np.float64(3.0),
            0.0,
            2.0,
            True,
            2.0,
        ),
        (  # 38) a NumPy float with both constraints that lead to an exception
            np.float64(0.0),
            1.0,
            2.0,
            False,
            ValueError("Expected 'value' to be >= 1.0, but got 0.0."),
        ),
        (  # 39) a NumPy float with both constraints that lead to an exception
            np.float64(3.0),
            0.0,
            2.0,
            False,
            ValueError("Expected 'value' to be <= 2.0, but got 3.0."),
        ),
        (  # 40) a NumPy float with flipped constraints
            np.float64(1.0),
            2.0,
            0.0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2.0 and max = 0.0."
            ),
        ),
        (  # 41) a NumPy integer with a minimum constraint that is satisfied
            np.int64(1),
            0,
            None,
            False,
            1.0,
        ),
        (  # 42) a NumPy integer with a minimum constraint that leads to clipping
            np.int64(1),
            2,
            None,
            True,
            2.0,
        ),
        (  # 43) a NumPy integer with a minimum constraint that leads to an exception
            np.int64(1),
            2,
            None,
            False,
            ValueError("Expected 'value' to be >= 2, but got 1."),
        ),
        (  # 44) a NumPy integer with a maximum constraint that is satisfied
            np.int64(1),
            None,
            2,
            False,
            1.0,
        ),
        (  # 45) a NumPy integer with a maximum constraint that leads to clipping
            np.int64(3),
            None,
            2,
            True,
            2.0,
        ),
        (  # 46) a NumPy integer with a maximum constraint that leads to an exception
            np.int64(3),
            None,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 47) a NumPy integer with both constraints that are satisfied
            np.int64(1),
            0,
            2,
            False,
            1.0,
        ),
        (  # 48) a NumPy integer with both constraints that lead to clipping
            np.int64(0),
            1,
            2,
            True,
            1.0,
        ),
        (  # 49) a NumPy integer with both constraints that lead to clipping
            np.int64(3),
            0,
            2,
            True,
            2.0,
        ),
        (  # 50) a NumPy integer with both constraints that lead to an exception
            np.int64(0),
            1,
            2,
            False,
            ValueError("Expected 'value' to be >= 1, but got 0."),
        ),
        (  # 51) a NumPy integer with both constraints that lead to an exception
            np.int64(3),
            0,
            2,
            False,
            ValueError("Expected 'value' to be <= 2, but got 3."),
        ),
        (  # 52) a NumPy integer with flipped constraints
            np.int64(1),
            2,
            0,
            False,
            ValueError(
                "Expected minimum value for 'value' to be <= maximum value, but got "
                "min = 2 and max = 0."
            ),
        ),
        (  # 53) a complex value with a minimum constraint that is satisfied
            complex(1.0, 1.0),
            0.0,
            None,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'int', but got 'complex'."
            ),
        ),
        (  # 54) a complex value with a maximum constraint that is satisfied
            complex(1.0, 1.0),
            None,
            2.0,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'int', but got 'complex'."
            ),
        ),
        (  # 55) a complex value with both constraints that are satisfied
            complex(1.0, 1.0),
            0.0,
            2.0,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'int', but got 'complex'."
            ),
        ),
        (  # 56) a complex value with flipped constraints
            complex(1.0, 1.0),
            2.0,
            0.0,
            False,
            TypeError(
                "Expected 'value' to be of type 'float' / 'int', but got 'complex'."
            ),
        ),
    ],
)
def test_real_numeric_validation(
    value: Any,
    min_value: Optional[float],
    max_value: Optional[float],
    clip: bool,
    expected: Union[float, Exception],
) -> None:
    """
    Test the function :func:`convert_to_validated_float` for various input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    # if an exception should be raised, the function is called and the exception is
    # checked
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=str(expected)):
            checked_value = convert_to_validated_python_float(
                value=value,
                name="value",
                min_value=min_value,
                max_value=max_value,
                clip=clip,
            )

        return

    # if no exception should be raised, the function is called and the output is checked
    checked_value = convert_to_validated_python_float(
        value=value,
        name="value",
        min_value=min_value,
        max_value=max_value,
        clip=clip,
    )

    assert py_isclose(
        checked_value,
        expected,
        abs_tol=1e-15,
        rel_tol=1e-15,
    )

    return
