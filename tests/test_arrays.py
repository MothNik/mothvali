"""
This test suite implements all tests for the module :mod:`mothvali.arrays`.

"""

# === Imports ===

from array import array
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd
import pytest

from mothvali import convert_to_validated_real_numeric_1d_array_like

# === Constants ===


class _DtypesInOtherFormatsDict(TypedDict):
    """
    The mapping of a Numpy dtype to its corresponding dtype representations in other
    formats.

    """

    python_array: str
    pandas_series: type


NUMPY_DTYPES_TO_OTHER_FORMATS: Dict[Type, _DtypesInOtherFormatsDict] = {
    np.float32: {
        "python_array": "f",
        "pandas_series": pd.Float32Dtype,
    },
    np.float64: {
        "python_array": "d",
        "pandas_series": pd.Float64Dtype,
    },
}


# === Auxiliary Models ===


class Array1DTestCaseSpecs(NamedTuple):
    """
    A convenience class for defining test case specifications for 1D Array-like inputs.

    """

    description: str
    value: Any
    min_size: Optional[int]
    max_size: Optional[int]
    output_dtype: Type
    expected: Union[np.ndarray, Exception]


@dataclass()
class MultiFormat1DFArrayLike:
    """
    A wrapper class for generating various 1D Array-like objects in various formats,
    namely:

    - a Numpy Array
    - a Python List
    - a Python Tuple
    - a Python Array
    - a Pandas Series

    """

    numpy_value: np.ndarray
    list_value: List = field(init=False)
    tuple_value: Tuple = field(init=False)
    py_array_value: array = field(init=False)
    pandas_series_value: pd.Series = field(init=False)

    def __post_init__(self) -> None:

        # --- Input Validation ---

        if self.numpy_value.ndim != 1:
            raise ValueError(
                f"Expected 'value' to be a 1D Array, but got a "
                f"{self.numpy_value.ndim}D Array."
            )

        dtype_in_other_formats = NUMPY_DTYPES_TO_OTHER_FORMATS.get(
            self.numpy_value.dtype.type, None
        )

        # --- Conversion to Other Formats ---

        if dtype_in_other_formats is None:
            raise ValueError(
                f"Unsupported dtype '{self.numpy_value.dtype}' of the provided "
                f"Numpy Array."
            )

        # NOTE: using ``.tolist()`` does not preserve the dtype of the original array
        value_as_list = [val for val in self.numpy_value]
        self.list_value = value_as_list
        self.tuple_value = tuple(self.numpy_value)

        self.py_array_value = array(
            dtype_in_other_formats["python_array"],
            value_as_list,
        )
        self.pandas_series_value = pd.Series(
            self.numpy_value,
            # NOTE: calling the pandas dtype suppresses the warning that the dtype
            #       needs to be instantiated beforehand
            dtype=dtype_in_other_formats["pandas_series"](),
        )

        return

    def __iter__(self) -> Generator[Tuple[Any, str], None, None]:

        yield self.numpy_value, ".0) a Numpy Array"
        yield self.list_value, ".1) a Python List"
        yield self.tuple_value, ".2) a Python Tuple"
        yield self.py_array_value, ".3) a Python Array"
        yield self.pandas_series_value, ".4) a Pandas Series"
        return


# === Tests ===


OneDimensionalFloat64TestArray = MultiFormat1DFArrayLike(
    numpy_value=np.array(
        [1.0],
        dtype=np.float64,
    )
)
OneDimensionalFloat32TestArray = MultiFormat1DFArrayLike(
    numpy_value=np.array(
        [1.0],
        dtype=np.float32,
    )
)


@pytest.mark.parametrize(
    "description, value, min_size, max_size, output_dtype, expected",
    [
        Array1DTestCaseSpecs(
            description="0{case} without any constraints",
            value=OneDimensionalFloat64TestArray,
            min_size=None,
            max_size=None,
            output_dtype=np.float64,
            expected=np.array([1.0], dtype=np.float64),
        ),
        Array1DTestCaseSpecs(
            description="1{case} with a satisfied minimum size",
            value=OneDimensionalFloat64TestArray,
            min_size=1,
            max_size=None,
            output_dtype=np.float64,
            expected=np.array([1.0], dtype=np.float64),
        ),
        Array1DTestCaseSpecs(
            description="2{case} with a violated minimum size",
            value=OneDimensionalFloat64TestArray,
            min_size=2,
            max_size=None,
            output_dtype=np.float64,
            expected=ValueError(
                "Expected 'value' to have a size between 2 and None for axis 0, but "
                "got a size of 1."
            ),
        ),
        Array1DTestCaseSpecs(
            description="3{case} with a satisfied maximum size",
            value=OneDimensionalFloat64TestArray,
            min_size=None,
            max_size=1,
            output_dtype=np.float64,
            expected=np.array([1.0], dtype=np.float64),
        ),
        Array1DTestCaseSpecs(
            description="4{case} with a violated maximum size",
            value=OneDimensionalFloat64TestArray,
            min_size=None,
            max_size=0,
            output_dtype=np.float64,
            expected=ValueError(
                "Expected 'value' to have a size between None and 0 for axis 0, but "
                "got a size of 1."
            ),
        ),
        Array1DTestCaseSpecs(
            description="5{case} with a satisfied fixed size",
            value=OneDimensionalFloat64TestArray,
            min_size=1,
            max_size=1,
            output_dtype=np.float64,
            expected=np.array([1.0], dtype=np.float64),
        ),
        Array1DTestCaseSpecs(
            description="6{case} with a violated fixed size",
            value=OneDimensionalFloat64TestArray,
            min_size=2,
            max_size=2,
            output_dtype=np.float64,
            expected=ValueError(
                "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
                "a size of 1."
            ),
        ),
        Array1DTestCaseSpecs(
            description=(
                "7{case} with a satisfied minimum size and a violated maximum size"
            ),
            value=OneDimensionalFloat64TestArray,
            min_size=1,
            max_size=0,
            output_dtype=np.float64,
            expected=ValueError(
                "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        Array1DTestCaseSpecs(
            description=(
                "8{case} with a violated minimum size and a satisfied maximum size"
            ),
            value=OneDimensionalFloat64TestArray,
            min_size=2,
            max_size=1,
            output_dtype=np.float64,
            expected=ValueError(
                "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
                "a size of 1."
            ),
        ),
        Array1DTestCaseSpecs(
            description=(
                "9{case} with a satisfied minimum size and a satisfied maximum size"
            ),
            value=OneDimensionalFloat64TestArray,
            min_size=1,
            max_size=1,
            output_dtype=np.float64,
            expected=np.array([1.0], dtype=np.float64),
        ),
        Array1DTestCaseSpecs(
            description=(
                "10{case} with a violated minimum size and a violated maximum size"
            ),
            value=OneDimensionalFloat64TestArray,
            min_size=2,
            max_size=0,
            output_dtype=np.float64,
            expected=ValueError(
                "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
                "a size of 1."
            ),
        ),
        Array1DTestCaseSpecs(
            description="11{case} with a valid type conversion",
            value=OneDimensionalFloat32TestArray,
            min_size=None,
            max_size=None,
            output_dtype=np.float64,
            expected=np.array([1.0], dtype=np.float64),
        ),
        Array1DTestCaseSpecs(
            description="12{case} with an invalid type conversion",
            value=OneDimensionalFloat32TestArray,
            min_size=None,
            max_size=None,
            output_dtype=np.int64,
            # NOTE: raw string with bracket escape for preventing regex errors
            #       https://stackoverflow.com/a/76565993/14814813
            expected=TypeError(
                r"Could not convert 'value' from a 'float32'- to a 'int64'-Array "
                r"\(uses 'safe' casting\)."
            ),
        ),
        Array1DTestCaseSpecs(
            description="13{case} that is empty without any constraints",
            value=np.array([]),
            min_size=None,
            max_size=None,
            output_dtype=np.float64,
            expected=ValueError("Expected 'value' to be a non-empty Array-like."),
        ),
        # Array1DTestCaseSpecs(
        #     description="14) a 2D NumPy Array without any constraints",
        #     value=np.array([[1.0]]),
        #     min_size=None,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     expected=ValueError(
        #         r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
        #         r"shape \(1, 1\)."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description="15) a Python List without any constraints",
        #     value=[1.0],
        #     min_size=None,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="16) a Python List with a satisfied minimum size",
        #     value=[1.0],
        #     min_size=1,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="17) a Python List with a violated minimum size",
        #     value=[1.0],
        #     min_size=2,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between 2 and None for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description="18) a Python List with a satisfied maximum size",
        #     value=[1.0],
        #     min_size=None,
        #     max_size=1,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="19) a Python List with a violated maximum size",
        #     value=[1.0],
        #     min_size=None,
        #     max_size=0,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between None and 0 for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description="20) a Python List with a satisfied fixed size",
        #     value=[1.0],
        #     min_size=1,
        #     max_size=1,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="21) a Python List with a violated fixed size",
        #     value=[1.0],
        #     min_size=2,
        #     max_size=2,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description=(
        #         "22) a Python List with a satisfied minimum size and a violated "
        #         "maximum size"
        #     ),
        #     value=[1.0],
        #     min_size=1,
        #     max_size=0,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description=(
        #         "23) a Python List with a violated minimum size and a satisfied "
        #         "maximum size"
        #     ),
        #     value=[1.0],
        #     min_size=2,
        #     max_size=1,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description=(
        #         "24) a Python List with a satisfied minimum size and a satisfied "
        #         "maximum size"
        #     ),
        #     value=[1.0],
        #     min_size=1,
        #     max_size=1,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description=(
        #         "25) a Python List with a violated minimum size and a violated "
        #         "maximum size"
        #     ),
        #     value=[1.0],
        #     min_size=2,
        #     max_size=0,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description="26) a Python List with a valid type conversion",
        #     value=[1],
        #     min_size=None,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="27) a Python List with an invalid type conversion",
        #     value=[1.0],
        #     min_size=None,
        #     max_size=None,
        #     output_dtype=np.int64,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     expected=TypeError(
        #         r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
        #         r"\(uses 'safe' casting\)."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description="28) an empty Python List without any constraints",
        #     value=[],
        #     min_size=None,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=ValueError("Expected 'value' to be a non-empty Array-like."),
        # ),
        # Array1DTestCaseSpecs(
        #     description="29) a 2D Python List without any constraints",
        #     value=[[1.0]],
        #     min_size=None,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     expected=ValueError(
        #         r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
        #         r"shape \(1, 1\)."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description="30) a Python Tuple without any constraints",
        #     value=(1.0,),
        #     min_size=None,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="31) a Python Tuple with a satisfied minimum size",
        #     value=(1.0,),
        #     min_size=1,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="32) a Python Tuple with a violated minimum size",
        #     value=(1.0,),
        #     min_size=2,
        #     max_size=None,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between 2 and None for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # Array1DTestCaseSpecs(
        #     description="33) a Python Tuple with a satisfied maximum size",
        #     value=(1.0,),
        #     min_size=None,
        #     max_size=1,
        #     output_dtype=np.float64,
        #     expected=np.array([1.0], dtype=np.float64),
        # ),
        # Array1DTestCaseSpecs(
        #     description="34) a Python Tuple with a violated maximum size",
        #     value=(1.0,),
        #     min_size=None,
        #     max_size=0,
        #     output_dtype=np.float64,
        #     expected=ValueError(
        #         "Expected 'value' to have a size between None and 0 for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # (  # 33) a Python Tuple with a satisfied maximum size
        #     (1.0,),
        #     None,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 34) a Python Tuple with a violated maximum size
        #     (1.0,),
        #     None,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between None and 0 for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # (  # 35) a Python Tuple with a satisfied fixed size
        #     (1.0,),
        #     1,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 36) a Python Tuple with a violated fixed size
        #     (1.0,),
        #     2,
        #     2,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 37) a Python Tuple with a satisfied minimum size and a violated maximum size  # noqa: E501
        #     (1.0,),
        #     1,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 38) a Python Tuple with a violated minimum size and a satisfied maximum size  # noqa: E501
        #     (1.0,),
        #     2,
        #     1,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 39) a Python Tuple with a satisfied minimum size and a satisfied maximum size  # noqa: E501
        #     (1.0,),
        #     1,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 40) a Python Tuple with a violated minimum size and a violated maximum size
        #     (1.0,),
        #     2,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 41) a Python Tuple with a valid type conversion
        #     (1.0,),
        #     None,
        #     None,
        #     np.float64,
        #     np.float64(1.0),
        # ),
        # (  # 42) a Python Tuple with an invalid type conversion
        #     (1.0,),
        #     None,
        #     None,
        #     np.int64,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     TypeError(
        #         r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
        #         r"\(uses 'safe' casting\)."
        #     ),
        # ),
        # (  # 43) an empty Python Tuple without any constraints
        #     tuple(),
        #     None,
        #     None,
        #     None,
        #     ValueError("Expected 'value' to be a non-empty Array-like."),
        # ),
        # (  # 44) a 2D Python Tuple without any constraints
        #     ((1.0,),),
        #     None,
        #     None,
        #     None,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     ValueError(
        #         r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
        #         r"shape \(1, 1\)."
        #     ),
        # ),
        # (  # 45) a Python Array without any constraints
        #     array("d", [1.0]),
        #     None,
        #     None,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 46) a Python Array with a satisfied minimum size
        #     array("d", [1.0]),
        #     1,
        #     None,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 47) a Python Array with a violated minimum size
        #     array("d", [1.0]),
        #     2,
        #     None,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and None for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # (  # 48) a Python Array with a satisfied maximum size
        #     array("d", [1.0]),
        #     None,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 49) a Python Array with a violated maximum size
        #     array("d", [1.0]),
        #     None,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between None and 0 for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # (  # 50) a Python Array with a satisfied fixed size
        #     array("d", [1.0]),
        #     1,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 51) a Python Array with a violated fixed size
        #     array("d", [1.0]),
        #     2,
        #     2,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 52) a Python Array with a satisfied minimum size and a violated maximum size  # noqa: E501
        #     array("d", [1.0]),
        #     1,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 53) a Python Array with a violated minimum size and a satisfied maximum size # noqa: E501
        #     array("d", [1.0]),
        #     2,
        #     1,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 54) a Python Array with a satisfied minimum size and a satisfied maximum size  # noqa: E501
        #     array("d", [1.0]),
        #     1,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 55) a Python Array with a violated minimum size and a violated maximum size
        #     array("d", [1.0]),
        #     2,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 56) a Python Array with a valid type conversion
        #     array("d", [1.0]),
        #     None,
        #     None,
        #     np.float64,
        #     np.float64(1.0),
        # ),
        # (  # 57) a Python Array with an invalid type conversion
        #     array("d", [1.0]),
        #     None,
        #     None,
        #     np.int64,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     TypeError(
        #         r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
        #         r"\(uses 'safe' casting\)."
        #     ),
        # ),
        # (  # 58) an empty Python Array without any constraints
        #     array("d", []),
        #     None,
        #     None,
        #     None,
        #     ValueError("Expected 'value' to be a non-empty Array-like."),
        # ),
        # (  # 59) a "2D Python Array" without any constraints
        #     [array("d", [1.0])],
        #     None,
        #     None,
        #     None,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     ValueError(
        #         r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
        #         r"shape \(1, 1\)."
        #     ),
        # ),
        # (  # 60) a Pandas Series without any constraints
        #     pd.Series([1.0]),
        #     None,
        #     None,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 61) a Pandas Series with a satisfied minimum size
        #     pd.Series([1.0]),
        #     1,
        #     None,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 62) a Pandas Series with a violated minimum size
        #     pd.Series([1.0]),
        #     2,
        #     None,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and None for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # (  # 63) a Pandas Series with a satisfied maximum size
        #     pd.Series([1.0]),
        #     None,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 64) a Pandas Series with a violated maximum size
        #     pd.Series([1.0]),
        #     None,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between None and 0 for axis 0, but "
        #         "got a size of 1."
        #     ),
        # ),
        # (  # 65) a Pandas Series with a satisfied fixed size
        #     pd.Series([1.0]),
        #     1,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 66) a Pandas Series with a violated fixed size
        #     pd.Series([1.0]),
        #     2,
        #     2,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 2 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 67) a Pandas Series with a satisfied minimum size and a violated maximum size  # noqa: E501
        #     pd.Series([1.0]),
        #     1,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 1 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 68) a Pandas Series with a violated minimum size and a satisfied maximum size  # noqa: E501
        #     pd.Series([1.0]),
        #     2,
        #     1,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 1 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 69) a Pandas Series with a satisfied minimum size and a satisfied maximum size  # noqa: E501
        #     pd.Series([1.0]),
        #     1,
        #     1,
        #     None,
        #     np.array([1.0]),
        # ),
        # (  # 70) a Pandas Series with a violated minimum size and a violated maximum size  # noqa: E501
        #     pd.Series([1.0]),
        #     2,
        #     0,
        #     None,
        #     ValueError(
        #         "Expected 'value' to have a size between 2 and 0 for axis 0, but got "
        #         "a size of 1."
        #     ),
        # ),
        # (  # 71) a Pandas Series with a valid type conversion
        #     pd.Series([1.0]),
        #     None,
        #     None,
        #     np.float64,
        #     np.float64(1.0),
        # ),
        # (  # 72) a Pandas Series with an invalid type conversion
        #     pd.Series([1.0]),
        #     None,
        #     None,
        #     np.int64,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     TypeError(
        #         r"Could not convert 'value' from a 'float64'- to a 'int64'-Array "
        #         r"\(uses 'safe' casting\)."
        #     ),
        # ),
        # (  # 73) an empty Pandas Series without any constraints
        #     pd.Series([]),
        #     None,
        #     None,
        #     None,
        #     ValueError("Expected 'value' to be a non-empty Array-like."),
        # ),
        # (  # 74) a "2D Pandas Series" without any constraints
        #     pd.DataFrame([1.0]),
        #     None,
        #     None,
        #     None,
        #     # NOTE: raw string with bracket escape for preventing regex errors
        #     #       https://stackoverflow.com/a/76565993/14814813
        #     ValueError(
        #         r"Expected 'value' to be a 1D Array-like, but got a 2D Array-like of "
        #         r"shape \(1, 1\)."
        #     ),
        # ),
        # (  # 75) a Python lists of lists with inconsistent sizes
        #     [[1.0], [2.0, 3.0]],
        #     None,
        #     None,
        #     None,
        #     ValueError("'value' could not be converted to a NumPy Array-like."),
        # ),
        # (  # 76) a NumPy Array of complex values
        #     np.array([1.0 + 1.0j]),
        #     None,
        #     None,
        #     None,
        #     TypeError(
        #         # NOTE: the round parenthesis would cause a regex error
        #         r"Expected 'value' to be a 1D Array-like of real numeric \(boolean "
        #         r"excluded\) values, but got a 1D Array-like not meeting this "
        #         r"requirement."
        #     ),
        # ),
        # (  # 77) a Python List of complex values
        #     [1.0 + 1.0j],
        #     None,
        #     None,
        #     None,
        #     TypeError(
        #         # NOTE: the round parenthesis would cause a regex error
        #         r"Expected 'value' to be a 1D Array-like of real numeric \(boolean "
        #         r"excluded\) values, but got a 1D Array-like not meeting this "
        #         r"requirement."
        #     ),
        # ),
        # (  # 78) a Python Tuple of complex values
        #     (1.0 + 1.0j,),
        #     None,
        #     None,
        #     None,
        #     TypeError(
        #         # NOTE: the round parenthesis would cause a regex error
        #         r"Expected 'value' to be a 1D Array-like of real numeric \(boolean "
        #         r"excluded\) values, but got a 1D Array-like not meeting this "
        #         r"requirement."
        #     ),
        # ),
        # (  # 79) a Pandas Series of complex values
        #     pd.Series([1.0 + 1.0j]),
        #     None,
        #     None,
        #     None,
        #     TypeError(
        #         # NOTE: the round parenthesis would cause a regex error
        #         r"Expected 'value' to be a 1D Array-like of real numeric \(boolean "
        #         r"excluded\) values, but got a 1D Array-like not meeting this "
        #         r"requirement."
        #     ),
        # ),
    ],
)
def test_real_numeric_1d_array_like_validation(
    description: str,
    value: Any,
    min_size: Optional[int],
    max_size: Optional[int],
    output_dtype: Type,
    expected: Union[np.ndarray, Exception],
) -> None:
    """
    Tests the function :func:`convert_to_validated_real_numeric_1d_array_like` for
    various input values for

    - passing for correct input values
    - raising exceptions for incorrect input values

    """

    if not isinstance(value, MultiFormat1DFArrayLike):
        new_value = ((value, None),)
    else:
        new_value = value

    for sub_value, sub_case in new_value:

        sub_description = description
        if sub_case is not None:
            sub_description = sub_description.format(case=sub_case)

        # if an exception should be raised, the function is called and the exception is
        # checked
        if isinstance(expected, Exception):
            with pytest.raises(type(expected), match=str(expected)):
                checked_value = convert_to_validated_real_numeric_1d_array_like(
                    value=sub_value,
                    name="value",
                    min_size=min_size,
                    max_size=max_size,
                    output_dtype=output_dtype,
                )

            continue

        # if no exception should be raised, the function is called and the output is
        # checked
        checked_value = convert_to_validated_real_numeric_1d_array_like(
            value=sub_value,
            name="value",
            min_size=min_size,
            max_size=max_size,
            output_dtype=output_dtype,
        )

        assert (
            checked_value.dtype == output_dtype
        ), f"Data type mismatch for test case '{sub_description}'."

        assert np.allclose(
            checked_value,
            expected,
            atol=1e-15,
            rtol=1e-15,
        ), f"Value mismatch for test case '{sub_description}'."

    return
