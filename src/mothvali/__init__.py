# === Imports ===

from pathlib import Path as __Path

from .numpy_arrays import (  # noqa: F401
    RealNumericArrayLike,
    convert_to_validated_numeric_nd_array_like,
    convert_to_validated_real_numeric_1d_array_like,
    convert_to_validated_real_numeric_2d_array_like,
)
from .scalars import (  # noqa: F401
    Integer,
    RealNumeric,
    convert_to_validated_python_float,
    convert_to_validated_python_integer,
    isinstance_incl_none,
)

# === Package Metadata ===

# the author and version are read using relative file locations
__current_file_directory = __Path(__file__).resolve().parent

with open(__current_file_directory / "VERSION.txt", "r") as version_file:
    __version__ = version_file.read().strip()

with open(__current_file_directory / "AUTHORS.txt", "r") as author_file:
    __author__ = author_file.read().strip()
