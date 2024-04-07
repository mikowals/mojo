# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements slice.

These are Mojo built-ins, so you don't need to import them.
"""

from sys.intrinsics import _mlirtype_is_eq
from collections import OptionalReg


@always_inline("nodebug")
fn _int_max_value() -> Int:
    # FIXME: The `slice` type should have the concept of `None` indices, but the
    # effect of a `None` end index is the same as a very large end index.
    return int(Int32.MAX)


@always_inline("nodebug")
fn _default_or[T: AnyRegType](value: T, default: Int) -> Int:
    # TODO: Handle `__index__` for other types when we have traits!
    @parameter
    if _mlirtype_is_eq[T, Int]():
        return __mlir_op.`kgen.rebind`[_type=Int](value)
    else:
        __mlir_op.`kgen.param.assert`[
            cond = (_mlirtype_is_eq[T, NoneType]()).__mlir_i1__(),
            message = "expected Int or NoneType".value,
        ]()
        return default


@register_passable("trivial")
struct Slice(Sized, Stringable, EqualityComparable):
    """Represents a slice expression.

    Objects of this type are generated when slice syntax is used within square
    brackets, e.g.:

    ```mojo
    var msg: String = "Hello Mojo"

    # Both are equivalent and print "Mojo".
    print(msg[6:])
    print(msg.__getitem__(Slice(6, len(msg))))
    ```
    """

    var start: Int
    """The starting index of the slice."""
    var end: Int
    """The end index of the slice."""
    var step: Int
    """The step increment value of the slice."""

    @always_inline("nodebug")
    fn __init__(start: Int, end: Int) -> Self:
        """Construct slice given the start and end values.

        Args:
            start: The start value.
            end: The end value.

        Returns:
            The constructed slice.
        """
        return Self {start: start, end: end, step: 1}

    @always_inline("nodebug")
    fn __init__[
        T0: AnyRegType, T1: AnyRegType, T2: AnyRegType
    ](start: T0, end: T1, step: T2) -> Self:
        """Construct slice given the start, end and step values.

        Parameters:
            T0: Type of the start value.
            T1: Type of the end value.
            T2: Type of the step value.

        Args:
            start: The start value.
            end: The end value.
            step: The step value.

        Returns:
            The constructed slice.
        """
        return Self {
            start: _default_or(start, 0),
            end: _default_or(end, _int_max_value()),
            step: _default_or(step, 1),
        }

    fn __str__(self) -> String:
        """Gets the string representation of the span.

        Returns:
            The string representation of the span.
        """
        var res = str(self.start)
        res += ":"
        if self._has_end():
            res += str(self.end)
        res += ":"
        res += str(self.step)
        return res

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Compare this slice to the other.

        Args:
            other: The slice to compare to.

        Returns:
            True if start, end, and step values of this slice match the
            corresponding values of the other slice and False otherwise.
        """
        return (
            self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Compare this slice to the other.

        Args:
            other: The slice to compare to.

        Returns:
            False if start, end, and step values of this slice match the
            corresponding values of the other slice and True otherwise.
        """
        return not (self == other)

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Return the length of the slice.

        Returns:
            The length of the slice.
        """

        return len(range(self.start, self.end, self.step))

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        """Get the slice index.

        Args:
            idx: The index.

        Returns:
            The slice index.
        """
        return self.start + idx * self.step

    @always_inline("nodebug")
    fn _has_end(self) -> Bool:
        return self.end != _int_max_value()


@value
@register_passable("trivial")
struct RawSlice:
    """Represents a slice expression with optional indices.
    Objects of this type are generated when slice syntax is used within
    square brackets, but some or all of the indices may be optional.
    Optional indices are represented as `None` values.
    """

    var start: OptionalReg[Int]
    """The starting index of the slice, or None."""
    var end: OptionalReg[Int]
    """The end index of the slice, or None."""
    var step: Int
    """The step increment value of the slice."""

    @always_inline("nodebug")
    fn __init__(
        inout self,
        start: OptionalReg[Int],
        end: OptionalReg[Int],
        step: OptionalReg[Int],
    ):
        """Construct slice given optional start and end indices along with a step value.

        Args:
            start: The optional start value.
            end: The optional end value.
            step: The step value.
        """
        self.start = start
        self.end = end
        self.step = step.value() if step else 1

    fn adjust(self, size: Int) -> Slice:
        """Convert a slice which may have None indices to one with numeric indices.

        Args:
            size: The size of the array/sequence being sliced.

        Returns:
            A Slice with numeric indices.  The indices will be non-negative integers
            representing actual indices within the length of the slice exept in the case where
            step is negative and end must be negative to allow inclusion of the element at index 0.
        """
        debug_assert(self.step == 0, "Step cannot be zero")

        var step = self.step
        var default_start = 0 if step > 0 else size - 1
        var default_end = size if step > 0 else -1
        var start: Int = default_start
        var end: Int = default_end
        if self.start:
            start = self.start.value()
            if start < 0:
                start += size

        if self.end:
            end = self.end.value()
            if end < 0:
                end += size

        if step < 0:
            start = start if start < default_start else default_start
        else:
            end = end if end < default_end else default_end

        return Slice(start, end, self.step)


@always_inline("nodebug")
fn slice(end: Int) -> Slice:
    """Construct slice given the end value.

    Args:
        end: The end value.

    Returns:
        The constructed slice.
    """
    return Slice(0, end)


@always_inline("nodebug")
fn slice(start: Int, end: Int) -> Slice:
    """Construct slice given the start and end values.

    Args:
        start: The start value.
        end: The end value.

    Returns:
        The constructed slice.
    """
    return Slice(start, end)


# TODO(30496): Modernize the slice type
@always_inline("nodebug")
fn slice[
    T0: AnyRegType, T1: AnyRegType, T2: AnyRegType
](start: T0, end: T1, step: T2) -> Slice:
    """Construct a Slice given the start, end and step values.

    Parameters:
        T0: Type of the start value.
        T1: Type of the end value.
        T2: Type of the step value.

    Args:
        start: The start value.
        end: The end value.
        step: The step value.

    Returns:
        The constructed slice.
    """
    return Slice(start, end, step)
