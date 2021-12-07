## Coverage of the Test Suite

This document outlines the coverage of the test suite over the
[spec](https://data-apis.org/array-api/) at a high level.

The following things are tested

* **Smoke tested** means that the function has a basic test that calls the
  function with some inputs, but does not imply any testing of the output
  value. This includes calling keyword arguments to the function, and checking
  that it takes the correct number of positional arguments. A smoke test will
  fail if the function is not implemented with the correct signature or raises
  an exception, but will not check any other aspect of the spec.

* **All Inputs** means that the function is tested with all possible inputs
  required by the spec (using hypothesis). This means all possible array
  shapes, all possible dtypes (that are required for the given function), and
  all possible values for the given dtype (omitting those whose behavior is
  undefined).

* **Output Shape** means that the result shape is tested. For functions that
  take more than one argument, this means the result shape should produced
  from
  [broadcasting](https://data-apis.org/array-api/latest/API_specification/broadcasting.html)
  the input shapes. For functions of a single argument, the result shape
  should be the same as the input shape.

* **Output Dtype** means that the result dtype is tested. For (most) functions
  with a single argument, the result dtype should be the same as the input.
  For functions with two arguments, there are different possibilities, such as
  performing [type
  promotion](https://data-apis.org/array-api/latest/API_specification/type_promotion.html)
  or always returning a specific dtype (e.g., `equals()` should always return
  a `bool` array).

* **Output Values** means that the exact output is tested in some way. For
  functions that operate on floating-point inputs, the spec does not require
  exact values, so a "Yes" in this case will mean only that the output value
  is checked to be "close" to the numerically correct result. The exception to
  this is special cases for elementwise functions, which are tested exactly.
  For functions that operate on non-floating-point inputs, or functions like
  manipulation functions or indexing that simply rearrange the same values of
  the input arrays, a "Yes" means that the exact values are tested. Note that
  in many cases, certain values of inputs are left unspecified, and are thus
  not tested (e.g., the behavior for division by integer 0 is unspecified).

* **Stacking** means that functions that operate on "stacks" of smaller data
  are tested to produce the same result on a stack as on the individual
  components. For example, an elementwise function on an array
  should produce the same output values as the same function called on each
  value individually, or a linalg function on a stack of matrices should
  produce the same value when called on individual matrices. Here "same" may
  only mean "close" when the input values are floating-point.

## Statistical Functions

|----------|------------|------------|--------------|--------------|---------------|----------|
| Function | Smoke Test | All Inputs | Output Shape | Result Dtype | Output Values | Stacking |
|----------|------------|------------|--------------|--------------|---------------|----------|
| max      | Yes        | Yes        | Yes          | Yes          |               |          |
| mean     | Yes        | Yes        | Yes          | Yes          |               |          |
| min      | Yes        | Yes        | Yes          | Yes          |               |          |
| prod     | Yes        | Yes        | Yes          | Yes (1)      |               |          |
| std      | Yes        | Yes        | Yes          | Yes          |               |          |
| sum      | Yes        | Yes        | Yes          | Yes (1)      |               |          |
| var      | Yes        | Yes        | Yes          | Yes          |               |          |

(1): `sum` and `prod` have special type promotion rules.
