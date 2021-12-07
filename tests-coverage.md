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

| Function | Smoke Test | All Inputs | Output Shape | Result Dtype | Output Values | Stacking |
|----------|------------|------------|--------------|--------------|---------------|----------|
| max      | Yes        | Yes        | Yes          | Yes          |               |          |
| mean     | Yes        | Yes        | Yes          | Yes          |               |          |
| min      | Yes        | Yes        | Yes          | Yes          |               |          |
| prod     | Yes        | Yes        | Yes          | Yes [^1]     |               |          |
| std      | Yes        | Yes        | Yes          | Yes          |               |          |
| sum      | Yes        | Yes        | Yes          | Yes [^1]     |               |          |
| var      | Yes        | Yes        | Yes          | Yes          |               |          |

[^1]: `sum` and `prod` have special type promotion rules.

## Additional Planned Features

In addition to getting full coverage of the spec, there are some additional
features and improvements for the test suite that are planned. Work on these features
will be guided primarily by concrete needs from library implementers, so if
you are someone using this test suite to test your library, please [let us
know](https://github.com/data-apis/array-api-tests/issues) the limitations you
come across.

- Making the test suite more usable for partially conforming libraries. Many
  tests rely on various functions in the array library to function. This means
  that if certain functions aren't implemented, for example, `asarray()` or
  `equals()`, then many tests will not function at all. We want to improve
  this situation, so that tests that don't strictly require these functions can
  still be run.

- Better reporting. The pytest output can be difficult to parse, especially
  when there are many failures. Additionally some error messages can be
  difficult to understand without prior knowledge of the test internals.
  Better reporting can also make it easier to compare different
  implementations by their conformance.

- Better tests for numerical outputs. Right now numerical outputs are either
  not tested at all, or only tested against very rough epsilons. This is
  partly due to the fact that the spec does not mandate any level of precision
  for most functions. However, it may be useful to, for instance, give a
  report of how off a given function is from the "expected" exact output.
