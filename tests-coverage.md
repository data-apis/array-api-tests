## What this suite actually tests

`array-api-tests` tests that an array library adopting the [standard](https://data-apis.org/array-api/) is indeed covering everything that is in scope.

## Primary tests

Every function—including array object methods—has a respective test method. We use [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) to generate a diverse set of valid inputs. This means array inputs will cover different dtypes and shapes, as well as contain interesting elements. These examples generate with interesting arrangements of non-array positional arguments and keyword arguments.

Each test case will cover the following areas if relevant:

* **Smoking**: We pass our generated examples to all functions. As these examples solely consist of *valid* inputs, we are testing that functions can be called using their documented inputs without raising errors.

* **Data type**: For functions returning/modifying arrays, we assert that output arrays have the correct data types. Most functions [type-promote](https://data-apis.org/array-api/latest/API_specification/type_promotion.html) input arrays and some functions have bespoke rules—in both cases we simulate the correct behaviour to find the expected data types.

* **Shape**: For functions returning/modifying arrays, we assert that output arrays have the correct shape. Most functions [broadcast](https://data-apis.org/array-api/latest/API_specification/broadcasting.html) input arrays and some functions have bespoke rules—in both cases we simulate the correct behaviour to find the expected shapes.

* **Values**: We assert output values (including the elements of returned/modified arrays) are as expected. Except for manipulation functions or special cases, the spec allows floating-point inputs to have inexact outputs, so with such examples we only assert values are roughly as expected.

## Additional tests

In addition to having one test case for each function, we test other properties of the functions and some miscellaneous things.

* **Special cases**: For functions with special case behaviour, we assert that these functions return the correct values.

* **Signatures**: We assert functions have the correct signatures.

* **Constants**: We assert that [constants](https://data-apis.org/array-api/latest/API_specification/constants.html) behave expectedly, are roughly the expected value, and that any related functions interact with them correctly.


TODO: future plans
