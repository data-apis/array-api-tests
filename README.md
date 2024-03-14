# Test Suite for Array API Compliance

This is the test suite for array libraries adopting the [Python Array API
standard](https://data-apis.org/array-api/latest).

Note the suite is still a **work in progress**. Feedback and contributions are
welcome!

## Quickstart

### Setup

Currently we pin the Array API specification repo [`array-api`](https://github.com/data-apis/array-api/)
as a git submodule. This might change in the future to better support vendoring
use cases (see [#107](https://github.com/data-apis/array-api-tests/issues/107)),
but for now be sure submodules are pulled too, e.g.

```bash
$ git submodule update --init
```

To run the tests, install the testing dependencies.

```bash
$ pip install -r requirements.txt
```

Ensure you have the array library that you want to test installed.

### Specifying the array module

You need to specify the array library to test. It can be specified via the
`ARRAY_API_TESTS_MODULE` environment variable, e.g.

```bash
$ export ARRAY_API_TESTS_MODULE=numpy.array_api
```

Alternately, import/define the `xp` variable in `array_api_tests/__init__.py`.

### Run the suite

Simply run `pytest` against the `array_api_tests/` folder to run the full suite.

```bash
$ pytest array_api_tests/
```

The suite tries to logically organise its tests. `pytest` allows you to only run
a specific test case, which is useful when developing functions.

```bash
$ pytest array_api_tests/test_creation_functions.py::test_zeros
```

## What the test suite covers

We are interested in array libraries conforming to the
[spec](https://data-apis.org/array-api/latest/API_specification/index.html).
Ideally this means that if a library has fully adopted the Array API, the test
suite passes. We take great care to _not_ test things which are out-of-scope,
so as to not unexpectedly fail the suite.

### Primary tests

Every function—including array object methods—has a respective test
method<sup>1</sup>. We use
[Hypothesis](https://hypothesis.readthedocs.io/en/latest/)
to generate a diverse set of valid inputs. This means array inputs will cover
different dtypes and shapes, as well as contain interesting elements. These
examples generate with interesting arrangements of non-array positional
arguments and keyword arguments.

Each test case will cover the following areas if relevant:

* **Smoking**: We pass our generated examples to all functions. As these
  examples solely consist of *valid* inputs, we are testing that functions can
  be called using their documented inputs without raising errors.

* **Data type**: For functions returning/modifying arrays, we assert that output
  arrays have the correct data types. Most functions
  [type-promote](https://data-apis.org/array-api/latest/API_specification/type_promotion.html)
  input arrays and some functions have bespoke rules—in both cases we simulate
  the correct behaviour to find the expected data types.

* **Shape**: For functions returning/modifying arrays, we assert that output
  arrays have the correct shape. Most functions
  [broadcast](https://data-apis.org/array-api/latest/API_specification/broadcasting.html)
  input arrays and some functions have bespoke rules—in both cases we simulate
  the correct behaviour to find the expected shapes.

* **Values**: We assert output values (including the elements of
  returned/modified arrays) are as expected. Except for manipulation functions
  or special cases, the spec allows floating-point inputs to have inexact
  outputs, so with such examples we only assert values are roughly as expected.

### Additional tests

In addition to having one test case for each function, we test other properties
of the functions and some miscellaneous things.

* **Special cases**: For functions with special case behaviour, we assert that
  these functions return the correct values.

* **Signatures**: We assert functions have the correct signatures.

* **Constants**: We assert that
  [constants](https://data-apis.org/array-api/latest/API_specification/constants.html)
  behave expectedly, are roughly the expected value, and that any related
  functions interact with them correctly.

Be aware that some aspects of the spec are impractical or impossible to actually
test, so they are not covered in the suite. <!-- TODO: note what these are -->

## Interpreting errors

First and foremost, note that most tests have to assume that certain aspects of
the Array API have been correctly adopted, as fundamental APIs such as array
creation and equalities are hard requirements for many assertions. This means a
test case for one function might fail because another function has bugs or even
no implementation.

This means adopting libraries at first will result in a vast number of errors
due to cascading errors. Generally the nature of the spec means many granular
details such as type promotion is likely going to also fail nearly-conforming
functions.

We hope to improve user experience in regards to "noisy" errors in
[#51](https://github.com/data-apis/array-api-tests/issues/51). For now, if an
error message involves `_UndefinedStub`, it means an attribute of the array
library (including functions) and it's objects (e.g. the array) is missing.

The spec is the suite's source of truth. If the suite appears to assume
behaviour different from the spec, or test something that is not documented,
this is a bug—please [report such
issues](https://github.com/data-apis/array-api-tests/issues/) to us.


## Running on CI

See our existing [GitHub Actions workflow for
Numpy](https://github.com/data-apis/array-api-tests/blob/master/.github/workflows/numpy.yml)
for an example of using the test suite on CI.

### Releases

We recommend pinning against a [release tag](https://github.com/data-apis/array-api-tests/releases)
when running on CI.

We use [calender versioning](https://calver.org/) for the releases. You should
expect that any version may be "breaking" compared to the previous one, in that
new tests (or improvements to existing tests) may cause a previously passing
library to fail.

### Configuration

#### API version

You can specify the API version to use when testing via the
`ARRAY_API_TESTS_VERSION` environment variable. Currently this defaults to the
array module's `__array_api_version__` value, and if that attribute doesn't
exist then we fallback to `"2021.12"`.

#### Data-dependent shapes

Use the `--disable-data-dependent-shapes` flag to skip testing functions which have
[data-dependent shapes](https://data-apis.org/array-api/latest/design_topics/data_dependent_output_shapes.html).

#### Extensions

By default, tests for the optional Array API extensions such as
[`linalg`](https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html)
will be skipped if not present in the specified array module. You can purposely
skip testing extension(s) via the `--disable-extension` option.

#### Skip or XFAIL test cases

Test cases you want to skip can be specified in a skips or XFAILS file. The
difference between skip and XFAIL is that XFAIL tests are still run and
reported as XPASS if they pass.

By default, the skips and xfails files are `skips.txt` and `fails.txt` in the root
of this repository, but any file can be specified with the `--skips-file` and
`--xfails-file` command line flags.

The files should list the test ids to be skipped/xfailed. Empty lines and
lines starting with `#` are ignored. The test id can be any substring of the
test ids to skip/xfail.

```
# skips.txt or xfails.txt
# Line comments can be denoted with the hash symbol (#)

# Skip specific test case, e.g. when argsort() does not respect relative order
# https://github.com/numpy/numpy/issues/20778
array_api_tests/test_sorting_functions.py::test_argsort

# Skip specific test case parameter, e.g. you forgot to implement in-place adds
array_api_tests/test_add[__iadd__(x1, x2)]
array_api_tests/test_add[__iadd__(x, s)]

# Skip module, e.g. when your set functions treat NaNs as non-distinct
# https://github.com/numpy/numpy/issues/20326
array_api_tests/test_set_functions.py
```

Here is an example GitHub Actions workflow file, where the xfails are stored
in `array-api-tests.xfails.txt` in the base of the `your-array-library` repo.

If you want, you can use `-o xfail_strict=True`, which causes XPASS tests (XFAIL
tests that actually pass) to fail the test suite. However, be aware that
XFAILures can be flaky (see below, so this may not be a good idea unless you
use some other mitigation of such flakyness).

If you don't want this behavior, you can remove it, or use `--skips-file`
instead of `--xfails-file`.

```yaml
# ./.github/workflows/array_api.yml
jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']

    steps:
    - name: Checkout <your array library>
      uses: actions/checkout@v3
      with:
        path: your-array-library

    - name: Checkout array-api-tests
      uses: actions/checkout@v3
      with:
        repository: data-apis/array-api-tests
        submodules: 'true'
        path: array-api-tests

    - name: Run the array API test suite
      env:
        ARRAY_API_TESTS_MODULE: your.array.api.namespace
      run: |
        export PYTHONPATH="${GITHUB_WORKSPACE}/your-array-library"
        cd ${GITHUB_WORKSPACE}/array-api-tests
        pytest -v -rxXfE --ci --xfails-file ${GITHUB_WORKSPACE}/your-array-library/array-api-tests-xfails.txt array_api_tests/
```

> **Warning**
>
> XFAIL tests that use Hypothesis (basically every test in the test suite except
> those in test_has_names.py) can be flaky, due to the fact that Hypothesis
> might not always run the test with an input that causes the test to fail.
> There are several ways to avoid this problem:
>
> - Increase the maximum number of examples, e.g., by adding `--max-examples
>   200` to the test command (the default is `100`, see below). This will
>   make it more likely that the failing case will be found, but it will also
>   make the tests take longer to run.
> - Don't use `-o xfail_strict=True`. This will make it so that if an XFAIL
>   test passes, it will alert you in the test summary but will not cause the
>   test run to register as failed.
> - Use skips instead of XFAILS. The difference between XFAIL and skip is that
>   a skipped test is never run at all, whereas an XFAIL test is always run
>   but ignored if it fails.
> - Save the [Hypothesis examples
>   database](https://hypothesis.readthedocs.io/en/latest/database.html)
>   persistently on CI. That way as soon as a run finds one failing example,
>   it will always re-run future runs with that example. But note that the
>   Hypothesis examples database may be cleared when a new version of
>   Hypothesis or the test suite is released.

#### Max examples

The tests make heavy use
[Hypothesis](https://hypothesis.readthedocs.io/en/latest/). You can configure
how many examples are generated using the `--max-examples` flag, which
defaults to `100`. Lower values can be useful for quick checks, and larger
values should result in more rigorous runs. For example, `--max-examples
10_000` may find bugs where default runs don't but will take much longer to
run.


## Contributing

### Remain in-scope

It is important that every test only uses APIs that are part of the standard.
For instance, when creating input arrays you should only use the [array creation
functions](https://data-apis.org/array-api/latest/API_specification/creation_functions.html)
that are documented in the spec. The same goes for testing arrays—you'll find
many utilities that parralel NumPy's own test utils in the `*_helpers.py` files.

### Tools

Hypothesis should almost always be used for the primary tests, and can be useful
elsewhere. Effort should be made so drawn arguments are labeled with their
respective names. For
[`st.data()`](https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.data),
draws should be accompanied with the `label` kwarg i.e. `data.draw(<strategy>,
label=<label>)`.

[`pytest.mark.parametrize`](https://docs.pytest.org/en/latest/how-to/parametrize.html)
should be used to run tests over multiple arguments. Parameterization should be
preferred over using Hypothesis when there are a small number of possible
inputs, as this allows better failure reporting. Note using both parametrize and
Hypothesis for a single test method is possible and can be quite useful.

### Error messages

Any assertion should be accompanied with a descriptive error message, including
the relevant values. Error messages should be self-explanatory as to why a given
test fails, as one should not need prior knowledge of how the test is
implemented.

### Generated files

Some files in the suite are automatically generated from the spec, and should
not be edited directly. To regenerate these files, run the script

    ./generate_stubs.py path/to/array-api

where `path/to/array-api` is the path to a local clone of the [`array-api`
repo](https://github.com/data-apis/array-api/). Edit `generate_stubs.py` to make
changes to the generated files.


### Release

To make a release, first make an annotated tag with the version, e.g.:

```
git tag -a 2022.01.01
```

Be sure to use the calver version number for the tag name. Don't worry too much
on the tag message, e.g. just write "2022.01.01".

Versioneer will automatically set the version number of the `array_api_tests`
package based on the git tag. Push the tag to GitHub:

```
git push --tags upstream 2022.1
```

Then go to the [tags page on
GitHub](https://github.com/data-apis/array-api-tests/tags) and convert the tag
into a release. If you want, you can add release notes, which GitHub can
generate for you.


## Future plans

Keeping full coverage of the spec is an on-going priority as the Array API
evolves.

Additionally, we have features and general improvements planned. Work on such
functionality is guided primarily by the concerete needs of developers
implementing and using the Array API—be sure to [let us
know](https://github.com/data-apis/array-api-tests/issues) any limitations you
come across.

* A dependency graph for every test case, which could be used to modify pytest's
  collection so that low-dependency tests are run first, and tests with faulty
  dependencies would skip/xfail.

* In some tests we've found it difficult to find appropaite assertion parameters
  for output values (particularly epsilons for floating-point outputs), so we
  need to review these and either implement assertions or properly note the lack
  thereof.

---

<sup>1</sup>The only exceptions to having just one primary test per function are:

* [`asarray()`](https://data-apis.org/array-api/latest/API_specification/generated/signatures.creation_functions.asarray.html),
  which is tested by `test_asarray_scalars` and `test_asarray_arrays` in
  `test_creation_functions.py`. Testing `asarray()` works with scalars (and
  nested sequences of scalars) is fundamental to testing that it works with
  arrays, as said arrays can only be generated by passing scalar sequences to
  `asarray()`.

* Indexing methods
  ([`__getitem__()`](https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.__getitem__.html)
  and
  [`__setitem__()`](https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.__setitem__.html)),
  which respectively have both a test for non-array indices and a test for
  boolean array indices. This is because [masking is
  opt-in](https://data-apis.org/array-api/latest/API_specification/indexing.html#boolean-array-indexing)
  (and boolean arrays need to be generated by indexing arrays anyway).
