# Array API Standard Test Suite

This is the test suite for the PyData Array APIs standard.

**NOTE: This test suite is still a work in progress.**

Feedback and contributions are welcome, but be aware that this suite is not
yet completed. In particular, there are still many parts of the array API
specification that are not yet tested here.

## Running the tests

To run the tests, first install the testing dependencies

    pip install pytest hypothesis

or

    conda install pytest hypothesis

as well as the array libraries that you want to test. To run the tests, you
need to set the array library that is to be tested. There are two ways to do
this. One way is to set the `ARRAY_API_TESTS_MODULE` environment variable. For
example

    ARRAY_API_TESTS_MODULE=numpy pytest

Alternately, edit the `array_api_tests/_array_module.py` file and change the
line

```py
array_module = None
```

to

```
import numpy as array_module
```

(replacing `numpy` with the array module namespace to be tested).

## Notes on Interpreting Errors

- Some tests cannot be run unless other tests pass first. This is because very
  basic APIs such as certain array creation APIs are required for a large
  fraction of the tests to run. TODO: Write which tests are required to pass
  first here.

- If an error message involves `_UndefinedStub`, it means some name that is
  required for the test to run is not defined in the array library.

- Due to the nature of the array api spec, virtually every array library will
  produce a large number of errors from nonconformance. It is still a work in
  progress to enable reporting the errors in a way that makes them easy to
  understand, even if there are a large number of them.

- The spec documents are the ground source of truth. If the test suite appears
  to be testing something that is different from the spec, or something that
  isn't actually mentioned in the spec, this is a bug. [Please report
  it](https://github.com/data-apis/array-api-tests/issues/new). Furthermore,
  be aware that some aspects of the spec are either impossible or extremely
  difficult to actually test, so they are not covered in the test suite (TODO:
  list what these are).

## Contributing

### Adding Tests

It is important that every test in the test suite only uses APIs that are part
of the standard. This means that, for instance, when creating test arrays, you
should only use array creation functions that are part of the spec, such as
`ones` or `full`. It also means that many array testing functions that are
built-in to libraries like numpy are reimplemented in the test suite (see
`array_api_tests/pytest_helpers.py` and
`array_api_tests/hypothesis_helpers.py`).

In order to enforce this, the `array_api_tests._array_module` should be used
everywhere in place of the actual array module that is being tested.

### Hypothesis

The test suite uses [Hypothesis](https://hypothesis.readthedocs.io/en/latest/)
to generate random input data. Any test that should be applied over all
possible array inputs should use hypothesis tests. Custom Hypothesis
strategies are in the `array_api_tests/hypothesis_helpers.py` file.

### Parameterization

Any test that applies over all functions in a module should use
`pytest.mark.parametrize` to parameterize over them. For example,

```py
from . import function_stubs

@pytest.mark.parametrize('name', function_stubs.__all__)
def test_whatever(name):
    ...
```

will parameterize `test_whatever` over all the functions stubs generated from
the spec. Parameterization should be preferred over using Hypothesis whenever
there are a finite number of input possibilities, as this will cause pytest to
report failures for all input values separately, as opposed to Hypothesis
which will only report one failure.

### Error Strings

Any assertion or exception should be accompanied with a useful error message.
The test suite is designed to be ran by people who are not familiar with the
test suite code, so the error messages should be self explanatory as to why
the module fails a given test.

### Meta-errors

Any error that indicates a bug in the test suite itself, rather than in the
array module not following the spec, should use `RuntimeError` whenever
possible.

(TODO: Update this policy to something better. See [#5](https://github.com/data-apis/array-api-tests/issues/5).)

### Automatically Generated Files

Some files in the test suite are automatically generated from the API spec
files. These files should not be edited directly. To regenerate these files,
run the script

    ./generate_stubs.py path/to/array-api

where `path/to/array-api` is the path to the local clone of the `array-api`
repo. To modify the automatically generated files, edit the code that
generates them in the `generate_stubs.py` script.
