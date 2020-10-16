#!/usr/bin/env python
"""
Generate stub files for the tests.

To run the script, first clone the https://github.com/data-apis/array-api
repo, then run

./generate_stubs.py path/to/clone/of/array-api

This will update the stub files in array_api_tests/function_stubs/
"""
import argparse
import os
import re
from collections import defaultdict

SIGNATURE_RE = re.compile(r'#+ (?:<.*>) ?(.*\(.*\))')
NAME_RE = re.compile(r'(.*)\(.*\)')

STUB_FILE_HEADER = '''\
"""
Function stubs for {title}.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.

See
https://github.com/data-apis/array-api/blob/master/spec/API_specification/{filename}

Note, all non-keyword-only arguments are positional-only. We don't include that
here because

1. The /, syntax for positional-only arguments is Python 3.8+ only, and
2. There is no real way to test that anyway.
"""
'''

INIT_HEADER = '''\
"""
Stub definitions for functions defined in the spec

These are used to test function signatures.

NOTE: This file is generated automatically by the generate_stubs.py script. Do
not modify it directly.
"""

__all__ = []
'''

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('array_api_repo', help="Path to clone of the array-api repository")
    parser.add_argument('--no-write', help="""Print what it would do but don't
    write any files""", action='store_false', dest='write')
    parser.add_argument('--quiet', help="""Don't print any output to the terminal""", action='store_true', dest='quiet')
    args = parser.parse_args()

    spec_dir = os.path.join(args.array_api_repo, 'spec', 'API_specification')
    modules = {}
    for filename in sorted(os.listdir(spec_dir)):
        with open(os.path.join(spec_dir, filename)) as f:
                  text = f.read()
        if filename == 'elementwise_functions.md':
            special_values = parse_special_values(text)

        signatures = SIGNATURE_RE.findall(text)
        if not signatures:
            continue
        if not args.quiet:
            print(f"Found signatures in {filename}")
        if not args.write:
            continue
        py_file = filename.replace('.md', '.py')
        py_path = os.path.join('array_api_tests', 'function_stubs', py_file)
        title = filename.replace('.md', '').replace('_', ' ')
        module_name = py_file.replace('.py', '')
        modules[module_name] = []
        if not args.quiet:
            print(f"Writing {py_path}")
        with open(py_path, 'w') as f:
            f.write(STUB_FILE_HEADER.format(filename=filename, title=title))
            for sig in signatures:
                if not args.quiet:
                    print(f"Writing stub for {sig}")
                f.write(f"""
def {sig.replace(', /', '')}:
    pass
""")
                func_name = NAME_RE.match(sig).group(1)
                modules[module_name].append(func_name)

            f.write('\n__all__ = [')
            f.write(', '.join(f"'{i}'" for i in modules[module_name]))
            f.write(']\n')

    init_path = os.path.join('array_api_tests', 'function_stubs', '__init__.py')
    if args.write:
        with open(init_path, 'w') as f:
            f.write(INIT_HEADER)
            for module_name in modules:
                f.write(f"\nfrom .{module_name} import ")
                f.write(', '.join(modules[module_name]))
                f.write('\n\n')
                f.write('__all__ += [')
                f.write(', '.join(f"'{i}'" for i in modules[module_name]))
                f.write(']\n')

_value = r"(?:`([^`]*)`|(finite)|(positive)|(negative)|(nonzero)|(a nonzero finite number)|(an integer value)|an implementation-dependent approximation to `([^`]*)`(?: \(rounded\))?|a (signed (?:infinity|zero)) with the sign determined by the rule already stated above)"
SPECIAL_VALUE_REGEXS = dict(
    ONE_ARG_EQUAL = re.compile(rf'- +If `x_i` is ?{_value}, the result is {_value}\.'),
    ONE_ARG_GREATER = re.compile(rf'- +If `x_i` is greater than {_value}, the result is {_value}\.'),
    ONE_ARG_LESS = re.compile(rf'- +If `x_i` is less than {_value}, the result is {_value}\.'),
    ONE_ARG_ALREADY_INTEGER_VALUED = re.compile(rf'- +If `x_i` is already integer-valued, the result is {_value}\.'),
    ONE_ARG_EITHER = re.compile(rf'- +If `x_i` is either {_value} or {_value}, the result is {_value}\.'),

    TWO_ARGS_EQUAL_EQUAL = re.compile(rf'- +If `x1_i` is {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_GREATER_EQUAL = re.compile(rf'- +If `x1_i` is greater than {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_GREATER_FINITE_EQUAL = re.compile(rf'- +If `x1_i` is greater than {_value}, `x1_i` is finite, and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_LESS_EQUAL = re.compile(rf'- +If `x1_i` is less than {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_LESS_FINITE_EQUAL = re.compile(rf'- +If `x1_i` is less than {_value}, `x1_i` is finite, and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_LESS_FINITE_EQUAL_NOTEQUAL = re.compile(rf'- +If `x1_i` is less than {_value}, `x1_i` is finite, `x2_i` is {_value}, and `x2_i` is not {_value}, the result is {_value}\.'),
    TWO_ARGS_EQUAL_GREATER = re.compile(rf'- +If `x1_i` is {_value} and `x2_i` is greater than {_value}, the result is {_value}\.'),
    TWO_ARGS_EQUAL_LESS = re.compile(rf'- +If `x1_i` is {_value} and `x2_i` is less than {_value}, the result is {_value}\.'),
    TWO_ARGS_EQUAL_NOTEQUAL = re.compile(rf'- +If `x1_i` is {_value} and `x2_i` is not (?:equal to )?{_value}, the result is {_value}\.'),
    TWO_ARGS_EQUAL_LESS_ODD = re.compile(rf'- +If `x1_i` is {_value}, `x2_i` is less than {_value}, and `x2_i` is an odd integer value, the result is {_value}\.'),
    TWO_ARGS_EQUAL_LESS_NOT_ODD = re.compile(rf'- +If `x1_i` is {_value}, `x2_i` is less than {_value}, and `x2_i` is not an odd integer value, the result is {_value}\.'),
    TWO_ARGS_EQUAL_GREATER_ODD = re.compile(rf'- +If `x1_i` is {_value}, `x2_i` is greater than {_value}, and `x2_i` is an odd integer value, the result is {_value}\.'),
    TWO_ARGS_EQUAL_GREATER_NOT_ODD = re.compile(rf'- +If `x1_i` is {_value}, `x2_i` is greater than {_value}, and `x2_i` is not an odd integer value, the result is {_value}\.'),
    TWO_ARGS_NOTEQUAL_EQUAL = re.compile(rf'- +If `x1_i` is not equal to {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_ABS_EQUAL_EQUAL = re.compile(rf'- +If `abs\(x1_i\)` is {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_ABS_GREATER_EQUAL = re.compile(rf'- +If `abs\(x1_i\)` is greater than {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_ABS_LESS_EQUAL = re.compile(rf'- +If `abs\(x1_i\)` is less than {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_EITHER = re.compile(rf'- +If either `x1_i` or `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_EITHER_1 = re.compile(rf'- +If `x1_i` is either {_value} or {_value} and `x2_i` is {_value}, the result is {_value}\.'),
    TWO_ARGS_EITHER_2 = re.compile(rf'- +If `x1_i` is {_value} and `x2_i` is either {_value} or {_value}, the result is {_value}\.'),
    TWO_ARGS_EITHER_12 = re.compile(rf'- +If `x1_i` is either {_value} or {_value} and `x2_i` is either {_value} or {_value}, the result is {_value}\.'),
    TWO_ARGS_SAME_SIGN = re.compile(rf'- +If both `x1_i` and `x2_i` have the same sign, the result is {_value}\.'),
    TWO_ARGS_DIFFERENT_SIGNS = re.compile(rf'- +If `x1_i` and `x2_i` have different signs, the result is {_value}\.'),
    TWO_ARGS_EVEN_IF_2 = re.compile(rf'- +If `x2_i` is {_value}, the result is {_value}, even if `x1_i` is {_value}\.'),

    TWO_INTEGERS_EQUALLY_CLOSE = re.compile(rf'- +If two integers are equally close to `x_i`, the result is whichever integer is farthest from {_value}\.'),
    REMAINING = re.compile(r"- +In the remaining cases, (.*)"),
)

def parse_special_values(spec_text):
    special_values = {}
    in_block = False
    for line in spec_text.splitlines():
        m = SIGNATURE_RE.match(line)
        if m:
            name = NAME_RE.match(m.group(1)).group(1)
            special_values[name] = defaultdict(list)
            continue
        if line == '#### Special Values':
            in_block = True
            continue
        elif line.startswith('#'):
            in_block = False
            continue
        if in_block:
            if '- ' not in line:
                continue
            for typ, reg in SPECIAL_VALUE_REGEXS.items():
                m = reg.match(line)
                if m:
                    special_values[name][typ].append(m)
                    break
            else:
                raise ValueError(f"Unrecognized special value string for '{name}':\n{line}")

    return special_values

if __name__ == '__main__':
    main()
