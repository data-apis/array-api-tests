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
import regex
from collections import defaultdict

SIGNATURE_RE = regex.compile(r'#+ (?:<.*>) ?(.*\(.*\))')
CONSTANT_RE = regex.compile(r'#+ (?:<.*>) ?([^\(\n]*)\n')
NAME_RE = regex.compile(r'(.*)\(.*\)')

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
            special_values = parse_special_values(text, verbose=True)
            for func in special_values:
                for typ in special_values[func]:
                    multiple = len(special_values[func][typ]) > 1
                    for i, m in enumerate(special_values[func][typ], 1):
                        test_name_extra = typ.lower()
                        if multiple:
                            test_name_extra += f"_{i}"
                        try:
                            print(generate_special_value_test(func, typ, m,
                                                              test_name_extra))
                        except:
                            print(f"Error with {func}() {typ}: {m.group(0)}:\n")
                            raise

        signatures = SIGNATURE_RE.findall(text)
        constants = CONSTANT_RE.findall(text)
        if not (signatures or constants):
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

            for const in constants:
                if not args.quiet:
                    print(f"Writing stub for {const}")
                f.write(f"\n{const} = None\n")
                modules[module_name].append(const)

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

# (?|...) is a branch reset (regex module only feature). It works like (?:...)
# except only the matched alternative is assigned group numbers, so \1, \2, and
# so on will always refer to a single match from _value.
_value = r"(?|`([^`]*)`|(finite)|(positive)|(negative)|(nonzero)|(a nonzero finite number)|(an integer value)|(an odd integer value)|an implementation-dependent approximation to `([^`]*)`(?: \(rounded\))?|a (signed (?:infinity|zero)) with the sign determined by the rule already stated above)"
SPECIAL_VALUE_REGEXS = dict(
    ONE_ARG_EQUAL = regex.compile(rf'^- +If `x_i` is {_value}, the result is {_value}\.$'),
    ONE_ARG_GREATER = regex.compile(rf'^- +If `x_i` is greater than {_value}, the result is {_value}\.$'),
    ONE_ARG_LESS = regex.compile(rf'^- +If `x_i` is less than {_value}, the result is {_value}\.$'),
    ONE_ARG_ALREADY_INTEGER_VALUED = regex.compile(rf'^- +If `x_i` is already integer-valued, the result is {_value}\.$'),
    ONE_ARG_EITHER = regex.compile(rf'^- +If `x_i` is either {_value} or {_value}, the result is {_value}\.$'),

    TWO_ARGS_EQUAL__EQUAL = regex.compile(rf'^- +If `x1_i` is {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_GREATER__EQUAL = regex.compile(rf'^- +If `x1_i` is greater than {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_GREATER_EQUAL__EQUAL = regex.compile(rf'^- +If `x1_i` is greater than {_value}, `x1_i` is {_value}, and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_LESS__EQUAL = regex.compile(rf'^- +If `x1_i` is less than {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_LESS_EQUAL__EQUAL = regex.compile(rf'^- +If `x1_i` is less than {_value}, `x1_i` is {_value}, and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_LESS_EQUAL__EQUAL_NOTEQUAL = regex.compile(rf'^- +If `x1_i` is less than {_value}, `x1_i` is {_value}, `x2_i` is {_value}, and `x2_i` is not {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__GREATER = regex.compile(rf'^- +If `x1_i` is {_value} and `x2_i` is greater than {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__LESS = regex.compile(rf'^- +If `x1_i` is {_value} and `x2_i` is less than {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__NOTEQUAL = regex.compile(rf'^- +If `x1_i` is {_value} and `x2_i` is not (?:equal to )?{_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__LESS_EQUAL = regex.compile(rf'^- +If `x1_i` is {_value}, `x2_i` is less than {_value}, and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__LESS_NOTEQUAL = regex.compile(rf'^- +If `x1_i` is {_value}, `x2_i` is less than {_value}, and `x2_i` is not {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__GREATER_EQUAL = regex.compile(rf'^- +If `x1_i` is {_value}, `x2_i` is greater than {_value}, and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__GREATER_NOTEQUAL = regex.compile(rf'^- +If `x1_i` is {_value}, `x2_i` is greater than {_value}, and `x2_i` is not {_value}, the result is {_value}\.$'),
    TWO_ARGS_NOTEQUAL__EQUAL = regex.compile(rf'^- +If `x1_i` is not equal to {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_ABS_EQUAL__EQUAL = regex.compile(rf'^- +If `abs\(x1_i\)` is {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_ABS_GREATER__EQUAL = regex.compile(rf'^- +If `abs\(x1_i\)` is greater than {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_ABS_LESS__EQUAL = regex.compile(rf'^- +If `abs\(x1_i\)` is less than {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EITHER = regex.compile(rf'^- +If either `x1_i` or `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EITHER__EQUAL = regex.compile(rf'^- +If `x1_i` is either {_value} or {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__EITHER = regex.compile(rf'^- +If `x1_i` is {_value} and `x2_i` is either {_value} or {_value}, the result is {_value}\.$'),
    TWO_ARGS_EITHER__EITHER = regex.compile(rf'^- +If `x1_i` is either {_value} or {_value} and `x2_i` is either {_value} or {_value}, the result is {_value}\.$'),
    TWO_ARGS_SAME_SIGN = regex.compile(rf'^- +If both `x1_i` and `x2_i` have the same sign, the result is {_value}\.$'),
    TWO_ARGS_DIFFERENT_SIGNS = regex.compile(rf'^- +If `x1_i` and `x2_i` have different signs, the result is {_value}\.$'),
    TWO_ARGS_EVEN_IF = regex.compile(rf'^- +If `x2_i` is {_value}, the result is {_value}, even if `x1_i` is {_value}\.$'),

    TWO_INTEGERS_EQUALLY_CLOSE = regex.compile(rf'^- +If two integers are equally close to `x_i`, the result is whichever integer is farthest from {_value}\.$'),
    REMAINING = regex.compile(r"^- +In the remaining cases, (.*)$"),
)


ONE_ARG_TEMPLATE = """
{decorator}
def test_{func}_special_values_{test_name_extra}(arg1):
    {doc}
    mask = {mask}
    assert_equal({func}(arg1)[mask], {result})
"""

TWO_ARGS_TEMPLATE = """
{decorator}
def test_{func}_special_values_{test_name_extra}(arg1, arg2):
    {doc}
    mask = {mask}
    assert_{assert_type}({func}(arg1, arg2)[mask], {result})
"""

TWO_INTEGERS_EQUALLY_CLOSE = "# TODO: Implement TWO_INTEGERS_EQUALLY_CLOSE"
REMAINING = "# TODO: Implement REMAINING"

def parse_values(values, arg):
    res = []
    for value in values:
        if value == 'NaN':
            res.append(f"NaN({arg}.dtype)")
        elif value == "+infinity":
            res.append(f"infinity({arg}.dtype)")
        elif value == "-infinity":
            res.append(f"-infinity({arg}.dtype)")
        elif value in ["0", "+0"]:
            res.append(f"zero({arg}.dtype)")
        elif value == "-0":
            res.append(f"-zero({arg}.dtype)")
        elif value in ["1", "+1"]:
            res.append(f"one({arg}.dtype)")
        elif value == "-1":
            res.append(f"-one({arg}.dtype)")
        elif 'π' in value:
            res.append(value.replace('π', f'π({arg}.dtype)'))
        else:
            raise RuntimeError(f"Unexpected input value {value}")
    return res

def generate_special_value_test(func, typ, m, test_name_extra):
    if typ.startswith("ONE_ARG"):
        decorator = "@given(numeric_arrays)"
        doc = f'''"""
    Special value test for {func}(x):

        {m.group(0)}

    """
'''
        if typ == "ONE_ARG_EQUAL":
            value1, result = parse_values(m.groups(), 'arg1')
            mask = f"equal(arg1, {value1})"
        elif typ == "ONE_ARG_GREATER":
            value1, result = parse_values(m.groups(), 'arg1')
            mask = f"greater(arg1, {value1})"
        elif typ == "ONE_ARG_LESS":
            value1, result = parse_values(m.groups(), 'arg1')
            mask = f"less(arg1, {value1})"
        elif typ == "ONE_ARG_EITHER":
            value1, value2, result = parse_values(m.groups(), 'arg1')
            mask = f"logical_or(equal(arg1, {value1}), equal(arg1, {value2}))"
        elif typ == "ONE_ARG_ALREADY_INTEGER_VALUED":
            return
        else:
            raise ValueError(f"Unrecognized special value type {typ}")
        return ONE_ARG_TEMPLATE.format(
            decorator=decorator,
            func=func,
            test_name_extra=test_name_extra,
            doc=doc,
            mask=mask,
            result=result,
        )

    if typ == "TWO_ARGS_EQUAL__EQUAL":
        pass
    if typ == "TWO_ARGS_GREATER__EQUAL":
        pass
    if typ == "TWO_ARGS_GREATER_EQUAL__EQUAL":
        pass
    if typ == "TWO_ARGS_LESS__EQUAL":
        pass
    if typ == "TWO_ARGS_LESS_EQUAL__EQUAL":
        pass
    if typ == "TWO_ARGS_LESS_EQUAL__EQUAL_NOTEQUAL":
        pass
    if typ == "TWO_ARGS_EQUAL__GREATER":
        pass
    if typ == "TWO_ARGS_EQUAL__LESS":
        pass
    if typ == "TWO_ARGS_EQUAL__NOTEQUAL":
        pass
    if typ == "TWO_ARGS_EQUAL__LESS_EQUAL":
        pass
    if typ == "TWO_ARGS_EQUAL__LESS_NOTEQUAL":
        pass
    if typ == "TWO_ARGS_EQUAL__GREATER_EQUAL":
        pass
    if typ == "TWO_ARGS_EQUAL__GREATER_NOTEQUAL":
        pass
    if typ == "TWO_ARGS_NOTEQUAL__EQUAL":
        pass
    if typ == "TWO_ARGS_ABS_EQUAL__EQUAL":
        pass
    if typ == "TWO_ARGS_ABS_GREATER__EQUAL":
        pass
    if typ == "TWO_ARGS_ABS_LESS__EQUAL":
        pass
    if typ == "TWO_ARGS_EITHER":
        pass
    if typ == "TWO_ARGS_EITHER__EQUAL":
        pass
    if typ == "TWO_ARGS_EQUAL__EITHER":
        pass
    if typ == "TWO_ARGS_EITHER__EITHER":
        pass
    if typ == "TWO_ARGS_SAME_SIGN":
        pass
    if typ == "TWO_ARGS_DIFFERENT_SIGNS":
        pass
    if typ == "TWO_ARGS_EVEN_IF":
        pass

    if typ == "TWO_INTEGERS_EQUALLY_CLOSE":
        pass
    if typ == "REMAINING":
        pass

def parse_special_values(spec_text, verbose=False):
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
                    if verbose:
                        print(f"Matched {typ} for {name}: {m.groups()}")
                    special_values[name][typ].append(m)
                    break
            else:
                raise ValueError(f"Unrecognized special value string for '{name}':\n{line}")

    return special_values

if __name__ == '__main__':
    main()
