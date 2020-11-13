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
import ast
from collections import defaultdict

import regex

FUNCTION_HEADER_RE = regex.compile(r'\(function-(.*?)\)')
FUNCTION_RE = regex.compile(r'\(function-.*\)=\n#+ ?(.*\(.*\))')
METHOD_RE = regex.compile(r'\(method-.*\)=\n#+ ?(.*\(.*\))')
CONSTANT_RE = regex.compile(r'\(constant-.*\)=\n#+ ?(.*)')
ATTRIBUTE_RE = regex.compile(r'\(attribute-.*\)=\n#+ ?(.*)')
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
        functions = FUNCTION_RE.findall(text)
        methods = METHOD_RE.findall(text)
        constants = CONSTANT_RE.findall(text)
        attributes = ATTRIBUTE_RE.findall(text)
        if not (functions or methods or constants or attributes):
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
        sigs = {}
        with open(py_path, 'w') as f:
            f.write(STUB_FILE_HEADER.format(filename=filename, title=title))
            for sig in functions + methods:
                ismethod = sig in methods
                sig = sig.replace(r'\_', '_')
                func_name = NAME_RE.match(sig).group(1)
                doc = ""
                if ismethod:
                    doc = f'''
    """
    Note: {func_name} is a method of the array object.
    """'''
                if not args.quiet:
                    print(f"Writing stub for {sig}")
                sig = sig.replace(', /', '')
                f.write(f"""
def {sig}:{doc}
    pass
""")
                modules[module_name].append(func_name)
                sigs[func_name] = sig
            for const in constants + attributes:
                if not args.quiet:
                    print(f"Writing stub for {const}")
                isattr = const in attributes
                if isattr:
                    f.write(f"\n# Note: {const} is an attribute of the array object.")
                f.write(f"\n{const} = None\n")
                modules[module_name].append(const)

            f.write('\n__all__ = [')
            f.write(', '.join(f"'{i}'" for i in modules[module_name]))
            f.write(']\n')

        if filename == 'elementwise_functions.md':
            special_cases = parse_special_cases(text, verbose=True)
            for func in special_cases:
                for typ in special_cases[func]:
                    multiple = len(special_cases[func][typ]) > 1
                    for i, m in enumerate(special_cases[func][typ], 1):
                        test_name_extra = typ.lower()
                        if multiple:
                            test_name_extra += f"_{i}"
                        try:
                            test = generate_special_case_test(func, typ, m,
                                                              test_name_extra, sigs)
                            if test is None:
                                raise NotImplementedError("Special case test not implemented")
                            print(test)
                        except:
                            print(f"Error with {func}() {typ}: {m.group(0)}:\n")
                            raise


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
_value = r"(?|`([^`]*)`|a (finite) number|a (positive \(i\.e\., greater than `0`\) finite) number|a (negative \(i\.e\., less than `0`\) finite) number|(finite)|(positive)|(negative)|(nonzero)|(?:a )?(nonzero finite) numbers?|an (integer) value|already (integer)-valued|an (odd integer) value|an implementation-dependent approximation to `([^`]*)`(?: \(rounded\))?|a (signed (?:infinity|zero)) with the mathematical sign determined by the rule already stated above|(positive) mathematical sign|(negative) mathematical sign)"
SPECIAL_CASE_REGEXS = dict(
    ONE_ARG_EQUAL = regex.compile(rf'^- +If `x_i` is {_value}, the result is {_value}\.$'),
    ONE_ARG_GREATER = regex.compile(rf'^- +If `x_i` is greater than {_value}, the result is {_value}\.$'),
    ONE_ARG_LESS = regex.compile(rf'^- +If `x_i` is less than {_value}, the result is {_value}\.$'),
    ONE_ARG_EITHER = regex.compile(rf'^- +If `x_i` is either {_value} or {_value}, the result is {_value}\.$'),
    ONE_ARG_TWO_INTEGERS_EQUALLY_CLOSE = regex.compile(rf'^- +If two integers are equally close to `x_i`, the result is whichever integer is farthest from {_value}\.$'),

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
    TWO_ARGS_ABSEQUAL__EQUAL = regex.compile(rf'^- +If `abs\(x1_i\)` is {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_ABSGREATER__EQUAL = regex.compile(rf'^- +If `abs\(x1_i\)` is greater than {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_ABSLESS__EQUAL = regex.compile(rf'^- +If `abs\(x1_i\)` is less than {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EITHER = regex.compile(rf'^- +If either `x1_i` or `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EITHER__EQUAL = regex.compile(rf'^- +If `x1_i` is either {_value} or {_value} and `x2_i` is {_value}, the result is {_value}\.$'),
    TWO_ARGS_EQUAL__EITHER = regex.compile(rf'^- +If `x1_i` is {_value} and `x2_i` is either {_value} or {_value}, the result is {_value}\.$'),
    TWO_ARGS_EITHER__EITHER = regex.compile(rf'^- +If `x1_i` is either {_value} or {_value} and `x2_i` is either {_value} or {_value}, the result is {_value}\.$'),
    TWO_ARGS_SAME_SIGN = regex.compile(rf'^- +If `x1_i` and `x2_i` have the same mathematical sign, the result has a {_value}\.$'),
    TWO_ARGS_SAME_SIGN_BOTH = regex.compile(rf'^- +If `x1_i` and `x2_i` have the same mathematical sign and are both {_value}, the result has a {_value}\.$'),
    TWO_ARGS_DIFFERENT_SIGNS = regex.compile(rf'^- +If `x1_i` and `x2_i` have different mathematical signs, the result has a {_value}\.$'),
    TWO_ARGS_DIFFERENT_SIGNS_BOTH = regex.compile(rf'^- +If `x1_i` and `x2_i` have different mathematical signs and are both {_value}, the result has a {_value}\.$'),
    TWO_ARGS_EVEN_IF = regex.compile(rf'^- +If `x2_i` is {_value}, the result is {_value}, even if `x1_i` is {_value}\.$'),

    REMAINING = regex.compile(r"^- +In the remaining cases, (.*)$"),
)


def parse_value(value, arg):
    if value == 'NaN':
        return f"NaN({arg}.dtype)"
    elif value == "+infinity":
        return f"infinity({arg}.dtype)"
    elif value == "-infinity":
        return f"-infinity({arg}.dtype)"
    elif value in ["0", "+0"]:
        return f"zero({arg}.dtype)"
    elif value == "-0":
        return f"-zero({arg}.dtype)"
    elif value in ["1", "+1"]:
        return f"one({arg}.dtype)"
    elif value == "-1":
        return f"-one({arg}.dtype)"
    # elif value == 'signed infinity':
    elif value == 'signed zero':
        return f"zero({arg}.dtype))"
    elif 'π' in value:
        value = regex.sub(r'(\d+)π', r'\1*π', value)
        return value.replace('π', f'π({arg}.dtype)')
    elif 'x1_i' in value or 'x2_i' in value:
        return value
    elif value.startswith('where('):
        return value
    elif value in ['finite', 'nonzero', 'nonzero finite',
                   "integer", "odd integer", "positive",
                   "negative"]:
        return value
    # There's no way to remove the parenthetical from the matching group in
    # the regular expression.
    elif value == "positive (i.e., greater than `0`) finite":
        return "positive finite"
    elif value == 'negative (i.e., less than `0`) finite':
        return "negative finite"
    else:
        raise RuntimeError(f"Unexpected input value {value!r}")

def _check_exactly_equal(typ, value):
    if not typ == 'exactly_equal':
        raise RuntimeError(f"Unexpected mask type {typ}: {value}")

def get_mask(typ, arg, value):

    if typ.startswith("not"):
        return f"logical_not({get_mask(typ[len('not'):], arg, value)})"
    if typ.startswith("abs"):
        return get_mask(typ[len("abs"):], f"abs({arg})", value)
    if value == 'finite':
        _check_exactly_equal(typ, value)
        return f"isfinite({arg})"
    elif value == 'nonzero':
        _check_exactly_equal(typ, value)
        return f"nonzero({arg})"
    elif value == 'positive finite':
        _check_exactly_equal(typ, value)
        return f"logical_and(isfinite({arg}), ispositive({arg}))"
    elif value == 'negative finite':
        _check_exactly_equal(typ, value)
        return f"logical_and(isfinite({arg}), isnegative({arg}))"
    elif value == 'nonzero finite':
        _check_exactly_equal(typ, value)
        return f"logical_and(isfinite({arg}), nonzero({arg}))"
    elif value == 'positive':
        _check_exactly_equal(typ, value)
        return f"ispositive({arg})"
    elif value == 'negative':
        _check_exactly_equal(typ, value)
        return f"isnegative({arg})"
    elif value == 'integer':
        _check_exactly_equal(typ, value)
        return f"isintegral({arg})"
    elif value == 'odd integer':
        _check_exactly_equal(typ, value)
        return f"isodd({arg})"
    elif 'x_i' in value:
        return f"{typ}({arg}, {value.replace('x_i', 'arg1')})"
    elif 'x1_i' in value:
        return f"{typ}({arg}, {value.replace('x1_i', 'arg1')})"
    elif 'x2_i' in value:
        return f"{typ}({arg}, {value.replace('x2_i', 'arg2')})"
    return f"{typ}({arg}, {value})"

def get_assert(typ, result):
    if result == "signed infinity":
        _check_exactly_equal(typ, result)
        return "assert_isinf(res[mask])"
    elif result == "positive":
        _check_exactly_equal(typ, result)
        return "assert_positive(res[mask])"
    elif 'x_i' in result:
        return f"assert_{typ}(res[mask], {result.replace('x_i', 'arg1')})"
    elif 'x1_i' in result:
        return f"assert_{typ}(res[mask], {result.replace('x1_i', 'arg1')})"
    elif 'x2_i' in result:
        return f"assert_{typ}(res[mask], {result.replace('x2_i', 'arg2')})"
    # TODO: Get use something better than arg1 here for the arg
    result = parse_value(result, "arg1")
    try:
        # This won't catch all unknown values, but will catch some.
        ast.parse(result)
    except SyntaxError:
        raise RuntimeError(f"Unexpected result value {result!r} for {typ} (bad syntax)")
    return f"assert_{typ}(res[mask], {result})"

ONE_ARG_TEMPLATE = """
{decorator}
def test_{func}_special_cases_{test_name_extra}(arg1):
    {doc}
    res = {func}(arg1)
    mask = {mask}
    {assertion}
"""

TWO_ARGS_TEMPLATE = """
{decorator}
def test_{func}_special_cases_{test_name_extra}(arg1, arg2):
    {doc}
    res = {func}(arg1, arg2)
    mask = {mask}
    {assertion}
"""

REMAINING_TEMPLATE = "# TODO: Implement REMAINING"

def generate_special_case_test(func, typ, m, test_name_extra, sigs):

    doc = f'''"""
    Special case test for `{sigs[func]}`:

        {m.group(0)}

    """'''
    if typ.startswith("ONE_ARG"):
        decorator = "@given(numeric_arrays)"
        if typ == "ONE_ARG_EQUAL":
            value1, result = m.groups()
            value1 = parse_value(value1, 'arg1')
            mask = get_mask("exactly_equal", "arg1", value1)
        elif typ == "ONE_ARG_GREATER":
            value1, result = m.groups()
            value1 = parse_value(value1, 'arg1')
            mask = get_mask("greater", "arg1", value1)
        elif typ == "ONE_ARG_LESS":
            value1, result = m.groups()
            value1 = parse_value(value1, 'arg1')
            mask = get_mask("less", "arg1", value1)
        elif typ == "ONE_ARG_EITHER":
            value1, value2, result = m.groups()
            value1 = parse_value(value1, 'arg1')
            value2 = parse_value(value2, 'arg1')
            mask1 = get_mask("exactly_equal", "arg1", value1)
            mask2 = get_mask("exactly_equal", "arg1", value2)
            mask = f"logical_or({mask1}, {mask2})"
        elif typ == "ONE_ARG_ALREADY_INTEGER_VALUED":
            result, = m.groups()
            mask = parse_value("integer", "arg1")
        elif typ == "ONE_ARG_TWO_INTEGERS_EQUALLY_CLOSE":
            result, = m.groups()
            mask = "equal(subtract(arg1, floor(arg1)), subtract(ceiling(arg1), arg1))"
            result = parse_value(result, "arg1")
            result = f"where(greater(abs(subtract({result}, floor(arg1))), abs(subtract(ceiling(arg1), {result}))), floor(arg1), ceil(arg1))"
        else:
            raise ValueError(f"Unrecognized special value type {typ}")
        assertion = get_assert("exactly_equal", result)
        return ONE_ARG_TEMPLATE.format(
            decorator=decorator,
            func=func,
            test_name_extra=test_name_extra,
            doc=doc,
            mask=mask,
            assertion=assertion,
        )

    elif typ.startswith("TWO_ARGS"):
        decorator = "@given(numeric_arrays, numeric_arrays)"
        if typ in [
                "TWO_ARGS_EQUAL__EQUAL",
                "TWO_ARGS_GREATER__EQUAL",
                "TWO_ARGS_LESS__EQUAL",
                "TWO_ARGS_EQUAL__GREATER",
                "TWO_ARGS_EQUAL__LESS",
                "TWO_ARGS_EQUAL__NOTEQUAL",
                "TWO_ARGS_NOTEQUAL__EQUAL",
                "TWO_ARGS_ABSEQUAL__EQUAL",
                "TWO_ARGS_ABSGREATER__EQUAL",
                "TWO_ARGS_ABSLESS__EQUAL",
                "TWO_ARGS_GREATER_EQUAL__EQUAL",
                "TWO_ARGS_LESS_EQUAL__EQUAL",
                "TWO_ARGS_EQUAL__LESS_EQUAL",
                "TWO_ARGS_EQUAL__LESS_NOTEQUAL",
                "TWO_ARGS_EQUAL__GREATER_EQUAL",
                "TWO_ARGS_EQUAL__GREATER_NOTEQUAL",
                "TWO_ARGS_LESS_EQUAL__EQUAL_NOTEQUAL",
                "TWO_ARGS_EITHER__EQUAL",
                "TWO_ARGS_EQUAL__EITHER",
                "TWO_ARGS_EITHER__EITHER",
        ]:
            arg1typs, arg2typs = [i.split('_') for i in typ[len("TWO_ARGS_"):].split("__")]
            if arg1typs == ["EITHER"]:
                arg1typs = ["EITHER_EQUAL", "EITHER_EQUAL"]
            if arg2typs == ["EITHER"]:
                arg2typs = ["EITHER_EQUAL", "EITHER_EQUAL"]
            *values, result = m.groups()
            if len(values) != len(arg1typs) + len(arg2typs):
                raise RuntimeError(f"Unexpected number of parsed values for {typ}: len({values}) != len({arg1typs}) + len({arg2typs})")
            arg1values, arg2values = values[:len(arg1typs)], values[len(arg1typs):]
            arg1values = [parse_value(value, 'arg1') for value in arg1values]
            arg2values = [parse_value(value, 'arg2') for value in arg2values]

            tomask = lambda t: t.lower().replace("either_equal", "equal").replace("equal", "exactly_equal")
            value1masks = [get_mask(tomask(t), 'arg1', v) for t, v in
                           zip(arg1typs, arg1values)]
            value2masks = [get_mask(tomask(t), 'arg2', v) for t, v in
                           zip(arg2typs, arg2values)]
            if len(value1masks) > 1:
                if arg1typs[0] == "EITHER_EQUAL":
                    mask1 = f"logical_or({value1masks[0]}, {value1masks[1]})"
                else:
                    mask1 = f"logical_and({value1masks[0]}, {value1masks[1]})"
            else:
                mask1 = value1masks[0]
            if len(value2masks) > 1:
                if arg2typs[0] == "EITHER_EQUAL":
                    mask2 = f"logical_or({value2masks[0]}, {value2masks[1]})"
                else:
                    mask2 = f"logical_and({value2masks[0]}, {value2masks[1]})"
            else:
                mask2 = value2masks[0]

            mask = f"logical_and({mask1}, {mask2})"
            assertion = get_assert("exactly_equal", result)

        elif typ == "TWO_ARGS_EITHER":
            value, result = m.groups()
            value = parse_value(value, "arg1")
            mask1 = get_mask("exactly_equal", "arg1", value)
            mask2 = get_mask("exactly_equal", "arg2", value)
            mask = f"logical_or({mask1}, {mask2})"
            assertion = get_assert("exactly_equal", result)
        elif typ == "TWO_ARGS_SAME_SIGN":
            result, = m.groups()
            mask = "same_sign(arg1, arg2)"
            assertion = get_assert("exactly_equal", result)
        elif typ == "TWO_ARGS_SAME_SIGN_BOTH":
            value, result = m.groups()
            mask1 = get_mask("exactly_equal", "arg1", value)
            mask2 = get_mask("exactly_equal", "arg2", value)
            mask = f"logical_and(same_sign(arg1, arg2), logical_and({mask1}, {mask2}))"
            assertion = get_assert("exactly_equal", result)
        elif typ == "TWO_ARGS_DIFFERENT_SIGNS":
            result, = m.groups()
            mask = "logical_not(same_sign(arg1, arg2))"
            assertion = get_assert("exactly_equal", result)
        elif typ == "TWO_ARGS_DIFFERENT_SIGNS_BOTH":
            value, result = m.groups()
            mask1 = get_mask("exactly_equal", "arg1", value)
            mask2 = get_mask("exactly_equal", "arg2", value)
            mask = f"logical_and(logical_not(same_sign(arg1, arg2)), logical_and({mask1}, {mask2}))"
            assertion = get_assert("exactly_equal", result)
        elif typ == "TWO_ARGS_EVEN_IF":
            value1, result, value2 = m.groups()
            value1 = parse_value(value1, "arg1")
            mask = get_mask("exactly_equal", "arg1", value1)
            assertion = get_assert("exactly_equal", result)
        else:
            raise ValueError(f"Unrecognized special value type {typ}")
        return TWO_ARGS_TEMPLATE.format(
            decorator=decorator,
            func=func,
            test_name_extra=test_name_extra,
            doc=doc,
            mask=mask,
            assertion=assertion,
        )

    elif typ == "REMAINING":
        return REMAINING_TEMPLATE
    else:
        raise RuntimeError(f"Unexpected type {typ}")

def parse_special_cases(spec_text, verbose=False):
    special_cases = {}
    in_block = False
    for line in spec_text.splitlines():
        m = FUNCTION_HEADER_RE.match(line)
        if m:
            name = m.group(1)
            special_cases[name] = defaultdict(list)
            continue
        if line == '#### Special Cases':
            in_block = True
            continue
        elif line.startswith('#'):
            in_block = False
            continue
        if in_block:
            if '- ' not in line:
                continue
            for typ, reg in SPECIAL_CASE_REGEXS.items():
                m = reg.match(line)
                if m:
                    if verbose:
                        print(f"Matched {typ} for {name}: {m.groups()}")
                    special_cases[name][typ].append(m)
                    break
            else:
                raise ValueError(f"Unrecognized special case string for '{name}':\n{line}")

    return special_cases

if __name__ == '__main__':
    main()
