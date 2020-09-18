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
    args = parser.parse_args()

    spec_dir = os.path.join(args.array_api_repo, 'spec', 'API_specification')
    modules = {}
    for filename in sorted(os.listdir(spec_dir)):
        with open(os.path.join(spec_dir, filename)) as f:
                  text = f.read()

        signatures = SIGNATURE_RE.findall(text)
        if not signatures:
            continue
        print(f"Found signatures in {filename}")
        if not args.write:
            continue
        py_file = filename.replace('.md', '.py')
        py_path = os.path.join('array_api_tests', 'function_stubs', py_file)
        title = filename.replace('.md', '').replace('_', ' ')
        module_name = py_file.replace('.py', '')
        modules[module_name] = []
        print(f"Writing {py_path}")
        with open(py_path, 'w') as f:
            f.write(STUB_FILE_HEADER.format(filename=filename, title=title))
            for sig in signatures:
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
if __name__ == '__main__':
    main()
