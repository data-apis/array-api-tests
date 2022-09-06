#!/usr/bin/env python
import sys
import json

import jsonschema

def main():
    if len(sys.argv) == 1:
        report_file = '.report.json'
    elif len(sys.argv) == 2:
        report_file = sys.argv[1]
    else:
        sys.exit("Usage: verify_report.py [json_report_file]")

    schema = json.load(open('report.schema.json'))

    jsonschema.Validator.check_schema(schema)

    with open(report_file) as f:
        report = json.load(f)

    jsonschema.validate(report, schema)

if __name__ == '__main__':
    main()
