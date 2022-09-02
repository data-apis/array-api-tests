#!/usr/bin/env python
import json
import jsonschema

def main():
    schema = json.load(open('report.schema.json'))

    jsonschema.Validator.check_schema(schema)

    with open('.report.json') as f:
        report = json.load(f)

    jsonschema.validate(report, schema)

if __name__ == '__main__':
    main()
