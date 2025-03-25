from uuid import UUID
from ipaddress import IPv4Address, IPv6Address
from jsonschema import Draft202012Validator, FormatChecker, SchemaError


format_checker = FormatChecker()


# TODO: add more format checks, based on what's in the dataset


@format_checker.checks("ipv4")
def ipv4_check(value):
    IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    UUID(value)


def is_json_schema_valid(schema):
    try:
        Draft202012Validator.check_schema(schema)
        return True
    except SchemaError:
        return False


def validate_json_object(instance, schema):
    if not is_json_schema_valid(schema):
        raise ValueError("Invalid JSON schema")

    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)

    # we catch all exceptions include ValidationError and Error from extension validators
    except Exception:
        return False
    return True
