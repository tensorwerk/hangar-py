import click


class StrOrIntType(click.ParamType):
    """Custom type for click to parse the sample name
    argument to integer or string
    """

    def convert(self, value, param, ctx):
        if not value:
            return None

        try:
            stype, sample = value.split(':') if ':' in value else ('str', value)
        except ValueError:
            self.fail(f"Sample name {value} not formatted properly", param, ctx)
        try:
            if stype not in ('str', 'int'):
                self.fail(f"type {stype} is not allowed", param, ctx)
            return int(sample) if stype == 'int' else str(sample)
        except (ValueError, TypeError):
            self.fail(f"{sample} is not a valid {stype}", param, ctx)


def parse_custom_arguments(click_args: list) -> dict:
    """
    Parse all the unknown arguments from click for downstream tasks. Used in
    user plugins for custom command line arguments.

    Parameters
    ----------
    click_args : list
        Unknown arguments from click

    Returns
    -------
    parsed : dict
        Parsed arguments stored as key value pair

    Note
    -----
    Unknown arguments must be long arguments i.e should start with --
    """
    parsed = {}
    for i in range(0, len(click_args), 2):
        key = click_args[i]
        val = click_args[i + 1]
        if not key.startswith('--'):
            raise RuntimeError(f"Could not parse argument {key}. It should be prefixed with `--`")
        parsed[key[2:]] = val
    return parsed
