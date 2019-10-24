
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

    Notes
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
