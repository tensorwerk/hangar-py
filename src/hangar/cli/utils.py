
def parse_custom_arguments(click_args):
    parsed = {}
    for i in range(0, len(click_args), 2):
        key = click_args[i]
        val = click_args[i + 1]
        if not key.startswith('--'):
            raise RuntimeError(f"Could not parse argument {key}. It should be prefixed with `--`")
        parsed[key[2:]] = val
    return parsed
