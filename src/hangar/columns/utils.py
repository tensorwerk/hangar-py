def valfilter(predicate, d, factory=dict):
    """ Filter items in dictionary by values that are true.

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilter(iseven, d)
    {1: 2, 3: 4}

    See Also:
        valfilterfalse
    """
    rv = factory()
    for k, v in d.items():
        if predicate(v):
            rv[k] = v
    return rv


def valfilterfalse(predicate, d, factory=dict):
    """ Filter items in dictionary by values which are false.

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilterfalse(iseven, d)
    {2: 3, 4: 5}

    See Also:
        valfilter
    """
    rv = factory()
    for k, v in d.items():
        if not predicate(v):
            rv[k] = v
    return rv
