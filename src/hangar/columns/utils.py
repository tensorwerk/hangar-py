import types

import wrapt


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


@wrapt.decorator
def writer_checkout_only(wrapped, instance, args, kwargs) -> types.MethodType:
    """Only allow a method to be called in a write-enable checkout.

    Parameters
    ----------
    wrapped
        bound method which is being called
    instance
        class object being operated on ie. ``instance is self``
        in both (equality and identify).
    args
        argument list passed to the method
    kwargs
        keyword args dict passed to the method.

    Returns
    -------
    types.MethodType
        If instance._mode == 'a' (write enabled checkout) then
        operation is allowed and pass through args and kwargs
        to the method as specified.

    Raises
    ------
    PermissionError
        If the checkout is opened in read-only mode, then deny
        ability to call and raise error explaining why to user.
    """

    if instance._mode != 'a':
        err = (f'Method "{wrapped.__func__.__name__}" '
               f'cannot be called in a read-only checkout.')
        raise PermissionError(err) from None
    else:
        return wrapped(*args, **kwargs)



@wrapt.decorator
def reader_checkout_only(wrapped, instance, args, kwargs) -> types.MethodType:
    """Only allow a method to be called in a read-only checkout.

    Parameters
    ----------
    wrapped
        bound method which is being called
    instance
        class object being operated on ie. ``instance is self``
        in both (equality and identify).
    args
        argument list passed to the method
    kwargs
        keyword args dict passed to the method.

    Returns
    -------
    types.MethodType
        If instance._mode == 'r' (read-only checkout) then
        operation is allowed and pass through args and kwargs
        to the method as specified.

    Raises
    ------
    PermissionError
        If the checkout is opened in write-enabled mode, then deny
        ability to call and raise error explaining why to user.
    """

    if instance._mode != 'r':
        err = (f'Method "{wrapped.__func__.__name__}" '
               f'cannot be called in a write-enabled checkout.')
        raise PermissionError(err) from None
    else:
        return wrapped(*args, **kwargs)
