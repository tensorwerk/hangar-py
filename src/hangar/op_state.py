import types
import sys

import wrapt


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
    try:
        if instance._mode == 'a':  # user facing classes hide attribute
            return wrapped(*args, **kwargs)
        else:
            err = (f'Method "{wrapped.__func__.__name__}" '
                   f'cannot be called in a read-only checkout.')
            raise PermissionError(err) from None

    except AttributeError:
        if instance.mode == 'a':  # internal classes don't hide attribute
            return wrapped(*args, **kwargs)
        else:
            err = (f'Method "{wrapped.__func__.__name__}" '
                   f'cannot be called in a read-only checkout.')
            raise PermissionError(err) from None


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
    try:
        if instance._mode == 'r':  # user facing classes hide attribute
            return wrapped(*args, **kwargs)
        else:
            err = (f'Method "{wrapped.__func__.__name__}" '
                   f'cannot be called in a write-enabled checkout.')
            raise PermissionError(err) from None

    except AttributeError:
        if instance.mode == 'r':  # internal classes don't hide attribute
            return wrapped(*args, **kwargs)
        else:
            err = (f'Method "{wrapped.__func__.__name__}" '
                   f'cannot be called in a write-enabled checkout.')
            raise PermissionError(err) from None


def tb_params_last_called(tb: types.TracebackType) -> dict:
    """Get parameters of the last function called before exception thrown.

    Parameters
    ----------
    tb : types.TracebackType
        traceback object returned as the third item from sys.exc_info()
        corresponding to an exception raised in the last stack frame.

    Returns
    -------
    dict
        parameters passed to the last function called before the exception was
        thrown.
    """
    while tb.tb_next:
        tb = tb.tb_next
    frame = tb.tb_frame
    code = frame.f_code
    argcount = code.co_argcount
    if code.co_flags & 4:  # *args
        argcount += 1
    if code.co_flags & 8:  # **kwargs
        argcount += 1
    names = code.co_varnames[:argcount]
    params = {}
    for name in names:
        params[name] = frame.f_locals.get(name, '<deleted>')
    return params


def report_corruption_risk_on_parsing_error(func):
    """Decorator adding try/except handling non-explicit exceptions.

    Explicitly raised RuntimeErrors generally point to corrupted data
    identified by a cryptographic hash mismatch. However, in order to get to
    the point where such quantities can be processes, a non-trivial amount of
    parsing machinery must be run. Should any error be thrown in the parse
    machinery due to corrupted values, this method raises the exception in a
    useful form; providing traceback context, likely root cause (displayed to
    users), and the offending arguments passed to the function which threw the
    error.
    """
    def wrapped(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(
                f'Corruption detected during {func.__name__}. Most likely this is the '
                f'result of unparsable record values. Exception msg `{str(e)}`. Params '
                f'`{tb_params_last_called(sys.exc_info()[2])}`') from e
    return wrapped
