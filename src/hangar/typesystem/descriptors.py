"""
Portions of this code have been taken and modified from the book:

Beazley, D. and B. K. Jones (2013). Python Cookbook, O’Reilly Media, Inc.

Chapter: 8.13. Implementing a Data Model or Type System

===============================================================================

Problem
-------

You want to define various kinds of data structures, but want to enforce
constraints on the values that are allowed to be assigned to certain
attributes.

Solution
--------

In this problem, you are basically faced with the task of placing checks or
assertions on the setting of certain instance attributes. To do this, you need
to customize the setting of attributes on a per-attribute basis. To do this,
you should use descriptors.

This recipe involves a number of advanced techniques, including descriptors,
mixin classes, the use of super(), class decorators, and metaclasses. Covering
the basics of all those topics is beyond what can be covered here; However,
there are a number of subtle points worth noting.

First, in the Descriptor base class, you will notice that there is a __set__()
method, but no corresponding __get__(). If a descriptor will do nothing more
than extract an identically named value from the underlying instance
dictionary, defining __get__() is unnecessary. In fact, defining __get__() will
just make it run slower. Thus, this recipe only focuses on the implementation
of __set__().

The overall design of the various descriptor classes is based on mixin classes.
For example, the Unsigned and MaxSized classes are meant to be mixed with the
other descriptor classes derived from Typed. To handle a specific kind of data
type, multiple inheritance is used to combine the desired functionality.

You will also notice that all __init__() methods of the various descriptors
have been programmed to have an identical signature involving keyword arguments
**opts. The class for MaxSized looks for its required attribute in opts, but
simply passes it along to the Descriptor base class, which actually sets it.
One tricky part about composing classes like this (especially mixins), is that
you don’t always know how the classes are going to be chained together or what
super() will invoke. For this reason, you need to make it work with any
possible combination of classes.

The definitions of the various type classes such as Integer, Float, and String
illustrate a useful technique of using class variables to customize an
implementation. The Ty ped descriptor merely looks for an expected_type
attribute that is provided by each of those subclasses.

The use of a class decorator or metaclass is often useful for simplifying the
specification by the user.

The code for the class decorator and metaclass simply scan the class dictionary
looking for descriptors. When found, they simply fill in the descriptor name
based on the key value.

As a final twist, a class decorator approach can also be used as a replacement
for mixin classes, multiple inheritance, and tricky use of the super() function

The classes defined in this alternative formulation work in exactly the same
manner as before (none of the earlier example code changes) except that
everything runs much faster. For example, a simple timing test of setting a
typed attribute reveals that the class decorator approach runs almost 100%
faster than the approach using mixins.
"""
from typing import Sequence


class Descriptor:
    # Base class. Uses a descriptor to set a value
    def __init__(self, name=None, **opts):
        self.name = name
        self.__dict__.update(opts)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


def Typed(expected_type, cls=None):
    # Decorator for applying type checking
    if cls is None:
        return lambda cls: Typed(expected_type, cls)

    super_set = cls.__set__

    def __set__(self, instance, value):
        if not isinstance(value, expected_type):
            raise TypeError('expected ' + str(expected_type))
        super_set(self, instance, value)

    cls.__set__ = __set__
    return cls


def TypedSequence(expected_element_types, cls=None):
    # Decorator enforcing that all elements in an sequence are specific type(s).
    # using the python ABC definition of "Sequence" (list, tuple)
    # https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence
    if cls is None:
        return lambda cls: TypedSequence(expected_element_types, cls)

    super_set = cls.__set__
    def __set__(self, instance, value):
        if not isinstance(value, Sequence):
            raise TypeError(f'input is not Sequence type, recieved {type(value)}')
        elif not all([isinstance(el, expected_element_types) for el in value]):
            raise TypeError(f'not all elements are {expected_element_types} type(s) in {value}')
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls


def OneOf(expected_values, cls=None):
    # Decorator for enforcing values
    if cls is None:
        return lambda cls: OneOf(expected_values, cls)

    super_set = cls.__set__
    def __set__(self, instance, value):
        if value not in expected_values:
            raise ValueError(f'expected one of {expected_values} recieved {value}')
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls


def MaxSized(cls):
    # Decorator for allowing sized values
    super_init = cls.__init__
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        self.size = opts['size']
        super_init(self, name, **opts)
    cls.__init__ = __init__
    super_set = cls.__set__
    def __set__(self, instance, value):
        if len(value) > self.size:
            raise ValueError('size must be < ' + str(self.size))
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls


def DictItems(expected_keys_required, expected_values, cls=None):
    # check a dictionary for the existence of keys. expected_keys should be a dictionary of keys,
    # with bool values set to indicate if they are required or not. expected_values should be
    # mapping of same keys to list of acceptable values.
    if cls is None:
        return lambda cls: DictItems(expected_keys_required, expected_values, cls)

    super_set = cls.__set__
    def __set__(self, instance, value):
        if not isinstance(value, dict):
            raise TypeError(f'expected {dict}, recieved {type(value)}')
        for expected_key, required in expected_keys_required.items():
            try:
                if value[expected_key] not in expected_values[expected_key]:
                    raise ValueError(f'{value[expected_key]} invalid for key {expected_key}')
            except KeyError as e:
                if required:
                    raise e
        for recieved_key in value.keys():
            if recieved_key not in expected_keys_required:
                raise TypeError(f'Not supposed to have key {recieved_key}')
        super_set(self, instance, value)
    cls.__set__ = __set__
    return cls


@Typed(str)
class String(Descriptor):
    pass


@DictItems(expected_keys_required={},
           expected_values={},)
class EmptyDict(Descriptor):
    pass


@Typed((dict, type(None)))
class OptionalDict(Descriptor):
    pass


@Typed((str, type(None)))
class OptionalString(Descriptor):
    pass


@Typed(tuple)
class Tuple(Descriptor):
    pass


@MaxSized
@TypedSequence(int)
class SizedIntegerTuple(Tuple):
    pass


class checkedmeta(type):
    # A metaclass that applies checking
    def __new__(cls, clsname, bases, methods):
        # Attach attribute names to the descriptors
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clsname, bases, methods)
