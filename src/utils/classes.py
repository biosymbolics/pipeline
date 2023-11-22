"""
Utils around classes
"""
from enum import Enum


def overrides(interface_class):
    """
    Overrides annotation

    example: @overrides(MySuperInterface)
    """

    def overrider(method):
        assert method.__name__ in dir(interface_class)
        return method

    return overrider


def nonoverride(method):
    """
    Non-override annotation

    example: @nonoverride
    """
    return method


class ByDefinitionOrderEnum(Enum):
    def __init__(self, *args):
        try:
            # attempt to initialize other parents in the hierarchy
            super().__init__(*args)
        except TypeError:
            # ignore -- there are no other parents
            pass
        ordered = len(self.__class__.__members__) + 1
        self._order = ordered

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self._order >= other._order
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self._order > other._order
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self._order <= other._order
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self._order < other._order
        return NotImplemented

    def __str__(self):
        return self.value

    def __repr__(self) -> str:
        return self.__str__()
