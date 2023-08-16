"""
Utils around classes
"""


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
