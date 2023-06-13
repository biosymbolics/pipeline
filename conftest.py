from src.system import init


def pytest_configure():
    init()


def pytest_unconfigure():
    # Code to run after all tests
    pass
