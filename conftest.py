from src.system import initialize


def pytest_configure():
    initialize()


def pytest_unconfigure():
    # Code to run after all tests
    pass
