"""
Setup for creating whl
"""

import setuptools

print(setuptools.find_packages())

setuptools.setup(
    author="Biosymbolics",
    author_email="kristin@biosymbolics.ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    description="Biosympbolics data modules",
    name="biosymbolics-data-modules",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    version="0.0.1",  # upon change, update requirements.txt as well
)
