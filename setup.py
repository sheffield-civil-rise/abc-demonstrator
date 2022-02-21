"""
This is a file which should need no introduction!
"""

# Non-standard imports.
from setuptools import setup, find_packages

setup(
    name="photogrammetry_e110a",
    version="1.0",
    packages=find_packages(exclude=["tests*"]),
    license="none",
    description="The photogrammetry code I worked on in room E110A",
    long_description=open("README.md").read(),
    install_requires=[],
    url="REPOSITORY_URL",
    author="AUTHOR_NAME",
    author_email="AUTHOR_EMAIL"
)
