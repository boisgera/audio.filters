#!/usr/bin/env python
# coding: utf-8

# Third-Party Libraries
from setuptools import setup
from Cython.Build import cythonize

# Source:
# https://github.com/cython/cython/wiki/PackageHierarchy

if __name__ == "__main__":
    setup(
      name = "audio.filters",
      packages = ["audio"],
      ext_modules = cythonize("audio/filters.pyx")
    )

