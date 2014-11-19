#!/usr/bin/env python
# coding: utf-8

# Third-Party Libraries
import setuptools
#from Cython.Build import cythonize

# Source:
# https://github.com/cython/cython/wiki/PackageHierarchy

if __name__ == "__main__":
    setuptools.setup(
      name = "audio.filters",
      version = "0.1.1",
      packages = setuptools.find_packages(),
      #ext_modules = cythonize("audio/filters.pyx")
    )

