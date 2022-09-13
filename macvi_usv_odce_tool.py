#!/usr/bin/env python3

# NOTE: this script is intended for use with git checkout of the toolkit, without having installed it first.
# However, even in such scenarios, it is preferrable to install the toolkit in a (virtual) environment (for
# example, using `pip install --editable /path/to/toolkit/directory` to install in editable mode) and use
# the setuptools-installed entry-point script (i.e., `macvi-usv-odce-tool`), assuming that the environment's
# scripts directory is in PATH.

from macvi_usv_odce_toolkit.__main__ import main as toolkit_main

toolkit_main()
