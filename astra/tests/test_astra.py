"""
Unit and regression test for the astra package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import astra


def test_astra_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "astra" in sys.modules
