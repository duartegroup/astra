import pytest
import sys
import astra


def test_astra_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "astra" in sys.modules
