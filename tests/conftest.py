"""Pytest bootstrap: put the repository root on sys.path.

Lets tests import ``core.*`` without installing the package.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
