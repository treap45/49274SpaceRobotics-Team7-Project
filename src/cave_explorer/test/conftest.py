"""Let the module tests import ``cave_explorer`` straight from the source tree.

Run from ``src/cave_explorer`` with ``python3 -m pytest test/`` after sourcing
``/opt/ros/humble/setup.bash``; no ``colcon build`` is needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
