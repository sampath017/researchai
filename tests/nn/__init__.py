import numpy as np
import sys
from pathlib import Path

paths = [
    Path.cwd() / "../../src/nn",
    Path.cwd() / "../.."
]

for path in paths:
    sys.path.append(path.resolve().as_posix())

