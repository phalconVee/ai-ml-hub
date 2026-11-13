import sys
from pathlib import Path

# Let tests do `from fakes import …` without making tests/ a package.
sys.path.insert(0, str(Path(__file__).parent))
