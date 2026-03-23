"""
bump_version.py
---------------
Reads the latest Git tag and writes the version to src/minas/_version.py
and pyproject.toml. Run before building: python bump_version.py
"""

import subprocess
import pathlib
import re

def get_version():
    result = subprocess.run(
        ['git', 'describe', '--tags', '--abbrev=0'],
        capture_output=True, text=True
    )
    tag = result.stdout.strip()
    if not tag.startswith('v'):
        raise ValueError(f"Tag '{tag}' does not start with 'v'")
    return tag[1:]

version = get_version()

p = pathlib.Path("pyproject.toml")
p.write_text(re.sub(r'^version = ".*"', f'version = "{version}"', p.read_text(), flags=re.MULTILINE))

pathlib.Path("src/minas/_version.py").write_text(f'__version__ = "{version}"\n')

print(f"✓ version set to {version}")
