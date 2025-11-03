# Version Management

This directory contains files for managing MINAS package versions.

## Files

- **VERSION**: Current package version
- **VERSIONING.md**: Complete versioning documentation
- **bump_version.py**: Script to automatically update versions

## Quick Usage

From the MINAS root directory:

```bash
# Bug fixes
python version/bump_version.py patch

# New features
python version/bump_version.py minor

# Breaking changes
python version/bump_version.py major
```

See [VERSIONING.md](VERSIONING.md) for complete documentation.
