# MINAS Versioning

MINAS uses **Semantic Versioning**: `MAJOR.MINOR.PATCH`

## Types of Changes

### MAJOR (1.x.x → 2.x.x)
**Significant changes / Breaking Changes**
- API changes that break backward compatibility
- Removal of features
- Complete restructuring of modules
- Changes that require modifications in users' code

**Examples:**
- Rename public functions
- Change function signatures (required parameters)
- Remove functions or modules

### MINOR (x.1.x → x.2.x)
**New features (backward compatible)**
- Add new functionalities
- Add new optional parameters
- Significant performance improvements
- New modules or classes

**Examples:**
- New visualization function
- New prediction algorithm
- New optional parameters in existing functions

### PATCH (x.x.1 → x.x.2)
**Bug fixes / Small changes**
- Bug fixes
- Documentation adjustments
- Small visual adjustments
- Code typo corrections

**Examples:**
- Fix error in plots (x-axis ticks)
- Adjust font size
- Fix metrics calculation

## How to Use

### 1. Update version automatically:

```bash
# For MAJOR changes (breaking changes)
python version/bump_version.py major

# For new features (MINOR)
python version/bump_version.py minor

# For bug fixes (PATCH)
python version/bump_version.py patch
```

### 2. Commit the changes:

```bash
git add version/VERSION pyproject.toml
git commit -m "Bump version to X.Y.Z"
git tag vX.Y.Z
git push && git push --tags
```

## Version History

### v1.0.0 (2025-09-03)
- Initial MINAS release
- Prediction features with RF and XGBoost
- Regression and bolometric correction plots
- Support for APOGEE, LAMOST, GALAH

### Examples of Future Versions

- **v1.0.1**: Fix ticks in residuals plots
- **v1.1.0**: Add support for new surveys
- **v2.0.0**: Complete API restructuring
