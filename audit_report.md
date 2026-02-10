# DeepV Codebase Audit Report

Generated: 2026-02-09 05:59

## Notes Format Explanation

Each file entry in the audit report follows a standardized format to ensure consistent evaluation and tracking of codebase quality improvements.

### Entry Fields

- **Status**: Current audit status
  - `completed`: File has been audited and any issues fixed
  - `in-progress`: Currently being audited
  - `pending`: Not yet audited

- **Category**: Functional category of the file
  - `cleaning`: Image preprocessing and cleaning modules
  - `vectorization`: Neural network models and training
  - `refinement`: Post-processing optimization
  - `merging`: Primitive consolidation
  - `other`: Utilities, scripts, configuration

- **Audited**: Date when the audit was performed (YYYY-MM-DD format)

- **Notes**: Detailed audit results in structured format
  - **AUDIT section**: Quality assessment checklist results
  - **Action section**: Either `FIXED:` (issues resolved) or `VERIFIED:` (no issues found)

### AUDIT Checklist Criteria

Each file is evaluated against these quality dimensions and must be fixed appropriately (or a reason added as to why it hasn't been after auditing):

- **Header&Doc**: Documentation completeness (docstrings, comments, examples)
- **Imports**: Import organization and cleanliness (no unused, proper grouping)
- **TypeHints**: Type annotation coverage and accuracy
- **ErrorHandling**: Exception handling patterns and specificity
- **CodeStyle**: PEP 8 compliance and formatting consistency
- **Performance**: Efficiency considerations and potential bottlenecks
- **Architecture**: Design patterns, coupling, and structural quality
- **Testing**: Testability, mocking requirements, and coverage potential
- **Security**: Input validation, safe operations, vulnerability prevention
- **Maintainability**: Code clarity, complexity, and future modification ease
- **ResearchCleanup**: Removal of experimental/research artifacts for production readiness

### Action Section Format

- **FIXED:** [Specific changes made to resolve identified issues]
- **VERIFIED:** [Confirmation that file meets standards with no required changes]

## Overall Progress

- Files audited: 9/152 (5.9%)

## Detailed File Status

### __init__.py
- Status: completed
- Category: other
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Added minimal package docstring and metadata to improve discoverability without changing runtime behavior.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — module docstring added.
    - **Imports:** ✅ Not applicable — no imports present.
    - **TypeHints:** ⚪ N/A — no functions or classes to annotate.
    - **ErrorHandling:** ⚪ N/A — no runtime code.
    - **CodeStyle:** ✅ Compliant — file passes formatting/lint checks.
    - **Performance:** ✅ Not applicable — trivial module.
    - **Architecture:** ✅ Improved — provides metadata and a place for top-level exports.
    - **Testing:** ⚪ N/A — nothing to test directly.
    - **Security:** ✅ No issues — no operations performed.
    - **Maintainability:** ✅ Improved — docstring and metadata aid discoverability.
    - **ResearchCleanup:** ✅ Clean — no experimental code present.
  - **Automated Checks Run:**
    - `flake8 __init__.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy __init__.py --ignore-missing-imports --no-strict-optional` — no issues
    - `black --check --diff __init__.py` — file unchanged
    - `isort --check-only --diff __init__.py` — no issues
  - **FIXED:**
    - Updated `e:\dv\DeepV\__init__.py` to include typed metadata:
      ```py
      """DeepV package init.

      Expose package metadata and provide a stable place for top-level exports.

      This file intentionally contains minimal runtime logic — keep heavy imports
      out of package-level initialization to avoid slow imports for CLI/tools.
      """

      __all__: list[str] = []

      __version__: str = "0.0.0"
      ```
    - Rationale: Minimal, safe change that improves package metadata visibility.

### cleaning\__init__.py
- Status: completed
- Category: cleaning
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Package initializer for `cleaning` module; added type annotation for `__all__` to satisfy type checks.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — module docstring describes purpose, submodules, and example usage.
    - **Imports:** ✅ Clean — uses relative imports for `scripts` and `utils`.
    - **TypeHints:** ✅ Improved — `__all__` annotated as `list[str]` to satisfy `mypy`.
    - **ErrorHandling:** ⚪ N/A — no runtime logic present.
    - **CodeStyle:** ✅ Compliant — formatting and style pass `flake8`/`black` checks.
    - **Performance:** ✅ Not applicable — import-only module.
    - **Architecture:** ✅ Good — exposes intended submodules via `__all__`.
    - **Testing:** ⚪ N/A — nothing to unit-test directly.
    - **Security:** ✅ No issues — no unsafe operations.
    - **Maintainability:** ✅ Good — clear docstring and explicit exports improve discoverability.
    - **ResearchCleanup:** ✅ Clean — no experimental code present.
  - **Automated Checks Run:**
    - `flake8 cleaning\__init__.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy cleaning\__init__.py --ignore-missing-imports --no-strict-optional` — no issues
    - `black --check --diff cleaning\__init__.py` — file unchanged
    - `isort --check-only --diff cleaning\__init__.py` — no issues
  - **FIXED:**
    - Edited `e:\dv\DeepV\cleaning\__init__.py` to annotate `__all__`:
      ```py
      __all__: list[str] = ["scripts", "utils"]
      ```
    - Rationale: Satisfies static type checking and clarifies exported API.

### dataset\__init__.py
- Status: completed
- Category: other
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Package initializer for `dataset` module; added type annotation for `__all__` for consistency.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — module docstring describes purpose, submodules, and example usage.
    - **Imports:** ✅ Clean — uses relative imports for `downloaders` and `processors`.
    - **TypeHints:** ✅ Improved — `__all__` annotated as `list[str]` for type safety.
    - **ErrorHandling:** ⚪ N/A — no runtime logic present.
    - **CodeStyle:** ✅ Compliant — formatting and style pass `flake8`/`black` checks.
    - **Performance:** ✅ Not applicable — import-only module.
    - **Architecture:** ✅ Good — exposes intended submodules via `__all__`.
    - **Testing:** ⚪ N/A — nothing to unit-test directly.
    - **Security:** ✅ No issues — no unsafe operations.
    - **Maintainability:** ✅ Good — clear docstring and explicit exports improve discoverability.
    - **ResearchCleanup:** ✅ Clean — no experimental code present.
  - **Automated Checks Run:**
    - `flake8 dataset\__init__.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy dataset\__init__.py --ignore-missing-imports --no-strict-optional` — no issues (submodule errors not relevant)
    - `black --check --diff dataset\__init__.py` — file unchanged
    - `isort --check-only --diff dataset\__init__.py` — no issues
  - **FIXED:**
    - Edited `e:\dv\DeepV\dataset\__init__.py` to annotate `__all__`:
      ```py
      __all__: list[str] = ["downloaders", "processors"]
      ```
    - Rationale: Ensures type safety and consistency with other package initializers. 

### merging\__init__.py
- Status: completed
- Category: merging
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Package initializer for `merging` module; added type annotation for `__all__` for consistency.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — module docstring describes purpose, submodules, and example usage.
    - **Imports:** ✅ Clean — uses relative import for `utils`.
    - **TypeHints:** ✅ Improved — `__all__` annotated as `list[str]` for type safety.
    - **ErrorHandling:** ⚪ N/A — no runtime logic present.
    - **CodeStyle:** ✅ Compliant — formatting and style pass `flake8`/`black` checks.
    - **Performance:** ✅ Not applicable — import-only module.
    - **Architecture:** ✅ Good — exposes intended submodule via `__all__`.
    - **Testing:** ⚪ N/A — nothing to unit-test directly.
    - **Security:** ✅ No issues — no unsafe operations.
    - **Maintainability:** ✅ Good — clear docstring and explicit exports improve discoverability.
    - **ResearchCleanup:** ✅ Clean — no experimental code present.
  - **Automated Checks Run:**
    - `flake8 merging\__init__.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy merging\__init__.py --ignore-missing-imports --no-strict-optional` — no issues
    - `black --check --diff merging\__init__.py` — file unchanged
    - `isort --check-only --diff merging\__init__.py` — no issues
  - **FIXED:**
    - Edited `e:\dv\DeepV\merging\__init__.py` to annotate `__all__`:
      ```py
      __all__: list[str] = ["utils"]
      ```
    - Rationale: Ensures type safety and consistency with other package initializers. 

### cleaning\scripts\__init__.py
- Status: completed
- Category: cleaning
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Package initializer for `cleaning/scripts` module; added type annotation for `__all__` for consistency.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — module docstring describes purpose and scripts.
    - **Imports:** ✅ Not applicable — no imports present.
    - **TypeHints:** ✅ Improved — `__all__` annotated as `list[str]` for type safety.
    - **ErrorHandling:** ⚪ N/A — no runtime logic present.
    - **CodeStyle:** ✅ Compliant — formatting and style pass `flake8`/`black` checks after adding newline.
    - **Performance:** ✅ Not applicable — import-only module.
    - **Architecture:** ✅ Good — correctly indicates scripts are not for importing.
    - **Testing:** ⚪ N/A — nothing to unit-test directly.
    - **Security:** ✅ No issues — no unsafe operations.
    - **Maintainability:** ✅ Good — clear docstring and explicit exports improve discoverability.
    - **ResearchCleanup:** ✅ Clean — no experimental code present.
  - **Automated Checks Run:**
    - `flake8 cleaning\scripts\__init__.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy cleaning\scripts\__init__.py --ignore-missing-imports --no-strict-optional` — no issues
    - `black --check --diff cleaning\scripts\__init__.py` — file unchanged
    - `isort --check-only --diff cleaning\scripts\__init__.py` — no issues
  - **FIXED:**
    - Edited `e:\dv\DeepV\cleaning\scripts\__init__.py` to annotate `__all__`:
      ```py
      __all__: list[str] = []
      ```
    - Rationale: Ensures type safety and consistency with other package initializers. 

### analysis\tracing.py
- Status: completed
- Category: other
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Cleaned up imports, fixed style issues, and formatted code for better maintainability.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — class and methods have docstrings with usage examples.
    - **Imports:** ✅ Complete — removed unused 'os' import.
    - **TypeHints:** ⚠️ Partial — basic type hints present but could be enhanced (e.g., offset parameter).
    - **ErrorHandling:** ✅ Complete — specific exceptions caught with appropriate fallbacks.
    - **CodeStyle:** ✅ Complete — formatted with black, no long lines or unused variables.
    - **Performance:** ✅ Complete — uses compressed saves, efficient for tracing.
    - **Architecture:** ✅ Complete — simple class with clear responsibilities.
    - **Testing:** ⚪ N/A — tracing utility, not directly unit testable.
    - **Security:** ✅ Complete — safe file I/O with error handling.
    - **Maintainability:** ✅ Complete — clear code structure and comments.
    - **ResearchCleanup:** ✅ Complete — no experimental artifacts.
  - **Automated Checks Run:**
    - `flake8 analysis/tracing.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy analysis/tracing.py --ignore-missing-imports --no-strict-optional` — no issues
    - `black --check --diff analysis/tracing.py` — file unchanged
    - `isort --check-only --diff analysis/tracing.py` — no issues
  - **FIXED:**
    - Removed unused 'os' import (F401).
    - Removed unused exception variable 'e' in except block (F841).
    - Applied black formatting to fix long line and improve style. 

### cad\export.py
- Status: completed
- Category: other
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Fixed critical bug where 'height' parameter was undefined in helper functions, breaking DXF export. Also fixed long line in SVG export.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — comprehensive docstrings for all functions.
    - **Imports:** ✅ Complete — proper imports with try/except for optional dependencies.
    - **TypeHints:** ⚠️ Partial — some type hints present, but could be more comprehensive.
    - **ErrorHandling:** ✅ Complete — try/except blocks with informative error messages.
    - **CodeStyle:** ✅ Complete — fixed long line, passes formatting checks.
    - **Performance:** ✅ Complete — efficient export without unnecessary computations.
    - **Architecture:** ✅ Complete — modular functions for different primitive types.
    - **Testing:** ⚠️ Partial — has example usage in __main__, but no unit tests.
    - **Security:** ✅ Complete — safe file operations with validation.
    - **Maintainability:** ✅ Complete — clear function separation and comments.
    - **ResearchCleanup:** ✅ Complete — production-ready code.
  - **Automated Checks Run:**
    - `flake8 cad/export.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy cad/export.py --ignore-missing-imports --no-strict-optional` — no issues
    - `black --check --diff cad/export.py` — file unchanged
    - `isort --check-only --diff cad/export.py` — no issues
  - **FIXED:**
    - Added 'height' parameter to all DXF export helper functions (_export_lines_to_dxf, _export_curves_to_dxf, _export_cubic_curves_to_dxf, _export_arcs_to_dxf) and updated calls to pass it.
    - Fixed long line (E501) in SVG arc export by breaking the f-string into multiple lines. 

### audit_tracker.py
- Status: completed
- Category: other
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Cleaned up imports, improved type annotations, and formatted code for better maintainability.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — comprehensive docstrings and usage examples.
    - **Imports:** ✅ Complete — removed unused imports, sorted properly.
    - **TypeHints:** ✅ Complete — added proper type annotations for return types and variables.
    - **ErrorHandling:** ⚪ N/A — no runtime operations requiring error handling.
    - **CodeStyle:** ✅ Complete — formatted with black, fixed indentation and blank lines.
    - **Performance:** ✅ Complete — efficient file operations and data structures.
    - **Architecture:** ✅ Complete — well-structured class with clear methods.
    - **Testing:** ⚪ N/A — utility script, not directly unit testable.
    - **Security:** ✅ Complete — safe file I/O operations.
    - **Maintainability:** ✅ Complete — clear code structure and comments.
    - **ResearchCleanup:** ✅ Complete — production-ready code.
  - **Automated Checks Run:**
    - `flake8 audit_tracker.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy audit_tracker.py --ignore-missing-imports --no-strict-optional` — no issues
    - `black --check --diff audit_tracker.py` — file unchanged
    - `isort --check-only --diff audit_tracker.py` — no issues
  - **FIXED:**
    - Removed unused imports: 'sys', 'Set', 'Tuple' (F401).
    - Added type annotations: Dict[str, Any] for load_progress return type, cast for json.load, Dict[str, int] for issue_counts.
    - Applied black formatting and isort import sorting. 

### cleaning\scripts\fine_tuning.py
- Status: completed
- Category: cleaning
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Cleaned up imports, fixed missing return statement in parse_args, and formatted code.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — comprehensive docstrings for functions.
    - **Imports:** ✅ Complete — removed unused imports ('gmtime', 'strftime', 'F', 'IOU').
    - **TypeHints:** ✅ Complete — added missing return statement in parse_args.
    - **ErrorHandling:** ✅ Complete — proper error checking and CUDA availability.
    - **CodeStyle:** ✅ Complete — formatted with black, fixed indentation.
    - **Performance:** ✅ Complete — efficient training loop with GPU utilization.
    - **Architecture:** ✅ Complete — well-structured training script with validation.
    - **Testing:** ⚪ N/A — training script, not directly unit testable.
    - **Security:** ✅ Complete — safe file operations and model loading.
    - **Maintainability:** ✅ Complete — clear function separation and comments.
    - **ResearchCleanup:** ⚠️ Partial — has TODO comment but otherwise clean.
  - **Automated Checks Run:**
    - `flake8 cleaning/scripts/fine_tuning.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy cleaning/scripts/fine_tuning.py --ignore-missing-imports --no-strict-optional` — no issues (errors in other files)
    - `black --check --diff cleaning/scripts/fine_tuning.py` — file unchanged
    - `isort --check-only --diff cleaning/scripts/fine_tuning.py` — no issues
  - **FIXED:**
    - Removed unused imports: 'gmtime', 'strftime', 'F', 'IOU' (F401).
    - Added missing return statement in parse_args() function.
    - Applied black formatting to fix indentation and style. 

### cleaning\scripts\fine_tuning_two_network_added_part.py
- Status: completed
- Category: cleaning
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Cleaned up imports, removed unused variables, and formatted code for better maintainability.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — comprehensive docstrings for functions.
    - **Imports:** ✅ Complete — removed unused imports ('gmtime', 'strftime', 'F', 'IOU').
    - **TypeHints:** ✅ Complete — proper type annotations present.
    - **ErrorHandling:** ✅ Complete — CUDA availability checks and proper error handling.
    - **CodeStyle:** ✅ Complete — formatted with black, fixed indentation.
    - **Performance:** ✅ Complete — efficient two-network training with GPU utilization.
    - **Architecture:** ✅ Complete — well-structured two-network training script.
    - **Testing:** ⚪ N/A — training script, not directly unit testable.
    - **Security:** ✅ Complete — safe file operations and model loading.
    - **Maintainability:** ✅ Complete — clear function separation and comments.
    - **ResearchCleanup:** ✅ Complete — production-ready code.
  - **Automated Checks Run:**
    - `flake8 cleaning/scripts/fine_tuning_two_network_added_part.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy cleaning/scripts/fine_tuning_two_network_added_part.py --ignore-missing-imports --no-strict-optional` — no issues (errors in other files)
    - `black --check --diff cleaning/scripts/fine_tuning_two_network_added_part.py` — file unchanged
    - `isort --check-only --diff cleaning/scripts/fine_tuning_two_network_added_part.py` — no issues
  - **FIXED:**
    - Removed unused imports: 'gmtime', 'strftime', 'F', 'IOU' (F401).
    - Removed unused variables: 'logits_restor' (assigned to _), 'unet_opt' (removed unused optimizer).
    - Applied black formatting to fix indentation and style. 

### cleaning\scripts\generate_synthetic_data.py
- Status: completed
- Category: cleaning
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Cleaned up imports, moved sys.path.append to proper location, and formatted code.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — docstrings for module and functions.
    - **Imports:** ✅ Complete — removed unused 'NoReturn', organized imports properly.
    - **TypeHints:** ✅ Complete — proper type annotations.
    - **ErrorHandling:** ⚪ N/A — simple script with no error handling needed.
    - **CodeStyle:** ✅ Complete — formatted with black, fixed indentation.
    - **Performance:** ✅ Complete — efficient data generation loop.
    - **Architecture:** ✅ Complete — simple script structure.
    - **Testing:** ⚪ N/A — data generation script, not unit testable.
    - **Security:** ✅ Complete — safe file operations.
    - **Maintainability:** ✅ Complete — clear and concise code.
    - **ResearchCleanup:** ✅ Complete — production-ready code.
  - **Automated Checks Run:**
    - `flake8 cleaning/scripts/generate_synthetic_data.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy cleaning/scripts/generate_synthetic_data.py --ignore-missing-imports --no-strict-optional` — no issues (errors in other files)
    - `black --check --diff cleaning/scripts/generate_synthetic_data.py` — file unchanged
    - `isort --check-only --diff cleaning/scripts/generate_synthetic_data.py` — no issues
  - **FIXED:**
    - Removed unused import: 'NoReturn' (F401).
    - Moved sys.path.append before third-party imports to fix E402.
    - Added noqa comments for necessary E402 violations.
    - Applied black formatting. 

### cleaning\scripts\main_cleaning.py
- Status: completed
- Category: cleaning
- Audited: 2026-02-10
- Notes: 
  - **Summary:** Cleaned up unnecessary TYPE_CHECKING import, added noqa comments for false positive undefined names, and ensured proper type annotations.
  - **AUDIT:**
    - **Header&Doc:** ✅ Complete — comprehensive docstrings for module and functions.
    - **Imports:** ✅ Complete — proper imports with lazy loading for heavy dependencies.
    - **TypeHints:** ✅ Complete — type annotations present, with necessary ignores.
    - **ErrorHandling:** ✅ Complete — proper CUDA checks and error handling.
    - **CodeStyle:** ✅ Complete — formatted with black, passes linting.
    - **Performance:** ✅ Complete — mixed precision and early stopping support.
    - **Architecture:** ✅ Complete — well-structured training script with modular functions.
    - **Testing:** ⚠️ Partial — has optional args for testing, but no unit tests.
    - **Security:** ✅ Complete — safe file operations and model loading.
    - **Maintainability:** ✅ Complete — clear function separation and comments.
    - **ResearchCleanup:** ⚠️ Partial — has TODO comment but otherwise clean.
  - **Automated Checks Run:**
    - `flake8 cleaning/scripts/main_cleaning.py --max-line-length=127 --extend-ignore=E203,W503` — no issues
    - `mypy cleaning/scripts/main_cleaning.py --ignore-missing-imports --no-strict-optional` — no issues (errors in other files)
    - `black --check --diff cleaning/scripts/main_cleaning.py` — file unchanged
    - `isort --check-only --diff cleaning/scripts/main_cleaning.py` — no issues
  - **FIXED:**
    - Removed unnecessary TYPE_CHECKING import for torch.
    - Added noqa comments for false positive F821 undefined names (np, torchvision) due to global imports.
    - Added type: ignore for mypy issues with numpy operations. 

### cleaning\scripts\run.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### cleaning\utils\__init__.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### cleaning\utils\dataloader.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### cleaning\utils\loss.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### refinement\our_refinement\refinement_for_curves.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### cleaning\utils\synthetic_data_generation.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\downloaders\__init__.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\downloaders\download_dataset.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\__init__.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\base.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\cadvgdrawing.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\cubicasa.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\cubicasa_temp.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\floorplancad.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\fplanpoly.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\msd.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\quickdraw.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\resplan.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\processors\sketchgraphs.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### pipeline_unified.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset\run_processor.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### dataset_downloaders.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### fast_file_list.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### merging\merging_for_curves.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### merging\merging_for_lines.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### merging\utils\merging_functions.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### refinement\our_refinement\optimization_classes.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### batch_audit.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### regenerate_splits.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### run_pipeline.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### run_pipeline_hydra.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### run_web_ui.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### run_web_ui_demo.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### cleaning\models\SmallUnet\unet.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### cleaning\models\Unet\unet_model.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### cleaning\models\Unet\unet_parts.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### refinement\our_refinement\lines_refinement_functions.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### refinement\our_refinement\refinement_for_lines.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\aggregate_metrics.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\analyze_outputs.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\benchmark_performance.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\benchmark_pipeline.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\check_cuda.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\comprehensive_analysis.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\create_floorplancad_splits.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\debug_train_step.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\download_and_verify_floorplancad.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\evaluation_suite.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\extract_floorplancad_ground_truth.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\generate_diagnostics.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\generate_trace_report.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\lint_code.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\list_floorplancad_files.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\postprocess_floorplancad.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\precompute_floorplancad_targets.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\profile_performance.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\profile_pipeline_performance.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\profile_refinement_bottlenecks.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\report_utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_all_downloaders_test.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_batch_pipeline.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_cleaning.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_fine_tuning.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_security_scan.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_single_test_image.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_tests_local.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\run_trace_for_random.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\test_discover.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\test_evaluation.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\train_floorplancad.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\validate_env.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### scripts\verify_downloads.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\benchmark_merging.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_bezier_splatting.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_config_manager.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_early_stopping_integration.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_file_utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_file_utils_paths.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_hydra_config.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_integration.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_main_cleaning_args.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_merging_clip_and_assemble.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_merging_functions.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_mixed_precision_integration.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_refinement_integration.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_refinement_smoke.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_refinement_utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_regression.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_smoke.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### tests\test_vectorization.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\cad_export.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\color_utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\config_manager.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\dataloading.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\early_stopping.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\evaluation_utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\exceptions.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\geometric.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\logging.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\mixed_precision.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\patchify.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\performance_profiler.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\tensorboard.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\visualization.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\warnings.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\chunked.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics_primitives.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\line_drawings_dataset.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\prefetcher.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\preprocessed.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\preprocessing.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\graphics.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\path.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\primitives.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\raster_embedded.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\units.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\utils\common.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\utils\parse.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\utils\raster_utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\data\graphics\utils\splitting.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\loss_functions\lovacz_losses.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\loss_functions\supervised.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\metrics\raster_metrics.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\metrics\skeleton_metrics.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\metrics\vector_metrics.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\optimization\parameters.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\rendering\bezier_splatting.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\rendering\cairo.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\rendering\gpu_line_renderer.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\rendering\skeleton.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\rendering\utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\simplification\curve.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\simplification\detect_overlaps.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\simplification\join_qb.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\simplification\polyline.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\simplification\simplify.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### util_files\simplification\utils.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\models\common.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\models\fully_conv_net.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\models\generic.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\models\lstm.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\modules\base.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\modules\conv_modules.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\modules\fully_connected.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\modules\maybe_module.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\modules\output.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\modules\transformer.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\modules\_transformer_modules.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### vectorization\scripts\train_vectorization.py
- Status: pending
- Category: 
- Audited: 
- Notes: 

### web_ui\app.py
- Status: pending
- Category: 
- Audited: 
- Notes:
