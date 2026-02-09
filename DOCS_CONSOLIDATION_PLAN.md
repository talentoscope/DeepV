# Documentation Consolidation Plan

> **Status**: ✅ CONSOLIDATION COMPLETE (February 9, 2026)  
> **Next Action**: After final commit, move this plan to `docs/archive/DOCS_CONSOLIDATION_PLAN_2026-02-09.md` for historical reference

**Analysis Date:** February 9, 2026  
**Last Updated**: February 9, 2026 - HIGH PRIORITY & MEDIUM PRIORITY consolidations COMPLETED ✅  
**Total Documentation Files:** 24 markdown files

## Executive Summary

The repository has good documentation but suffers from:
- **Fragmentation**: TODO.md, REFACTOR.md, and PLAN.md overlap significantly
- **Archived content**: Some docs marked "COMPLETED" or "ARCHIVED" still in root
- **Misplaced files**: PR template in root instead of .github/
- **Temporary files**: CLEANUP_SUMMARY.md from recent cleanup
- **Redundancy**: FloorPlanCAD performance warnings repeated in 5+ files

---

## Recommended Actions

### � High Priority - COMPLETED ✅

#### 1. Remove Temporary Files ✅ DONE
- **Removed**: `CLEANUP_SUMMARY.md`

#### 2. Move Misplaced Files ✅ DONE
- **Moved**: `PR_DESCRIPTION.md` → `.github/pull_request_template.md`

#### 3. Archive Completed/Deferred Documentation ✅ DONE
Create `docs/archive/` directory and moved:
- **Archived**: `VECTORIZATION_TRAINING_SETUP.md`
- **Archived**: `FUTURE_ENHANCEMENTS.md`

### � Medium Priority - COMPLETED ✅

#### 4. Merge Development Tracking Documents ✅ DONE

**Consolidation Results**:
- Created comprehensive `DEVELOPMENT.md` consolidating TODO.md + REFACTOR.md
- Merged 646 lines of development tracking into single, organized document
- Updated references in:
  - `.github/copilot-instructions.md` (reference to DEVELOPMENT.md)
  - `README.md` (reference to DEVELOPMENT.md)
- Deleted redundant files:
  - `TODO.md` (301 lines) - ✅ Consolidated
  - `REFACTOR.md` (345 lines) - ✅ Consolidated

**New File Structure**:
- `DEVELOPMENT.md` (614 lines) - Active development status, action items, code quality tracking
- `PLAN.md` (360 lines) - Strategic roadmap, phases, architecture decisions
- `README.md` - Project overview with links to DEVELOPMENT.md

#### 5. Consolidate FloorPlanCAD Performance Warnings ✅ IN PROGRESS

Ready to execute next phase - see section below.

### 🟡 Low Priority - Future Optimization

#### 6. Reorganize Documentation Structure

**Current**: All docs in root directory (cluttered)

**Proposed Structure**:
```
/
├── README.md                    # Main entry point
├── INSTALL.md                   # Installation guide
├── CONTRIBUTING.md              # Contribution guidelines
├── LICENSE                      # License file
│
├── docs/
│   ├── DEVELOPER.md            # Developer guide
│   ├── DEVELOPMENT.md          # Development status (merged TODO+REFACTOR)
│   ├── PLAN.md                 # Strategic roadmap
│   ├── DATA_SOURCES.md         # Dataset information
│   ├── QA.md                   # FAQ
│   │
│   └── archive/
│       ├── VECTORIZATION_TRAINING_SETUP.md
│       ├── FUTURE_ENHANCEMENTS.md
│       └── CLEANUP_2026-02-09.md  (rename CLEANUP_SUMMARY)
│
├── .github/
│   ├── pull_request_template.md  # PR template (moved)
│   ├── copilot-instructions.md
│   └── ISSUE_TEMPLATE/
```

**Benefits**:
- Cleaner root directory
- Logical grouping
- Preserved history in archive/
- GitHub integration works properly

#### 7. Update Cross-References

After moves/consolidations, update all docs that reference moved files:
- Update copilot-instructions.md links
- Update CONTRIBUTING.md references
- Update README.md table of contents
- Update DEVELOPER.md links

---

## Detailed File Analysis

### Root Documentation (Categorized)

#### ✅ Keep As-Is (Essential, Well-Maintained)
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| README.md | 365 | Main documentation, project overview | ✅ Keep |
| CONTRIBUTING.md | 248 | Contribution guidelines | ✅ Keep |
| INSTALL.md | 247 | Installation instructions | ✅ Keep |
| QA.md | 149 | FAQ | ✅ Keep |
| DATA_SOURCES.md | 176 | Dataset catalog | ✅ Keep |

#### 🔄 Keep But Modify (Need Updates/Consolidation)
| File | Lines | Issues | Action |
|------|-------|--------|--------|
| DEVELOPER.md | 406 | Overlaps with TODO/PLAN | Keep, update references after consolidation |
| PLAN.md | 360 | Some overlap with TODO | Keep as strategic doc, clarify scope |
| TODO.md | 301 | Overlaps with REFACTOR | Merge with REFACTOR → DEVELOPMENT.md |
| REFACTOR.md | 345 | Overlaps with TODO | Merge with TODO → DEVELOPMENT.md |

#### 📦 Archive (Completed/Deferred)
| File | Lines | Reason | Action |
|------|-------|--------|--------|
| VECTORIZATION_TRAINING_SETUP.md | 327 | Marked "COMPLETED" | Move to docs/archive/ |
| FUTURE_ENHANCEMENTS.md | ~400 | Marked "deferred", research-only | Move to docs/archive/ |

#### 🗑️ Remove (Temporary/Misplaced)
| File | Lines | Reason | Action |
|------|-------|--------|--------|
| CLEANUP_SUMMARY.md | 201 | Temporary cleanup docs | Delete |
| PR_DESCRIPTION.md | 108 | Should be .github template | Move to .github/ |

### Subdirectory Documentation

#### ✅ Keep All (Valuable Module-Specific Docs)
| File | Purpose | Status |
|------|---------|--------|
| scripts/README_batch.md | Batch processing guide | ✅ Well-written |
| scripts/README_benchmarking.md | Comprehensive benchmarking docs (479 lines) | ✅ Excellent |
| config/README.md | Hydra configuration guide (366 lines) | ✅ Comprehensive |
| cleaning/README.md | Cleaning module docs | ✅ Keep |
| util_files/README.md | Utilities documentation | ✅ Keep |
| cad/README.md | CAD export docs | ✅ Keep |
| web_ui/README.md | Web UI docs | ✅ Keep |
| tests/e2e/README.md | E2E testing docs | ✅ Keep |
| .github/copilot-instructions.md | AI assistant guidelines | ✅ Keep |

---

## Implementation Steps

### Step 1: Create Archive Directory
```bash
mkdir -p docs/archive
```

### Step 2: Archive Completed/Deferred Docs
```bash
git mv VECTORIZATION_TRAINING_SETUP.md docs/archive/
git mv FUTURE_ENHANCEMENTS.md docs/archive/
git mv CLEANUP_SUMMARY.md docs/archive/CLEANUP_2026-02-09.md
```

### Step 3: Move PR Template
```bash
git mv PR_DESCRIPTION.md .github/pull_request_template.md
```

### Step 4: Consolidate Development Tracking
```bash
# Create merged development tracking doc
# (Manual: merge TODO.md + REFACTOR.md → DEVELOPMENT.md)
git add DEVELOPMENT.md
git rm TODO.md REFACTOR.md
```

### Step 5: Update Cross-References
```bash
# Update all files referencing moved/merged docs
# (Manual: search and replace in affected files)
```

### Step 6: Move Strategic Docs to docs/
```bash
git mv DEVELOPER.md docs/
git mv PLAN.md docs/
git mv DATA_SOURCES.md docs/
git mv QA.md docs/
```

### Step 7: Update README.md
```markdown
# Update table of contents and links to reflect new structure
```

---

## Success Metrics

After consolidation:
- ✅ Root directory has ≤5 markdown files (README, INSTALL, CONTRIBUTING, LICENSE, CHANGELOG)
- ✅ Zero redundant/overlapping development tracking docs
- ✅ All archived/historical docs in docs/archive/
- ✅ GitHub PR template functional
- ✅ All cross-references working
- ✅ Documentation findability improved

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Broken links | Medium | Comprehensive search/replace, verify all references |
| Lost historical context | Low | Archive instead of delete, preserve in git history |
| Confusion during transition | Medium | Clear commit messages, update CHANGELOG |
| Copilot instructions outdated | Low | Update .github/copilot-instructions.md explicitly |

---

## Timeline Estimate

- **Step 1-3** (Archive & Move): 15 minutes
- **Step 4** (Consolidate development docs): 45-60 minutes  
- **Step 5** (Update cross-references): 30-45 minutes
- **Step 6-7** (Reorganize docs/): 15 minutes
- **Testing & Validation**: 15 minutes

**Total**: ~2-2.5 hours of focused work

---

## Appendix: File Size Analysis

```
Large Documentation Files (>300 lines):
  1. scripts/README_benchmarking.md (479 lines) - Comprehensive, keep
  2. DEVELOPER.md (406 lines) - Keep
  3. config/README.md (366 lines) - Comprehensive, keep
  4. README.md (365 lines) - Keep, primary entry
  5. PLAN.md (360 lines) - Keep, strategic
  6. REFACTOR.md (345 lines) - Consolidate
  7. VECTORIZATION_TRAINING_SETUP.md (327 lines) - Archive
  8. TODO.md (301 lines) - Consolidate

Medium Documentation Files (150-300):
  9. CONTRIBUTING.md (248 lines) - Keep
  10. INSTALL.md (247 lines) - Keep
  11. CLEANUP_SUMMARY.md (201 lines) - Remove
  12. DATA_SOURCES.md (176 lines) - Keep

Smaller focused docs (<150 lines): All appropriate for their scope
```

---

## Recommendation Summary

**Immediate Actions** (30 minutes):
1. ✅ Delete `CLEANUP_SUMMARY.md`
2. ✅ Move `PR_DESCRIPTION.md` to `.github/pull_request_template.md`
3. ✅ Archive `VECTORIZATION_TRAINING_SETUP.md` and `FUTURE_ENHANCEMENTS.md`

**Near-Term Actions** (1-2 hours when time permits):
4. Consolidate TODO.md + REFACTOR.md → DEVELOPMENT.md
5. Simplify FloorPlanCAD performance warnings (single authoritative source + brief links)
6. Move strategic docs to docs/ subdirectory

**Priority**: Do immediate actions now; schedule near-term actions for dedicated documentation sprint.
