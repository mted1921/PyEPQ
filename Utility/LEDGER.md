# Utility Package Conversion Ledger

Tracks every class in `gov.nist.microanalysis.Utility` through the
Java → Python port pipeline.

**Status symbols**
- `✓` — complete and follows current `_ver{G}_{N}_{F}` naming scheme
- `~` — port exists but filename does not follow the versioning scheme (old `_ver1` or user-renamed)
- `✗` — not yet started

**Columns**
- **Port file** — Python source relative to `Utility/`
- **Test harness** — parity test relative to `Utility/tests/`
- **Unresolved deps** — intra-Utility dependencies not yet ported (blocks this class)

---

## Tier 0 — Foundation (no intra-Utility dependencies)

| Class | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|---|:---:|---|---|
| `UtilException` | ~ | `UtilException.py` | ~ | `test_parity_utilexception.py` | — |
| `FindRoot` | ~ | `FindRoot.py` | ~ | `test_parity_findroot.py` | — |
| `HalfUpFormat` | ~ | `HalfUpFormat.py` | ~ | `test_parity_halfupformat.py` | — |
| `LazyEvaluate` | ~ | `LazyEvaluate.py` | ✓ | `test_parity_lazyevaluate_ver1_1_0.py` | — |

---

## Tier 1 — Depends only on Tier 0

| Class | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|---|:---:|---|---|
| `Math2` | ✓ | `Math2.py` | ✓ | `test_parity_math2.py` | — |
| `AdaptiveRungeKutta` | ~ | `AdaptiveRungeKutta.py` | ✓ | `test_parity_adaptiverungekutta_ver1_1_0.py` | — |
| `Simplex` | ~ | `Simplex.py` | ✓ | `test_parity_simplex_ver1_1_0.py` | — |
| `ExponentFormat` | ✗ | — | ✓ | `test_parity_exponentformat_ver1_1_0.py` | — |
| `HTMLFormat` | ✗ | — | ✓ | `test_parity_htmlformat_ver1_1_0.py` | — |
| `Integrator` | ✗ | — | ✓ | `test_parity_integrator_ver1_1_0.py` | — |
| `MultiDHistogram` | ~ | `MultiDHistogram.py` | ✓ | `test_parity_multidhistogram_ver1_1_0.py` | — |
| `PoissonDeviate` | ~ | `PoissonDeviate.py` | ✓ | `test_parity_poissondeviate_ver1_1_0.py` | — |
| `ProgressEvent` | ~ | `ProgressEvent.py` | ✓ | `test_parity_progressevent_ver1_1_0.py` | — |
| `MCIntegrator` | ✗ | — | ✓ | `test_parity_mcintegrator_ver1_1_0.py` | — |

---

## Tier 2 — Depends on Tier 1

| Class | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|---|:---:|---|---|
| `LinearRegression` | ✗ | — | ✗ | — | `LazyEvaluate` |
| `UncertainValue2` | ✗ | — | ✗ | — | `ExponentFormat` |
| `StageRelocation` | ✗ | — | ✗ | — | `Simplex` |
| `Translate2D` | ✗ | — | ✗ | — | `Simplex` |

---

## Tier 3 — Depends on Tier 2

| Class | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|---|:---:|---|---|
| `Constraint` | ✗ | — | ✗ | — | `UncertainValue2` |
| `DescriptiveStatistics` | ✗ | — | ✗ | — | `UncertainValue2` |
| `LevenbergMarquardt2` | ✗ | — | ✗ | — | `UncertainValue2` |
| `LinearLeastSquares` | ✗ | — | ✗ | — | `UncertainValue2` |
| `UncertainValue` | ✗ | — | ✗ | — | `UncertainValue2` |
| `UncertainValueMC` | ✗ | — | ✗ | — | `UncertainValue2` |

---

## Tier 4 — Depends on Tier 3

| Class | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|---|:---:|---|---|
| `LevenbergMarquardtConstrained` | ✗ | — | ✗ | — | `LevenbergMarquardt2`, `Constraint` |
| `LinearLeastSquaresMS` | ✗ | — | ✗ | — | `LinearLeastSquares` |
| `MCUncertaintyEngine` | ✗ | — | ✗ | — | `UncertainValueMC`, `DescriptiveStatistics` |

---

## Tier 5 — Depends on Tier 4

| Class | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|---|:---:|---|---|
| `LevenbergMarquardtParameterized` | ✗ | — | ✗ | — | `LevenbergMarquardtConstrained` |

---

## Grey nodes — Deferred (GUI / isolated, no dependency edges)

These classes have no edges in `Utility.dot` and depend on AWT/Swing or are
standalone utilities with no consumers inside Utility.  Port after all
connected classes are done.

| Class | Notes |
|---|---|
| `AutoComplete` | Swing autocomplete widget |
| `ComboBoxCellEditor` | Swing table cell editor |
| `CSVReader` | Standalone CSV parser |
| `EachRowEditor` | Swing table helper |
| `ElementComboBoxModel` | Swing combo-box model |
| `ElementTreePanel` | Swing tree panel |
| `Histogram` | 1-D histogram, no Utility deps |
| `HTMLList` | HTML list renderer |
| `HtmlSelection` | Clipboard HTML selection |
| `Interval` | Simple interval value object |
| `MemberSet` | Bit-set utility |
| `Pair` | Generic pair |
| `PrintUtilities` | AWT print helper |
| `SpectrumPropertiesTableModel` | Swing table model |
| `TextUtilities` | String formatting helpers |
| `Transform3D` | 3-D affine transform |

---

## Progress summary

| Tier | Classes | Port (✓/~) | Test ✓ |
|---|:---:|:---:|:---:|
| 0 — Foundation | 4 | 4 | 4 |
| 1 — Core | 10 | 6 | 10 |
| 2 | 4 | 0 | 0 |
| 3 | 6 | 0 | 0 |
| 4 | 3 | 0 | 0 |
| 5 | 1 | 0 | 0 |
| **Connected total** | **28** | **10** | **14** |
| Grey / deferred | 16 | 0 | 0 |
| **Grand total** | **44** | **10** | **14** |

> **Next milestone:** port remaining Tier 1 classes (`ExponentFormat`, `HTMLFormat`,
> `Integrator`, `MCIntegrator`) to unlock Tier 2.
> `UncertainValue2` is the highest-leverage single class: it unblocks 6 Tier 3 dependents.
