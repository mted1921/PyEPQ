# HalfUpFormat Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.HalfUpFormat`

Source: `src/gov/nist/microanalysis/Utility/HalfUpFormat.java`

---

## Inbound dependencies (Java imports)
- `java.math.RoundingMode` — `HALF_UP` constant
- `java.text.DecimalFormat` — superclass; no Python equivalent
- `java.text.DecimalFormatSymbols` — used to set a custom grouping separator
- `java.text.NumberFormat` — declared type in `adaptiveFormat`

---

## Outbound dependents (callers of public methods)

Used throughout EPQ for number formatting in reports, tables, and GUI labels.
`adaptiveFormat` is the primary public API called by reporting code.
Any EPQ class that constructs `new HalfUpFormat(pattern)` or
`new HalfUpFormat(pattern, true)` to format individual values.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `HalfUpFormat()` | `__init__(self)` | No-arg constructor |
| `HalfUpFormat(boolean spaceGrouping)` | `from_space_grouping(cls, spaceGrouping: bool)` | `@classmethod` |
| `HalfUpFormat(String pattern)` | `from_pattern(cls, pattern: str)` | `@classmethod` |
| `HalfUpFormat(String pattern, boolean spaceGrouping)` | `from_pattern_space_grouping(cls, pattern: str, spaceGrouping: bool)` | `@classmethod` |
| `HalfUpFormat(String pattern, DecimalFormatSymbols symbols)` | `from_pattern_symbols(cls, pattern: str, symbols)` | `@classmethod` |
| `String format(double number)` | `format(self, number: float) -> str` | Core format method |
| `static String adaptiveFormat(double value, int digits, double threshold)` | `@staticmethod adaptiveFormat(value: float, digits: int, threshold: float) -> str` | |

`setGroupingSize` and `setGroupingUsed` are also ported (called internally by
the constructors through `super()` chains).

---

## Private / protected members

| Java | Python |
|---|---|
| `DecimalFormat` state (pattern, grouping, symbols) | `self._pattern: str`, `self._space_grouping: bool`, `self._grouping_sep: str` |

---

## Overloaded methods (split plan)
Five constructor overloads → `__init__` + four `@classmethod` factory methods (R4).
The no-arg `__init__` covers the `HalfUpFormat()` case.
Each classmethod delegates back to `__init__` after resolving its arguments.

---

## Mutable-output methods
None.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- **`java.text.DecimalFormat`** — no Python equivalent. Reimplemented from scratch
  using `decimal.Decimal` with `ROUND_HALF_UP` for correct rounding.
- **`java.text.DecimalFormatSymbols`** — minimal Python shim (`DecimalFormatSymbols`)
  in the same module. Only `setGroupingSeparator` / `getGroupingSeparator` and
  `getInstance()` are needed.
- **`java.text.NumberFormat`** — declared type only; replaced by `HalfUpFormat`
  directly (duck typing, no shim needed).
- **`RoundingMode.HALF_UP`** — `decimal.ROUND_HALF_UP` from Python stdlib.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends DecimalFormat` | Standalone class; no equivalent in Python | R2 |
| Five constructor overloads | `__init__` + four `@classmethod` factory methods | R4 |
| `DecimalFormatSymbols` | Minimal shim class in same module | R3 |
| `RoundingMode.HALF_UP` | `decimal.ROUND_HALF_UP` | R2 |
| `Math.max`, `Math.abs`, `Math.floor`, `Math.log10`, `Math.pow`, `Math.round` | `max()`, `abs()`, `math.floor`, `math.log10`, `math.pow`, `round()` | R2 |
| Pattern `"0"*n + "." + "0"*n + "E0"` subset | `_apply_pattern` + `_format_scientific_halfup` helpers | R2 |
| MEDIUM MATHEMATICAL SPACE (`U+2006`) grouping separator | `self._grouping_sep = " "` when `spaceGrouping=True` | R2 |

---

## Suspected Java bugs
None. The class exists specifically to fix Java's default `HALF_EVEN` (banker's)
rounding; `HALF_UP` is the intended behaviour. `BUG_LEDGER` is empty.

---

## Static init order
None. No static blocks or cross-class static references.

---

## Thread safety
Not documented. Pattern state is set at construction and not mutated thereafter;
instances are effectively immutable after construction and safe to share across threads.
