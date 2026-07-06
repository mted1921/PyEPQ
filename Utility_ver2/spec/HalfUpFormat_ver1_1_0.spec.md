# HalfUpFormat Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.HalfUpFormat`

Source: `src/gov/nist/microanalysis/Utility/HalfUpFormat.java`

---

## Inbound dependencies (Java imports)
- `java.math.RoundingMode` — used to set HALF_UP rounding on the formatter
- `java.text.DecimalFormat` — base class
- `java.text.DecimalFormatSymbols` — used to set the grouping separator to U+2006
- `java.text.NumberFormat` — used as return/parameter type in `adaptiveFormat`

No EPQ-internal imports.

---

## Outbound dependents (callers of public methods)
`ExponentFormat` constructs a `HalfUpFormat` internally. `UncertainValue2.formatComponent` also constructs one. Broadly used throughout the EPQ library for formatted output.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `public HalfUpFormat()` | `__init__(self, space_grouping: bool = False)` | Default: no pattern, no grouping |
| `public HalfUpFormat(boolean spaceGrouping)` | covered by default arg above | |
| `public HalfUpFormat(String pattern)` | `__init__(self, pattern: str, space_grouping: bool = False)` | Overload dispatched by arg type |
| `public HalfUpFormat(String pattern, boolean spaceGrouping)` | covered by default arg above | |
| `public HalfUpFormat(String pattern, DecimalFormatSymbols symbols)` | `__init__(self, pattern: str, symbols: Optional[dict] = None)` | `symbols` maps grouping-separator and decimal-separator overrides |
| `static public String adaptiveFormat(double number, int precision, double sciRange)` | `@staticmethod def adaptiveFormat(number: float, precision: int, sciRange: float) -> str` | Contains Math.round — see bugs |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static final long serialVersionUID` | Discard — not applicable in Python |
| `private final static char MEDIUM_MATHEMATICAL_SPACE = ' '` | `_MEDIUM_MATHEMATICAL_SPACE: str = ' '` |

---

## Overloaded methods (split plan)
Five Java constructors collapse to two Python `__init__` signatures dispatched by argument type:

- **No-arg / bool**: `__init__(self, space_grouping: bool = False)` — handles `HalfUpFormat()` and `HalfUpFormat(boolean)`.
- **Pattern / pattern+bool**: `__init__(self, pattern: str, space_grouping: bool = False)` — handles `HalfUpFormat(String)` and `HalfUpFormat(String, boolean)`.
- **Pattern + symbols**: `__init__(self, pattern: str, symbols: Optional[dict])` — handles `HalfUpFormat(String, DecimalFormatSymbols)`.

`__init__` dispatches on `isinstance(pattern, str)` to select the path. Classmethods provided for Java API surface:
- `from_bool(space_grouping: bool) -> HalfUpFormat`
- `from_pattern(pattern: str, space_grouping: bool = False) -> HalfUpFormat`
- `from_pattern_symbols(pattern: str, symbols: dict) -> HalfUpFormat`

`adaptiveFormat` is static, no overloads.

---

## Mutable-output methods
None. All methods return new strings or new formatter instances.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. `java.text.*` has no Jama/Swing touchpoints.

---

## Abstract class strategy
Not applicable. Class is not abstract.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends DecimalFormat` | No Python equivalent for `DecimalFormat`. Port as a standalone class with `format(number: float) -> str` method using Python `decimal` module with `ROUND_HALF_UP`. | Java class hierarchy not portable |
| `RoundingMode.HALF_UP` | `decimal.ROUND_HALF_UP` | Python `round()` uses HALF_EVEN (banker's rounding); must use `decimal` module explicitly |
| `DecimalFormatSymbols` (grouping separator) | `symbols` dict: `{"grouping_separator": " ", ...}` | Java class has no Python equivalent |
| `format(number)` on a NumberFormat instance | `def format(self, number: float) -> str` | Primary public interface |
| `setGroupingSize(3)` / `setGroupingUsed(true)` | Apply in `format()` using Python string formatting with grouping | |
| Pattern string (e.g. `"0.000E0"`) | Translate to Python format spec inside `format()` | Pattern translation is the main complexity |
| `Math.floor(Math.log10(Math.abs(number))) + 1` | `int(math.floor(math.log10(abs(number)))) + 1` | Zero input: `log10(0) = -inf`; guard against zero separately |

---

## Suspected Java bugs

**Bug 1 — `adaptiveFormat` uses `Math.round` (HALF_UP-floor) not HALF_EVEN.**
Java source line: `final double rounded = Math.round(number / div) * div;`
Java `Math.round(x)` is `floor(x + 0.5)`, not Python `round()` (HALF_EVEN). This diverges at 0.5 boundaries.
Disposition: **Preserve** — the literal port uses `int(math.floor(number/div + 0.5)) * div`. The scipy primary may use Python's `round()` but must document the divergence as `SCIPY-DEV-1`.

**Bug 2 — `adaptiveFormat` undefined for `number == 0`.**
Java source line: `final int nn = (int) Math.floor(Math.log10(Math.abs(number))) + 1;`
When `number == 0`, `Math.log10(0)` is `-Infinity`, `Math.floor(-Inf)` is `-Inf`, cast to `int` yields `Integer.MIN_VALUE` in Java (implementation-defined). The sciRange branch will be taken (since `0 < 1/sciRange`), so this path is actually unreachable for `number == 0` when `sciRange > 0`. Not a true bug at runtime but the inner path is broken for zero.
Disposition: **Flag** — annotate the unreachable zero path; do not fix silently.

---

## Static init order
None. `MEDIUM_MATHEMATICAL_SPACE` is a compile-time constant.

---

## Thread safety
Instances of `HalfUpFormat` are not shared across threads in normal use. `adaptiveFormat` is static and creates a local formatter each call — safe.
