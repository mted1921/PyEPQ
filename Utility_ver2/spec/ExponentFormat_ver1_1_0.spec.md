# ExponentFormat Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.ExponentFormat`

Source: `src/gov/nist/microanalysis/Utility/ExponentFormat.java`

---

## Inbound dependencies (Java imports)
- `java.text.FieldPosition` — parameter to `format(double, StringBuffer, FieldPosition)` — no Python equivalent; use `int` offset pair
- `java.text.NumberFormat` — base class
- `java.text.ParsePosition` — parameter to `parse(String, ParsePosition)` — no Python equivalent; use `int` index
- `gov.nist.microanalysis.Utility.HalfUpFormat` — used internally to construct `mFormatter`

---

## Outbound dependents (callers of public methods)
`UncertainValue2.formatComponent(String, int)` constructs an `ExponentFormat` when the component value is very small. Not exhaustively audited beyond this.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `public ExponentFormat(int places)` | `__init__(self, places: int) -> None` | Clamps `places` to `max(1, places)` |
| `public StringBuffer format(double arg0, StringBuffer arg1, FieldPosition arg2)` | `def format(self, arg0: float) -> str` | Primary interface; `StringBuffer`/`FieldPosition` collapsed to simple return |
| `public StringBuffer format(long arg0, StringBuffer arg1, FieldPosition arg2)` | `def format_long(self, arg0: int) -> str` | Delegates to `format(float(arg0))` |
| `public Number parse(String arg0, ParsePosition arg1)` | `def parse(self, arg0: str, arg1: int = 0) -> Optional[float]` | `arg1` is the start index; returns `None` if parse fails |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static final long serialVersionUID` | Discard |
| `private static final String FMT2 = "</sup>"` | `_FMT2: str = "</sup>"` |
| `private static final String FMT1 = "&middot;10<sup>"` | `_FMT1: str = "&middot;10<sup>"` |
| `private final int mPlaces` | `self._mPlaces: int` |
| `private final NumberFormat mFormatter` | `self._mFormatter: HalfUpFormat` |

---

## Overloaded methods (split plan)
Two `format` overloads:
- `format(double, ...)` → `format(self, arg0: float) -> str` — primary
- `format(long, ...)` → `format_long(self, arg0: int) -> str` — delegates to `format(float(arg0))`

`parse` has no overload.

---

## Mutable-output methods
None. `format` appends to a new string. `parse` does not mutate caller-supplied state (the `ParsePosition` index update is folded into the return value).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. `java.text.*` is formatting-only.

---

## Abstract class strategy
Not applicable. Class is not abstract.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends NumberFormat` | Standalone class with `format(number: float) -> str` | No Python `NumberFormat` hierarchy |
| `StringBuffer arg1, FieldPosition arg2` params | Collapse to `-> str` return; position tracking dropped | R10 deviation — document in CHANGES |
| `ParsePosition arg1` (mutable index tracker) | `arg1: int = 0` start index; updated index returned as `Optional[tuple[float, int]]` | R10 deviation |
| `HalfUpFormat(sb.toString())` inside constructor | `HalfUpFormat(pattern=sb_str)` | Sibling module import |
| `Math.signum(arg0) >= 0` | `0.0 if arg0 == 0.0 else math.copysign(1.0, arg0) >= 0` | R8 — `signum(0)==0`, so sign="" for zero |
| `(int) la` where `la = Math.log10(|arg0|)` | `int(la)` — Python truncates toward zero, Java truncates toward zero; same behaviour | |
| `la > 0.0 ? (int) la : ((int) la) - 1` | `int(la) if la > 0.0 else int(la) - 1` | Preserve verbatim |

---

## Suspected Java bugs

**Bug 1 — `format(double)`: undefined output for `arg0 == 0`.**
Java source line: `final double la = Math.log10(Math.abs(arg0));`
When `arg0 == 0`, `Math.log10(0) = -Infinity`. Subsequent `int exp = la > 0.0 ? (int) la : ((int) la) - 1` would give `Integer.MIN_VALUE - 1` (undefined behavior in Java). In practice callers do not pass zero, but the method is not guarded.
Disposition: **Flag** — add a guard: `if arg0 == 0.0: return "<nobr>0</nobr>"` in the primary port and document as R10 deviation.

**Bug 2 — `Math.signum(arg0) >= 0` gives empty sign for zero.**
Java source line: `final String sign = Math.signum(arg0) >= 0 ? "" : "-";`
For `arg0 == 0`, `Math.signum(0) == 0`, so `sign = ""`. For `-0.0`, `Math.signum(-0.0) == -0.0`, which is `>= 0` in Java IEEE-754, so `sign = ""`. `-0.0` would format without a leading minus.
Disposition: **Preserve** — port the `>=0` comparison verbatim; document as `# JAVA-BUG-1`.

---

## Static init order
`FMT1` and `FMT2` are Java `private static final String` constants — initialized at class load. Port as class-level string constants. No ordering concern.

---

## Thread safety
Instances are not shared across threads in normal use. `mFormatter` is a final field initialized in the constructor — safe for read-only concurrent formatting after construction.
