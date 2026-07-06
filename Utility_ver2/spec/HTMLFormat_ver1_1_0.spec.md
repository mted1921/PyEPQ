# HTMLFormat Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.HTMLFormat`

Source: `src/gov/nist/microanalysis/Utility/HTMLFormat.java`

---

## Inbound dependencies (Java imports)
- `java.text.NumberFormat` — **superclass**; `HTMLFormat extends NumberFormat`. No direct Python equivalent — see Abstract class / base-class strategy below.
- `java.text.DecimalFormat` — type of the wrapped formatter field and of one constructor parameter. The Python port wraps a `HalfUpFormat` port instance (which already replaces `DecimalFormat`); accept any object exposing `format`/`parse`.
- `java.text.FieldPosition` — formatting cursor argument; **drop** from the Python API (Java-idiom only).
- `java.text.ParsePosition` — parse cursor argument; **drop** / simplify from the Python API.
- `java.text.ParseException` — caught internally in `format`; map to the `parse` failure path.
- `gov.nist.microanalysis.Utility.HalfUpFormat` — instantiated in the `String`-pattern constructor; import from the sibling port (filename from `UTILITY_LEDGER.md`).

---

## Outbound dependents (callers of public methods)
Used to render numbers in HTML scientific notation (`m &times; 10<sup>e</sup>`). Consumers call `format(double)`. Not exhaustively audited.

---

## Public API surface

`HTMLFormat extends NumberFormat`. The Java `NumberFormat` contract is expressed through `StringBuffer`/`FieldPosition`/`ParsePosition`; the Python port collapses these idioms to direct `str` returns.

### Static constants

| Java declaration | Python name | Notes |
|---|---|---|
| `private static final long serialVersionUID = 3292195639010835754L` | `serialVersionUID: int = 3292195639010835754` | **R1 exception** — port WITHOUT `_` prefix, expose as a public class attribute regardless of the Java `private` modifier |

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public HTMLFormat(String pattern)` | `__init__(self, pattern: str)` | Wraps `HalfUpFormat(pattern, False)`; dispatch by `isinstance(arg, str)` |
| `public HTMLFormat(DecimalFormat df)` | `__init__(self, df)` (formatter object) | Stores the supplied formatter directly; dispatch when the first arg is a formatter, not a `str` |

`__init__` must dispatch on argument type (P4 discipline): a `str` first argument takes the pattern path (build a `HalfUpFormat`); any other (a formatter object) takes the wrap-directly path.

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public StringBuffer format(double number, StringBuffer toAppendTo, FieldPosition pos)` | `def format(self, number: float) -> str` | **Primary method.** Returns the HTML scientific-notation string. Drop the `StringBuffer`/`FieldPosition` params (R10 deviation) |
| `public StringBuffer format(long number, StringBuffer toAppendTo, FieldPosition pos)` | `def format_long(self, number: int) -> str` | R4 — integer overload; delegates to the wrapped formatter with no HTML scientific notation |
| `public Number parse(String source, ParsePosition parsePosition)` | `def parse(self, source: str) -> float` | Delegates to the wrapped formatter's parse; drop `ParsePosition` (R10 deviation) |

`NumberFormat` also exposes convenience overloads (`format(double)`, `format(long)`) inherited from the base class. Provide a `format` that takes a single `float`/`int` and routes to `format` / `format_long` accordingly; document this as the Python public entry point replacing the inherited `NumberFormat.format(...)` family.

---

## Private / protected members

| Java | Python |
|---|---|
| `private final DecimalFormat mFormat` | `self._mFormat` — a `HalfUpFormat` port instance (or supplied formatter) exposing `format(float) -> str` and `parse(str) -> float` |

---

## Overloaded methods (split plan)
- `format` has a `double` overload and a `long` overload → `format` (float, HTML scientific) / `format_long` (int, plain delegate). R4.
- Constructor has a `String` overload and a `DecimalFormat` overload → single `__init__` dispatching on `isinstance(arg, str)`.

---

## Mutable-output methods
None. The Java versions append to a caller-supplied `StringBuffer`; the Python port returns a new `str` instead (R10 deviation). No R5 guard needed.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. The `java.text.*` types are formatting-only and are replaced/simplified per the dependency table above.

---

## Abstract class / base-class strategy
IS_ABSTRACT = False (concrete class), but it **extends the Java stdlib `NumberFormat`**, which has no Python equivalent.

Do **not** attempt to subclass a Python `NumberFormat`. Port `HTMLFormat` as a standalone `class HTMLFormat:` that wraps a `HalfUpFormat` port instance and exposes `format` / `format_long` / `parse`. Document the dropped `NumberFormat` inheritance and the dropped `StringBuffer`/`FieldPosition`/`ParsePosition` arguments as an R10 deviation in the docstring `CHANGES` section.

**Parity-test note (P6).** The Java parity side instantiates `HTMLFormat` and calls `format(double)`. The test must compare the Python `format(number)` string against the Java `format(number, new StringBuffer(), new FieldPosition(0)).toString()`. The test must **not** import any `NumberFormat`/`DecimalFormat`/`FieldPosition` symbol from the port — those names are intentionally absent. Verify the **format output string**, not the wrapped-formatter type.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends NumberFormat` | standalone class; no base | R10 deviation — Java stdlib base has no Python analogue |
| `StringBuffer format(double, StringBuffer, FieldPosition)` | `format(self, number: float) -> str` | R10 — drop Java I/O-cursor idioms |
| `serialVersionUID` (private) | `serialVersionUID: int = ...` (public) | R1 exception |
| `new HalfUpFormat(pattern, false)` | `HalfUpFormat(pattern, False)` from sibling port | R3 / sibling import |
| `(int) Math.log10(Math.abs(number))` | `int(math.log10(abs(number)))` | R7 — Java `(int)` truncates toward zero |
| `Math.pow(10.0, pow)` | `math.pow(10.0, pow)` or `10.0 ** pow` | |
| `Integer.toString(pow)` | `str(pow)` | |
| `mFormat.parse(...).doubleValue()` | `self._mFormat.parse(...)` returning `float` | |
| `catch (ParseException e)` | `try/except` around the parse, swallow as the Java does (`// Just ignore it...`) | Faithful port |
| string concat `" &times; 10<sup>" + ... + "</sup>"` | f-string; preserve the exact HTML markup verbatim | Exact output parity |

---

## Suspected Java bugs

**JAVA-BUG-1 — always-true scientific-notation guard makes the plain-format branch dead code.**
Java source line: `if ((pow <= 2) || (pow >= 2)) {`
For any integer `pow`, `pow <= 2 || pow >= 2` is **always true** (every integer is `≤ 2` or `≥ 2`), so the `else` branch (`return mFormat.format(number, toAppendTo, pos)`) is unreachable. Every value is rendered in HTML scientific notation, even ordinary mid-range numbers. The intended guard was almost certainly `(pow <= -2) || (pow >= 2)` (scientific notation only for very small or very large magnitudes).
Disposition: **Preserve** — port the always-true condition verbatim behind a `# JAVA-BUG-1` marker quoting the exact line; add a `BUG_LEDGER` entry (`has_strict_variant=True`). The `format_strict` companion uses `(pow <= -2) or (pow >= 2)` so the plain-format branch is reachable.

Note also the `pow` adjustment block (`n >= 10.0` → `pow += 1`; `n < 1.0` → `pow -= 1`) re-normalises the mantissa into `[1, 10)`; port it verbatim — it is correct behaviour, not a bug.

---

## Static init order
`serialVersionUID` is a compile-time constant — initialize in the class body. No ordering concern.

---

## Thread safety
Not thread-safe in the strict sense (the wrapped `DecimalFormat`/`HalfUpFormat` is not), but `HTMLFormat` holds no mutable instance state beyond the final wrapped formatter. Treat as effectively immutable after construction; document that the wrapped formatter is not safe for concurrent `format`/`parse` from multiple threads.
