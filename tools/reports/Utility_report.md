# Dependency Map: gov.nist.microanalysis.Utility

- **Files analyzed**: 44
- **Nodes (classes)**: 44
- **Internal edges**: 42
- **Weakly-connected components**: 17
- **Strongly-connected clusters (size > 1)**: 0
- **Simple cycles enumerated**: 0
- **Dead-code candidates**: 16

## Recommended port order (first 50)

Topological depth ascending, then in_degree descending. Port foundations first; classes inside a cycle must be ported as a group.

| # | Class | Depth | In | Out | In Cycle |
|---|-------|------:|---:|---:|:--------:|
| 1 | FindRoot | 0 | 1 | 0 |  |
| 2 | HalfUpFormat | 0 | 4 | 0 |  |
| 3 | Math2 | 1 | 10 | 1 |  |
| 4 | UtilException | 0 | 5 | 0 |  |
| 5 | ExponentFormat | 1 | 1 | 1 |  |
| 6 | UncertainValue2 | 2 | 9 | 4 |  |
| 7 | Constraint | 3 | 2 | 1 |  |
| 8 | LevenbergMarquardt2 | 3 | 2 | 2 |  |
| 9 | Simplex | 1 | 2 | 1 |  |
| 10 | DescriptiveStatistics | 3 | 1 | 1 |  |
| 11 | UncertainValueMC | 3 | 1 | 1 |  |
| 12 | LazyEvaluate | 0 | 1 | 0 |  |
| 13 | LinearLeastSquares | 3 | 1 | 2 |  |
| 14 | LevenbergMarquardtConstrained | 4 | 1 | 3 |  |
| 15 | AdaptiveRungeKutta | 1 | 1 | 1 |  |
| 16 | UncertainValue | 3 | 0 | 1 |  |
| 17 | Translate2D | 2 | 0 | 4 |  |
| 18 | Transform3D | 0 | 0 | 0 |  |
| 19 | TextUtilities | 0 | 0 | 0 |  |
| 20 | StageRelocation | 2 | 0 | 2 |  |
| 21 | SpectrumPropertiesTableModel | 0 | 0 | 0 |  |
| 22 | ProgressEvent | 2 | 0 | 1 |  |
| 23 | PrintUtilities | 0 | 0 | 0 |  |
| 24 | PoissonDeviate | 2 | 0 | 1 |  |
| 25 | Pair | 0 | 0 | 0 |  |
| 26 | MultiDHistogram | 2 | 0 | 1 |  |
| 27 | MemberSet | 0 | 0 | 0 |  |
| 28 | MCUncertaintyEngine | 4 | 0 | 3 |  |
| 29 | MCIntegrator | 2 | 0 | 1 |  |
| 30 | LinearRegression | 2 | 0 | 2 |  |
| 31 | LinearLeastSquaresMS | 4 | 0 | 1 |  |
| 32 | LevenbergMarquardtParameterized | 5 | 0 | 4 |  |
| 33 | Interval | 0 | 0 | 0 |  |
| 34 | Integrator | 2 | 0 | 2 |  |
| 35 | HtmlSelection | 0 | 0 | 0 |  |
| 36 | HTMLList | 0 | 0 | 0 |  |
| 37 | HTMLFormat | 1 | 0 | 1 |  |
| 38 | Histogram | 0 | 0 | 0 |  |
| 39 | ElementTreePanel | 0 | 0 | 0 |  |
| 40 | ElementComboBoxModel | 0 | 0 | 0 |  |
| 41 | EachRowEditor | 0 | 0 | 0 |  |
| 42 | CSVReader | 0 | 0 | 0 |  |
| 43 | ComboBoxCellEditor | 0 | 0 | 0 |  |
| 44 | AutoComplete | 0 | 0 | 0 |  |

## Most depended-upon (high in_degree)

| Class | In | Instability | PageRank |
|-------|---:|------------:|---------:|
| Math2 | 10 | 0.091 | 0.1053 |
| UncertainValue2 | 9 | 0.308 | 0.0943 |
| UtilException | 5 | 0.000 | 0.0723 |
| HalfUpFormat | 4 | 0.000 | 0.0730 |
| Constraint | 2 | 0.333 | 0.0192 |
| LevenbergMarquardt2 | 2 | 0.500 | 0.0192 |
| Simplex | 2 | 0.333 | 0.0202 |
| AdaptiveRungeKutta | 1 | 0.500 | 0.0176 |
| DescriptiveStatistics | 1 | 0.500 | 0.0158 |
| ExponentFormat | 1 | 0.500 | 0.0324 |

## Most depending (high out_degree)

| Class | Out | Instability | Reachable |
|-------|----:|------------:|----------:|
| LevenbergMarquardtParameterized | 4 | 1.000 | 9 |
| Translate2D | 4 | 1.000 | 5 |
| UncertainValue2 | 4 | 0.308 | 5 |
| LevenbergMarquardtConstrained | 3 | 0.750 | 8 |
| MCUncertaintyEngine | 3 | 1.000 | 8 |
| Integrator | 2 | 1.000 | 2 |
| LevenbergMarquardt2 | 2 | 0.500 | 6 |
| LinearLeastSquares | 2 | 0.667 | 6 |
| LinearRegression | 2 | 1.000 | 3 |
| StageRelocation | 2 | 1.000 | 4 |

## Dead-code candidates

(Zero internal in-degree AND zero internal out-degree -- might be entry points, test fixtures, or genuinely unused.)

- AutoComplete
- ComboBoxCellEditor
- CSVReader
- EachRowEditor
- ElementComboBoxModel
- ElementTreePanel
- Histogram
- HTMLList
- HtmlSelection
- Interval
- MemberSet
- Pair
- PrintUtilities
- SpectrumPropertiesTableModel
- TextUtilities
- Transform3D