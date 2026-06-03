# Dependency Map: gov.nist.microanalysis.Utility

- **Files analyzed**: 44
- **Nodes (classes)**: 44
- **Internal edges**: 3
- **Weakly-connected components**: 41
- **Strongly-connected clusters (size > 1)**: 0
- **Simple cycles enumerated**: 0
- **Dead-code candidates**: 38

## Recommended port order (first 50)

Topological depth ascending, then in_degree descending. Port foundations first; classes inside a cycle must be ported as a group.

| # | Class | Depth | In | Out | In Cycle |
|---|-------|------:|---:|---:|:--------:|
| 1 | LinearLeastSquares | 0 | 1 | 0 |  |
| 2 | LevenbergMarquardt2 | 0 | 1 | 0 |  |
| 3 | AdaptiveRungeKutta | 0 | 1 | 0 |  |
| 4 | UtilException | 0 | 0 | 0 |  |
| 5 | UncertainValueMC | 0 | 0 | 0 |  |
| 6 | UncertainValue2 | 0 | 0 | 0 |  |
| 7 | UncertainValue | 0 | 0 | 0 |  |
| 8 | Translate2D | 0 | 0 | 0 |  |
| 9 | Transform3D | 0 | 0 | 0 |  |
| 10 | TextUtilities | 0 | 0 | 0 |  |
| 11 | StageRelocation | 0 | 0 | 0 |  |
| 12 | SpectrumPropertiesTableModel | 0 | 0 | 0 |  |
| 13 | Simplex | 0 | 0 | 0 |  |
| 14 | ProgressEvent | 0 | 0 | 0 |  |
| 15 | PrintUtilities | 0 | 0 | 0 |  |
| 16 | PoissonDeviate | 0 | 0 | 0 |  |
| 17 | Pair | 0 | 0 | 0 |  |
| 18 | MultiDHistogram | 0 | 0 | 0 |  |
| 19 | MemberSet | 0 | 0 | 0 |  |
| 20 | MCUncertaintyEngine | 0 | 0 | 0 |  |
| 21 | MCIntegrator | 0 | 0 | 0 |  |
| 22 | Math2 | 0 | 0 | 0 |  |
| 23 | LinearRegression | 0 | 0 | 0 |  |
| 24 | LinearLeastSquaresMS | 1 | 0 | 1 |  |
| 25 | LevenbergMarquardtParameterized | 0 | 0 | 0 |  |
| 26 | LevenbergMarquardtConstrained | 1 | 0 | 1 |  |
| 27 | LazyEvaluate | 0 | 0 | 0 |  |
| 28 | Interval | 0 | 0 | 0 |  |
| 29 | Integrator | 1 | 0 | 1 |  |
| 30 | HtmlSelection | 0 | 0 | 0 |  |
| 31 | HTMLList | 0 | 0 | 0 |  |
| 32 | HTMLFormat | 0 | 0 | 0 |  |
| 33 | Histogram | 0 | 0 | 0 |  |
| 34 | HalfUpFormat | 0 | 0 | 0 |  |
| 35 | FindRoot | 0 | 0 | 0 |  |
| 36 | ExponentFormat | 0 | 0 | 0 |  |
| 37 | ElementTreePanel | 0 | 0 | 0 |  |
| 38 | ElementComboBoxModel | 0 | 0 | 0 |  |
| 39 | EachRowEditor | 0 | 0 | 0 |  |
| 40 | DescriptiveStatistics | 0 | 0 | 0 |  |
| 41 | CSVReader | 0 | 0 | 0 |  |
| 42 | Constraint | 0 | 0 | 0 |  |
| 43 | ComboBoxCellEditor | 0 | 0 | 0 |  |
| 44 | AutoComplete | 0 | 0 | 0 |  |

## Most depended-upon (high in_degree)

| Class | In | Instability | PageRank |
|-------|---:|------------:|---------:|
| AdaptiveRungeKutta | 1 | 0.000 | 0.0397 |
| LevenbergMarquardt2 | 1 | 0.000 | 0.0397 |
| LinearLeastSquares | 1 | 0.000 | 0.0397 |
| AutoComplete | 0 | 0.000 | 0.0215 |
| ComboBoxCellEditor | 0 | 0.000 | 0.0215 |
| Constraint | 0 | 0.000 | 0.0215 |
| CSVReader | 0 | 0.000 | 0.0215 |
| DescriptiveStatistics | 0 | 0.000 | 0.0215 |
| EachRowEditor | 0 | 0.000 | 0.0215 |
| ElementComboBoxModel | 0 | 0.000 | 0.0215 |

## Most depending (high out_degree)

| Class | Out | Instability | Reachable |
|-------|----:|------------:|----------:|
| Integrator | 1 | 1.000 | 1 |
| LevenbergMarquardtConstrained | 1 | 1.000 | 1 |
| LinearLeastSquaresMS | 1 | 1.000 | 1 |
| AdaptiveRungeKutta | 0 | 0.000 | 0 |
| AutoComplete | 0 | 0.000 | 0 |
| ComboBoxCellEditor | 0 | 0.000 | 0 |
| Constraint | 0 | 0.000 | 0 |
| CSVReader | 0 | 0.000 | 0 |
| DescriptiveStatistics | 0 | 0.000 | 0 |
| EachRowEditor | 0 | 0.000 | 0 |

## Dead-code candidates

(Zero internal in-degree AND zero internal out-degree -- might be entry points, test fixtures, or genuinely unused.)

- AutoComplete
- ComboBoxCellEditor
- Constraint
- CSVReader
- DescriptiveStatistics
- EachRowEditor
- ElementComboBoxModel
- ElementTreePanel
- ExponentFormat
- FindRoot
- HalfUpFormat
- Histogram
- HTMLFormat
- HTMLList
- HtmlSelection
- Interval
- LazyEvaluate
- LevenbergMarquardtParameterized
- LinearRegression
- Math2
- MCIntegrator
- MCUncertaintyEngine
- MemberSet
- MultiDHistogram
- Pair
- PoissonDeviate
- PrintUtilities
- ProgressEvent
- Simplex
- SpectrumPropertiesTableModel
- StageRelocation
- TextUtilities
- Transform3D
- Translate2D
- UncertainValue
- UncertainValue2
- UncertainValueMC
- UtilException