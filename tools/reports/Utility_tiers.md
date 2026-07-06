# Dependency Tiers: Utility

Classes grouped by dependency tier. **Tier 0** depends on nothing else inside Utility (the foundations); a class in tier _N_ depends only on classes in lower tiers. Classes bound together in a dependency cycle share a tier and are marked _(cycle)_.

- **Classes**: 44
- **Tiers**: 6
- **Internal edges**: 42
- **Cyclic clusters**: 0

## Tier 0 — foundations (20 classes)

- AutoComplete
- CSVReader
- ComboBoxCellEditor
- EachRowEditor
- ElementComboBoxModel
- ElementTreePanel
- FindRoot
- HTMLList
- HalfUpFormat
- Histogram
- HtmlSelection
- Interval
- LazyEvaluate
- MemberSet
- Pair
- PrintUtilities
- SpectrumPropertiesTableModel
- TextUtilities
- Transform3D
- UtilException

## Tier 1 (5 classes)

- AdaptiveRungeKutta
- ExponentFormat
- HTMLFormat
- Math2
- Simplex

## Tier 2 (9 classes)

- Integrator
- LinearRegression
- MCIntegrator
- MultiDHistogram
- PoissonDeviate
- ProgressEvent
- StageRelocation
- Translate2D
- UncertainValue2

## Tier 3 (6 classes)

- Constraint
- DescriptiveStatistics
- LevenbergMarquardt2
- LinearLeastSquares
- UncertainValue
- UncertainValueMC

## Tier 4 (3 classes)

- LevenbergMarquardtConstrained
- LinearLeastSquaresMS
- MCUncertaintyEngine

## Tier 5 (1 classes)

- LevenbergMarquardtParameterized
