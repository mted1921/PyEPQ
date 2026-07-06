# Interconnection: Utility → EPQLibrary

Every point where **Utility** depends on **EPQLibrary**. Each row is a EPQLibrary connection point and the Utility classes that reach into it (most-shared first).

- **EPQLibrary connection points**: 6
- **Utility classes that cross over**: 12
- **Interconnection edges**: 16

| EPQLibrary class | Used by | Utility classes |
|---|---:|---|
| `EPQException` | 10 | `Histogram`, `LevenbergMarquardt2`, `LevenbergMarquardtConstrained`, `LevenbergMarquardtParameterized`, `LinearLeastSquares`, `LinearLeastSquaresMS`, `Math2`, `SpectrumPropertiesTableModel`, `StageRelocation`, `UncertainValue2` |
| `EPQFatalException` | 2 | `CSVReader`, `Histogram` |
| `Composition` | 1 | `SpectrumPropertiesTableModel` |
| `Element` | 1 | `ElementComboBoxModel` |
| `ISpectrumData` | 1 | `SpectrumPropertiesTableModel` |
| `SpectrumProperties` | 1 | `SpectrumPropertiesTableModel` |