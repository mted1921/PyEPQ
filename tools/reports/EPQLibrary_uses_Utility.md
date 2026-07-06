# Interconnection: EPQLibrary → Utility

Every point where **EPQLibrary** depends on **Utility**. Each row is a Utility connection point and the EPQLibrary classes that reach into it (most-shared first).

- **Utility connection points**: 22
- **EPQLibrary classes that cross over**: 92
- **Interconnection edges**: 171

| Utility class | Used by | EPQLibrary classes |
|---|---:|---|
| `Math2` | 52 | `AbsoluteIonizationCrossSection`, `AbsorptionCorrection`, `Armstrong1982Base`, `Armstrong1982Correction`, `Armstrong1982ParticleCorrection`, `Armstrong1982ParticleMC`, `AverageSpectrum`, `Bremsstrahlung`, `BremsstrahlungAnalytic`, `BremsstrahlungAngularDistribution`, `BrowningEmpiricalCrossSection`, `Composition`, `CorrectionAlgorithm`, `DetectorProperties`, `EDSDetector`, `EdgeEnergy`, `ElectronProbe`, `FilteredSpectrum`, `FittingFilter`, `Fluorescence`, `GasScatteringCrossSection`, `GaussianSumSpectrum`, `IonizationCrossSection`, `IterationAlgorithm`, `KRatioSet`, `MapImage`, `MassAbsorptionCoefficient`, `MicrocalCalibration`, `MicrocalSpectrumFitter`, `NISTMottScatteringAngle`, `Oxidizer`, `PAP1991`, `ParticleSignature`, `PeakROISearch`, `PeakStripping`, `QuantificationOutline`, `ROISpectrumNaive`, `Riveros1993`, `SamplePreparation`, `SampleShape`, `SpectrumFitResult`, `SpectrumFitter8`, `SpectrumProperties`, `SpectrumSimulator`, `SpectrumUtils`, `StandardsDatabase2`, `StoppingPower`, `VariableWidthFittingFilter`, `VectorSet`, `XPP1989Ext`, `XPP1991`, `XRayWindow3` |
| `UncertainValue2` | 29 | `BremsstrahlungAnalytic`, `Composition`, `CompositionFromKRatios`, `CompositionOptimizer`, `ComputeZAF`, `CorrectionAlgorithm`, `FilterFit`, `FromSI`, `KRatioSet`, `MLLSQSignature`, `MassAbsorptionCoefficient`, `MeasuredCalibration`, `MicrocalSpectrumFitter`, `Oxidizer`, `ParticleSignature`, `PeakROISearch`, `QuantificationOptimizer2`, `QuantificationOutline`, `QuantifyUsingSTEMinSEM`, `QuantifyUsingStandards`, `QuantifyUsingZetaFactors`, `STEMinSEMCorrection`, `SpectrumFitResult`, `SpectrumFitter8`, `SpectrumProperties`, `SpectrumUtils`, `ToSI`, `XPP1991`, `ZetaFactor` |
| `HalfUpFormat` | 27 | `Armstrong1982ParticleCorrection`, `BackscatterFactor`, `CITZAF`, `Composition`, `CompositionFromKRatios`, `ConductiveCoating`, `EPMAOptimizer`, `FittingFilter`, `Gas`, `MassAbsorptionCoefficient`, `Material`, `MicrocalSpectrumFitter`, `MultiSpectrumMetrics`, `ParticleSignature`, `QuantificationOutline`, `QuantificationPlan`, `QuantifyUsingStandards`, `QuantifyUsingZetaFactors`, `RegionOfInterestSet`, `SamplePreparation`, `SampleShape`, `SiLiCalibration`, `SpectrumFitResult`, `SpectrumProperties`, `SpectrumUtils`, `StageCoordinate`, `XRayWindowFactory` |
| `CSVReader` | 17 | `AbsoluteIonizationCrossSection`, `AtomicShell`, `BremsstrahlungAngularDistribution`, `EdgeEnergy`, `Element`, `FluorescenceYield`, `FluorescenceYieldMean`, `MassAbsorptionCoefficient`, `MeanIonizationPotential`, `MuCal`, `Oxidizer`, `PandPDatabase`, `TransitionEnergy`, `TransitionProbabilities`, `XRayTransition`, `XRayWindow2`, `XRayWindow3` |
| `TextUtilities` | 8 | `CompositionOptimizer`, `EPMAOptimizer`, `QuantificationOutline`, `QuantificationPlan`, `RegionOfInterestSet`, `SpectrumProperties`, `StandardsDatabase2`, `XRayTransition` |
| `Pair` | 5 | `QuantifyUsingSTEMinSEM`, `QuantifyUsingStandards`, `QuantifyUsingZetaFactors`, `STEMinSEMCorrection`, `ZetaFactor` |
| `Interval` | 4 | `BremsstrahlungAnalytic`, `FilterFit`, `FilteredSpectrum`, `SpectrumUtils` |
| `LinearLeastSquares` | 4 | `BremsstrahlungAnalytic`, `LinearSpectrumFit`, `MicrocalSpectrumFitter`, `SpectrumFitter8` |
| `UtilException` | 4 | `IonizationCrossSection`, `KRatioSet`, `MeasuredCalibration`, `SpectrumFitter8` |
| `Constraint` | 3 | `MeasuredCalibration`, `MicrocalSpectrumFitter`, `SpectrumFitter8` |
| `DescriptiveStatistics` | 3 | `MapImage`, `Oxidizer`, `SpectrumUtils` |
| `LinearRegression` | 3 | `DuaneHuntLimit`, `ROISpectrumNaive`, `SpectrumUtils` |
| `LevenbergMarquardt2` | 2 | `MeasuredCalibration`, `MicrocalSpectrumFitter` |
| `PoissonDeviate` | 2 | `NoisySpectrum`, `SpectrumUtils` |
| `HTMLFormat` | 1 | `SpectrumFitResult` |
| `Integrator` | 1 | `Armstrong1982ParticleCorrection` |
| `LevenbergMarquardtConstrained` | 1 | `MeasuredCalibration` |
| `LevenbergMarquardtParameterized` | 1 | `SpectrumFitter8` |
| `MCIntegrator` | 1 | `Armstrong1982ParticleMC` |
| `ProgressEvent` | 1 | `MicrocalSpectrumFitter` |
| `Simplex` | 1 | `IonizationCrossSection` |
| `Translate2D` | 1 | `StageCoordinate` |