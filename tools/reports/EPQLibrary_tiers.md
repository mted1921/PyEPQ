# Dependency Tiers: EPQLibrary

Classes grouped by dependency tier. **Tier 0** depends on nothing else inside EPQLibrary (the foundations); a class in tier _N_ depends only on classes in lower tiers. Classes bound together in a dependency cycle share a tier and are marked _(cycle)_.

- **Classes**: 143
- **Tiers**: 11
- **Internal edges**: 1093
- **Cyclic clusters**: 2

## Tier 0 — foundations (7 classes)

- CaveatBase
- EPQException
- EPQFatalException
- FittingFilter
- ITransform
- PhysicalConstants
- SampleShape

## Tier 1 (3 classes)

- FromSI
- StageCoordinate
- ToSI

## Tier 2 (85 classes)

- AbsorptionCorrection _(cycle)_
- AlgorithmClass _(cycle)_
- AlgorithmUser _(cycle)_
- Armstrong1982Base _(cycle)_
- Armstrong1982Correction _(cycle)_
- Armstrong1982ParticleCorrection _(cycle)_
- AtomicShell _(cycle)_
- BackscatterCoefficient _(cycle)_
- BackscatterFactor _(cycle)_
- BaseSpectrum _(cycle)_
- BasicSiLiLineshape _(cycle)_
- BetheElectronEnergyLoss _(cycle)_
- Bremsstrahlung _(cycle)_
- BremsstrahlungAnalytic _(cycle)_
- BremsstrahlungAngularDistribution _(cycle)_
- Composition _(cycle)_
- ConductiveCoating _(cycle)_
- CorrectionAlgorithm _(cycle)_
- DerivedSpectrum _(cycle)_
- DetectorCalibration _(cycle)_
- DetectorLineshapeModel _(cycle)_
- DetectorProperties _(cycle)_
- DuaneHuntLimit _(cycle)_
- EDSCalibration _(cycle)_
- EDSDetector _(cycle)_
- EdgeEnergy _(cycle)_
- EditableSpectrum _(cycle)_
- ElectronProbe _(cycle)_
- ElectronRange _(cycle)_
- Element _(cycle)_
- FanoSiLiLineshape _(cycle)_
- FilterFit _(cycle)_
- FilteredSpectrum _(cycle)_
- Fluorescence _(cycle)_
- FluorescenceYield _(cycle)_
- FluorescenceYieldMean _(cycle)_
- GridMountedWindow _(cycle)_
- ISpectrumData _(cycle)_
- ISpectrumTransformation _(cycle)_
- IXRayDetector _(cycle)_
- IXRayWindowProperties _(cycle)_
- IonizationCrossSection _(cycle)_
- IonizationDepthRatio _(cycle)_
- JumpRatio _(cycle)_
- KRatioSet _(cycle)_
- LenardCoefficient _(cycle)_
- LinearSpectrumFit _(cycle)_
- LitReference _(cycle)_
- MassAbsorptionCoefficient _(cycle)_
- Material _(cycle)_
- MaterialFactory _(cycle)_
- MeanIonizationPotential _(cycle)_
- MicrocalCalibration _(cycle)_
- Oxidizer _(cycle)_
- PAP1991 _(cycle)_
- ParticleSignature _(cycle)_
- PeakROISearch _(cycle)_
- PeakStripping _(cycle)_
- ProportionalIonizationCrossSection _(cycle)_
- QuantifyUsingSTEMinSEM _(cycle)_
- ROISpectrum _(cycle)_
- ROISpectrumNaive _(cycle)_
- RegionOfInterestSet _(cycle)_
- Riveros1993 _(cycle)_
- SDDCalibration _(cycle)_
- STEMinSEMCorrection _(cycle)_
- SiLiCalibration _(cycle)_
- SpectrumMath _(cycle)_
- SpectrumProperties _(cycle)_
- SpectrumSmoothing _(cycle)_
- SpectrumUtils _(cycle)_
- StoppingPower _(cycle)_
- Strategy _(cycle)_
- SurfaceIonization _(cycle)_
- TransitionEnergy _(cycle)_
- TransitionProbabilities _(cycle)_
- VariableWidthFittingFilter _(cycle)_
- XPP1989Ext _(cycle)_
- XPP1991 _(cycle)_
- XRayTransition _(cycle)_
- XRayTransitionSet _(cycle)_
- XRayWindow _(cycle)_
- XRayWindow2 _(cycle)_
- XRayWindow3 _(cycle)_
- XRayWindowFactory _(cycle)_

## Tier 3 (27 classes)

- AbsoluteIonizationCrossSection
- Armstrong1982ParticleMC
- AverageSpectrum
- BadgerFilmExport
- BrowningEmpiricalCrossSection
- CITZAF
- ComputeZAF
- CzyzewskiMottCrossSection
- ExtremumSpectrum
- Gas
- GaussianSumSpectrum
- IterationAlgorithm
- MACCache
- MLLSQSignature
- MajorMinorTrace
- MicrocalSpectrumFitter
- MuCal
- MultiSpectrumMetrics
- NISTXRayTransitionDB
- NoisySpectrum
- PandPDatabase
- RandomizedScatter
- SamplePreparation
- SpectrumFitResult
- StandardsDatabase2
- VectorSet
- ZetaFactor

## Tier 4 (11 classes)

- CharacteristicXRayGeneration
- CompositionFromKRatios
- CzyzewskiMottScatteringAngle _(cycle)_
- GasMixture
- GasScatteringCrossSection _(cycle)_
- NISTMottScatteringAngle _(cycle)_
- QuantifyUsingZetaFactors
- RandomizedScatterFactory _(cycle)_
- ScreenedRutherfordScatteringAngle _(cycle)_
- SpectrumFitter8
- SpectrumSimulator

## Tier 5 (4 classes)

- EPMAOptimizer
- MapImage
- MeasuredCalibration
- QuantificationOutline

## Tier 6 (2 classes)

- CompositionOptimizer
- StandardBundle

## Tier 7 (1 classes)

- QuantifyUsingStandards

## Tier 8 (1 classes)

- QuantificationPlan

## Tier 9 (1 classes)

- QuantificationOptimizer

## Tier 10 (1 classes)

- QuantificationOptimizer2

## Cyclic clusters

Each group imports itself (directly or transitively) and must be ported together — no member stands alone.

1. (85) AbsorptionCorrection ↔ AlgorithmClass ↔ AlgorithmUser ↔ Armstrong1982Base ↔ Armstrong1982Correction ↔ Armstrong1982ParticleCorrection ↔ AtomicShell ↔ BackscatterCoefficient ↔ BackscatterFactor ↔ BaseSpectrum ↔ BasicSiLiLineshape ↔ BetheElectronEnergyLoss ↔ Bremsstrahlung ↔ BremsstrahlungAnalytic ↔ BremsstrahlungAngularDistribution ↔ Composition ↔ ConductiveCoating ↔ CorrectionAlgorithm ↔ DerivedSpectrum ↔ DetectorCalibration ↔ DetectorLineshapeModel ↔ DetectorProperties ↔ DuaneHuntLimit ↔ EDSCalibration ↔ EDSDetector ↔ EdgeEnergy ↔ EditableSpectrum ↔ ElectronProbe ↔ ElectronRange ↔ Element ↔ FanoSiLiLineshape ↔ FilterFit ↔ FilteredSpectrum ↔ Fluorescence ↔ FluorescenceYield ↔ FluorescenceYieldMean ↔ GridMountedWindow ↔ ISpectrumData ↔ ISpectrumTransformation ↔ IXRayDetector ↔ IXRayWindowProperties ↔ IonizationCrossSection ↔ IonizationDepthRatio ↔ JumpRatio ↔ KRatioSet ↔ LenardCoefficient ↔ LinearSpectrumFit ↔ LitReference ↔ MassAbsorptionCoefficient ↔ Material ↔ MaterialFactory ↔ MeanIonizationPotential ↔ MicrocalCalibration ↔ Oxidizer ↔ PAP1991 ↔ ParticleSignature ↔ PeakROISearch ↔ PeakStripping ↔ ProportionalIonizationCrossSection ↔ QuantifyUsingSTEMinSEM ↔ ROISpectrum ↔ ROISpectrumNaive ↔ RegionOfInterestSet ↔ Riveros1993 ↔ SDDCalibration ↔ STEMinSEMCorrection ↔ SiLiCalibration ↔ SpectrumMath ↔ SpectrumProperties ↔ SpectrumSmoothing ↔ SpectrumUtils ↔ StoppingPower ↔ Strategy ↔ SurfaceIonization ↔ TransitionEnergy ↔ TransitionProbabilities ↔ VariableWidthFittingFilter ↔ XPP1989Ext ↔ XPP1991 ↔ XRayTransition ↔ XRayTransitionSet ↔ XRayWindow ↔ XRayWindow2 ↔ XRayWindow3 ↔ XRayWindowFactory
2. (5) CzyzewskiMottScatteringAngle ↔ GasScatteringCrossSection ↔ NISTMottScatteringAngle ↔ RandomizedScatterFactory ↔ ScreenedRutherfordScatteringAngle
