# Dependency Map: gov.nist.microanalysis.EPQLibrary

- **Files analyzed**: 143
- **Nodes (classes)**: 143
- **Internal edges**: 193
- **Weakly-connected components**: 35
- **Strongly-connected clusters (size > 1)**: 2
- **Simple cycles enumerated**: 3
- **Dead-code candidates**: 31

## Recommended port order (first 50)

Topological depth ascending, then in_degree descending. Port foundations first; classes inside a cycle must be ported as a group.

| # | Class | Depth | In | Out | In Cycle |
|---|-------|------:|---:|---:|:--------:|
| 1 | ISpectrumData | 0 | 3 | 0 |  |
| 2 | BaseSpectrum | 1 | 4 | 1 |  |
| 3 | XRayTransition | 0 | 6 | 0 |  |
| 4 | SpectrumProperties | 0 | 14 | 2 | Y |
| 5 | IXRayWindowProperties | 0 | 4 | 1 | Y |
| 6 | IXRayDetector | 0 | 3 | 1 | Y |
| 7 | XRayTransitionSet | 0 | 2 | 0 |  |
| 8 | FromSI | 0 | 4 | 0 |  |
| 9 | Element | 0 | 8 | 0 |  |
| 10 | EditableSpectrum | 2 | 2 | 1 |  |
| 11 | EPQException | 0 | 8 | 0 |  |
| 12 | AlgorithmUser | 0 | 5 | 0 |  |
| 13 | DetectorCalibration | 1 | 1 | 2 |  |
| 14 | EDSDetector | 3 | 16 | 10 | Y |
| 15 | SpectrumUtils | 3 | 7 | 1 | Y |
| 16 | Composition | 0 | 3 | 0 |  |
| 17 | AlgorithmClass | 1 | 34 | 1 |  |
| 18 | EDSCalibration | 4 | 5 | 3 |  |
| 19 | ToSI | 0 | 8 | 0 |  |
| 20 | MaterialFactory | 0 | 5 | 0 |  |
| 21 | Material | 1 | 9 | 1 |  |
| 22 | MassAbsorptionCoefficient | 2 | 3 | 1 |  |
| 23 | AtomicShell | 0 | 3 | 0 |  |
| 24 | SiLiCalibration | 5 | 1 | 11 |  |
| 25 | DetectorLineshapeModel | 4 | 8 | 1 |  |
| 26 | SDDCalibration | 6 | 1 | 4 |  |
| 27 | EPQFatalException | 0 | 3 | 0 |  |
| 28 | XPP1991 | 0 | 1 | 0 |  |
| 29 | ISpectrumTransformation | 0 | 1 | 0 |  |
| 30 | DerivedSpectrum | 2 | 5 | 1 |  |
| 31 | RandomizedScatter | 2 | 4 | 1 |  |
| 32 | QuantificationOutline | 4 | 1 | 1 |  |
| 33 | QuantificationOptimizer | 0 | 1 | 0 |  |
| 34 | IonizationCrossSection | 2 | 2 | 1 |  |
| 35 | LinearSpectrumFit | 5 | 1 | 2 |  |
| 36 | SpectrumFitter8 | 5 | 1 | 2 |  |
| 37 | SpectrumFitResult | 7 | 1 | 2 |  |
| 38 | XRayWindow | 2 | 1 | 10 |  |
| 39 | EPMAOptimizer | 4 | 1 | 1 |  |
| 40 | Armstrong1982Base | 0 | 3 | 0 |  |
| 41 | ZetaFactor | 2 | 0 | 1 |  |
| 42 | XPP1989Ext | 1 | 0 | 1 |  |
| 43 | VectorSet | 0 | 0 | 0 |  |
| 44 | VariableWidthFittingFilter | 5 | 0 | 1 |  |
| 45 | TransitionProbabilities | 2 | 0 | 1 |  |
| 46 | TransitionEnergy | 2 | 0 | 1 |  |
| 47 | SurfaceIonization | 2 | 0 | 1 |  |
| 48 | Strategy | 0 | 0 | 0 |  |
| 49 | StoppingPower | 2 | 0 | 1 |  |
| 50 | STEMinSEMCorrection | 0 | 0 | 0 |  |
| ... | (+93 more) | | | | |

## Most depended-upon (high in_degree)

| Class | In | Instability | PageRank |
|-------|---:|------------:|---------:|
| AlgorithmClass | 34 | 0.029 | 0.0936 |
| EDSDetector | 16 | 0.385 | 0.0512 |
| SpectrumProperties | 14 | 0.125 | 0.0954 |
| Material | 9 | 0.100 | 0.0098 |
| DetectorLineshapeModel | 8 | 0.111 | 0.0147 |
| Element | 8 | 0.000 | 0.0096 |
| EPQException | 8 | 0.000 | 0.0093 |
| ToSI | 8 | 0.000 | 0.0062 |
| SpectrumUtils | 7 | 0.125 | 0.0255 |
| XRayTransition | 6 | 0.000 | 0.0108 |

## Most depending (high out_degree)

| Class | Out | Instability | Reachable |
|-------|----:|------------:|----------:|
| MeasuredCalibration | 13 | 1.000 | 28 |
| MicrocalCalibration | 13 | 1.000 | 24 |
| SiLiCalibration | 11 | 0.917 | 23 |
| EDSDetector | 10 | 0.385 | 12 |
| XRayWindow | 10 | 0.909 | 12 |
| XRayWindow2 | 8 | 1.000 | 9 |
| XRayWindowFactory | 8 | 1.000 | 11 |
| GridMountedWindow | 4 | 1.000 | 13 |
| SDDCalibration | 4 | 0.800 | 24 |
| XRayWindow3 | 4 | 1.000 | 5 |

## Cycles (top 10 by length)

1. (2 nodes) SpectrumUtils -> EDSDetector -> SpectrumUtils
2. (2 nodes) IXRayDetector -> SpectrumProperties -> IXRayDetector
3. (2 nodes) IXRayWindowProperties -> SpectrumProperties -> IXRayWindowProperties

## Dead-code candidates

(Zero internal in-degree AND zero internal out-degree -- might be entry points, test fixtures, or genuinely unused.)

- BadgerFilmExport
- Bremsstrahlung
- BrowningEmpiricalCrossSection
- CaveatBase
- CITZAF
- ConductiveCoating
- CzyzewskiMottCrossSection
- FittingFilter
- ITransform
- KRatioSet
- LitReference
- MACCache
- MajorMinorTrace
- MapImage
- MuCal
- MultiSpectrumMetrics
- NISTXRayTransitionDB
- Oxidizer
- PandPDatabase
- PAP1991
- ParticleSignature
- PhysicalConstants
- QuantificationPlan
- Riveros1993
- SamplePreparation
- SampleShape
- StageCoordinate
- StandardsDatabase2
- STEMinSEMCorrection
- Strategy
- VectorSet