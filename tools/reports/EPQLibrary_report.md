# Dependency Map: gov.nist.microanalysis.EPQLibrary

- **Files analyzed**: 143
- **Nodes (classes)**: 143
- **Internal edges**: 1093
- **Weakly-connected components**: 1
- **Strongly-connected clusters (size > 1)**: 2
- **Simple cycles enumerated**: 1000
- **Dead-code candidates**: 0

## Recommended port order (first 50)

Topological depth ascending, then in_degree descending. Port foundations first; classes inside a cycle must be ported as a group.

| # | Class | Depth | In | Out | In Cycle |
|---|-------|------:|---:|---:|:--------:|
| 1 | EPQException | 0 | 64 | 0 |  |
| 2 | PhysicalConstants | 0 | 9 | 0 |  |
| 3 | StageCoordinate | 1 | 2 | 1 |  |
| 4 | FittingFilter | 0 | 2 | 0 |  |
| 5 | EPQFatalException | 0 | 20 | 0 |  |
| 6 | SampleShape | 0 | 10 | 0 |  |
| 7 | ToSI | 1 | 51 | 1 |  |
| 8 | CaveatBase | 0 | 10 | 0 |  |
| 9 | FromSI | 1 | 51 | 1 |  |
| 10 | Element | 2 | 86 | 7 | Y |
| 11 | SpectrumProperties | 2 | 58 | 13 | Y |
| 12 | Composition | 2 | 54 | 5 | Y |
| 13 | XRayTransition | 2 | 52 | 6 | Y |
| 14 | AtomicShell | 2 | 43 | 3 | Y |
| 15 | SpectrumUtils | 2 | 42 | 18 | Y |
| 16 | AlgorithmClass | 2 | 40 | 2 | Y |
| 17 | ISpectrumData | 2 | 37 | 1 | Y |
| 18 | LitReference | 2 | 35 | 1 | Y |
| 19 | XRayTransitionSet | 2 | 30 | 5 | Y |
| 20 | Material | 2 | 28 | 4 | Y |
| 21 | MassAbsorptionCoefficient | 2 | 23 | 13 | Y |
| 22 | RegionOfInterestSet | 2 | 19 | 10 | Y |
| 23 | EDSDetector | 2 | 17 | 18 | Y |
| 24 | AlgorithmUser | 2 | 16 | 10 | Y |
| 25 | KRatioSet | 2 | 15 | 5 | Y |
| 26 | CorrectionAlgorithm | 2 | 13 | 25 | Y |
| 27 | DetectorLineshapeModel | 2 | 12 | 1 | Y |
| 28 | DerivedSpectrum | 2 | 11 | 4 | Y |
| 29 | IXRayWindowProperties | 2 | 10 | 1 | Y |
| 30 | EditableSpectrum | 2 | 8 | 3 | Y |
| 31 | MaterialFactory | 2 | 7 | 5 | Y |
| 32 | MeanIonizationPotential | 2 | 7 | 7 | Y |
| 33 | XPP1991 | 2 | 7 | 16 | Y |
| 34 | DetectorProperties | 2 | 6 | 5 | Y |
| 35 | EDSCalibration | 2 | 6 | 8 | Y |
| 36 | BackscatterFactor | 2 | 5 | 9 | Y |
| 37 | BaseSpectrum | 2 | 5 | 3 | Y |
| 38 | ConductiveCoating | 2 | 5 | 8 | Y |
| 39 | EdgeEnergy | 2 | 5 | 6 | Y |
| 40 | FilterFit | 2 | 5 | 22 | Y |
| 41 | SpectrumMath | 2 | 5 | 5 | Y |
| 42 | StoppingPower | 2 | 5 | 9 | Y |
| 43 | BackscatterCoefficient | 2 | 4 | 6 | Y |
| 44 | BremsstrahlungAnalytic | 2 | 4 | 16 | Y |
| 45 | LinearSpectrumFit | 2 | 4 | 13 | Y |
| 46 | ProportionalIonizationCrossSection | 2 | 4 | 7 | Y |
| 47 | SurfaceIonization | 2 | 4 | 5 | Y |
| 48 | XRayWindow | 2 | 4 | 10 | Y |
| 49 | Armstrong1982Base | 2 | 3 | 16 | Y |
| 50 | ElectronRange | 2 | 3 | 9 | Y |
| ... | (+93 more) | | | | |

## Most depended-upon (high in_degree)

| Class | In | Instability | PageRank |
|-------|---:|------------:|---------:|
| Element | 86 | 0.075 | 0.0619 |
| EPQException | 64 | 0.000 | 0.0383 |
| SpectrumProperties | 58 | 0.183 | 0.0447 |
| Composition | 54 | 0.085 | 0.0282 |
| XRayTransition | 52 | 0.103 | 0.0276 |
| FromSI | 51 | 0.019 | 0.0346 |
| ToSI | 51 | 0.019 | 0.0362 |
| AtomicShell | 43 | 0.065 | 0.0216 |
| SpectrumUtils | 42 | 0.300 | 0.0216 |
| AlgorithmClass | 40 | 0.048 | 0.0276 |

## Most depending (high out_degree)

| Class | Out | Instability | Reachable |
|-------|----:|------------:|----------:|
| CorrectionAlgorithm | 25 | 0.658 | 93 |
| QuantifyUsingStandards | 24 | 0.923 | 100 |
| FilterFit | 22 | 0.815 | 93 |
| Fluorescence | 19 | 0.950 | 93 |
| SpectrumFitResult | 19 | 0.905 | 94 |
| CompositionFromKRatios | 18 | 0.783 | 95 |
| EDSDetector | 18 | 0.514 | 93 |
| MLLSQSignature | 18 | 1.000 | 94 |
| PAP1991 | 18 | 0.857 | 93 |
| QuantificationOutline | 18 | 0.783 | 98 |

## Cycles (top 10 by length)

1. (24 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> XPP1989Ext -> XPP1991 -> PAP1991 -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
2. (24 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> Armstrong1982ParticleCorrection -> Armstrong1982Correction -> Armstrong1982Base -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
3. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> XPP1989Ext -> XPP1991 -> PAP1991 -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
4. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> Armstrong1982ParticleCorrection -> Armstrong1982Correction -> Armstrong1982Base -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
5. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> XPP1989Ext -> XPP1991 -> PAP1991 -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
6. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> Armstrong1982ParticleCorrection -> Armstrong1982Correction -> Armstrong1982Base -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
7. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> XPP1991 -> PAP1991 -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
8. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> XPP1989Ext -> XPP1991 -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
9. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> XPP1989Ext -> PAP1991 -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum
10. (23 nodes) FilteredSpectrum -> DerivedSpectrum -> BaseSpectrum -> SpectrumUtils -> EDSDetector -> XRayTransition -> AtomicShell -> Element -> BetheElectronEnergyLoss -> MeanIonizationPotential -> AlgorithmClass -> LitReference -> Riveros1993 -> MassAbsorptionCoefficient -> Material -> Composition -> AlgorithmUser -> CorrectionAlgorithm -> Armstrong1982Correction -> Armstrong1982Base -> SpectrumProperties -> QuantifyUsingSTEMinSEM -> FilterFit -> FilteredSpectrum