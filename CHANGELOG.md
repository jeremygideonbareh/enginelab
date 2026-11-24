# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project uses [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-24

First public release.

### Added
- CAD breakdown: STEP/IGES assemblies through OpenCASCADE XCAF (tree, instances, multi-body components),
  native SolidWorks `.SLDPRT` / zipped `.SLDASM` (2015+ and older OLE files), meshes (STL, OBJ, PLY, GLB,
  3MF) with shell splitting and spectral cuts of fused meshes.
- Part identification: engine part classifier (20 classes, trained on generated engines) and a component
  classifier for robots and machines (11 families, trained on ~9,100 real labelled parts), with calibrated
  confidence and a separate model for files that hold a single part.
- Measurements: bore, stroke, rod length, journals, gears, springs, threads; parts list (BOM) with quantities,
  designations, masses and supplier links; CSV export.
- Materials: 36 engineering materials, physics-based ranking, live what-if preview.
- Testing: part test bench with pressure sweep, voxel FEA (axial, bending, torsion, pressure).
- Engine life: thermodynamic cycle, duty cycles, Monte Carlo with Latin-hypercube sampling and confidence
  intervals, balance and torsional vibration analysis.
- Design improvement: per-part suggestions and a self-improving optimiser (surrogate model + polishing).
- Robotics: mass properties, URDF export, joint torque sizing against common actuators.
- Sourcing: prices and shops by country, live exchange rates, shopping plan.
- Web app with a sakura-themed UI, keyboard shortcuts, recent analyses, picture export.
- Docker / Hugging Face Space packaging and a one-click public beta script (Cloudflare quick tunnel).
