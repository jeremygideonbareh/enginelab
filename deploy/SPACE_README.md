---
title: EngineLab (beta)
emoji: ⚙️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# EngineLab — beta

Upload a CAD assembly (STEP / IGES, native SolidWorks `.SLDPRT` or a `.zip` of an `.SLDASM` with its parts,
or STL / OBJ / GLB / 3MF meshes). EngineLab:

- breaks it into its parts (assembly tree, instances, parts list with quantities);
- identifies standard components (fasteners, bearings, rails, motors, gears...) with supplier search links;
- for engines: measures bore, stroke, rod length and journals from the geometry, runs the
  thermodynamic cycle and a Monte Carlo life simulation, a component test bench and voxel FEA;
- suggests better materials and design changes, each one simulated, and runs a self-improving
  design optimiser;
- for robots: mass properties and URDF export.

No model handy? Click **Generate demo engine**, or open one of the sample robot arms.

**Beta notes.** Results are engineering estimates for comparing designs (handbook material data,
standard machine-design models), not a certified analysis. Uploads are processed on a shared server
and kept in memory only; please do not upload confidential CAD. Large STEP files take about
1–2 s per MB on first read.

Sample CAD and their licences: see `NOTICE.md`.
