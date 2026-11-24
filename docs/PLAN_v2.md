# EngineLab v2: a design, test and sourcing tool for robotics, mechanical and automotive engineers

## Who it is for and what they need to get done

| User | Real problem | What the tool must do |
|---|---|---|
| Robotics engineer | Turn a CAD robot into a simulation model; size motors and gearboxes; check that links and brackets don't bend or break | Exact per-part mass, centre of mass and inertia; detect joints (coaxial bearings and shafts); **export URDF/MJCF**; joint torque sizing (static + dynamic) and motor/gearbox suggestions |
| Mechanical designer | Get a BOM out of a big assembly; find and buy the standard parts; check bolts, bearings, shafts and gears | **Exact breakdown** with assembly tree, instances and quantities; standard-part recognition (ISO fasteners, bearings, extrusions, gears, motors); supplier search links; bearing L10 life, bolt joints, shaft deflection, gear rating |
| Automotive engineer | Engine and driveline durability, materials, weak spots | The existing engine cycle, life simulation, test bench and FEA, extended to gearboxes and suspension parts |

## Status (21 Nov 2026)

- Stage A: done for STEP/IGES via OCP XCAF (tree, instances, multi-body folding, exact B-rep volume/genus/cylinders, fallback face meshing, cache, background jobs). Validated on the real Voron toolhead (31/31 parts with standards) and run on the 241 MB Voron 2.4. Still open: mesh-file contact-face splitting, Three.js instancing for 5,000+ parts, a parallel geometry transfer for faster first reads.
- Stage B: name + geometry recognition for fasteners, bearings, rails, extrusions, motors, pulleys/gears, fans, fittings and plates is in. Still open: ML model for generic parts and more geometry rules (nuts/washers without names, belts, PCBs).
- Stage C: verified search links + CSV BOM done. Live price/stock needs an API key (Digi-Key / Nexar).
- Stages D–E: next.

## Stage A: perfect breakdown of big CAD assemblies (do this first)

1. STEP/IGES: read the **product structure** (sub-assemblies, part names, instances with transforms, colours) rather than a flat list of solids. Options: gmsh labels (path names like `Shapes/Arm/Link2`) now; OCP/pythonOCC XCAF if wheels exist for Python 3.14 (check, or fall back to Python 3.12).
2. Instance detection: identical part definitions give one BOM line with a quantity.
3. Scale: 1,000–10,000 parts. Use coarser tessellation for tiny parts, parallel feature extraction, Three.js instancing, and lazy specs.
4. Mesh files: split where bodies touch with coincident triangles (orientation-aware pairing at non-manifold edges), keep tiny fasteners, and merge cavity shells.
5. Synthetic stress-test assemblies written as STEP *with names and hierarchy*: 6-DOF robot arm, mobile robot, gearbox, 3D-printer frame, engine, all with 500–3,000 parts. Score them on exact part count, BOM quantities, tree depth and timing.
6. Validation on real public CAD, which needs the user's go-ahead to download (see "Needed from the user").

## Stage B: generic component recognition

- Wider taxonomy: fasteners (SHCS, hex bolt, nut, washer, set screw, standoff, dowel), bearings (ball races, balls, cage, flanged), shafts, keys, couplings, spur/helical/bevel gears, pulleys and belts, lead/ball screws, linear rails and carriages, extrusions (2020/3030/4040), motors (NEMA, BLDC, servo), strain-wave/planetary gearboxes, brackets, plates, links, housings, wheels, PCBs, connectors.
- Name and part-number parsing: McMaster numbers, DIN/ISO codes, `M5x20`, `6204-2RS`, `MGN12`, `GT2`, `NEMA17`.
- Geometry to standard size, e.g. a measured 20×47×14 bearing becomes 6204, and a measured 5 mm shank and 20 mm length becomes M5×20.
- Retrain on the new generators plus single and partial uploads.

## Stage C: sourcing

- Per BOM line, a precise query (standard + size + material) with search links to McMaster-Carr, Misumi, Digi-Key, RS, TraceParts and GrabCAD. URL formats must be verified (web search was unavailable at the usage limit).
- BOM export (CSV/XLSX) with quantities, mass, material and links.

## Stage D: robotics tools

- Mass properties per part and per sub-assembly using chosen materials.
- Joint inference from coaxial bearing/shaft/motor stacks, with the user confirming.
- URDF/MJCF export (meshes, inertias, joint axes and limits).
- Joint torque sizing over the workspace with a payload; motor + gearbox suggestions; link deflection via FEA.

## Stage E: general mechanical checks

Bearing L10 (ISO 281), bolted joints (VDI 2230-lite), shaft bending/torsion and critical speed, gear bending/pitting, and an FEA test bench for any part.

## Needed from the user

1. Permission to download public test CAD (open robot arm STEP files, NIST/FreeCAD sample assemblies), or the user's own STEP assemblies.
2. Optional: a Nexar/Octopart or Digi-Key API key, to show live prices and stock instead of search links.
3. Optional: Python 3.12 alongside 3.14, if OCP/pythonOCC wheels aren't available for 3.14 (gives exact B-rep: true hole and cylinder radii, the full assembly tree).
