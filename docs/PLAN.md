# EngineLab — plan

Upload a 3D model of a machine (an engine, gearbox, pump…), break it into its
component parts, identify each part, give specifications, recommend materials,
and simulate thousands of operating hours to find where and when it fails.

## Approach (why it is built this way)

Getting the parts out accurately matters more than anything else, so the
pipeline uses the most reliable signal available first and only falls back to
guessing when it has to.

| Step | Method | Why |
|---|---|---|
| 1. Ingest | STEP/IGES through **gmsh** (OpenCASCADE kernel); STL/OBJ/PLY/GLB/3MF through **trimesh** | STEP keeps every solid body and its name, which is the most accurate part split possible. Mesh formats lose that, so we rebuild it. |
| 2. Segment | (a) CAD solid bodies / named groups, (b) connected shells after vertex welding, (c) spectral clustering on the face graph for fused meshes | CAD exports of assemblies keep each part as its own closed shell, so (b) is exact for those. (c) is the fallback for meshes that were merged into one shell. |
| 3. Describe | ~40 geometric features per part: oriented box, volume, area, compactness, convexity, genus (holes), rotational symmetry, radial profile, shape-distribution histogram, plus **assembly context**: how many identical copies, alignment with the main axis, neighbours | A piston on its own looks like a cup. Four identical cups in a row along the crank axis are almost certainly pistons. Context is where much of the accuracy comes from. |
| 4. Classify | Ensemble: name keywords (when the CAD file has names) → gradient-boosted classifier trained on procedurally generated engines → engineering rules → assembly-level reasoning (pistons ↔ bores ↔ conrods ↔ crank) | No public labelled dataset of engine parts exists, so we generate one: a parametric engine generator produces thousands of varied, labelled engines. |
| 5. Specify | Dimensions, mass, inertia, bore/stroke/displacement inferred from matched parts | Specs come from the geometry plus engineering relationships. |
| 6. Materials | Database of ~30 engineering materials with real properties; per-part requirement profiles; weighted (Ashby-style) ranking | Explains *why* a material fits each part. |
| 7. Simulate | Slider-crank dynamics with a Wiebe combustion pressure model → per-part load histories → stress (analytical models + voxel FEA) → fatigue (Basquin S-N, Goodman mean-stress, Marin factors, Miner damage), wear (Archard), creep (Larson-Miller), bearing life → duty-cycle time-stepping over thousands of hours with Monte Carlo scatter | Gives a "where and when will it break" answer with probabilities, not a single number. |
| 8. UI | FastAPI backend + browser UI with a Three.js viewer: exploded view, colour by part type / safety factor / damage, per-part test bench | |

## Stages

1. Foundation: mesh core, IO, primitives, segmentation, features — with tests.
2. Parametric engine generator (test data + training data + demo).
3. Classifier (rules + ML + context) and accuracy measurement.
4. Specs + materials database + recommender.
5. Physics: kinematics, loads, fatigue, wear, creep, life simulation, voxel FEA.
6. API + web UI.
7. End-to-end verification in the browser; fix what breaks.
8. Later: more machine families (gearboxes, pumps, turbines), learned point-cloud
   model (PointNet-style) once real labelled data exists, thermal FEA.

## Honest limits

- Analytical and voxel-FEA results are engineering estimates for comparison and
  early design, not a certified analysis.
- The classifier is trained on generated engines. Real CAD from other sources
  will vary; uploads where the user corrects labels can be saved as new
  training data.
