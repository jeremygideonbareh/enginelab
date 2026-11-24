import sys
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IMeshTools import IMeshTools_Parameters
from OCP.TopLoc import TopLoc_Location
from OCP.ShapeFix import ShapeFix_Shape
from OCP.BRepTools import BRepTools
def nulls(shape):
    exp = TopExp_Explorer(shape, TopAbs_FACE); k = 0
    while exp.More():
        t = BRep_Tool.Triangulation_s(TopoDS.Face(exp.Current()), TopLoc_Location())
        k += (t is None or t.NbTriangles() == 0); exp.Next()
    return k
r = STEPControl_Reader(); r.ReadFile(sys.argv[1]); r.TransferRoots(); base = r.OneShape()
p = IMeshTools_Parameters(); p.Deflection = 0.05; p.Angle = 0.35; p.InParallel = True
p.AllowQualityDecrease = True; p.ControlSurfaceDeflection = True
BRepTools.Clean_s(base); BRepMesh_IncrementalMesh(base, p); print("AllowQualityDecrease:", nulls(base))
fx = ShapeFix_Shape(base); fx.Perform(); fixed = fx.Shape()
BRepTools.Clean_s(fixed); BRepMesh_IncrementalMesh(fixed, 0.05, False, 0.35, True); print("after ShapeFix:", nulls(fixed))
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepAdaptor import BRepAdaptor_Surface
exp = TopExp_Explorer(base, TopAbs_FACE); tot = 0; badA = []
while exp.More():
    f = TopoDS.Face(exp.Current()); g = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, g); a = g.Mass(); tot += a
    t = BRep_Tool.Triangulation_s(f, TopLoc_Location())
    if t is None or t.NbTriangles() == 0:
        s = BRepAdaptor_Surface(f); c = s.Cone()
        badA.append((round(a, 4), round(c.RefRadius(), 3), round(c.SemiAngle() * 57.3, 2), round(s.FirstVParameter(), 4), round(s.LastVParameter(), 4)))
    exp.Next()
print("total area", round(tot), "null faces (area, ref radius, half-angle deg, v-range):", badA)
