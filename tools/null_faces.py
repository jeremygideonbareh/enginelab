import sys
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.TopLoc import TopLoc_Location
from OCP.BRepCheck import BRepCheck_Analyzer
r = STEPControl_Reader(); r.ReadFile(sys.argv[1]); r.TransferRoots(); shape = r.OneShape()
BRepMesh_IncrementalMesh(shape, 0.05, False, 0.35, True)
exp = TopExp_Explorer(shape, TopAbs_FACE); bad = {}; n = 0
while exp.More():
    f = TopoDS.Face(exp.Current()); n += 1
    t = BRep_Tool.Triangulation_s(f, TopLoc_Location())
    if t is None or t.NbTriangles() == 0:
        k = str(BRepAdaptor_Surface(f).GetType()); bad[k] = bad.get(k, 0) + 1
    exp.Next()
print("faces", n, "null triangulation by surface type:", bad)
print("shape valid:", BRepCheck_Analyzer(shape).IsValid())
