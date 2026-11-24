import sys, time
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.collections import Sequence_TDF_Label
key, val = sys.argv[2], sys.argv[3]
if key != "none":
    ok = Interface_Static.SetCVal_s(key, val) if not val.lstrip("-").isdigit() else Interface_Static.SetIVal_s(key, int(val))
    print("set", key, val, ok)
doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
r = STEPCAFControl_Reader(); r.SetNameMode(True); r.SetColorMode(True)
t = time.time(); r.ReadFile(sys.argv[1]); t1 = time.time(); r.Transfer(doc); t2 = time.time()
st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()); labs = Sequence_TDF_Label(); st.GetShapes(labs)
print(f"{key}={val}: read {t1-t:.1f}s transfer {t2-t1:.1f}s  shapes {labs.Length()}", flush=True)
