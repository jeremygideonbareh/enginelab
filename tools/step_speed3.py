import sys, time
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.STEPControl import STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDocStd import TDocStd_Document
mode = sys.argv[2]
t = time.time()
if mode == "plain":
    r = STEPControl_Reader(); r.ReadFile(sys.argv[1]); t1 = time.time(); r.TransferRoots(); t2 = time.time()
else:
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    r = STEPCAFControl_Reader()
    r.SetColorMode("c" in mode); r.SetNameMode("n" in mode); r.SetLayerMode("l" in mode)
    r.SetPropsMode(False); r.SetGDTMode(False); r.SetMatMode(False); r.SetViewMode(False); r.SetSHUOMode(False)
    r.ReadFile(sys.argv[1]); t1 = time.time(); r.Transfer(doc); t2 = time.time()
print(f"{mode:8s} read {t1-t:.1f}s transfer {t2-t1:.1f}s", flush=True)
