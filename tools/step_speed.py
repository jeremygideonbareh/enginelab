"""Time STEP reading with different OpenCASCADE settings: python tools/step_speed.py file.step"""
import sys
import time

from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDocStd import TDocStd_Document


def run(label, setup=None):
    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    r = STEPCAFControl_Reader()
    r.SetNameMode(True)
    r.SetColorMode(True)
    if setup:
        setup(r)
    t = time.time()
    assert r.ReadFile(sys.argv[1]) == IFSelect_RetDone
    t1 = time.time()
    r.Transfer(doc)
    t2 = time.time()
    print(f"{label:40s} read {t1 - t:6.1f}s  transfer {t2 - t1:6.1f}s", flush=True)


def no_heal(r):
    flags, ok = r.GetShapeProcessFlags()
    flags.reset() if hasattr(flags, "reset") else None
    r.SetShapeProcessFlags(flags)


run("default")
for name in ("read.step.sequence", "read.step.resource.name"):
    try:
        print(name, "=", Interface_Static.CVal_s(name))
    except Exception as e:
        print(name, "n/a", e)
try:
    run("shape-process flags cleared", no_heal)
except Exception as e:
    print("flags API failed:", e)
