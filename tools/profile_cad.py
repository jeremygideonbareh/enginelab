"""Profile the CAD reader on a big STEP file and cache the result: python tools/profile_cad.py file.step out.pkl"""
import cProfile, pickle, pstats, sys, time, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from enginelab.cad import read_cad
pr = cProfile.Profile(); pr.enable(); t = time.time()
a = read_cad(sys.argv[1])
pr.disable()
print(a.info, round(time.time() - t, 1), "s")
pstats.Stats(pr).sort_stats("cumtime").print_stats(18)
pickle.dump(a, open(sys.argv[2], "wb"))
