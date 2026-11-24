"""Start EngineLab: python run.py [port]

Environment (used by the hosted beta): HOST (default 127.0.0.1), PORT (default 8000),
ENGINELAB_BETA=1 (public-server limits: smaller uploads, fewer concurrent jobs).
"""
import os
import sys
import warnings

import uvicorn

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("enginelab.api:app", host=host, port=port, reload=False)
