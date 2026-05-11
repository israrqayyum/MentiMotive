import sys
import os
import traceback

sys.path.insert(0, os.path.abspath("backend"))

try:
    from main import app
    print("SUCCESS")
except Exception as e:
    print("FAILED")
    traceback.print_exc()
    sys.exit(1)
