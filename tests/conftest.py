import os

# GUI tests run without a display (CI, clusters).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
