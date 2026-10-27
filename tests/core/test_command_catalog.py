"""The catalogue tool: commands the guides do not describe yet take their title, text
and default from the comment of their default in the program's ini.cpp."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _tool():
    spec = importlib.util.spec_from_file_location(
        "command_catalog", ROOT / "tools/command_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_undocumented_commands_from_the_defaults(tmp_path):
    tool = _tool()
    ini = tmp_path / "ini.cpp"
    ini.write_text(
        "    Q68=1;  // int bedload layer: bed shear stress, 0 log law, 1 max(log law, 0.3 rho k)\n"
        "    Q76=-1.0e20;        // double fixed (non-erodible) parcel bed upstream of x = Q76\n"
        "    A225=1;    // int SFLOW Riemann solver: 1 HLL, 2 HLLC\n"
        "    A10=6;\n"
    )  # fmt: skip
    defaults = tool.read_defaults(ini)
    assert defaults["Q 76"] == (
        "double",
        "fixed (non-erodible) parcel bed upstream of x = Q76",
        "-1.0e20",
    )
    blank = {"title": "", "repeat": False, "params": [], "values": 1, "options": {},
             "default": [], "text": "Not in the user guide yet.", "documented": False}  # fmt: skip
    commands = {
        "Q 68": dict(blank),
        "Q 76": dict(blank),
        "A 225": dict(blank),
        "A 10": {"title": "Model", "values": 1, "params": [], "default": ["6"], "text": "Guide."},
    }
    tool.describe_from_defaults(commands, defaults)
    assert commands["Q 68"]["title"] == "Bedload layer: bed shear stress"
    assert commands["Q 68"]["text"].endswith("0.3 rho k). Not in the user guide yet.")
    assert commands["Q 68"]["params"] == [{"name": "", "type": "int"}]
    assert commands["Q 68"]["default"] == ["1"]
    assert commands["Q 76"]["default"] == ["-1.0e20"]
    assert commands["A 225"]["title"] == "SFLOW Riemann solver"
    assert commands["A 10"]["title"] == "Model"  # described by the guide: kept


def test_the_bedload_layer_commands_are_in_the_catalogue():
    commands = json.loads(
        (ROOT / "src/lagoonviz/core/data/catalog/reef3d.json").read_text(encoding="utf-8")
    )["commands"]
    for code in ("Q 68", "Q 69", "Q 70", "Q 72", "Q 74", "Q 75", "Q 76", "Q 77", "Q 78", "Q 79"):
        assert commands[code]["title"] and commands[code]["values"] == 1, code
