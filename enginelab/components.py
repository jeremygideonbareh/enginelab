"""Recognise standard / off-the-shelf components and build sourcing links.

Two independent signals:
  1. the part's name in the CAD file (designers usually name bought parts by their
     standard or part number: "M3x8 SHCS", "DIN 912 M5x20", "MR85", "MGN12H", "NEMA17"),
  2. its exact geometry (from the B-rep): thread size from the shank cylinder, bearing
     designation from bore x OD x width, extrusion size from the cross-section, etc.

The result is a category, a standard designation, a search query, and links to
suppliers and CAD-model libraries (URL formats checked in a browser, Nov 2026).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from urllib.parse import quote, quote_plus

import numpy as np

# ---------------------------------------------------------------- reference tables
METRIC = [1.6, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 27, 30, 36]

# Deep-groove / miniature ball bearings: designation -> (bore, OD, width) mm (ISO 15).
BEARINGS = {
    "MR52": (2, 5, 2.5), "MR63": (3, 6, 2.5), "MR74": (4, 7, 2.5), "MR85": (5, 8, 2.5),
    "MR95": (5, 9, 3), "MR105": (5, 10, 4), "MR115": (5, 11, 4), "MR117": (7, 11, 3), "MR128": (8, 12, 3.5),
    "683": (3, 7, 3), "684": (4, 9, 4), "685": (5, 11, 5), "686": (6, 13, 5), "687": (7, 14, 5),
    "688": (8, 16, 5), "689": (9, 17, 5), "693": (3, 8, 4), "694": (4, 11, 4), "695": (5, 13, 4),
    "696": (6, 15, 5), "697": (7, 17, 5), "698": (8, 19, 6), "623": (3, 10, 4), "624": (4, 13, 5),
    "625": (5, 16, 5), "626": (6, 19, 6), "627": (7, 22, 7), "608": (8, 22, 7), "609": (9, 24, 7),
    "6800": (10, 19, 5), "6801": (12, 21, 5), "6802": (15, 24, 5), "6803": (17, 26, 5), "6804": (20, 32, 7),
    "6900": (10, 22, 6), "6901": (12, 24, 6), "6902": (15, 28, 7), "6903": (17, 30, 7), "6904": (20, 37, 9),
    "6000": (10, 26, 8), "6001": (12, 28, 8), "6002": (15, 32, 9), "6003": (17, 35, 10), "6004": (20, 42, 12),
    "6005": (25, 47, 12), "6006": (30, 55, 13), "6200": (10, 30, 9), "6201": (12, 32, 10), "6202": (15, 35, 11),
    "6203": (17, 40, 12), "6204": (20, 47, 14), "6205": (25, 52, 15), "6206": (30, 62, 16), "6207": (35, 72, 17),
    "6300": (10, 35, 11), "6301": (12, 37, 12), "6302": (15, 42, 13), "6303": (17, 47, 14), "6304": (20, 52, 15),
    "6805": (25, 37, 7), "6806": (30, 42, 7), "6807": (35, 47, 7), "6808": (40, 52, 7), "6809": (45, 58, 7),
    "6810": (50, 65, 7), "6905": (25, 42, 9), "6906": (30, 47, 9),
    "16001": (12, 28, 7), "16002": (15, 32, 8), "16003": (17, 35, 8), "16004": (20, 42, 8), "16005": (25, 47, 8),
}
# 618xx / 619xx are the ISO names of the 68xx / 69xx thin-section series.
for _k in [k for k in list(BEARINGS) if len(k) == 4 and k[:2] in ("68", "69")]:
    BEARINGS["61" + _k[1:]] = BEARINGS[_k]
# Miniature profile rails: series -> (rail width, rail height) mm.
RAILS = {"MGN7": (7, 4.7), "MGN9": (9, 6.5), "MGN12": (12, 8), "MGN15": (15, 10), "HGR15": (15, 15),
         "HGR20": (20, 17.5), "HGR25": (23, 22)}
NEMA = {"NEMA 8": 20.3, "NEMA 11": 28.2, "NEMA 14": 35.2, "NEMA 17": 42.3, "NEMA 23": 56.4, "NEMA 34": 86.0}
EXTRUSIONS = [15, 20, 30, 40, 45, 60, 80]


@dataclass
class Component:
    category: str                  # e.g. "fastener / socket head cap screw"
    family: str                    # fastener | bearing | linear motion | transmission | motor | structure | electronics | spring | seal | custom
    designation: str = ""          # e.g. "ISO 4762 M3x8"
    query: str = ""                # search text for suppliers
    size: dict = field(default_factory=dict)
    source: str = "name"           # name | geometry
    confidence: float = 0.8
    standard_part: bool = True     # bought off the shelf (vs made for this design)
    links: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def links_for(query: str, family: str) -> dict:
    """Search links. Electronics go to electronics distributors, mechanics to mechanical ones."""
    q = query.strip()
    slug = re.sub(r"\s+", "-", q.lower())
    out = {
        "McMaster-Carr": f"https://www.mcmaster.com/{quote(slug)}/",
        "MISUMI": f"https://us.misumi-ec.com/vona2/result/?Keyword={quote_plus(q)}",
        "TraceParts (CAD)": f"https://www.traceparts.com/en/search?Keywords={quote_plus(q)}",
        "GrabCAD (CAD)": f"https://grabcad.com/library?query={quote(q)}",
        "Google Shopping": f"https://www.google.com/search?tbm=shop&q={quote_plus(q)}",
    }
    if family in ("electronics", "motor"):
        out = {"Digi-Key": f"https://www.digikey.com/en/products/result?keywords={quote(q)}", **out}
        if family == "electronics":
            out.pop("McMaster-Carr", None)
    return out


def _metric(d: float, tol: float = 0.35) -> float | None:
    best = min(METRIC, key=lambda m: abs(m - d))
    return best if abs(best - d) <= tol * max(1.0, best / 6) else None


# ---------------------------------------------------------------- name rules
_N = lambda p: re.compile(p, re.IGNORECASE)
# ROBOTIS Dynamixel CAD names its servo housing pieces DC15_A01_CASE_M_DUMMY etc.: pieces of a
# bought servo, although "case" otherwise marks a made-for-this-design part.
ELECTRONIC_PART = _N(r"\b(capacitor|resistor|inductor|diode|crystal|smd|qfn\d*|sop[\s-]*\d+|soic\d*|sot[\s-]*\d+|"
                     r"sod[\s-]*\d+|to[\s-]*22\d|(?:[crl]\s?)?(?:01005|0201|0402|0603|0805|1206|1210|2512)(?=\s|$|-))")   # SMD footprint codes
ROBOTIS_HOUSING = _N(r"\bdc\d{2}[\s_]a\d{2}[\s_](case|idle\w*[\s_]cap|dummy$)")
NAME_RULES: list[tuple[re.Pattern, str, str, str]] = [
    # (pattern, family, category, standard prefix for the designation)
    (ELECTRONIC_PART, "electronics", "electronic component", ""),
    (_N(r"\b(b\d+b[\s-](eh|ph|xh|zr|sh)\w*|jst)\b"), "electronics", "connector", ""),
    (_N(r"\b(phs|bts\d?|pan\s*head|binding\s*head)\b"), "fastener", "screw", ""),
    (ROBOTIS_HOUSING, "motor", "servo", ""),
    (_N(r"\b(shcs|socket\s*head\s*cap|iso\s*4762|din\s*912|cap\s*screw|cap[\s-]*head)\b"), "fastener", "socket head cap screw", "ISO 4762"),
    (_N(r"\b(bhcs|bhs|button\s*head|iso\s*7380)\b"), "fastener", "button head screw", "ISO 7380"),
    (_N(r"\b(fhcs|fhs|flat\s*head|countersunk|csk|iso\s*10642|din\s*7991)\b"), "fastener", "countersunk screw", "ISO 10642"),
    (_N(r"(^|[_\s-])shs([_\s-]|$)"), "fastener", "socket head cap screw", "ISO 4762"),
    (_N(r"\b(set\s*screw|grub|iso\s*402[6-9]|din\s*91[3-6])\b"), "fastener", "set screw", "ISO 4026"),
    (_N(r"\b(hex\s*bolt|iso\s*401[47]|din\s*93[13])\b"), "fastener", "hex bolt", "ISO 4017"),
    (_N(r"\b(self[\s-]*tapping|thread[\s-]*forming|pt\s*screw)\b"), "fastener", "self-tapping screw", ""),
    (_N(r"\b(t[\s-]*nut|hammer\s*nut|roll[\s-]*in\s*nut|drop[\s-]*in\s*nut)\b"), "fastener", "T-slot nut", ""),
    (_N(r"\b(lock\s*nut|nyloc|iso\s*7040|din\s*985)\b"), "fastener", "lock nut", "ISO 7040"),
    (_N(r"\b(hex\s*nut|hext\s*nut|nut|iso\s*4032|din\s*934)\b"), "fastener", "hex nut", "ISO 4032"),
    (_N(r"\b(washer|shim|iso\s*7089|din\s*125|din\s*433|din\s*988)\b"), "fastener", "washer", "ISO 7089"),
    (_N(r"\b(heat[\s-]*set|threaded\s*insert|m\d+(\.\d)?\s*insert)\b"), "fastener", "heat-set threaded insert", ""),
    (_N(r"\b(pneumatic|push[\s-]*(in|fit)|pc4|ptfe\s*coupler|tube\s*fitting|bowden\s*collet|ecas)\b"), "fluid", "pneumatic / tube fitting", ""),
    (_N(r"\b(dowel|iso\s*2338|iso\s*8734|din\s*6325)\b|(^|[_\s-])pin[_\s-]\d"), "fastener", "dowel pin", "ISO 8734"),
    (_N(r"(^|[_\s-])fasteners?([_\s-]|$)"), "fastener", "screw", ""),
    (_N(r"\b(standoff|spacer)\b"), "fastener", "standoff / spacer", ""),
    (_N(r"\b(rivet)\b"), "fastener", "rivet", ""),
    (_N(r"\b(screw|bolt)\b"), "fastener", "screw", ""),
    (_N(r"(^|[^a-z])m\d+(\.\d+)?\s*[x×]\s*\d+"), "fastener", "screw", ""),     # "M3x16-..." is a screw
    (_N(r"(^|[\s_-])[a-z]{0,2}m\d(\.\d)?-\d{1,3}(\s|$)"), "fastener", "screw", ""),   # "7-HM3-8", "KM3-7" style
    (_N(r"\b(lm\d+\w*uu|linear\s*bearing|igus|drylin)\b"), "linear motion", "linear bushing", ""),
    (_N(r"\b(mg[nw]\d+\w*|hg[rhw]\d+\w*|linear\s*(rail|guide)|rail\s*carriage)\b"), "linear motion", "linear guide", ""),
    (_N(r"\b(lead\s*screw|leadscrew)\s*(stepper|motor)|integrated\s*lead\s*screw"), "motor", "stepper motor", ""),
    (_N(r"\b(lead\s*screw|leadscrew|tr\s?\d+x\d+|t8\b|acme)\b"), "linear motion", "lead screw", ""),
    (_N(r"\b(ball\s*screw|sfu\d+)\b"), "linear motion", "ball screw", ""),
    (_N(r"\b(needle[_\s-]*bearing)"), "bearing", "needle bearing", ""),
    (_N(r"\b(thrust\s*bearing)"), "bearing", "thrust bearing", ""),
    (_N(r"(ball\s*bearing|ballbearing|bearing|\bmr\d{2,3}\b|\b(16|6)[0-9]{2,4}[\s-]*(zz|2rs|rs|2z|z)\b|\b6[0-9]{2,3}\b(?!\s*(housing|fan|x))|\b608\b|\b625\b|\bf6\d{2}\b)"), "bearing", "ball bearing", ""),
    (_N(r"(pulley|idler|sprocket|(?<!cont)roller)"), "transmission", "pulley / idler", ""),   # not "controller"
    (_N(r"\b(gt2|gt3|htd|timing\s*belt|belt)\b"), "transmission", "timing belt / belt", ""),
    (_N(r"\b(coupler|coupling)\b"), "transmission", "shaft coupling", ""),
    (_N(r"\b(harmonic|strain\s*wave|planetary|gearbox|reducer)\b"), "transmission", "gearbox / reducer", ""),
    (_N(r"\b(gear|pinion|rack|worm)\b"), "transmission", "gear", ""),
    (_N(r"\bhorn\b|舵盘"), "transmission", "servo horn", ""),
    (_N(r"\b(shaft|axle)\b"), "transmission", "shaft / rod", ""),
    (_N(r"\b(nema[\s-]*\d+|stepper|step\s*motor)\b"), "motor", "stepper motor", ""),
    (_N(r"\b(bldc|brushless|outrunner|gimbal\s*motor|stator)\b"), "motor", "brushless motor", ""),
    (_N(r"\b(servo|dynamixel|mg99\d\w*|sg90|sts?\d{4}\w*|scs\d{4}\w*|(xl|xc|xm|xh|xw)[\s-]?\d{3}\w*|ax[\s-]?12\w*)\b"), "motor", "servo", ""),
    (_N(r"\b(gear\s*motor|dc\s*motor|motor)\b"), "motor", "motor", ""),
    (_N(r"\b(\d{4}\s*(extrusion|profile)|extrusion|v[\s-]*slot|hfs\d|alu(minium)?\s*profile)\b|\b(1515|2020|2040|2060|2080|3030|3060|4040|4080)\b"), "structure", "aluminium extrusion", ""),
    (_N(r"\b(spring)\b"), "spring", "spring", ""),
    (_N(r"\b(o[\s-]*ring|seal|gasket|grommet)\b"), "seal", "seal / O-ring", ""),
    (_N(r"\b(magnet)\b"), "structure", "magnet", ""),
    (_N(r"\b(wheel|tire|tyre|caster|castor)\b"), "transmission", "wheel", ""),
    (_N(r"\b(fan|blower)\b"), "electronics", "fan", ""),
    (_N(r"\b(pcb|board|raspberry|arduino|skr|octopus|mainboard|controller|driver)\b"), "electronics", "circuit board", ""),
    (_N(r"\b(psu|power\s*supply|lrs-\d+)\b"), "electronics", "power supply", ""),
    (_N(r"\b(connector|jst|xt60|xt30|terminal|plug|socket|wago|221-4\d\d|molex)\b"), "electronics", "connector", ""),
    (_N(r"\b(ptfe|bowden)\s*(guide\s*)?tube\b"), "fluid", "PTFE / Bowden tube", ""),
    (_N(r"\b(switch|endstop|limit\s*switch|microswitch|push\s*button|button)\b"), "electronics", "switch", ""),
    (_N(r"\b(sensor|probe|encoder|thermistor|camera|imu|lidar)\b"), "electronics", "sensor", ""),
    (_N(r"\b(led|display|screen|lcd)\b"), "electronics", "display / LED", ""),
    (_N(r"\b(heater|heat\s*cartridge|hotend|nozzle|heatsink)\b"), "electronics", "heater / hotend", ""),
    (_N(r"\b(cable\s*chain|drag\s*chain|chain\s*link|cable|wire)\b"), "electronics", "cable / cable chain", ""),
    (_N(r"\b(shaft|axle|rod)\b"), "transmission", "shaft / rod", ""),
    (_N(r"\b(panel|plate|sheet|bed)\b"), "structure", "plate / panel", ""),
    (_N(r"\b(bracket|mount|holder|clamp)\b"), "structure", "bracket / mount", ""),
]
# Names that mark a part made for this design (printed / machined), even if they mention a
# standard part it fits ("Cowling_PC4_M10 v6" is a printed cowling for a PC4 fitting).
# Parts made for this design: printed/machined nouns win over any standard part the name
# mentions ("BEARING-CLAMP", "STEPPER-BRACKET", "probe_bracket", "Cable Frame Anchor" are
# brackets, not bearings / motors / sensors / cables).
CUSTOM_HINT = _N(r"(_a\b|\[a\]|_x\d\b|printed|\bpla\b|petg|\babs\b|\basa\b|cowling|shroud|duct|"
                 r"housing|enclosure|frame_|adapter|bracket|clamp|mount(?!ing\s*screw)|holder|cover|"
                 r"anchor|grill|retainer|support|guard|\bcase\b|_case|knob|foot|feet|skirt|panel|door|"
                 r"spool|tensioner|\bblock\b|_block|cradle|\bz_carriage|\bx_carriage|^carriage$|^[xyz]_carriage|"
                 r"(motor|stepper|servo)[\s_-]*(plate|panel)|\b[abxyz]_stepper_(upper|lower|left|right|rear|front)|"
                 r"^[xyz][\s_]motor$|redesign|remix|fixture|braket|flange(?!d|\s*bearing)|bearing[\s_-]*cap|"
                 r"^\d+-[a-z0-9-]*(stator|rotor))")      # reBot-style printed actuator shells "2-RSM2-STATOR-1"


def _size_from_name(name: str) -> dict:
    s: dict = {}
    m = re.search(r"\bM(\d+(?:[.,]\d+)?)\s*[x×X*]\s*(\d+(?:[.,]\d+)?)", name) or \
        re.search(r"(?:^|[_\s-])[Mm](\d+(?:[.,]\d+)?)_(\d+(?:[.,]\d+)?)(?:[_\s-]|$)", name)
    if m:
        s["thread"] = f"M{m.group(1).replace(',', '.')}"
        s["length_mm"] = float(m.group(2).replace(",", "."))
    else:
        m = re.search(r"\bM(\d+(?:\.\d+)?)\b", name)
        if m:
            s["thread"] = f"M{m.group(1)}"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)", name)
    if m:
        s["dims_mm"] = [float(m.group(i)) for i in (1, 2, 3)]
    m = re.search(r"\b(MR\d{2,3}|F?16\d{3}|F?6\d{2,4}|608)(?:[\s-]?(ZZ|2RS|RS|2Z|Z))?\b", name, re.I)
    if m:
        s["bearing"] = m.group(1).upper()
    m = re.search(r"\b(MG[NW]\d+|HG[RHW]\d+)([CH])?\b", name, re.I)
    if m:
        s["rail"] = m.group(1).upper() + (m.group(2) or "").upper()
    m = re.search(r"\bNEMA[\s_-]*(\d+)", name, re.I)
    if m:
        s["frame"] = f"NEMA {m.group(1)}"
    m = re.search(r"\b(GT2|GT3|HTD\d?M?)\b", name, re.I)
    if m:
        s["belt_profile"] = m.group(1).upper()
    m = re.search(r"\b(\d{2,3})\s*T\b", name)
    if m:
        s["teeth"] = int(m.group(1))
    m = re.search(r"\b(1515|2020|2040|2060|2080|3030|3060|4040|4080)\b", name)
    if m:
        s["profile"] = m.group(1)
    return s


def from_name(name: str | None) -> Component | None:
    if not name:
        return None
    n = name.replace("_", " ")
    for pat, family, cat, std in NAME_RULES:
        if pat.search(n):
            size = _size_from_name(name)
            desig = _designation(cat, std, size, name)
            custom = family == "structure" and cat in ("plate / panel", "bracket / mount")
            return Component(category=cat, family=family, designation=desig,
                             query=desig or f"{cat} {name}", size=size, source="name",
                             confidence=0.9, standard_part=not custom)
    return None


def _designation(cat: str, std: str, size: dict, name: str) -> str:
    if cat in ("socket head cap screw", "button head screw", "countersunk screw", "set screw", "hex bolt",
               "self-tapping screw", "screw") and "thread" in size:
        L = f"x{size['length_mm']:g}" if "length_mm" in size else ""
        return f"{std + ' ' if std else ''}{size['thread']}{L} {cat}".strip()
    if cat in ("hex nut", "lock nut", "washer", "heat-set threaded insert", "T-slot nut", "standoff / spacer") \
            and "thread" in size:
        return f"{std + ' ' if std else ''}{size['thread']} {cat}".strip()
    if cat == "ball bearing":
        if "bearing" in size:
            d = BEARINGS.get(size["bearing"].lstrip("F"))
            dims = f" {d[0]:g}x{d[1]:g}x{d[2]:g} mm" if d else ""
            return f"{size['bearing']} ball bearing{dims}"
        if "dims_mm" in size:
            des = match_bearing(*size["dims_mm"])
            return f"{des + ' ' if des else ''}ball bearing {'x'.join(f'{v:g}' for v in size['dims_mm'])} mm"
    if cat == "linear guide" and "rail" in size:
        return f"{size['rail']} linear guide"
    if cat == "stepper motor" and "frame" in size:
        return f"{size['frame']} stepper motor"
    if cat == "aluminium extrusion" and "profile" in size:
        return f"{size['profile']} aluminium extrusion"
    if cat == "timing belt / belt" and "belt_profile" in size:
        return f"{size['belt_profile']} timing belt"
    if cat == "fan":
        m = re.search(r"\b(\d{2})(\d{2})\b|(\d{2})\s*x\s*(\d{2})\s*x\s*(\d{1,2})", name, re.I)
        if m:
            a, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(5))
            kind = "blower fan" if re.search("blower|radial", name, re.I) else "axial fan"
            return f"{a}{b} {kind} ({a}x{a}x{b} mm)"
    if cat == "pulley / idler" and ("belt_profile" in size or "teeth" in size):
        return f"{size.get('belt_profile', '')} {str(size['teeth']) + 'T ' if 'teeth' in size else ''}pulley".strip()
    return ""


def match_bearing(bore: float, od: float, width: float, tol: float = 0.35) -> str | None:
    best, err = None, 1e9
    for k, (d, D, B) in BEARINGS.items():
        e = abs(d - bore) + abs(D - od) + abs(B - width)
        if e < err:
            best, err = k, e
    return best if err <= 3 * tol else None


# ---------------------------------------------------------------- geometry rules
def from_geometry(info, brep: dict | None, specs: dict) -> Component | None:
    """Recognise common standard parts from exact geometry (B-rep cylinders) and shape."""
    ext = np.sort(np.asarray(info.extents))[::-1]
    L1, L2, L3 = ext
    cyl = (brep or {}).get("cylinders", [])
    holes = sorted({c["r"] for c in cyl if c["hole"]})
    shafts = sorted({c["r"] for c in cyl if not c["hole"]})
    sym = info.sym_score > 0.7 and info.features.get("cyl_frac", 0) > 0.25
    # Ring: bearing / washer / spacer.
    if sym and info.genus == 1 and holes and shafts:
        bore, od = 2 * min(holes), 2 * max(shafts)
        width = info.sym_length
        des = match_bearing(bore, od, width)
        if des:
            return Component("ball bearing", "bearing", f"{des} ball bearing {bore:g}x{od:g}x{width:g} mm",
                             f"{des} bearing", {"bore_mm": bore, "od_mm": od, "width_mm": width},
                             "geometry", 0.75)
        if width < 0.2 * od:
            m = _metric(bore - 0.3, 0.6)
            if m:
                return Component("washer", "fastener", f"ISO 7089 M{m:g} washer", f"M{m:g} washer",
                                 {"thread": f"M{m:g}", "od_mm": od, "thickness_mm": width}, "geometry", 0.6)
    # Screw / bolt: slender, symmetric, solid shank.
    if sym and info.features.get("slender", 0) > 1.3 and shafts and info.genus == 0:
        shank = min(shafts, key=lambda r: abs(r - min(shafts)))
        d = 2 * shank
        m = _metric(d, 0.25)
        head = 2 * max(shafts)
        if m and head > 1.2 * d:
            # Head height from the head's cylindrical face (lateral area = 2 pi r h) separates
            # ISO 4762 cap heads (h = d) from ISO 7380 button heads (h ~ 0.55 d); diameters alone can't.
            head_faces = [c for c in cyl if not c["hole"] and abs(c["r"] - max(shafts)) < 1e-3]
            h = sum(c["area"] for c in head_faces) / (2 * math.pi * max(shafts)) if head_faces else None
            if h is not None and h > 0.75 * m:
                kind, std, head_h = "socket head cap screw", "ISO 4762", m
            elif h is not None:
                kind, std, head_h = "button head screw", "ISO 7380", 0.55 * m
            else:
                kind = "button head screw" if head / m > 1.85 else "socket head cap screw"
                std = "ISO 7380" if kind == "button head screw" else "ISO 4762"
                head_h = 0.55 * m if kind == "button head screw" else m
            length = round(info.sym_length - head_h)
            return Component(kind, "fastener", f"{std} M{m:g}x{length:g} {kind}", f"M{m:g}x{length:g} {kind}",
                             {"thread": f"M{m:g}", "length_mm": length}, "geometry", 0.6)
        if m and abs(head - d) < 0.1 * d and info.sym_length > 8 * d:
            return Component("shaft / rod", "transmission", f"Ø{d:g} x {info.sym_length:.0f} mm shaft",
                             f"{d:g}mm linear shaft rod", {"diameter_mm": d, "length_mm": round(info.sym_length)},
                             "geometry", 0.6)
    # Hex nut: short, hole, hexagon outline.
    if info.genus == 1 and holes and info.features.get("hex", 0) > 0.3 and info.sym_length < 1.2 * 2 * min(holes) * 1.6:
        m = _metric(2 * min(holes) * 1.0 + 0.0, 0.6) or _metric(2 * min(holes) + 0.5, 0.6)
        if m:
            return Component("hex nut", "fastener", f"ISO 4032 M{m:g} hex nut", f"M{m:g} hex nut",
                             {"thread": f"M{m:g}"}, "geometry", 0.55)
    # Long constant prisms: extrusion / rail.
    if L1 > 6 * L2 and L2 < 1.6 * L3:
        sq = round(L2)
        if sq in EXTRUSIONS and abs(L2 - L3) < 0.6:
            return Component("aluminium extrusion", "structure", f"{sq}{sq} aluminium extrusion, {L1:.0f} mm",
                             f"{sq}{sq} aluminium extrusion", {"profile": f"{sq}{sq}", "length_mm": round(L1)},
                             "geometry", 0.6)
        for k, (w, h) in RAILS.items():
            if abs(L2 - w) < 0.4 and abs(L3 - h) < 0.6:
                return Component("linear guide", "linear motion", f"{k} linear rail, {L1:.0f} mm",
                                 f"{k} linear rail", {"rail": k, "length_mm": round(L1)}, "geometry", 0.55)
    # Stepper motor: square body ~NEMA size with a round shaft.
    if shafts and abs(L2 - L3) < 1.0:
        for k, side in NEMA.items():
            if abs(L2 - side) < 0.8 and 2 * min(shafts) in (5, 6.35, 8, 4):
                return Component("stepper motor", "motor", f"{k} stepper motor", f"{k} stepper motor",
                                 {"frame": k, "shaft_mm": 2 * min(shafts)}, "geometry", 0.5)
    # Toothed wheels.
    if specs.get("teeth") and sym:
        z = specs["teeth"]
        bore = 2 * min(holes) if holes else None
        if specs.get("module_mm", 1) < 0.9 and bore and bore <= 8:
            return Component("pulley / idler", "transmission", f"GT2 {z}T pulley, bore {bore:g} mm",
                             f"GT2 {z}T pulley {bore:g}mm bore", {"teeth": z, "bore_mm": bore}, "geometry", 0.5)
        return Component("gear", "transmission", f"spur gear m{specs.get('module_mm', 0):.2g} z{z}",
                         f"spur gear module {specs.get('module_mm', 0):.2g} {z} teeth", {"teeth": z}, "geometry", 0.5)
    # Thin flat parts: plates and panels (custom).
    if L3 < 0.08 * L2:
        return Component("plate / panel", "structure", f"plate {L1:.0f} x {L2:.0f} x {L3:.1f} mm",
                         f"{L3:.1f}mm plate", {"thickness_mm": round(L3, 2)}, "geometry", 0.5, standard_part=False)
    return None


def recognise(name: str | None, info, brep: dict | None, specs: dict, path: list | None = None) -> Component | None:
    """Best guess of what standard component this is (None for custom parts).
    The part's own name is tried first; the sub-assembly it sits in is the fallback, and a
    bought sub-assembly (fitting, fan, carriage...) passes its identity to its pieces."""
    by_name = from_name(name)
    strong = by_name is not None and (by_name.family == "fastener" and bool(by_name.designation) or
                                      bool(ROBOTIS_HOUSING.search(name or "")) or
                                      by_name.category == "pulley / idler" and re.search("roller|idler|pulley", name or "", re.I))
    if name and CUSTOM_HINT.search(name) and not strong:
        by_name = None          # made-for-this-design part, whatever standard part it mentions
        path = None
    parent = from_name(path[-1]) if path else None
    if parent and parent.standard_part and parent.family not in ("fastener", "structure"):
        # A bought sub-assembly (fan, fitting, carriage...) names its own pieces, but screws,
        # inserts and bearings inside it are still what they are.
        if by_name is None or (by_name.family != parent.family
                               and by_name.family not in ("fastener", "bearing")):
            parent.query = parent.designation or f"{parent.category} {path[-1]}"
            parent.size["bought_assembly"] = path[-1]
            by_name = parent
    elif by_name is None and parent is not None and parent.family == "fastener":
        by_name = parent
    by_geo = None
    try:
        by_geo = from_geometry(info, brep, specs)
    except Exception:
        by_geo = None
    comp = by_name
    if by_name and by_geo and by_geo.family == by_name.family:
        # Geometry fills in what the name left out (e.g. a bearing number from its dimensions).
        if not by_name.designation and by_geo.designation:
            by_name.designation = by_geo.designation
            by_name.query = by_geo.query
        by_name.size = {**by_geo.size, **by_name.size}
        by_name.confidence = 0.97
    elif comp is None:
        comp = by_geo
    if comp is None:
        return None
    if name and CUSTOM_HINT.search(name) and comp.family == "structure":
        comp.standard_part = False
    comp.query = re.sub(r"\s+", " ", re.sub(r"[()]", " ", comp.query)).strip()
    comp.links = links_for(comp.query, comp.family) if comp.standard_part else {}
    return comp
