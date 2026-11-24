"""Where to buy each part, what it should cost, delivered to the user's country.

For every parts-list line:
  * a typical price range (catalogue / marketplace prices for that kind and size of part,
    in USD, converted to the local currency); custom parts get a 3D-printing or CNC estimate;
  * the shops that sell it in or to that country, best first (local shops before importing),
    each with a ready search link;
  * a landed-cost estimate when the shop ships from abroad (shipping + import taxes);
  * a shopping plan: the fewest shops that cover the whole list.

Prices are estimates, stated as ranges. Live prices need distributor API keys (e.g. MOUSER_API_KEY):
when a key is set, "check live price" asks that API for real price breaks and stock.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np

# --------------------------------------------------------------------------- regions
# fx: units of local currency per USD (fallback values; refreshed from a free FX API when possible)
REGIONS = {
    "IN": {"name": "India", "currency": "INR", "symbol": "₹", "fx": 86.0, "import_tax": 0.28,
           "intl_ship_usd": 12.0, "local_ship_usd": 1.0},
    "US": {"name": "United States", "currency": "USD", "symbol": "$", "fx": 1.0, "import_tax": 0.25,
           "intl_ship_usd": 15.0, "local_ship_usd": 6.0},
    "GB": {"name": "United Kingdom", "currency": "GBP", "symbol": "£", "fx": 0.79, "import_tax": 0.20,
           "intl_ship_usd": 12.0, "local_ship_usd": 4.0},
    "DE": {"name": "Germany / EU", "currency": "EUR", "symbol": "€", "fx": 0.92, "import_tax": 0.21,
           "intl_ship_usd": 12.0, "local_ship_usd": 5.0},
    "CN": {"name": "China", "currency": "CNY", "symbol": "¥", "fx": 7.2, "import_tax": 0.13,
           "intl_ship_usd": 15.0, "local_ship_usd": 1.0},
    "AU": {"name": "Australia", "currency": "AUD", "symbol": "A$", "fx": 1.52, "import_tax": 0.10,
           "intl_ship_usd": 15.0, "local_ship_usd": 8.0},
    "CA": {"name": "Canada", "currency": "CAD", "symbol": "C$", "fx": 1.38, "import_tax": 0.13,
           "intl_ship_usd": 15.0, "local_ship_usd": 10.0},
    "OTHER": {"name": "Other country", "currency": "USD", "symbol": "$", "fx": 1.0, "import_tax": 0.20,
              "intl_ship_usd": 15.0, "local_ship_usd": 8.0},
}

# families: fastener, bearing, linear_motion, transmission, motor, electronics, structure, spring, fluid,
# seal, printed (3D printing service), machined (CNC service), engine (engine spares)
ALL = {"fastener", "bearing", "linear_motion", "transmission", "motor", "electronics", "structure",
       "spring", "fluid", "seal", "engine"}
SHOPS = {
    # India
    "robu": ("Robu.in", "https://robu.in/?s={q}&post_type=product", "IN",
             {"fastener", "bearing", "linear_motion", "transmission", "motor", "electronics", "structure", "spring"}),
    "robocraze": ("Robocraze", "https://robocraze.com/search?q={q}", "IN",
                  {"motor", "electronics", "transmission", "linear_motion", "bearing"}),
    "industrybuying": ("Industrybuying", "https://www.industrybuying.com/search/?q={q}", "IN",
                       {"fastener", "bearing", "transmission", "seal", "spring", "structure", "fluid", "engine"}),
    "indiamart": ("IndiaMART (bulk / local suppliers)", "https://dir.indiamart.com/search.mp?ss={q}", "IN", ALL),
    "amazon_in": ("Amazon.in", "https://www.amazon.in/s?k={q}", "IN", ALL),
    "element14_in": ("element14 India", "https://in.element14.com/search?st={q}", "IN", {"electronics"}),
    "misumi_in": ("MISUMI India", "https://in.misumi-ec.com/vona2/result/?Keyword={q}", "IN",
                  {"fastener", "bearing", "linear_motion", "transmission", "structure", "spring"}),
    # United States
    "mcmaster": ("McMaster-Carr", "https://www.mcmaster.com/{q}", "US",
                 {"fastener", "bearing", "linear_motion", "transmission", "structure", "spring", "seal", "fluid"}),
    "amazon_us": ("Amazon.com", "https://www.amazon.com/s?k={q}", "US", ALL),
    "misumi_us": ("MISUMI USA", "https://us.misumi-ec.com/vona2/result/?Keyword={q}", "US",
                  {"fastener", "bearing", "linear_motion", "transmission", "structure", "spring"}),
    "ebay_us": ("eBay", "https://www.ebay.com/sch/i.html?_nkw={q}", "US", ALL),
    # UK / EU
    "rs_uk": ("RS Components", "https://uk.rs-online.com/web/c/?searchTerm={q}", "GB",
              {"fastener", "bearing", "electronics", "motor", "transmission", "seal", "spring"}),
    "amazon_uk": ("Amazon.co.uk", "https://www.amazon.co.uk/s?k={q}", "GB", ALL),
    "ebay_uk": ("eBay UK", "https://www.ebay.co.uk/sch/i.html?_nkw={q}", "GB", ALL),
    "amazon_de": ("Amazon.de", "https://www.amazon.de/s?k={q}", "DE", ALL),
    "misumi_de": ("MISUMI Europe", "https://de.misumi-ec.com/vona2/result/?Keyword={q}", "DE",
                  {"fastener", "bearing", "linear_motion", "transmission", "structure", "spring"}),
    "ebay_de": ("eBay.de", "https://www.ebay.de/sch/i.html?_nkw={q}", "DE", ALL),
    # Australia / Canada / China
    "amazon_au": ("Amazon.com.au", "https://www.amazon.com.au/s?k={q}", "AU", ALL),
    "amazon_ca": ("Amazon.ca", "https://www.amazon.ca/s?k={q}", "CA", ALL),
    "taobao": ("Taobao", "https://s.taobao.com/search?q={q}", "CN", ALL),
    # Global (ship worldwide)
    "aliexpress": ("AliExpress", "https://www.aliexpress.com/wholesale?SearchText={q}", "CN", ALL),
    "lcsc": ("LCSC", "https://www.lcsc.com/search?q={q}", "CN", {"electronics"}),
    "digikey": ("Digi-Key", "https://www.digikey.com/en/products/result?keywords={q}", "US", {"electronics", "motor"}),
    "mouser": ("Mouser", "https://www.mouser.com/c/?q={q}", "US", {"electronics", "motor"}),
}
LOCAL_ORDER = {
    "IN": ["robu", "industrybuying", "amazon_in", "robocraze", "misumi_in", "element14_in", "indiamart"],
    "US": ["mcmaster", "amazon_us", "misumi_us", "ebay_us", "digikey", "mouser"],
    "GB": ["rs_uk", "amazon_uk", "ebay_uk"],
    "DE": ["amazon_de", "misumi_de", "ebay_de"],
    "CN": ["taobao", "lcsc"],
    "AU": ["amazon_au"],
    "CA": ["amazon_ca"],
    "OTHER": [],
}
GLOBAL = ["aliexpress", "lcsc", "digikey", "mouser"]
SERVICES = {   # custom parts: instant-quote manufacturing services
    "printed": [("Craftcloud (compares 3D-printing services)", "https://craftcloud3d.com/"),
                ("JLC3DP", "https://jlc3dp.com/"), ("PCBWay 3D printing", "https://www.pcbway.com/rapid-prototyping/3D-Printing/")],
    "machined": [("Xometry", "https://www.xometry.com/"), ("JLCCNC", "https://jlccnc.com/"),
                 ("PCBWay CNC", "https://www.pcbway.com/rapid-prototyping/CNC-machining/")],
}
FAMILY_OF_LABEL = {
    "fastener": "fastener", "bolt": "fastener", "bearing": "bearing", "main_bearing": "engine",
    "linear_motion": "linear_motion", "transmission": "transmission", "timing_gear": "engine",
    "motor": "motor", "electronics": "electronics", "spark_plug": "engine", "structure": "structure",
    "spring": "spring", "valve_spring": "engine", "fluid": "fluid", "seal": "seal", "head_gasket": "engine",
    "piston_ring": "engine", "piston": "engine", "piston_pin": "engine", "connecting_rod": "engine",
    "crankshaft": "engine", "camshaft": "engine", "valve": "engine",
}

# --------------------------------------------------------------------------- typical prices (USD, one unit)
NEMA_PRICE = {"NEMA 8": (12, 30), "NEMA 11": (12, 30), "NEMA 14": (9, 25), "NEMA 17": (6, 25),
              "NEMA 23": (18, 60), "NEMA 34": (45, 140)}


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def typical_price(line: dict) -> tuple[float, float, str] | None:
    """(low, high, basis) USD per unit for a standard part, from its category and size."""
    cat = (line.get("category") or "").lower()
    size = line.get("size") or {}
    name = f"{line.get('name', '')} {line.get('designation', '')}".lower()
    dims = sorted(line.get("dims_mm") or [0, 0, 0], reverse=True)
    thread = size.get("thread") or (re.search(r"\bm(\d+(?:\.\d+)?)", name) or [None, None])[1]
    t = _num(str(thread).lstrip("Mm"), None) if thread else None
    if "screw" in cat or "bolt" in cat:
        t = t or 4.0
        L = _num(size.get("length_mm"), 3 * t)
        base = 0.012 * t * t * (1 + L / (8 * t))
        return base * 0.6, base * 2.5, f"M{t:g} screw, pack price per piece"
    if cat in ("hex nut", "lock nut", "washer", "t-slot nut"):
        t = t or 4.0
        base = 0.006 * t * t + (0.15 if cat == "t-slot nut" else 0)
        return base * 0.6, base * 2.5, "pack price per piece"
    if "insert" in cat:
        return 0.04, 0.25, "heat-set insert, pack price per piece"
    if "standoff" in cat or "spacer" in cat or "dowel" in cat or "rivet" in cat:
        return 0.05, 0.6, "pack price per piece"
    if "bearing" in cat:
        od = dims[0] if dims[0] else 22
        lo = 0.25 + 0.012 * od
        return lo, lo * 5, f"OD {od:.0f} mm ball bearing; budget brand to SKF/NSK"
    if "linear guide" in cat or "rail" in cat:
        s = str(size.get("rail") or name)
        w = _num((re.search(r"(\d+)", s) or [None, None])[1], 12)
        if "carriage" in name or "block" in name or (dims[0] and dims[0] < 80):
            return 3 + 0.7 * w, 10 + 2.2 * w, f"MGN{w:g} carriage"
        per100 = 1.2 + 0.25 * w
        return per100 * dims[0] / 100, 3 * per100 * dims[0] / 100, f"MGN{w:g} rail, {dims[0]:.0f} mm"
    if "linear bushing" in cat:
        return 0.8, 6, "linear bushing"
    if "lead screw" in cat or "ball screw" in cat:
        return 5 + dims[0] / 60, 15 + dims[0] / 15, f"{cat}, {dims[0]:.0f} mm"
    if "stepper" in cat:
        lo, hi = NEMA_PRICE.get(size.get("frame", ""), (8, 30))
        return lo, hi, f"{size.get('frame', 'stepper')} motor"
    if re.search(r"adapter|board|driver|controller|hat|shield", name) and "servo" in cat:
        return 5, 40, "servo driver / adapter board"
    if "servo" in cat:
        if re.search(r"dynamixel|xl430|xm|xh", name):
            return 45, 300, "Dynamixel-class smart servo"
        if re.search(r"xl330|xc330|dc15", name):
            return 22, 35, "XL330-class smart servo"
        if re.search(r"sts3215|st3215|scs", name):
            return 14, 25, "STS3215 bus servo"
        return 3, 25, "hobby servo"
    if "brushless" in cat:
        return 12, 90, "BLDC / gimbal motor"
    if cat == "motor":
        return 3, 40, "DC / gear motor"
    if "pulley" in cat or "idler" in cat:
        return 0.6, 6, "pulley / idler"
    if "belt" in cat:
        return 1.0, 5.0, "timing belt, per metre"
    if "gear" in cat:
        return 1, 15, "gear"
    if "coupling" in cat:
        return 1.5, 12, "shaft coupling"
    if "shaft" in cat or "rod" in cat:
        return 1 + dims[0] / 80, 4 + dims[0] / 25, f"ground shaft, {dims[0]:.0f} mm"
    if "extrusion" in cat:
        return 2.5 * dims[0] / 1000 + 0.5, 9 * dims[0] / 1000 + 1, f"aluminium extrusion, {dims[0]:.0f} mm"
    if cat == "fan":
        return 1.5, 10, "fan"
    if "circuit board" in cat:
        return 5, 60, "controller board"
    if "connector" in cat or "electronic component" in cat:
        return 0.05, 2, "connector / component"
    if "switch" in cat or "sensor" in cat:
        return 0.3, 25, cat
    if "heater" in cat:
        return 3, 40, "hotend / heater part"
    if "spring" in cat:
        return 0.1, 2, "spring"
    if "seal" in cat or "o-ring" in cat:
        return 0.05, 3, "seal / O-ring"
    if "pneumatic" in cat or "fitting" in cat or "ptfe" in cat:
        return 0.3, 3, "tube fitting"
    if "magnet" in cat:
        return 0.05, 1, "magnet"
    if "wheel" in cat:
        return 1, 15, "wheel"
    return None


def custom_part_cost(line: dict, region: dict) -> tuple[float, float, str, str]:
    """Estimate for a made-for-this-design part: 3D printed (polymers) or CNC machined (metals)."""
    mass = float(line.get("unit_mass_kg") or 0.0)
    mat = (line.get("material") or "").lower()
    dims = sorted(line.get("dims_mm") or [0, 0, 0], reverse=True)
    if any(k in mat for k in ("pla", "petg", "asa", "abs", "nylon", "pa12", "pa-cf", "polycarbonate", "pc ")) \
            or "printed" in (line.get("category") or ""):
        # filament ~ $20-30/kg at home; services charge ~ $0.15-0.35 per gram all-in
        return mass * 1000 * 0.03, mass * 1000 * 0.30 + 2, "3D printing (home filament to print service)", "printed"
    vol_cm3 = mass / 7.85e-3 if mass else (dims[0] * dims[1] * dims[2]) / 1e3 * 0.3
    box = max(dims[0] * dims[1] * dims[2] / 1e3, 1.0)
    lo = 15 + 0.6 * box ** 0.75
    return lo, lo * 3.5 + 0.02 * vol_cm3, "CNC machining (one-off, online quote)", "machined"


# --------------------------------------------------------------------------- FX
_fx_cache = {"t": 0.0, "rates": None}


def fx_rates() -> dict:
    """Local currency per USD. Tries a free FX API once a day; falls back to the table above."""
    if _fx_cache["rates"] and time.time() - _fx_cache["t"] < 86400:
        return _fx_cache["rates"]
    rates = {r["currency"]: r["fx"] for r in REGIONS.values()}
    try:
        with urllib.request.urlopen("https://open.er-api.com/v6/latest/USD", timeout=4) as fh:
            live = json.load(fh).get("rates", {})
        for cur in list(rates):
            if cur in live:
                rates[cur] = float(live[cur])
        _fx_cache["live"] = True
    except Exception:
        _fx_cache["live"] = False
    _fx_cache.update(t=time.time(), rates=rates)
    return rates


# --------------------------------------------------------------------------- per line
def _query(line: dict) -> str:
    q = line.get("designation") or line.get("name") or line.get("category") or ""
    q = re.sub(r"\(.*?\)|\bv\d+\b", " ", q)
    return " ".join(q.split())[:80]


def shops_for(family: str, region: str) -> list[tuple[str, bool]]:
    """[(shop key, ships from abroad)], local shops first."""
    out = [(k, False) for k in LOCAL_ORDER.get(region, []) if family in SHOPS[k][3]]
    out += [(k, SHOPS[k][2] != region) for k in GLOBAL if family in SHOPS[k][3] and k not in dict(out)]
    return out


def source_line(line: dict, region_code: str, rates: dict) -> dict:
    region = REGIONS.get(region_code, REGIONS["OTHER"])
    fx = rates.get(region["currency"], region["fx"])
    family = FAMILY_OF_LABEL.get(line.get("label"), "structure")
    qty = int(line.get("qty") or 1)
    out = {"item": line.get("item"), "name": line.get("name"), "qty": qty, "family": family,
           "standard_part": bool(line.get("standard_part"))}
    if line.get("standard_part"):
        tp = typical_price(line)
        q = _query(line)
        shops = []
        for k, abroad in shops_for(family, region_code)[:5]:
            name, url, ships_from, _ = SHOPS[k]
            shops.append({"shop": name, "url": url.format(q=urllib.parse.quote_plus(q)), "ships_from": ships_from,
                          "import": abroad})
        out.update(query=q, shops=shops)
        if tp:
            lo, hi, basis = tp
            disc = 0.75 if qty >= 10 else 1.0           # pack / quantity pricing
            out.update(unit_low=round(lo * disc * fx, 2), unit_high=round(hi * fx, 2), basis=basis,
                       total_low=round(lo * disc * fx * qty, 2), total_high=round(hi * fx * qty, 2))
    else:
        lo, hi, basis, kind = custom_part_cost(line, region)
        out.update(unit_low=round(lo * fx, 2), unit_high=round(hi * fx, 2), basis=basis, make=kind,
                   total_low=round(lo * fx * qty, 2), total_high=round(hi * fx * qty, 2),
                   shops=[{"shop": n, "url": u, "ships_from": "global", "import": True} for n, u in SERVICES[kind]])
    return out


def shopping_plan(lines: list[dict], region_code: str) -> list[dict]:
    """Fewest shops that cover every standard part (greedy set cover, local shops preferred)."""
    need = {ln["item"]: ln["family"] for ln in lines if ln["standard_part"]}
    plan = []
    while need:
        best, cover = None, []
        for k in LOCAL_ORDER.get(region_code, []) + GLOBAL:
            c = [i for i, f in need.items() if f in SHOPS[k][3]]
            bonus = 0.5 if SHOPS[k][2] == region_code else 0.0
            if c and (best is None or len(c) + bonus > len(cover) + (0.5 if SHOPS[best][2] == region_code else 0)):
                best, cover = k, c
        if best is None:
            break
        plan.append({"shop": SHOPS[best][0], "ships_from": SHOPS[best][2], "items": sorted(cover),
                     "import": SHOPS[best][2] != region_code})
        for i in cover:
            need.pop(i)
    return plan


def collapse_bought(bom: list[dict]) -> list[dict]:
    """A bought component modelled with its insides (a servo with its PCB, motor and shells; an LED
    with its pads) is bought once, as a whole: replace its pieces by one line for it."""
    from .breakdown import FAMILY_TO_LABEL, display_name
    from .components import CUSTOM_HINT, from_name
    groups: dict = {}
    keep = []
    for ln in bom:
        segs = [s for s in (ln.get("path") or "").split(" / ") if s]
        parent = None
        for s in reversed(segs[1:]):                    # the root is the whole assembly
            n = display_name(s) or s
            c = from_name(n)
            if c and c.standard_part and c.family not in ("fastener", "structure") and not CUSTOM_HINT.search(n):
                parent = n
                break
        if parent:
            groups.setdefault(parent, []).append(ln)
        else:
            keep.append(ln)
    for parent, lines in groups.items():
        c = from_name(parent)
        dims = np.max([l.get("dims_mm") or [0, 0, 0] for l in lines], axis=0).tolist()
        keep.append({"item": min(l["item"] for l in lines), "name": parent, "qty": max(set(q := [l["qty"] for l in lines]), key=lambda v: (q.count(v), -v)),   # most common piece count
                     "label": FAMILY_TO_LABEL.get(c.family, "custom"), "category": c.category,
                     "designation": c.designation or parent, "standard_part": True, "size": c.size, "dims_mm": dims,
                     "material": "", "unit_mass_kg": None, "part_ids": [i for l in lines for i in l.get("part_ids", [])],
                     "includes": len(lines)})
    keep.sort(key=lambda l: l["item"])
    return keep


def source_assembly(asm, region_code: str = "IN") -> dict:
    from .breakdown import build_bom
    region_code = region_code if region_code in REGIONS else "OTHER"
    region = REGIONS[region_code]
    rates = fx_rates()
    fx = rates.get(region["currency"], region["fx"])
    bom = collapse_bought(build_bom(asm))
    lines = [source_line(ln, region_code, rates) for ln in bom]
    for ln, src in zip(lines, bom):
        ln["part_ids"] = src.get("part_ids", [])
        if src.get("includes"):
            ln["includes"] = src["includes"]
    std = [l for l in lines if l["standard_part"] and "total_low" in l]
    cus = [l for l in lines if not l["standard_part"]]
    plan = shopping_plan(lines, region_code)
    ship = sum(region["intl_ship_usd"] if p["import"] else region["local_ship_usd"] for p in plan) * fx
    imported = sum(l.get("total_high", 0) for l in std if l.get("shops") and l["shops"][0]["import"])
    return {
        "region": region_code, "region_name": region["name"], "currency": region["currency"], "symbol": region["symbol"],
        "fx_per_usd": round(fx, 4), "fx_live": bool(_fx_cache.get("live")),
        "lines": lines, "plan": plan,
        "totals": {"standard_low": round(sum(l["total_low"] for l in std), 2),
                   "standard_high": round(sum(l["total_high"] for l in std), 2),
                   "custom_low": round(sum(l["total_low"] for l in cus), 2),
                   "custom_high": round(sum(l["total_high"] for l in cus), 2),
                   "shipping_estimate": round(ship, 2),
                   "import_tax_estimate": round(imported * region["import_tax"], 2),
                   "priced_lines": len(std), "unpriced_lines": sum(1 for l in lines if l["standard_part"] and "total_low" not in l)},
        "live_prices": live_providers(),
        "note": ("Typical catalogue / marketplace price ranges, not quotes. Shipping and import tax are rough "
                 "estimates for small parcels. Use the shop links for real prices."),
    }


# --------------------------------------------------------------------------- live prices (optional API keys)
def live_providers() -> list[str]:
    return [n for n, env in (("Mouser", "MOUSER_API_KEY"),) if os.environ.get(env)]


def live_price(query: str, qty: int = 1) -> list[dict]:
    """Real price breaks and stock from distributor APIs whose keys are configured."""
    out = []
    key = os.environ.get("MOUSER_API_KEY")
    if key:
        body = json.dumps({"SearchByKeywordRequest": {"keyword": query, "records": 5, "startingRecord": 0}}).encode()
        req = urllib.request.Request(f"https://api.mouser.com/api/v1/search/keyword?apiKey={urllib.parse.quote(key)}",
                                     data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=8) as fh:
                data = json.load(fh)
            for p in (data.get("SearchResults") or {}).get("Parts", [])[:5]:
                breaks = p.get("PriceBreaks") or []
                best = None
                for b in breaks:
                    if int(b.get("Quantity", 1)) <= max(qty, 1):
                        best = b
                out.append({"source": "Mouser", "part": p.get("ManufacturerPartNumber"),
                            "maker": p.get("Manufacturer"), "stock": p.get("Availability"),
                            "price": (best or (breaks[0] if breaks else {})).get("Price"),
                            "currency": (best or (breaks[0] if breaks else {})).get("Currency"),
                            "url": p.get("ProductDetailUrl")})
        except Exception as exc:
            out.append({"source": "Mouser", "error": f"{exc.__class__.__name__}"})
    return out
