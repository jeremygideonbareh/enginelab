"""Engineering materials database.

Values are typical room-temperature handbook figures (ASM / MatWeb / supplier
datasheets) for the stated condition. They are good for comparison and early
design; check the supplier's certified data before final sign-off.

Units: density kg/m^3, E GPa, strengths MPa, conductivity W/m-K,
CTE 1e-6/K, temperatures degC, hardness HV, cost relative to plain carbon
steel per kg.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    category: str
    density: float
    E: float
    poisson: float
    yield_strength: float
    uts: float
    endurance: float            # fully reversed, polished specimen, ~1e7 cycles
    conductivity: float
    cte: float
    max_temp: float             # max continuous service temperature
    soft_temp: float            # temperature where strength starts to fall noticeably
    hardness_hv: float
    archard_k: float            # dimensionless wear coefficient, lubricated sliding
    cost: float
    brittle: bool = False
    bearing_pressure: float = 0.0   # allowable specific bearing load (MPa), bearing alloys only
    notes: str = ""

    def strength_factor(self, temp_c: float) -> float:
        """Fraction of room-temperature strength kept at `temp_c`."""
        if temp_c <= self.soft_temp:
            return 1.0
        span = max(self.max_temp * 1.25 - self.soft_temp, 1.0)
        return max(0.15, 1.0 - 0.75 * (temp_c - self.soft_temp) / span)

    def to_dict(self) -> dict:
        return asdict(self)


_M = [
    # --- carbon & alloy steels
    Material("steel_1018", "AISI 1018 low-carbon steel", "steel", 7870, 205, 0.29, 370, 440, 220, 51.9, 12.0, 400, 300, 130, 2e-8, 0.8,
             notes="Cheap, weldable, formable. Stamped covers, pans, brackets."),
    Material("steel_1045", "AISI 1045 medium-carbon steel (normalised)", "steel", 7850, 205, 0.29, 450, 655, 300, 49.8, 11.5, 400, 300, 200, 1e-8, 1.0,
             notes="General shafts; can be induction hardened."),
    Material("steel_38mnvs6", "38MnVS6 micro-alloyed forging steel", "steel", 7850, 210, 0.29, 600, 900, 420, 42.0, 12.0, 400, 300, 270, 8e-9, 1.2,
             notes="Air-cooled forging steel for crankshafts and conrods; no heat treatment needed."),
    Material("steel_4140", "AISI 4140 Cr-Mo steel (Q&T)", "steel", 7850, 205, 0.29, 850, 1000, 480, 42.6, 12.3, 450, 350, 310, 6e-9, 1.5,
             notes="Tough, strong; crankshafts, conrods, gears, bolts."),
    Material("steel_4340", "AISI 4340 Ni-Cr-Mo steel (Q&T)", "steel", 7850, 205, 0.29, 1080, 1250, 600, 44.5, 12.3, 450, 350, 380, 5e-9, 2.2,
             notes="High-performance crankshafts and conrods."),
    Material("steel_31crmov9n", "31CrMoV9 nitrided steel", "steel", 7850, 210, 0.29, 900, 1100, 560, 40.0, 12.0, 450, 380, 750, 2e-9, 2.8,
             notes="Nitrided surface (~750 HV) for racing crankshafts and cams; high fatigue strength."),
    Material("steel_8620c", "AISI 8620 carburised", "steel", 7850, 205, 0.29, 600, 900, 450, 46.6, 12.0, 400, 300, 700, 2e-9, 1.4,
             notes="Hard case, tough core: gears, pins, cams."),
    Material("steel_52100", "AISI 52100 bearing steel (hardened)", "steel", 7810, 210, 0.30, 1900, 2200, 750, 46.6, 11.9, 175, 150, 750, 1e-9, 2.0,
             notes="Rolling bearings, piston pins."),
    Material("steel_crsi", "Cr-Si spring steel (SAE 9254)", "steel", 7850, 207, 0.29, 1650, 1900, 650, 40.0, 12.0, 250, 200, 560, 5e-9, 1.8,
             notes="Valve springs and piston rings; shot-peened for fatigue."),
    Material("steel_bolt88", "Bolt steel ISO 8.8", "steel", 7850, 205, 0.29, 640, 800, 380, 45.0, 12.0, 300, 250, 250, 1e-8, 1.2,
             notes="Standard high-strength fasteners."),
    Material("steel_bolt109", "Bolt steel ISO 10.9 / 12.9", "steel", 7850, 205, 0.29, 940, 1040, 450, 42.0, 12.0, 300, 250, 330, 8e-9, 1.6,
             notes="Head bolts, conrod bolts, flywheel bolts."),
    Material("steel_pf", "Powder-forged steel (HS150)", "steel", 7800, 200, 0.28, 700, 980, 380, 38.0, 12.0, 400, 300, 290, 1e-8, 1.1,
             notes="Mass-produced conrods with fracture-split caps."),
    # --- stainless & valve steels, superalloys
    Material("ss_304", "Stainless 304", "stainless", 8000, 193, 0.29, 215, 505, 240, 16.2, 17.3, 870, 500, 160, 3e-8, 3.0,
             notes="Corrosion resistant; exhaust parts, fittings."),
    Material("ss_301fh", "Stainless 301 full-hard (MLS gasket)", "stainless", 7880, 193, 0.28, 965, 1275, 450, 16.2, 16.9, 800, 450, 400, 2e-8, 3.5,
             notes="Multi-layer steel head gaskets with embossed beads."),
    Material("ss_174ph", "17-4PH stainless (H900)", "stainless", 7810, 197, 0.27, 1170, 1310, 620, 17.9, 10.8, 480, 350, 420, 5e-9, 4.5,
             notes="Strong + corrosion resistant; pump shafts, valves."),
    Material("valve_21_4n", "21-4N austenitic valve steel", "valve steel", 7750, 200, 0.29, 650, 1000, 450, 14.0, 16.0, 815, 600, 330, 5e-9, 4.0,
             notes="Standard exhaust valve head material."),
    Material("inconel_751", "Inconel 751", "superalloy", 8220, 214, 0.29, 780, 1200, 500, 12.0, 13.0, 870, 700, 360, 4e-9, 25.0,
             notes="Heavy-duty / turbo exhaust valves."),
    Material("ti_6al4v", "Ti-6Al-4V", "titanium", 4430, 114, 0.34, 880, 950, 510, 6.7, 8.6, 400, 300, 340, 4e-8, 20.0,
             notes="Racing conrods and valves: high strength-to-weight, poor wear without coating."),
    # --- cast irons
    Material("ci_gray", "Grey cast iron (Class 30 / G3000)", "cast iron", 7200, 100, 0.26, 190, 214, 97, 52.0, 11.0, 450, 350, 210, 3e-8, 0.6, brittle=True,
             notes="Excellent damping and machinability; classic blocks, liners, flywheels."),
    Material("ci_cgi", "Compacted graphite iron (GJV-450)", "cast iron", 7100, 145, 0.26, 315, 450, 200, 37.0, 11.0, 500, 400, 230, 2.5e-8, 0.9,
             notes="Diesel blocks and heads: ~75% stronger than grey iron."),
    Material("ci_ductile", "Ductile iron (80-55-06)", "cast iron", 7100, 169, 0.27, 379, 552, 250, 32.0, 11.0, 450, 350, 250, 2e-8, 0.8,
             notes="Cast crankshafts and camshafts in cost-sensitive engines."),
    Material("ci_chilled", "Chilled cast iron (camshaft grade)", "cast iron", 7300, 150, 0.26, 270, 300, 130, 40.0, 11.0, 400, 350, 550, 4e-9, 0.9, brittle=True,
             notes="Very hard chilled lobe surfaces for flat-tappet camshafts."),
    # --- aluminium
    Material("al_a356", "A356-T6 cast aluminium", "aluminium", 2685, 72.4, 0.33, 205, 262, 90, 151, 21.5, 250, 170, 90, 5e-7, 2.5,
             notes="Heads and blocks: light, conducts heat well."),
    Material("al_a319", "A319-T6 cast aluminium", "aluminium", 2790, 74.0, 0.33, 165, 250, 76, 109, 21.5, 250, 170, 95, 5e-7, 2.2,
             notes="Common block and head alloy."),
    Material("al_a380", "A380 die-cast aluminium", "aluminium", 2710, 71.0, 0.33, 160, 324, 138, 96, 21.8, 175, 130, 80, 6e-7, 2.0,
             notes="Thin-wall die castings: covers, oil pans, housings."),
    Material("al_4032", "4032-T6 forged piston alloy", "aluminium", 2680, 79.0, 0.33, 315, 380, 110, 155, 19.4, 300, 200, 120, 2e-7, 3.0,
             notes="High-silicon forged pistons: low expansion, quiet."),
    Material("al_2618", "2618-T61 forged aluminium", "aluminium", 2760, 74.5, 0.33, 370, 440, 125, 146, 22.3, 300, 210, 130, 3e-7, 3.5,
             notes="Racing pistons: tougher at high temperature."),
    Material("al_6061", "6061-T6 aluminium", "aluminium", 2700, 68.9, 0.33, 276, 310, 96, 167, 23.6, 170, 120, 107, 6e-7, 2.0,
             notes="General machined parts and brackets."),
    Material("al_7075", "7075-T6 aluminium", "aluminium", 2810, 71.7, 0.33, 503, 572, 159, 130, 23.6, 120, 100, 175, 5e-7, 3.5,
             notes="High-strength aluminium for drag-race conrods (limited fatigue life)."),
    # --- magnesium, polymers, bearing alloys, ceramics
    Material("mg_az91", "AZ91D magnesium", "magnesium", 1810, 45, 0.35, 160, 230, 97, 72, 26.0, 120, 100, 63, 1e-6, 3.0,
             notes="Ultra-light covers and housings."),
    Material("pa66gf30", "PA66-GF30 (glass-filled nylon)", "polymer", 1370, 9.5, 0.35, 120, 160, 50, 0.3, 25.0, 150, 120, 30, 5e-6, 3.5,
             notes="Moulded valve covers and intake manifolds."),
    Material("brg_alsn", "Steel-backed AlSn20 bearing", "bearing alloy", 3100, 70, 0.33, 120, 180, 60, 120, 20.0, 160, 140, 45, 3e-8, 4.0, bearing_pressure=45,
             notes="Bi-metal plain bearings for passenger-car mains."),
    Material("brg_cupb", "Tri-metal CuPb24Sn bearing", "bearing alloy", 8900, 100, 0.34, 140, 230, 90, 60, 18.0, 180, 150, 70, 2e-8, 5.0, bearing_pressure=70,
             notes="Heavy-duty and diesel plain bearings."),
    Material("bronze_c954", "Aluminium bronze C95400", "copper alloy", 7450, 110, 0.32, 241, 586, 200, 59, 16.2, 350, 250, 170, 2e-8, 6.0, bearing_pressure=40,
             notes="Bushings, valve guides, worm gears."),
    Material("brass_c360", "Free-machining brass C360", "copper alloy", 8500, 97, 0.31, 310, 385, 140, 115, 20.5, 200, 150, 120, 3e-8, 4.0,
             notes="Heat-set inserts, fittings, bushings; machines very well."),
    # --- 3D-printing and engineering plastics (typical as-printed / moulded values)
    Material("pla", "PLA (3D printed)", "polymer", 1240, 3.5, 0.36, 50, 55, 15, 0.13, 68.0, 50, 45, 15, 1e-5, 2.0,
             notes="Easy to print and stiff, but softens above ~55 C and creeps under load."),
    Material("petg", "PETG (3D printed)", "polymer", 1270, 2.1, 0.38, 45, 50, 14, 0.2, 60.0, 70, 60, 12, 1e-5, 2.0,
             notes="Tougher than PLA, good layer adhesion, ~70 C limit."),
    Material("asa", "ASA / ABS (3D printed)", "polymer", 1070, 2.0, 0.35, 40, 45, 12, 0.17, 90.0, 90, 80, 10, 1e-5, 2.5,
             notes="Common for printer and robot parts near motors or heat; UV stable (ASA)."),
    Material("pa12", "PA12 nylon (SLS / MJF)", "polymer", 1010, 1.8, 0.39, 48, 50, 16, 0.25, 110.0, 110, 90, 10, 5e-6, 6.0,
             notes="Strong, tough printed nylon for functional robot parts and snap fits."),
    Material("pa_cf", "PA-CF carbon-fibre nylon (3D printed)", "polymer", 1150, 7.0, 0.35, 85, 95, 28, 0.4, 25.0, 120, 100, 14, 5e-6, 8.0,
             notes="Stiff, light and heat resistant; popular for robot arms and brackets."),
    Material("pc", "Polycarbonate", "polymer", 1200, 2.4, 0.37, 62, 66, 20, 0.2, 68.0, 115, 100, 15, 1e-5, 4.0,
             notes="Very tough; transparent grades for covers and guards."),
    Material("si3n4", "Silicon nitride ceramic", "ceramic", 3200, 310, 0.27, 800, 800, 500, 30, 3.2, 1200, 1000, 1500, 1e-10, 50.0, brittle=True,
             notes="Hybrid bearing balls, experimental valves; brittle."),
    Material("alumina", "Alumina 96% ceramic", "ceramic", 3720, 300, 0.22, 330, 330, 200, 25, 8.2, 1500, 1200, 1400, 1e-10, 8.0, brittle=True,
             notes="Spark plug insulators."),
]

MATERIALS: dict[str, Material] = {m.key: m for m in _M}


def get(key: str) -> Material:
    if key not in MATERIALS:
        raise KeyError(f"Unknown material '{key}'")
    return MATERIALS[key]


def all_materials() -> list[dict]:
    return [m.to_dict() for m in _M]
