import argparse
import csv
import glob
import os

from robotics_take_home.collision import DEFAULT_BUDGET, METHODS
from robotics_take_home.materials import MaterialTable
from robotics_take_home.pipeline import make_sim_ready
from robotics_take_home.priors import load_priors


def main():
    parser = argparse.ArgumentParser(description="Turn reconstructed meshes into MuJoCo assets.")
    parser.add_argument("assets", nargs="*", help="asset names, e.g. asset03 (default: all)")
    parser.add_argument("--models", default="data/models")
    parser.add_argument("--priors", default="data/priors.json")
    parser.add_argument("--materials", default="data/materials.json")
    parser.add_argument("--out", default="output")
    parser.add_argument("--collision", choices=sorted(METHODS), default="coacd")
    parser.add_argument("--hull-budget", type=int, default=DEFAULT_BUDGET, help="max convex parts per asset")
    parser.add_argument("--no-render", action="store_true", help="skip preview.png and drop.gif")
    parser.add_argument("--no-validate", action="store_true", help="skip the simulation checks")
    args = parser.parse_args()

    priors = load_priors(args.priors)
    materials = MaterialTable(args.materials)
    paths = sorted(glob.glob(os.path.join(args.models, "*.glb")))
    if args.assets:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in args.assets]

    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in priors:
            print(f"{name}: no prior in {args.priors}, skipping")
            continue
        summary = make_sim_ready(
            path,
            priors[name],
            materials,
            os.path.join(args.out, name),
            collision_method=args.collision,
            hull_budget=args.hull_budget,
            render_outputs=not args.no_render,
            run_validation=not args.no_validate,
        )
        c = summary["collision"]
        print(
            f"{name}: x{summary['scale_factor']:.3g} scale, {summary['mass']['total_mass_kg']:.3g} kg, "
            f"{c['parts']} collision parts ({c['seconds']}s)"
            + (f", repairs: {summary['repairs']}" if summary["repairs"] else "")
        )
        if "validation" in summary:
            v = summary["validation"]
            rows.append(validation_row(name, args.collision, v))
            print("  " + ", ".join(f"{k}={rows[-1][k]}" for k in VALIDATION_COLUMNS[2:]))

    if rows:
        path = os.path.join(args.out, "validation.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=VALIDATION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"{sum(r['passed'] for r in rows)}/{len(rows)} assets passed; table in {path}")


VALIDATION_COLUMNS = [
    "asset", "collision", "coverage", "rest", "drop_impact", "drop", "probe_gap", "probe",
    "ray_agree", "rays", "slide", "grasp", "export", "passed",
]


def validation_row(name: str, method: str, v: dict) -> dict:
    slide, grasp = v["slide"], v["grasp"]
    return {
        "asset": name,
        "collision": method,
        "coverage": round(v["coverage"], 3),
        "rest": v["rest"]["passed"],
        "drop_impact": round(v["drop"]["impact_penetration"], 4),
        "drop": v["drop"]["passed"],
        "probe_gap": round(v["probe"]["max_abs_gap"], 3),
        "probe": v["probe"]["passed"],
        "ray_agree": v["rays"]["agree"],
        "rays": v["rays"]["passed"],
        "slide": (f"mu={slide['mu']}: {slide['below']['outcome']}/{slide['above']['outcome']}"
                  if "mu" in slide else slide.get("reason")) + ("" if slide["passed"] else " FAIL"),
        "grasp": f"{grasp['strategy']}: {grasp.get('outcome', grasp.get('reason'))}",
        "export": v["export"]["passed"],
        "passed": v["passed"],
    }


if __name__ == "__main__":
    main()
