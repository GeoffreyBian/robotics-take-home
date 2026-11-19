import argparse
import csv
import glob
import os

from robotics_take_home.collision import METHODS
from robotics_take_home.pipeline import make_sim_ready
from robotics_take_home.priors import load_priors


def main():
    parser = argparse.ArgumentParser(description="Turn reconstructed meshes into MuJoCo assets.")
    parser.add_argument("assets", nargs="*", help="asset names, e.g. asset03 (default: all)")
    parser.add_argument("--models", default="data/models")
    parser.add_argument("--priors", default="data/priors.json")
    parser.add_argument("--out", default="output")
    parser.add_argument("--collision", choices=sorted(METHODS), default="coacd")
    parser.add_argument("--no-render", action="store_true", help="skip preview.png and drop.gif")
    parser.add_argument("--no-validate", action="store_true", help="skip the simulation checks")
    args = parser.parse_args()

    priors = load_priors(args.priors)
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
            os.path.join(args.out, name),
            collision_method=args.collision,
            render_outputs=not args.no_render,
            run_validation=not args.no_validate,
        )
        c = summary["collision"]
        print(
            f"{name}: x{summary['scale_factor']:.3g} scale, {summary['mass_kg']:.3g} kg, "
            f"{c['parts']} parts (vol ratio {c['volume_ratio']:.2f}, {c['seconds']}s)"
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
    "asset", "collision", "coverage", "rest_tilt_deg", "rest_penetration",
    "drop_energy_gain", "drop_impact_penetration", "drop_settled", "probe_max_gap", "passed",
]


def validation_row(name: str, method: str, v: dict) -> dict:
    return {
        "asset": name,
        "collision": method,
        "coverage": round(v["coverage"], 3),
        "rest_tilt_deg": round(v["rest"]["tilt_deg"], 2),
        "rest_penetration": round(v["rest"]["penetration"], 4),
        "drop_energy_gain": round(v["drop"]["energy_gain"], 3),
        "drop_impact_penetration": round(v["drop"]["impact_penetration"], 4),
        "drop_settled": v["drop"]["final_speed"] < 0.05 and v["drop"]["final_spin"] < 0.5,
        "probe_max_gap": round(v["probe"]["max_abs_gap"], 3),
        "passed": v["passed"],
    }


if __name__ == "__main__":
    main()
