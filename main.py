import argparse
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
    args = parser.parse_args()

    priors = load_priors(args.priors)
    paths = sorted(glob.glob(os.path.join(args.models, "*.glb")))
    if args.assets:
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in args.assets]

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
        )
        c = summary["collision"]
        print(
            f"{name}: x{summary['scale_factor']:.3g} scale, {summary['mass_kg']:.3g} kg, "
            f"{c['parts']} parts (vol ratio {c['volume_ratio']:.2f}, {c['seconds']}s)"
            + (f", repairs: {summary['repairs']}" if summary["repairs"] else "")
        )


if __name__ == "__main__":
    main()
