"""
Entry point: load hemoglobin, simulate the dipolar magnetic field, launch GUI.

Usage
-----
    python run.py                  # default: 2HHB (deoxyhemoglobin)
    python run.py --pdb 1A3N       # alternative human Hb structure
    python run.py --grid-n 80      # finer field grid (slower)
    python run.py --extent 40      # larger sampling volume (Angstroms)
"""

from __future__ import annotations

import argparse
import sys

from src.pdb_loader import DEFAULT_PDB_ID, load_structure
from src.visualizer import ProteinFieldGUI


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Simulate and visualize the magnetic field around hemoglobin's "
            "paramagnetic Fe centers (the physics behind BOLD fMRI contrast)."
        )
    )
    p.add_argument("--pdb", default=DEFAULT_PDB_ID,
                   help="4-letter RCSB PDB id (default: %(default)s)")
    p.add_argument("--grid-n", type=int, default=60,
                   help="Samples per axis on the 3D field grid (default: 60)")
    p.add_argument("--extent", type=float, default=None,
                   help="Half-extent of the field grid in Angstroms. "
                        "If omitted, the grid sizes itself to fit each "
                        "selected protein.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[run] Loading PDB {args.pdb} ...")
    structure = load_structure(args.pdb)
    print(f"[run] Loaded {structure.pdb_id}: "
          f"{structure.iron_positions.shape[0]} Fe atoms, "
          f"{sum(a.shape[0] for a in structure.backbone_by_chain.values())} C-alphas")
    if structure.iron_positions.shape[0] == 0:
        print("[run] No iron atoms found in this PDB - aborting.", file=sys.stderr)
        return 1

    gui = ProteinFieldGUI(
        structure,
        grid_n=args.grid_n,
        grid_half_extent=args.extent,
    )
    print("[run] Launching interactive viewer. Close the window to exit.")
    gui.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
