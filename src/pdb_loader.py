"""
PDB structure loader.

Downloads a PDB file (default: 2HHB, human deoxyhemoglobin) from the RCSB
Protein Data Bank if not already cached, then extracts:

- iron (Fe) atom positions  -> these are the paramagnetic dipole centers
- HEM (heme) atom positions -> for visualization of the four heme groups
- C-alpha backbone positions per chain -> for protein backbone ribbon

Coordinates are reported in Angstroms in the PDB's native frame.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DEFAULT_PDB_ID = "2HHB"  # Human deoxyhemoglobin, 4 paramagnetic Fe(II) centers.

# Residue names for "iron cofactor" groups we want to highlight as
# semi-transparent spheres. Covers all PDB heme variants plus the most
# common iron-sulfur cluster residue names.
IRON_COFACTOR_RESNAMES = frozenset({
    "HEM",   # heme B  - hemoglobin, myoglobin, P450, catalase
    "HEC",   # heme C  - cytochrome c (covalently attached)
    "HEA",   # heme A  - cytochrome c oxidase
    "HEB",   # heme B variant
    "HEO",   # heme O
    "HAS",   # heme A_S
    "FES",   # [2Fe-2S] cluster
    "SF4",   # [4Fe-4S] cluster
    "F3S",   # [3Fe-4S] cluster
    "FE",    # bare iron ion residue (some structures)
    "FEO",   # iron-oxo cluster
})


@dataclass(frozen=True)
class ProteinEntry:
    """One curated entry in the dropdown library of Fe-containing proteins."""
    pdb_id: str
    short_name: str   # display label in the dropdown
    blurb: str        # one-line biological role


# Curated library of biologically iconic Fe-containing proteins. Each entry
# was chosen for pedagogical value (different Fe coordination chemistries,
# different roles) and verified to download cleanly from RCSB with at least
# one Fe atom in the parsed structure.
PROTEIN_LIBRARY = (
    ProteinEntry(
        "2HHB", "Hemoglobin (deoxy)",
        "Human deoxyhemoglobin - oxygen transport in blood. 4 Fe(II) hemes "
        "in the alpha2-beta2 tetramer; high-spin paramagnetic - basis of "
        "BOLD fMRI contrast.",
    ),
    ProteinEntry(
        "1MBN", "Myoglobin",
        "Sperm-whale myoglobin - oxygen storage in muscle. 1 Fe heme. "
        "First protein ever solved by X-ray crystallography (Kendrew, 1958).",
    ),
    ProteinEntry(
        "1HRC", "Cytochrome c",
        "Horse-heart cytochrome c - mobile electron carrier in the "
        "mitochondrial respiratory chain. 1 low-spin Fe heme.",
    ),
    ProteinEntry(
        "2CPP", "Cytochrome P450cam",
        "P. putida cytochrome P450cam with camphor bound - archetype of "
        "the P450 superfamily that runs most drug metabolism. 1 Fe heme.",
    ),
    ProteinEntry(
        "1A3N", "Hemoglobin (R-state)",
        "Alternative crystal form of human hemoglobin - useful for "
        "comparing the T (tense) and R (relaxed) allosteric states.",
    ),
    ProteinEntry(
        "5RXN", "Rubredoxin",
        "Clostridium pasteurianum rubredoxin - the smallest known iron-"
        "sulfur protein (~6 kDa). 1 high-spin Fe(III) in a [Fe-Cys4] site.",
    ),
    ProteinEntry(
        "3FXC", "Ferredoxin [2Fe-2S]",
        "Spirulina platensis ferredoxin - photosynthetic electron carrier. "
        "1 [2Fe-2S] cluster (two coupled iron atoms).",
    ),
    ProteinEntry(
        "1A8E", "Transferrin (N-lobe)",
        "Human serum transferrin N-lobe - iron transport in blood. "
        "Binds Fe(III) tightly at one site per lobe.",
    ),
)


def find_protein_entry(pdb_id: str) -> "ProteinEntry | None":
    pid = pdb_id.upper()
    for e in PROTEIN_LIBRARY:
        if e.pdb_id.upper() == pid:
            return e
    return None


@dataclass
class ProteinStructure:
    """Parsed subset of a PDB file relevant to the magnetic-field simulation."""

    pdb_id: str
    iron_positions: np.ndarray              # (N_fe, 3) float, Angstroms
    heme_positions: np.ndarray              # (N_hem_atoms, 3) float, Angstroms
    backbone_by_chain: Dict[str, np.ndarray] = field(default_factory=dict)
    title: str = ""

    @property
    def center(self) -> np.ndarray:
        if self.iron_positions.size:
            return self.iron_positions.mean(axis=0)
        if self.heme_positions.size:
            return self.heme_positions.mean(axis=0)
        all_bb = np.concatenate(list(self.backbone_by_chain.values()), axis=0)
        return all_bb.mean(axis=0)

    @property
    def extent(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bounding box (min, max) over all loaded atoms."""
        chunks: List[np.ndarray] = []
        if self.iron_positions.size:
            chunks.append(self.iron_positions)
        if self.heme_positions.size:
            chunks.append(self.heme_positions)
        for arr in self.backbone_by_chain.values():
            if arr.size:
                chunks.append(arr)
        all_pts = np.concatenate(chunks, axis=0)
        return all_pts.min(axis=0), all_pts.max(axis=0)


def _cache_path(pdb_id: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{pdb_id.upper()}.pdb")


def download_pdb(pdb_id: str = DEFAULT_PDB_ID, cache_dir: str = "data") -> str:
    """Return path to a locally cached PDB file, downloading if necessary."""
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(pdb_id, cache_dir)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    print(f"[pdb_loader] Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    with open(path, "wb") as fh:
        fh.write(data)
    print(f"[pdb_loader] Saved to {path} ({len(data)/1024:.1f} kB)")
    return path


def _parse_pdb_atoms(path: str) -> Tuple[List[dict], str]:
    """
    Minimal PDB ATOM/HETATM parser. Returns a list of atom dicts plus the TITLE.

    Each atom dict has keys:
        record   : 'ATOM' or 'HETATM'
        name     : atom name (e.g. 'CA', 'FE')
        resname  : residue name (e.g. 'HEM', 'HIS')
        chain    : chain id
        resseq   : residue sequence number (int)
        element  : element symbol (e.g. 'C', 'FE')
        xyz      : np.ndarray, shape (3,), Angstroms
    """
    atoms: List[dict] = []
    title_parts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "TITLE":
                title_parts.append(line[10:80].strip())
                continue
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21:22].strip() or " "
                resseq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip()
                if not element:
                    # Fall back to atom name (works for FE, CA, etc.).
                    element = name.lstrip("0123456789")[:2].strip().upper()
            except (ValueError, IndexError):
                continue
            atoms.append({
                "record": rec,
                "name": name,
                "resname": resname,
                "chain": chain,
                "resseq": resseq,
                "element": element.upper(),
                "xyz": np.array([x, y, z], dtype=float),
            })
    return atoms, " ".join(title_parts).strip()


def load_structure(pdb_id: str = DEFAULT_PDB_ID, cache_dir: str = "data") -> ProteinStructure:
    """Download (if needed) and parse the named PDB into a ProteinStructure."""
    path = download_pdb(pdb_id, cache_dir=cache_dir)
    atoms, title = _parse_pdb_atoms(path)

    iron_xyz: List[np.ndarray] = []
    heme_xyz: List[np.ndarray] = []
    bb: Dict[str, List[Tuple[int, np.ndarray]]] = {}

    for a in atoms:
        if a["element"] == "FE":
            iron_xyz.append(a["xyz"])
        if a["resname"] in IRON_COFACTOR_RESNAMES:
            heme_xyz.append(a["xyz"])
        if a["record"] == "ATOM" and a["name"] == "CA":
            bb.setdefault(a["chain"], []).append((a["resseq"], a["xyz"]))

    backbone: Dict[str, np.ndarray] = {}
    for chain, items in bb.items():
        items.sort(key=lambda kv: kv[0])
        backbone[chain] = np.stack([xyz for _, xyz in items], axis=0)

    return ProteinStructure(
        pdb_id=pdb_id.upper(),
        iron_positions=np.array(iron_xyz, dtype=float).reshape(-1, 3),
        heme_positions=np.array(heme_xyz, dtype=float).reshape(-1, 3),
        backbone_by_chain=backbone,
        title=title,
    )


if __name__ == "__main__":
    s = load_structure()
    print(f"PDB:    {s.pdb_id}")
    print(f"Title:  {s.title[:80]}")
    print(f"Irons:  {s.iron_positions.shape[0]} atoms")
    print(f"Heme:   {s.heme_positions.shape[0]} atoms")
    print(f"Chains: {sorted(s.backbone_by_chain)}")
    for c, arr in s.backbone_by_chain.items():
        print(f"  chain {c}: {arr.shape[0]} C-alphas")
    lo, hi = s.extent
    print(f"Extent (A): {lo} ... {hi}")
    print(f"Center (A): {s.center}")
