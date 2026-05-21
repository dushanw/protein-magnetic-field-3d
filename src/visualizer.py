"""
Interactive 3D visualizer for the magnetic field around iron-containing
biological proteins (default: hemoglobin).

Uses PyVista's native widgets (no Qt dependency) so this runs anywhere
PyVista runs. The GUI offers:

    * 1 dropdown  : pick from a curated library of Fe proteins
                    (hemoglobin, myoglobin, cytochrome c, P450,
                    rubredoxin, ferredoxin, transferrin, ...).
    * 7 sliders   : log10(isosurface threshold in nT), B0 direction (X/Y/Z),
                    paramagnetic moment scale (1.0 = deoxy, 0.0 = oxy),
                    streamline density, and three slice-plane offsets
                    (one per X / Y / Z axis - scan a contour cross-section
                    through the field volume).
    * 6 buttons   : toggle isosurface / vectors / streamlines / X-slice /
                    Y-slice / Z-slice.
    * Always on   : protein backbone ribbon (per chain),
                    iron cofactors (heme / [Fe-S] cluster, semi-transparent),
                    Fe atoms (orange spheres) + dipole arrows.

Each slice is drawn as a colored cross-section plus white contour-lines of
log10|dB|, so it is a true contour plot of the magnetic-field magnitude.

Coordinates are in Angstroms; field is in Tesla but displayed in nano-Tesla
because protein-scale dipolar fields fall in the 1 - 10^5 nT range over
the visualization volume.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pyvista as pv

from .magnetic_field import (
    DEOXY_FE_MOMENT_BOHR,
    FieldResult,
    dipole_field,
    induced_moments,
    make_grid,
)
from .pdb_loader import (
    PROTEIN_LIBRARY,
    ProteinEntry,
    ProteinStructure,
    find_protein_entry,
    load_structure,
)

# Color palette - colorblind-friendly, dark-background friendly.
CHAIN_COLORS = {
    "A": "#4C72B0",  # alpha-1
    "B": "#DD8452",  # beta-1
    "C": "#55A467",  # alpha-2
    "D": "#C44E52",  # beta-2
}
FE_COLOR = "#FF8C00"     # orange
HEM_COLOR = "#8E44AD"    # violet
DIPOLE_ARROW_COLOR = "#FFD400"

# Per-layer colors used to color-code every label, slider title, and toggle
# button. The same color is shared between a layer's visualization actor
# and its UI controls, so the GUI is self-documenting.
ISO_COLOR        = "#00E5FF"   # cyan
VECTORS_COLOR    = "#FF8C00"   # orange
STREAM_COLOR     = "#55E07F"   # mint green
SLICE_X_COLOR    = "#FF5C5C"   # bright red
SLICE_Y_COLOR    = "#7CFC8A"   # bright lime
SLICE_Z_COLOR    = "#5DADE2"   # bright sky blue
B0_TITLE_COLOR   = "#FFD400"   # gold - matches the dipole arrows
MOMENT_COLOR     = "#FFA940"   # warm amber
INFO_TEXT_COLOR  = "#F0F0F0"   # near-white for static info labels


def _compute_adaptive_extent(structure: ProteinStructure, padding: float = 10.0,
                             lo: float = 20.0, hi: float = 60.0) -> float:
    """Pick a sensible field-grid half-extent for the given protein.

    Half-extent = half of the bounding-box diagonal + `padding` Angstroms,
    clamped to [`lo`, `hi`] so the grid stays manageable for both very
    small (rubredoxin, ~37 A) and very large (P450, ~94 A) proteins.
    """
    bb_lo, bb_hi = structure.extent
    diag = float(np.linalg.norm(bb_hi - bb_lo))
    extent = 0.5 * diag + padding
    return float(max(lo, min(hi, extent)))


class ProteinFieldGUI:
    """Stateful interactive plotter for the dipole-field visualization.

    Supports live switching between any of the proteins in PROTEIN_LIBRARY
    via a dropdown widget at the top of the window. On switch we
    transparently:

      * download / parse the new PDB,
      * recompute an adaptive field-grid extent and rebuild the grid,
      * tear down all protein-specific actors (per-chain backbones,
        cofactors, Fe spheres, dipole arrows) and re-add them for the
        new structure,
      * recompute the dipole field,
      * update the slice-offset slider ranges to match the new extent,
      * and reset the camera so the new protein fits in view.
    """

    def __init__(
        self,
        structure: ProteinStructure,
        grid_n: int = 60,
        grid_half_extent: Optional[float] = None,
    ):
        self.structure = structure
        self.grid_n = grid_n
        # If no explicit half-extent is given, pick one adaptively for
        # the initial structure (and we'll re-adapt on every switch).
        self.grid_half_extent = (
            float(grid_half_extent)
            if grid_half_extent is not None
            else _compute_adaptive_extent(structure)
        )

        # Mutable simulation state (driven by the sliders/buttons).
        # Default iso threshold of 10^5.5 nT (~316 uT) gives a beautiful
        # bilobed surface around the heme groups for the deoxy state at 30 A.
        # Slice offsets are measured in Angstroms from the protein center,
        # ranging over [-grid_half_extent, +grid_half_extent].
        self.state = {
            "b0_axis": 2,                  # 0=x, 1=y, 2=z
            "moment_scale": 1.0,           # 1.0 = deoxy, 0.0 = oxy
            "iso_log10_nT": 5.5,           # log10 of isosurface threshold in nT
            "stream_density": 8,           # number of seed points per side
            "n_contour_levels": 12,        # contour lines drawn on each slice
            "show_iso": True,
            "show_vectors": False,
            "show_streamlines": False,
            "show_slice_x": False,
            "show_slice_y": False,
            "show_slice_z": True,
            "slice_x_offset": 0.0,
            "slice_y_offset": 0.0,
            "slice_z_offset": 0.0,
        }

        # Pre-build the spatial grid once (independent of moment direction).
        gx, gy, gz = make_grid(
            structure.center, half_extent=self.grid_half_extent, n=grid_n
        )
        self.grid_x, self.grid_y, self.grid_z = gx, gy, gz

        self.field: Optional[FieldResult] = None
        self.image_data: Optional[pv.ImageData] = None

        self.plotter = pv.Plotter(window_size=(1280, 860))
        self.plotter.set_background("black", top="#15233a")

        # Actor handles so we can add/remove them on toggles & resimulations.
        self._actors: Dict[str, object] = {}
        # Names of every actor added by `_add_protein_actors`, so we can
        # cleanly tear them down on a protein switch.
        self._protein_actor_names: List[str] = []
        # Slice-offset slider widget refs, keyed by axis name. Used to
        # update slider ranges when the field grid changes size.
        self._slice_sliders: Dict[str, object] = {}

        self._add_protein_actors()
        self._recompute_field_and_redraw()
        self._add_widgets()
        self._add_title_and_legend()

    # ------------------------------------------------------------------ setup

    # Default colors for chains beyond A-D (we sometimes load proteins
    # with chain IDs L/H/E etc.). Cycled through in name order.
    _EXTRA_CHAIN_COLORS = (
        "#9B59B6", "#1ABC9C", "#F39C12", "#34495E",
        "#E67E22", "#16A085", "#7F8C8D", "#D35400",
    )

    def _chain_color(self, chain: str, idx: int) -> str:
        return CHAIN_COLORS.get(chain) or self._EXTRA_CHAIN_COLORS[
            idx % len(self._EXTRA_CHAIN_COLORS)
        ]

    def _add_protein_actors(self) -> None:
        """Add the static protein geometry (backbone + cofactors + Fe).

        Every actor name is also recorded in `self._protein_actor_names`
        so it can be torn down cleanly when the user switches proteins.
        """
        self._protein_actor_names = []

        for idx, (chain, ca) in enumerate(sorted(self.structure.backbone_by_chain.items())):
            if ca.shape[0] < 2:
                continue
            spline = pv.Spline(ca, n_points=max(ca.shape[0] * 4, 64))
            tube = spline.tube(radius=0.55)
            color = self._chain_color(chain, idx)
            name = f"backbone_{chain}"
            self.plotter.add_mesh(
                tube,
                color=color,
                smooth_shading=True,
                specular=0.3,
                name=name,
            )
            self._protein_actor_names.append(name)

        if self.structure.heme_positions.size:
            heme_pc = pv.PolyData(self.structure.heme_positions)
            heme_glyphs = heme_pc.glyph(
                geom=pv.Sphere(radius=0.45), scale=False, orient=False
            )
            self.plotter.add_mesh(
                heme_glyphs,
                color=HEM_COLOR,
                opacity=0.55,
                specular=0.6,
                name="heme",
            )
            self._protein_actor_names.append("heme")

        if self.structure.iron_positions.size:
            fe_pc = pv.PolyData(self.structure.iron_positions)
            fe_glyphs = fe_pc.glyph(
                geom=pv.Sphere(radius=1.4), scale=False, orient=False
            )
            self.plotter.add_mesh(
                fe_glyphs,
                color=FE_COLOR,
                specular=1.0,
                smooth_shading=True,
                name="iron",
            )
            self._protein_actor_names.append("iron")

    def _update_dipole_arrows(self) -> None:
        """(Re)draw the magnetic-moment arrows on each Fe atom."""
        if "dipole_arrows" in self._actors:
            self.plotter.remove_actor(self._actors.pop("dipole_arrows"))

        if not self.structure.iron_positions.size:
            return

        b0 = np.zeros(3)
        b0[self.state["b0_axis"]] = 1.0
        m_scale = float(self.state["moment_scale"])
        if m_scale <= 0.001:
            return  # nothing to draw in the diamagnetic (oxy) limit

        arrow_dir = b0 * m_scale
        n = self.structure.iron_positions.shape[0]
        vectors = np.tile(arrow_dir, (n, 1))

        pdata = pv.PolyData(self.structure.iron_positions)
        pdata["vectors"] = vectors
        arrows = pdata.glyph(
            orient="vectors",
            scale=False,
            factor=6.0,                # Angstrom-scale arrows
            geom=pv.Arrow(tip_length=0.30, tip_radius=0.13, shaft_radius=0.05),
        )
        actor = self.plotter.add_mesh(
            arrows, color=DIPOLE_ARROW_COLOR, specular=0.8, name="dipole_arrows"
        )
        self._actors["dipole_arrows"] = actor

    # ----------------------------------------------------------- core update

    def _build_image_data(self) -> pv.ImageData:
        """Wrap the FieldResult into a PyVista ImageData volume."""
        assert self.field is not None
        f = self.field
        nx, ny, nz = f.Bx.shape
        sx, sy, sz = f.spacing
        ox, oy, oz = f.origin
        img = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(sx, sy, sz),
            origin=(ox, oy, oz),
        )
        mag_nT = f.magnitude_nT
        vec = np.stack([f.Bx, f.By, f.Bz], axis=-1).reshape(-1, 3, order="F")
        img["|dB| [nT]"] = mag_nT.flatten(order="F")
        img["dB"] = vec
        img["log10|dB| [nT]"] = np.log10(np.clip(mag_nT, 1e-3, None)).flatten(order="F")
        return img

    def _recompute_field_and_redraw(self) -> None:
        b0 = np.zeros(3)
        b0[self.state["b0_axis"]] = 1.0
        moments = induced_moments(
            n_dipoles=self.structure.iron_positions.shape[0],
            b0_direction=b0,
            moment_per_fe_bohr=DEOXY_FE_MOMENT_BOHR,
            moment_scale=self.state["moment_scale"],
        )
        self.field = dipole_field(
            dipole_positions=self.structure.iron_positions,
            dipole_moments=moments,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            grid_z=self.grid_z,
        )
        self.image_data = self._build_image_data()
        self._update_dipole_arrows()
        self._redraw_field_actors()

    def _redraw_field_actors(self) -> None:
        """Rebuild the toggleable field actors from the current image_data."""
        cleanup_keys = ["iso", "vectors", "streamlines"]
        for ax in ("x", "y", "z"):
            cleanup_keys.append(f"slice_{ax}")
            cleanup_keys.append(f"slice_{ax}_contour")
        for key in cleanup_keys:
            if key in self._actors:
                self.plotter.remove_actor(self._actors.pop(key))

        if self.image_data is None or self.state["moment_scale"] <= 0.001:
            self._refresh_status_text()
            return

        # --- Isosurface ---
        if self.state["show_iso"]:
            thresh = 10.0 ** float(self.state["iso_log10_nT"])
            mag_max = float(np.nanmax(self.image_data["|dB| [nT]"]))
            if thresh < mag_max:
                try:
                    iso = self.image_data.contour(
                        isosurfaces=[thresh], scalars="|dB| [nT]"
                    )
                    if iso.n_points > 0:
                        actor = self.plotter.add_mesh(
                            iso,
                            color="#00E5FF",
                            opacity=0.35,
                            smooth_shading=True,
                            specular=0.5,
                            name="iso",
                        )
                        self._actors["iso"] = actor
                except Exception:  # noqa: BLE001
                    pass

        # --- Vector glyphs (subsampled) ---
        if self.state["show_vectors"]:
            stride = max(1, self.grid_n // 14)
            sub = self.image_data.extract_subset(
                (0, self.grid_n - 1, 0, self.grid_n - 1, 0, self.grid_n - 1),
                rate=(stride, stride, stride),
            )
            # The raw |dB| dynamic range is ~5 orders of magnitude, so we
            # can't use it directly for arrow length. Instead, build a unit
            # direction field and let glyph(scale=False) draw fixed-size
            # arrows, colored by true magnitude on a log scale.
            db = np.asarray(sub["dB"], dtype=float).reshape(-1, 3)
            mag = np.linalg.norm(db, axis=1)
            safe = np.where(mag > 0, mag, 1.0)
            unit = db / safe[:, None]
            sub["dB_unit"] = unit
            sub.set_active_vectors("dB_unit")
            glyphs = sub.glyph(
                orient="dB_unit",
                scale=False,
                factor=2.5,                # uniform 2.5 A arrows
                geom=pv.Arrow(tip_length=0.3, tip_radius=0.12, shaft_radius=0.04),
            )
            actor = self.plotter.add_mesh(
                glyphs,
                scalars="log10|dB| [nT]",
                cmap="plasma",
                name="vectors",
                show_scalar_bar=False,
            )
            self._actors["vectors"] = actor

        # --- Streamlines ---
        if self.state["show_streamlines"]:
            density = int(self.state["stream_density"])
            try:
                lo, hi = self.structure.extent
                center = self.structure.center
                radius = 0.8 * self.grid_half_extent
                seed = pv.Sphere(
                    radius=radius,
                    center=tuple(center),
                    phi_resolution=density,
                    theta_resolution=density,
                )
                streams = self.image_data.streamlines_from_source(
                    seed,
                    vectors="dB",
                    integration_direction="both",
                    initial_step_length=0.5,
                    terminal_speed=1e-30,
                )
                if streams.n_points > 0:
                    tubed = streams.tube(radius=0.18)
                    actor = self.plotter.add_mesh(
                        tubed,
                        scalars="|dB| [nT]",
                        cmap="viridis",
                        name="streamlines",
                        show_scalar_bar=False,
                    )
                    self._actors["streamlines"] = actor
            except Exception:  # noqa: BLE001
                pass

        # --- Cross-section contour slices (one per axis, independent) ---
        # Shared color limits so all three slices share the same colormap.
        log_mag = self.image_data["log10|dB| [nT]"]
        finite = log_mag[np.isfinite(log_mag)]
        if finite.size:
            color_lo = float(np.percentile(finite, 5))
            color_hi = float(np.percentile(finite, 99))
        else:
            color_lo, color_hi = 3.0, 8.0

        # Contour levels (log10 nT) - evenly spaced across the chosen band.
        n_levels = int(self.state["n_contour_levels"])
        levels = list(np.linspace(color_lo, color_hi, n_levels))

        first_drawn = True
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            if not self.state[f"show_slice_{axis_name}"]:
                continue
            normal = [0.0, 0.0, 0.0]
            normal[axis_idx] = 1.0
            origin = list(self.structure.center)
            origin[axis_idx] += float(self.state[f"slice_{axis_name}_offset"])
            try:
                sl = self.image_data.slice(normal=normal, origin=tuple(origin))
                if sl.n_points == 0:
                    continue

                # Filled colored cross-section.
                scalar_bar_args = {
                    "title": "log10 |dB| [nT]",
                    "vertical": True,
                    "position_x": 0.88,
                    "position_y": 0.20,
                    "width": 0.05,
                    "height": 0.55,
                    "color": "white",
                } if first_drawn else None
                actor = self.plotter.add_mesh(
                    sl,
                    scalars="log10|dB| [nT]",
                    cmap="magma",
                    opacity=0.80,
                    clim=[color_lo, color_hi],
                    name=f"slice_{axis_name}",
                    show_scalar_bar=first_drawn,
                    scalar_bar_args=scalar_bar_args,
                )
                self._actors[f"slice_{axis_name}"] = actor
                first_drawn = False

                # White contour lines on top (the "contour plot").
                contours = sl.contour(
                    isosurfaces=levels, scalars="log10|dB| [nT]"
                )
                if contours.n_points > 0:
                    contour_actor = self.plotter.add_mesh(
                        contours,
                        color="white",
                        line_width=2.0,
                        render_lines_as_tubes=True,
                        name=f"slice_{axis_name}_contour",
                    )
                    self._actors[f"slice_{axis_name}_contour"] = contour_actor
            except Exception:  # noqa: BLE001
                pass

        self._refresh_status_text()

    # ----------------------------------------------------------- widgets/UI

    def _styled_slider(
        self,
        callback,
        rng,
        value,
        title,
        pointa,
        pointb,
        title_color,
        fmt=None,
    ):
        """Add a slider with consistent, high-contrast styling.

        The title and the active value label both render in `title_color`
        with a slightly bold sans font, and the slider knob is shrunk so
        it does not visually clobber the title text.
        """
        kwargs = dict(
            callback=callback,
            rng=rng,
            value=value,
            title=title,
            pointa=pointa,
            pointb=pointb,
            style="modern",
            title_color=title_color,
            title_height=0.022,
            title_opacity=1.0,
            slider_width=0.025,         # smaller knob -> less overlap with title
            tube_width=0.006,           # slightly thicker track for visibility
        )
        if fmt is not None:
            kwargs["fmt"] = fmt
        slider = self.plotter.add_slider_widget(**kwargs)

        # Also color and bold the slider's value-label text via the underlying
        # VTK representation (PyVista's add_slider_widget doesn't expose it).
        try:
            rep = slider.GetRepresentation()
            for prop in (rep.GetLabelProperty(), rep.GetTitleProperty()):
                r = int(title_color[1:3], 16) / 255.0
                g = int(title_color[3:5], 16) / 255.0
                b = int(title_color[5:7], 16) / 255.0
                prop.SetColor(r, g, b)
                prop.SetBold(True)
                prop.SetFontFamilyToArial()
                prop.SetShadow(False)
        except Exception:  # noqa: BLE001
            pass
        return slider

    def _add_widgets(self) -> None:
        # Top: two rows of "what to compute" sliders.
        self._styled_slider(
            callback=self._set_iso_log10,
            rng=[3.5, 8.5],
            value=self.state["iso_log10_nT"],
            title="log10 |dB| iso [nT]",
            pointa=(0.03, 0.88), pointb=(0.47, 0.88),
            title_color=ISO_COLOR,
        )
        self._styled_slider(
            callback=self._set_b0_axis,
            rng=[0.0, 2.0],
            value=float(self.state["b0_axis"]),
            title="B0 axis  (0=X  1=Y  2=Z)",
            pointa=(0.55, 0.88), pointb=(0.97, 0.88),
            fmt="%.0f",
            title_color=B0_TITLE_COLOR,
        )
        self._styled_slider(
            callback=self._set_moment_scale,
            rng=[0.0, 1.0],
            value=self.state["moment_scale"],
            title="moment scale  (1=deoxy, 0=oxy)",
            pointa=(0.03, 0.78), pointb=(0.47, 0.78),
            title_color=MOMENT_COLOR,
        )
        self._styled_slider(
            callback=self._set_stream_density,
            rng=[3, 20],
            value=self.state["stream_density"],
            title="streamline seed density",
            pointa=(0.55, 0.78), pointb=(0.97, 0.78),
            fmt="%.0f",
            title_color=STREAM_COLOR,
        )

        # Bottom: one row of "where is the slice plane" sliders, one per
        # axis. They scan the corresponding cross-section through the field
        # grid (range = +/- the grid half-extent, in Angstroms). We keep
        # references so we can update the ranges when the protein changes.
        he = float(self.grid_half_extent)
        self._slice_sliders["x"] = self._styled_slider(
            callback=self._set_slice_x_offset,
            rng=[-he, he],
            value=self.state["slice_x_offset"],
            title="X-slice offset [A]",
            pointa=(0.03, 0.15), pointb=(0.27, 0.15),
            fmt="%.1f",
            title_color=SLICE_X_COLOR,
        )
        self._slice_sliders["y"] = self._styled_slider(
            callback=self._set_slice_y_offset,
            rng=[-he, he],
            value=self.state["slice_y_offset"],
            title="Y-slice offset [A]",
            pointa=(0.30, 0.15), pointb=(0.54, 0.15),
            fmt="%.1f",
            title_color=SLICE_Y_COLOR,
        )
        self._slice_sliders["z"] = self._styled_slider(
            callback=self._set_slice_z_offset,
            rng=[-he, he],
            value=self.state["slice_z_offset"],
            title="Z-slice offset [A]",
            pointa=(0.57, 0.15), pointb=(0.80, 0.15),
            fmt="%.1f",
            title_color=SLICE_Z_COLOR,
        )

        # Protein-selector dropdown (PyVista's text slider: cycles through
        # the curated PROTEIN_LIBRARY entries). Lives at the very top of
        # the window, above all the other sliders.
        protein_names = [e.short_name for e in PROTEIN_LIBRARY]
        current = find_protein_entry(self.structure.pdb_id)
        try:
            current_index = protein_names.index(current.short_name) if current else 0
        except ValueError:
            current_index = 0
        self._protein_slider = self.plotter.add_text_slider_widget(
            callback=self._on_protein_selected,
            data=protein_names,
            value=current_index,
            pointa=(0.18, 0.965), pointb=(0.97, 0.965),
            style="modern",
        )
        # Apply our high-contrast styling: bold yellow value text, smaller
        # knob so the protein name is fully legible.
        try:
            rep = self._protein_slider.GetRepresentation()
            for prop in (rep.GetLabelProperty(), rep.GetTitleProperty()):
                prop.SetColor(1.0, 0.84, 0.0)         # gold
                prop.SetBold(True)
                prop.SetFontFamilyToArial()
                prop.SetShadow(False)
            rep.SetSliderWidth(0.020)
            rep.SetTubeWidth(0.006)
        except Exception:  # noqa: BLE001
            pass
        # Static "Protein:" label sitting at the upper-left edge.
        self.plotter.add_text(
            "Protein:", position="upper_left", font_size=14,
            color="#FFD400", font="arial", name="lbl_protein",
        )

        # Toggle buttons (bottom row, left to right). Each button's label
        # is color-matched to the layer it toggles so the GUI is
        # self-documenting at a glance.
        self.plotter.add_checkbox_button_widget(
            self._toggle_iso, value=self.state["show_iso"],
            position=(20, 20), size=30, border_size=2,
            color_on=ISO_COLOR, color_off="#3a3a3a",
        )
        self.plotter.add_text(
            "Iso", position=(58, 24), font_size=12, color=ISO_COLOR,
            font="arial", name="lbl_iso",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_vectors, value=self.state["show_vectors"],
            position=(120, 20), size=30, border_size=2,
            color_on=VECTORS_COLOR, color_off="#3a3a3a",
        )
        self.plotter.add_text(
            "Vectors", position=(158, 24), font_size=12, color=VECTORS_COLOR,
            font="arial", name="lbl_vec",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_streamlines, value=self.state["show_streamlines"],
            position=(245, 20), size=30, border_size=2,
            color_on=STREAM_COLOR, color_off="#3a3a3a",
        )
        self.plotter.add_text(
            "Streamlines", position=(283, 24), font_size=12, color=STREAM_COLOR,
            font="arial", name="lbl_str",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_slice_x, value=self.state["show_slice_x"],
            position=(415, 20), size=30, border_size=2,
            color_on=SLICE_X_COLOR, color_off="#3a3a3a",
        )
        self.plotter.add_text(
            "slice X", position=(453, 24), font_size=12, color=SLICE_X_COLOR,
            font="arial", name="lbl_sx",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_slice_y, value=self.state["show_slice_y"],
            position=(540, 20), size=30, border_size=2,
            color_on=SLICE_Y_COLOR, color_off="#3a3a3a",
        )
        self.plotter.add_text(
            "slice Y", position=(578, 24), font_size=12, color=SLICE_Y_COLOR,
            font="arial", name="lbl_sy",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_slice_z, value=self.state["show_slice_z"],
            position=(665, 20), size=30, border_size=2,
            color_on=SLICE_Z_COLOR, color_off="#3a3a3a",
        )
        self.plotter.add_text(
            "slice Z", position=(703, 24), font_size=12, color=SLICE_Z_COLOR,
            font="arial", name="lbl_sz",
        )

    def _add_title_and_legend(self) -> None:
        # The protein-selector dropdown at the top already shows the
        # current protein name, so we skip a redundant title widget.
        self._rebuild_legend()
        # Orientation axes widget (stays put across protein switches).
        self.plotter.add_axes(interactive=False)

    def _rebuild_legend(self) -> None:
        """(Re)build the bottom-right legend for the current structure."""
        legend_entries: list = []
        for idx, chain in enumerate(sorted(self.structure.backbone_by_chain)):
            label = f"chain {chain}" if chain.strip() else "backbone"
            legend_entries.append([label, self._chain_color(chain, idx)])
        if self.structure.heme_positions.size:
            legend_entries.append(["Fe cofactor", HEM_COLOR])
        legend_entries.append(["Fe atom", FE_COLOR])
        legend_entries.append(["dipole moment", DIPOLE_ARROW_COLOR])

        # Drop any prior legend then add the new one. PyVista's
        # add_legend replaces the existing legend automatically by
        # internal name but we go through remove_legend() defensively.
        try:
            self.plotter.remove_legend()
        except Exception:  # noqa: BLE001
            pass
        self.plotter.add_legend(
            legend_entries,
            bcolor=(0, 0, 0),
            face=None,
            loc="lower right",
            size=(0.18, 0.20),
        )

    def _refresh_status_text(self) -> None:
        if self.field is None:
            return
        mag = self.field.magnitude_nT
        finite = mag[np.isfinite(mag)]
        if finite.size == 0:
            return
        axis_name = "XYZ"[self.state["b0_axis"]]
        n_fe = self.structure.iron_positions.shape[0]
        text = (
            f"{self.structure.pdb_id}  ({n_fe} Fe)    "
            f"B0: +{axis_name}    moment: {self.state['moment_scale']:.2f}    "
            f"|dB| max: {finite.max():.2e} nT    "
            f"|dB| median: {np.median(finite):.2e} nT"
        )
        # Anchor by pixel so it sits cleanly above the toggle-button row.
        self.plotter.add_text(
            text, position=(15, 60), font_size=11, color=INFO_TEXT_COLOR,
            font="arial", name="status",
        )

    # ----------------------------------------------------- widget callbacks

    def _set_iso_log10(self, value: float) -> None:
        self.state["iso_log10_nT"] = float(value)
        self._redraw_field_actors()

    def _set_b0_axis(self, value: float) -> None:
        axis = int(round(float(value)))
        axis = max(0, min(2, axis))
        if axis == self.state["b0_axis"]:
            return
        self.state["b0_axis"] = axis
        self._recompute_field_and_redraw()

    def _set_moment_scale(self, value: float) -> None:
        self.state["moment_scale"] = float(value)
        self._recompute_field_and_redraw()

    def _set_stream_density(self, value: float) -> None:
        self.state["stream_density"] = int(round(float(value)))
        if self.state["show_streamlines"]:
            self._redraw_field_actors()

    def _toggle_iso(self, value: bool) -> None:
        self.state["show_iso"] = bool(value)
        self._redraw_field_actors()

    def _toggle_vectors(self, value: bool) -> None:
        self.state["show_vectors"] = bool(value)
        self._redraw_field_actors()

    def _toggle_streamlines(self, value: bool) -> None:
        self.state["show_streamlines"] = bool(value)
        self._redraw_field_actors()

    def _toggle_slice_x(self, value: bool) -> None:
        self.state["show_slice_x"] = bool(value)
        self._redraw_field_actors()

    def _toggle_slice_y(self, value: bool) -> None:
        self.state["show_slice_y"] = bool(value)
        self._redraw_field_actors()

    def _toggle_slice_z(self, value: bool) -> None:
        self.state["show_slice_z"] = bool(value)
        self._redraw_field_actors()

    def _set_slice_x_offset(self, value: float) -> None:
        self.state["slice_x_offset"] = float(value)
        if self.state["show_slice_x"]:
            self._redraw_field_actors()

    def _set_slice_y_offset(self, value: float) -> None:
        self.state["slice_y_offset"] = float(value)
        if self.state["show_slice_y"]:
            self._redraw_field_actors()

    def _set_slice_z_offset(self, value: float) -> None:
        self.state["slice_z_offset"] = float(value)
        if self.state["show_slice_z"]:
            self._redraw_field_actors()

    # --------------------------------------------------- protein switching

    def _on_protein_selected(self, name: str) -> None:
        """Callback for the top-of-window protein dropdown."""
        entry = next((e for e in PROTEIN_LIBRARY if e.short_name == name), None)
        if entry is None or entry.pdb_id.upper() == self.structure.pdb_id.upper():
            return
        self._load_protein(entry)

    def _load_protein(self, entry: ProteinEntry) -> None:
        """Replace the active structure with `entry`, rebuilding everything."""
        print(f"[gui] Switching to {entry.pdb_id} ({entry.short_name}) ...")
        try:
            new_structure = load_structure(entry.pdb_id)
        except Exception as ex:  # noqa: BLE001
            print(f"[gui] Failed to load {entry.pdb_id}: {ex}")
            return

        if new_structure.iron_positions.shape[0] == 0:
            print(f"[gui] {entry.pdb_id} has no Fe atoms - keeping current protein.")
            return

        # 1. Tear down old protein actors (backbones + heme + Fe).
        for name in list(self._protein_actor_names):
            try:
                self.plotter.remove_actor(name)
            except Exception:  # noqa: BLE001
                pass
        self._protein_actor_names = []
        if "dipole_arrows" in self._actors:
            try:
                self.plotter.remove_actor(self._actors.pop("dipole_arrows"))
            except Exception:  # noqa: BLE001
                pass

        # 2. Tear down old field actors so they don't reference a stale grid.
        cleanup_keys = ["iso", "vectors", "streamlines"]
        for ax in ("x", "y", "z"):
            cleanup_keys.append(f"slice_{ax}")
            cleanup_keys.append(f"slice_{ax}_contour")
        for key in cleanup_keys:
            if key in self._actors:
                try:
                    self.plotter.remove_actor(self._actors.pop(key))
                except Exception:  # noqa: BLE001
                    pass

        # 3. Swap in the new structure and rebuild the field grid.
        self.structure = new_structure
        self.grid_half_extent = _compute_adaptive_extent(new_structure)
        gx, gy, gz = make_grid(
            new_structure.center, half_extent=self.grid_half_extent, n=self.grid_n,
        )
        self.grid_x, self.grid_y, self.grid_z = gx, gy, gz

        # Reset slice offsets to 0 (their old absolute positions are
        # meaningless in the new protein's frame) and update slider ranges.
        for ax in ("x", "y", "z"):
            self.state[f"slice_{ax}_offset"] = 0.0
            self._update_slice_slider_range(ax, self.grid_half_extent)

        # 4. Re-add protein actors and recompute the dipole field.
        self._add_protein_actors()
        self._rebuild_legend()
        self._recompute_field_and_redraw()

        # 5. Reset the camera so the new protein fits in the window.
        try:
            self.plotter.reset_camera()
        except Exception:  # noqa: BLE001
            pass

        # 6. Push the new selection into the dropdown widget so the GUI
        # label updates even when this method is invoked programmatically.
        try:
            names = [e.short_name for e in PROTEIN_LIBRARY]
            idx = names.index(entry.short_name)
            self._protein_slider.GetRepresentation().SetValue(float(idx))
        except Exception:  # noqa: BLE001
            pass

    def _update_slice_slider_range(self, axis: str, half_extent: float) -> None:
        """Push a new value range onto an existing slice-offset slider."""
        slider = self._slice_sliders.get(axis)
        if slider is None:
            return
        try:
            rep = slider.GetRepresentation()
            rep.SetMinimumValue(-half_extent)
            rep.SetMaximumValue(+half_extent)
            rep.SetValue(0.0)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ entrypoint

    def show(self) -> None:
        self.plotter.show()


# Backwards-compatible alias - the GUI is no longer hemoglobin-specific,
# but external scripts may still import the old name.
HemoglobinFieldGUI = ProteinFieldGUI
