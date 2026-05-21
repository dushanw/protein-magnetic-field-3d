"""
Interactive 3D visualizer for the magnetic field around hemoglobin.

Uses PyVista's native widgets (no Qt dependency) so this runs anywhere
PyVista runs. The GUI offers:

    * 7 sliders   : log10(isosurface threshold in nT), B0 direction (X/Y/Z),
                    paramagnetic moment scale (1.0 = deoxy, 0.0 = oxy),
                    streamline density, and three slice-plane offsets
                    (one per X / Y / Z axis - scan a contour cross-section
                    through the field volume).
    * 6 buttons   : toggle isosurface / vectors / streamlines / X-slice /
                    Y-slice / Z-slice.
    * Always on   : protein backbone ribbon (per chain),
                    heme groups (semi-transparent),
                    Fe atoms (orange spheres) + dipole arrows.

Each slice is drawn as a colored cross-section plus white contour-lines of
log10|dB|, so it is a true contour plot of the magnetic-field magnitude.

Coordinates are in Angstroms; field is in Tesla but displayed in nano-Tesla
because hemoglobin-scale dipolar fields fall in the 1 - 10^5 nT range over
the visualization volume.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyvista as pv

from .magnetic_field import (
    DEOXY_FE_MOMENT_BOHR,
    FieldResult,
    dipole_field,
    induced_moments,
    make_grid,
)
from .pdb_loader import ProteinStructure

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


class HemoglobinFieldGUI:
    """Stateful interactive plotter for the dipole-field visualization."""

    def __init__(
        self,
        structure: ProteinStructure,
        grid_n: int = 60,
        grid_half_extent: float = 30.0,
    ):
        self.structure = structure
        self.grid_n = grid_n
        self.grid_half_extent = grid_half_extent

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
        center = structure.center
        gx, gy, gz = make_grid(center, half_extent=grid_half_extent, n=grid_n)
        self.grid_x, self.grid_y, self.grid_z = gx, gy, gz

        self.field: Optional[FieldResult] = None
        self.image_data: Optional[pv.ImageData] = None

        self.plotter = pv.Plotter(window_size=(1280, 860))
        self.plotter.set_background("black", top="#15233a")

        # Actor handles so we can add/remove them on toggles & resimulations.
        self._actors: Dict[str, object] = {}

        self._add_protein_actors()
        self._recompute_field_and_redraw()
        self._add_widgets()
        self._add_title_and_legend()

    # ------------------------------------------------------------------ setup

    def _add_protein_actors(self) -> None:
        """Add the static protein geometry (backbone + hemes + Fe spheres)."""
        for chain, ca in self.structure.backbone_by_chain.items():
            if ca.shape[0] < 2:
                continue
            spline = pv.Spline(ca, n_points=max(ca.shape[0] * 4, 64))
            tube = spline.tube(radius=0.55)
            color = CHAIN_COLORS.get(chain, "#888888")
            self.plotter.add_mesh(
                tube,
                color=color,
                smooth_shading=True,
                specular=0.3,
                name=f"backbone_{chain}",
            )

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

    def _add_widgets(self) -> None:
        # Top: two rows of "what to compute" sliders.
        self.plotter.add_slider_widget(
            callback=self._set_iso_log10,
            rng=[3.5, 8.5],
            value=self.state["iso_log10_nT"],
            title="log10 |dB| iso [nT]",
            pointa=(0.03, 0.88), pointb=(0.47, 0.88),
            style="modern",
        )
        self.plotter.add_slider_widget(
            callback=self._set_b0_axis,
            rng=[0.0, 2.0],
            value=float(self.state["b0_axis"]),
            title="B0 axis (0=X 1=Y 2=Z)",
            pointa=(0.55, 0.88), pointb=(0.97, 0.88),
            fmt="%.0f",
            style="modern",
        )
        self.plotter.add_slider_widget(
            callback=self._set_moment_scale,
            rng=[0.0, 1.0],
            value=self.state["moment_scale"],
            title="moment scale (1=deoxy, 0=oxy)",
            pointa=(0.03, 0.78), pointb=(0.47, 0.78),
            style="modern",
        )
        self.plotter.add_slider_widget(
            callback=self._set_stream_density,
            rng=[3, 20],
            value=self.state["stream_density"],
            title="streamline seed density",
            pointa=(0.55, 0.78), pointb=(0.97, 0.78),
            fmt="%.0f",
            style="modern",
        )

        # Bottom: one row of "where is the slice plane" sliders, one per
        # axis. They scan the corresponding cross-section through the field
        # grid (range = +/- the grid half-extent, in Angstroms).
        he = float(self.grid_half_extent)
        self.plotter.add_slider_widget(
            callback=self._set_slice_x_offset,
            rng=[-he, he],
            value=self.state["slice_x_offset"],
            title="X-slice offset [A]",
            pointa=(0.03, 0.15), pointb=(0.27, 0.15),
            fmt="%.1f",
            style="modern",
        )
        self.plotter.add_slider_widget(
            callback=self._set_slice_y_offset,
            rng=[-he, he],
            value=self.state["slice_y_offset"],
            title="Y-slice offset [A]",
            pointa=(0.30, 0.15), pointb=(0.54, 0.15),
            fmt="%.1f",
            style="modern",
        )
        self.plotter.add_slider_widget(
            callback=self._set_slice_z_offset,
            rng=[-he, he],
            value=self.state["slice_z_offset"],
            title="Z-slice offset [A]",
            pointa=(0.57, 0.15), pointb=(0.80, 0.15),
            fmt="%.1f",
            style="modern",
        )

        # Toggle buttons (bottom row, left to right).
        self.plotter.add_checkbox_button_widget(
            self._toggle_iso, value=self.state["show_iso"],
            position=(20, 20), size=28, border_size=2,
            color_on="#00E5FF", color_off="#444444",
        )
        self.plotter.add_text(
            "Iso", position=(55, 22), font_size=10, color="white", name="lbl_iso"
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_vectors, value=self.state["show_vectors"],
            position=(120, 20), size=28, border_size=2,
            color_on="#FF8C00", color_off="#444444",
        )
        self.plotter.add_text(
            "Vectors", position=(155, 22), font_size=10, color="white",
            name="lbl_vec",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_streamlines, value=self.state["show_streamlines"],
            position=(240, 20), size=28, border_size=2,
            color_on="#55E07F", color_off="#444444",
        )
        self.plotter.add_text(
            "Streamlines", position=(275, 22), font_size=10, color="white",
            name="lbl_str",
        )
        # Three independent slice toggles (X red, Y green, Z blue - standard
        # axis colors). Each one enables/disables its own cross-section.
        self.plotter.add_checkbox_button_widget(
            self._toggle_slice_x, value=self.state["show_slice_x"],
            position=(400, 20), size=28, border_size=2,
            color_on="#E74C3C", color_off="#444444",
        )
        self.plotter.add_text(
            "slice X", position=(435, 22), font_size=10, color="white",
            name="lbl_sx",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_slice_y, value=self.state["show_slice_y"],
            position=(510, 20), size=28, border_size=2,
            color_on="#2ECC71", color_off="#444444",
        )
        self.plotter.add_text(
            "slice Y", position=(545, 22), font_size=10, color="white",
            name="lbl_sy",
        )
        self.plotter.add_checkbox_button_widget(
            self._toggle_slice_z, value=self.state["show_slice_z"],
            position=(620, 20), size=28, border_size=2,
            color_on="#3498DB", color_off="#444444",
        )
        self.plotter.add_text(
            "slice Z", position=(655, 22), font_size=10, color="white",
            name="lbl_sz",
        )

    def _add_title_and_legend(self) -> None:
        title = (
            f"Magnetic field around {self.structure.pdb_id} "
            f"(hemoglobin) — {self.structure.iron_positions.shape[0]} Fe centers"
        )
        self.plotter.add_text(
            title, position="upper_left", font_size=12, color="white",
            name="title",
        )

        legend_entries = []
        for chain in sorted(self.structure.backbone_by_chain):
            legend_entries.append(
                [f"chain {chain}", CHAIN_COLORS.get(chain, "#888888")]
            )
        legend_entries.append(["heme", HEM_COLOR])
        legend_entries.append(["Fe", FE_COLOR])
        legend_entries.append(["dipole moment", DIPOLE_ARROW_COLOR])
        # Lower-right of window so the legend never overlaps the colorbar
        # (the colorbar lives on the right edge between y=0.20 and y=0.75).
        self.plotter.add_legend(
            legend_entries,
            bcolor=(0, 0, 0),
            face=None,
            loc="lower right",
            size=(0.18, 0.18),
        )

        # Orientation axes widget.
        self.plotter.add_axes(interactive=False)

    def _refresh_status_text(self) -> None:
        if self.field is None:
            return
        mag = self.field.magnitude_nT
        finite = mag[np.isfinite(mag)]
        if finite.size == 0:
            return
        axis_name = "XYZ"[self.state["b0_axis"]]
        text = (
            f"B0 axis: +{axis_name}    "
            f"moment scale: {self.state['moment_scale']:.2f}    "
            f"|dB| max: {finite.max():.2e} nT    "
            f"|dB| median: {np.median(finite):.2e} nT"
        )
        # Anchor by pixel so it sits cleanly under the title, well clear
        # of the sliders (which occupy the top ~16% of the window).
        self.plotter.add_text(
            text, position=(15, 60), font_size=10, color="white", name="status",
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

    # ------------------------------------------------------------ entrypoint

    def show(self) -> None:
        self.plotter.show()
