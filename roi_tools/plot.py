"""Visualization tools for patch-grid image analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


class PatchGridPlotter:
    """Thin Matplotlib helper for patch-grid overlays."""

    def __init__(
        self,
        signal,
        grid,
        figsize=(10, 10),
        colorscale="gray",
        title=None,
        fig=None,
        ax=None,
        show_image=True,
        **imshow_kwargs,
    ):
        self.signal = signal
        self.grid = grid
        self.figsize = figsize
        self.colorscale = colorscale
        self.fig = fig
        self.ax = ax
        self.base_image_artist = None
        self._init_axes()
        if show_image:
            self.show_image(title=title, **imshow_kwargs)

    def _init_axes(self):
        if self.ax is not None:
            self.fig = self.ax.figure
            return
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            return
        self.ax = self.fig.add_subplot(111)

    def show_image(self, title=None, **imshow_kwargs):
        """Show the source image on the plotter axes."""
        imshow_kwargs.setdefault("cmap", self.colorscale)
        self.base_image_artist = self.ax.imshow(self._signal_data, **imshow_kwargs)
        if title is not None:
            self.ax.set_title(title)
        return self.base_image_artist

    @property
    def _signal_data(self):
        if hasattr(self.signal, "data"):
            return np.asarray(self.signal.data)
        return np.asarray(self.signal)

    def add_vertical_troughs(self, color="magenta", linestyle="--", linewidth=1, **kwargs):
        """Overlay detected vertical troughs."""
        if self.grid.vertical_troughs is None:
            return []
        artists = []
        for x in self.grid.vertical_troughs:
            artists.append(
                self.ax.axvline(
                    x,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    **kwargs,
                )
            )
        return artists

    def add_horizontal_troughs(self, color="magenta", linestyle="--", linewidth=1, **kwargs):
        """Overlay detected horizontal troughs."""
        if self.grid.horizontal_troughs is None:
            return []
        artists = []
        for y in self.grid.horizontal_troughs:
            artists.append(
                self.ax.axhline(
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    **kwargs,
                )
            )
        return artists

    def add_vertical_peaks(self, color="magenta", linestyle="--", linewidth=1, **kwargs):
        """Overlay detected vertical peaks."""
        if self.grid.vertical_peaks is None:
            return []
        artists = []
        for x in self.grid.vertical_peaks:
            artists.append(
                self.ax.axvline(
                    x,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    **kwargs,
                )
            )
        return artists

    def add_horizontal_peaks(self, color="magenta", linestyle="--", linewidth=1, **kwargs):
        """Overlay detected horizontal peaks."""
        if self.grid.horizontal_peaks is None:
            return []
        artists = []
        for y in self.grid.horizontal_peaks:
            artists.append(
                self.ax.axhline(
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    **kwargs,
                )
            )
        return artists

    def add_patch_edges(
        self,
        atom_type="all",
        values=None,
        color="r",
        cmap="viridis",
        vmin=None,
        vmax=None,
        center_cmap=False,
        linewidth=1,
        alpha=1.0,
        label=None,
        **kwargs,
    ):
        """Overlay patch edges, optionally colored by values."""
        _, segments, aligned_values = self._get_patch_geometries(atom_type, values)
        if not segments:
            return None

        kwargs.setdefault("linewidth", linewidth)
        kwargs.setdefault("alpha", alpha)
        kwargs.setdefault("label", label)
        if aligned_values is None:
            kwargs.setdefault("colors", color)
            collection = LineCollection(segments, **kwargs)
        else:
            collection = LineCollection(segments, cmap=cmap, **kwargs)
            collection.set_array(aligned_values)
            self._set_clim(collection, aligned_values, vmin, vmax, center_cmap)

        self.ax.add_collection(collection)
        return collection

    def add_patch_faces(
        self,
        atom_type="all",
        values=None,
        color="r",
        cmap="viridis",
        vmin=None,
        vmax=None,
        center_cmap=False,
        alpha=1.0,
        label=None,
        **kwargs,
    ):
        """Overlay patch faces, optionally colored by values."""
        rects, _, aligned_values = self._get_patch_geometries(atom_type, values)
        if aligned_values is not None:
            mask = np.isfinite(aligned_values)
            rects = [rect for rect, keep in zip(rects, mask) if keep]
            aligned_values = aligned_values[mask]
        if not rects:
            return None

        kwargs.setdefault("edgecolor", "none")
        kwargs.setdefault("alpha", alpha)
        kwargs.setdefault("label", label)
        if aligned_values is None:
            kwargs.setdefault("facecolor", color)
            collection = PatchCollection(rects, **kwargs)
        else:
            collection = PatchCollection(rects, cmap=cmap, **kwargs)
            collection.set_array(aligned_values)
            self._set_clim(collection, aligned_values, vmin, vmax, center_cmap)

        self.ax.add_collection(collection)
        return collection

    def add_boolean_edges(
        self,
        boolean_map,
        atom_type="all",
        color="r",
        linewidth=2,
        alpha=1.0,
        label=None,
        **kwargs,
    ):
        """Overlay patch edges where a boolean map is true."""
        rects, _, aligned_bools = self._get_patch_geometries(atom_type, boolean_map)
        if aligned_bools is None:
            return None
        true_rects = [rect for rect, flag in zip(rects, aligned_bools) if flag]
        if not true_rects:
            return None

        collection = PatchCollection(
            true_rects,
            edgecolor=kwargs.pop("edgecolor", color),
            facecolor=kwargs.pop("facecolor", "none"),
            linewidth=kwargs.pop("linewidth", linewidth),
            alpha=kwargs.pop("alpha", alpha),
            joinstyle=kwargs.pop("joinstyle", "miter"),
            antialiased=kwargs.pop("antialiased", True),
            label=kwargs.pop("label", label),
            **kwargs,
        )
        self.ax.add_collection(collection)
        return collection

    def add_boolean_faces(
        self,
        boolean_map,
        atom_type="all",
        color="r",
        alpha=0.5,
        label=None,
        **kwargs,
    ):
        """Overlay patch faces where a boolean map is true."""
        rects, _, aligned_bools = self._get_patch_geometries(atom_type, boolean_map)
        if aligned_bools is None:
            return None
        true_rects = [rect for rect, flag in zip(rects, aligned_bools) if flag]
        if not true_rects:
            return None

        collection = PatchCollection(
            true_rects,
            facecolor=kwargs.pop("facecolor", color),
            edgecolor=kwargs.pop("edgecolor", "none"),
            alpha=kwargs.pop("alpha", alpha),
            label=kwargs.pop("label", label),
            **kwargs,
        )
        self.ax.add_collection(collection)
        return collection

    def add_positions(
        self,
        positions,
        atom_type="all",
        size=5,
        values=None,
        color="r",
        cmap="viridis",
        vmin=None,
        vmax=None,
        center_cmap=False,
        label=None,
        **kwargs,
    ):
        """Overlay fitted atom positions, optionally colored by values."""
        positions, aligned_values = self._get_positions(atom_type, positions, values)
        if positions is None:
            return None

        valid = np.isfinite(positions).all(axis=1)
        positions = positions[valid]
        if len(positions) == 0:
            return None
        if aligned_values is not None:
            aligned_values = np.asarray(aligned_values)[valid]

        kwargs.setdefault("s", size)
        kwargs.setdefault("label", label)
        if aligned_values is None:
            kwargs.setdefault("c", color)
        else:
            kwargs.setdefault("c", aligned_values)
            kwargs.setdefault("cmap", cmap)
            if center_cmap:
                limit = vmax if vmax is not None else np.nanmax(np.abs(aligned_values))
                kwargs["vmin"] = -limit
                kwargs["vmax"] = limit
            else:
                if vmin is not None:
                    kwargs["vmin"] = vmin
                if vmax is not None:
                    kwargs["vmax"] = vmax

        return self.ax.scatter(positions[:, 0], positions[:, 1], **kwargs)

    def add_colorbar(self, artist=None, label=None, fraction=0.05, pad=0.04, **kwargs):
        """Add a colorbar for an artist or the base image."""
        target = artist if artist is not None else self.base_image_artist
        if target is None:
            return None
        if hasattr(target, "get_array") and target.get_array() is None:
            return None

        cbar = self.fig.colorbar(target, ax=self.ax, fraction=fraction, pad=pad, **kwargs)
        if label is not None:
            cbar.set_label(label)
        return cbar

    def add_legend(self, loc="upper left", bbox_to_anchor=(1.05, 1.0), **kwargs):
        """Add a legend for labeled plot artists."""
        handles, labels = self.ax.get_legend_handles_labels()
        if not handles:
            handles, labels = self._collection_legend_handles()

        if handles:
            return self.ax.legend(
                handles,
                labels,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                frameon=False,
                **kwargs,
            )
        return None

    def show(
        self,
        scale=True,
        scale_nm=2,
        scale_linewidth=8,
        scale_fontsize=20,
        scale_loc="left",
        scale_x_shift=0.0,
        scale_y_shift=-0.02,
        axis_on=None,
        show=True,
        **scale_kwargs,
    ):
        """Optionally add a scale bar and render the current figure."""
        pixel_size = self._pixel_size()
        if scale and pixel_size is not None:
            self.add_scale_bar(
                pixel_size,
                scale_nm,
                scale_linewidth,
                scale_fontsize,
                scale_loc,
                scale_x_shift,
                scale_y_shift,
                **scale_kwargs,
            )
        if axis_on is not None:
            self.ax.axis("on" if axis_on else "off")
        if show:
            plt.show()
        return self.fig, self.ax

    def _get_patch_geometries(self, atom_type, values):
        self._validate_atom_type(atom_type)
        aligned_values = self._aligned_values(atom_type, values) if values is not None else None
        rects = []
        segments = []

        for patch in self._selected_patches(atom_type):
            x0, x1 = patch.col_edges
            y0, y1 = patch.row_edges
            rects.append(Rectangle((x0, y0), patch.width, patch.height))
            segments.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])

        return rects, segments, aligned_values

    def _get_positions(self, atom_type, positions, values):
        self._validate_atom_type(atom_type)
        positions = np.asarray(positions, dtype=float)
        if positions.shape[:2] != self.grid.shape or positions.shape[-1] != 2:
            raise ValueError("positions must have shape grid.shape + (2,).")

        selected_positions = []
        mask = self.grid.atom_type_mask(atom_type)
        for index in np.ndindex(self.grid.shape):
            if mask[index]:
                selected_positions.append(positions[index])

        if not selected_positions:
            return None, None
        aligned_values = self._aligned_values(atom_type, values) if values is not None else None
        return np.asarray(selected_positions, dtype=float), aligned_values

    def _selected_patches(self, atom_type):
        mask = self.grid.atom_type_mask(atom_type)
        return [
            patch
            for index, patch in np.ndenumerate(self.grid.patches)
            if mask[index]
        ]

    def _aligned_values(self, atom_type, values):
        values = np.asarray(values)
        if values.shape[:2] != self.grid.shape:
            raise ValueError("values must have the same first two dimensions as the grid.")

        aligned = []
        mask = self.grid.atom_type_mask(atom_type)
        for index in np.ndindex(self.grid.shape):
            if mask[index]:
                aligned.append(values[index])
        return np.asarray(aligned)

    def _set_clim(self, collection, values, vmin, vmax, center_cmap):
        if center_cmap:
            limit = vmax if vmax is not None else np.nanmax(np.abs(values))
            collection.set_clim(-limit, limit)
        else:
            collection.set_clim(vmin, vmax)

    def add_scale_bar(
        self,
        pixel_size=None,
        scale_nm=2,
        linewidth=8,
        fontsize=20,
        loc="left",
        x_shift=0.0,
        y_shift=-0.02,
        color="black",
        text=None,
        line_kwargs=None,
        text_kwargs=None,
    ):
        """Add a scale bar and return its line and text artists."""
        if pixel_size is None:
            pixel_size = self._pixel_size()
        if pixel_size is None or self.base_image_artist is None:
            return None, None

        line_kwargs = {} if line_kwargs is None else dict(line_kwargs)
        text_kwargs = {} if text_kwargs is None else dict(text_kwargs)
        image_width_pixels = self.base_image_artist.get_array().shape[1]
        image_width_nm = image_width_pixels * pixel_size
        bar_fraction = scale_nm / image_width_nm

        if loc == "right":
            base_x = 1.0 - bar_fraction
        elif loc == "center":
            base_x = 0.5 - bar_fraction / 2
        else:
            base_x = 0.0

        start_x = base_x + x_shift
        end_x = start_x + bar_fraction
        line_kwargs.setdefault("color", color)
        line_kwargs.setdefault("linewidth", linewidth)
        line_kwargs.setdefault("clip_on", False)
        text_kwargs.setdefault("color", color)
        text_kwargs.setdefault("fontsize", fontsize)
        text_kwargs.setdefault("ha", "center")
        text_kwargs.setdefault("va", "top")
        text_kwargs.setdefault("clip_on", False)

        line = self.ax.plot(
            [start_x, end_x],
            [y_shift, y_shift],
            transform=self.ax.transAxes,
            **line_kwargs,
        )
        label = f"{scale_nm} nm" if text is None else text
        text_artist = self.ax.text(
            start_x + bar_fraction / 2,
            y_shift * 2,
            label,
            transform=self.ax.transAxes,
            **text_kwargs,
        )
        return line[0], text_artist

    def _collection_legend_handles(self):
        handles = []
        labels = []
        for collection in self.ax.collections:
            label = collection.get_label()
            if not label or label == "_nolegend_":
                continue
            ec = collection.get_edgecolor()
            fc = collection.get_facecolor()
            edge_color = ec[0] if len(ec) > 0 else "black"
            face_color = fc[0] if len(fc) > 0 else "none"
            handles.append(
                mpatches.Rectangle(
                    (0, 0),
                    1,
                    1,
                    edgecolor=edge_color,
                    facecolor=face_color,
                    linewidth=2,
                )
            )
            labels.append(label)
        return handles, labels

    def _validate_atom_type(self, atom_type):
        if atom_type not in ["all", "Fe", "Lu"]:
            raise ValueError("atom_type must be 'all', 'Fe', or 'Lu'.")

    def _pixel_size(self):
        if hasattr(self.signal, "axes_manager"):
            return self.signal.axes_manager[0].scale
        return None


ROIPlotter = PatchGridPlotter
