"""Patch data structures used by ROI image analysis."""

import numpy as np


class Patch:
    """Store a materialized image patch cache and its current image bounds."""

    def __init__(self, image, col_edges, row_edges, atom_type=None):
        # Keep the cropped HyperSpy signal or array materialized so repeated
        # metrics and fits do not need to re-slice the source image.
        self.image = image
        self.col_edges = np.asarray(col_edges, dtype=int)
        self.row_edges = np.asarray(row_edges, dtype=int)
        self._grid = None
        self._grid_index = None
        self._atom_type = atom_type

    @property
    def data(self):
        """Return cached patch image data as a NumPy array."""
        if hasattr(self.image, "data"):
            return np.asarray(self.image.data)
        return np.asarray(self.image)

    @property
    def width(self):
        """Return the patch width in pixels."""
        return int(self.col_edges[1] - self.col_edges[0])

    @property
    def height(self):
        """Return the patch height in pixels."""
        return int(self.row_edges[1] - self.row_edges[0])

    @property
    def sum_intensity(self):
        """Return the summed patch intensity."""
        return float(np.sum(self.data))

    @property
    def mean_intensity(self):
        """Return the mean patch intensity."""
        return float(np.mean(self.data))

    @property
    def max_intensity(self):
        """Return the maximum patch intensity."""
        return float(np.max(self.data))

    @property
    def atom_type(self):
        """Return the grid-level atom type for this patch."""
        if self._grid is None or self._grid_index is None:
            return self._atom_type
        return self._grid.atom_types[self._grid_index]

    @atom_type.setter
    def atom_type(self, atom_type):
        """Set the atom type on the parent grid when attached."""
        if self._grid is None or self._grid_index is None:
            self._atom_type = atom_type
        else:
            self._grid.atom_types[self._grid_index] = atom_type

    def _attach_grid(self, grid, index):
        """Attach this patch to its owning grid and index."""
        self._grid = grid
        self._grid_index = tuple(index)


class PatchLayout:
    """Store reusable patch-grid boundaries detected from an image."""

    def __init__(
        self,
        horizontal_peaks=None,
        horizontal_troughs=None,
        vertical_peaks=None,
        vertical_troughs=None,
        patch_col_edges=None,
        patch_row_edges=None,
    ):
        self.horizontal_peaks = horizontal_peaks
        self.horizontal_troughs = (
            None if horizontal_troughs is None else np.asarray(horizontal_troughs, dtype=int)
        )
        self.vertical_peaks = vertical_peaks
        self.vertical_troughs = (
            None if vertical_troughs is None else np.asarray(vertical_troughs, dtype=int)
        )
        self.patch_col_edges = self._normalize_patch_edges(patch_col_edges)
        self.patch_row_edges = self._normalize_patch_edges(patch_row_edges)

        if self.patch_col_edges is None or self.patch_row_edges is None:
            self._init_patch_edges_from_troughs()
        self._validate_patch_edges()

    @property
    def col_edges(self):
        """Return one column-edge pair per patch-grid column."""
        if self.patch_col_edges is None:
            return []
        return [tuple(edges) for edges in self.patch_col_edges[:, 0, :].astype(int)]

    @property
    def row_edges(self):
        """Return one row-edge pair per patch-grid row."""
        if self.patch_row_edges is None:
            return []
        return [tuple(edges) for edges in self.patch_row_edges[0, :, :].astype(int)]

    @property
    def shape(self):
        """Return the grid shape produced by this layout."""
        if self.patch_col_edges is not None:
            return self.patch_col_edges.shape[:2]
        return 0, 0

    def iter_patch_edges(self):
        """Iterate over grid indices and current per-patch edge bounds."""
        for index in np.ndindex(self.shape):
            yield (
                index,
                tuple(self.patch_col_edges[index].astype(int)),
                tuple(self.patch_row_edges[index].astype(int)),
            )

    @classmethod
    def from_patch_grid(cls, grid, keep_peaks=True):
        """Build a layout from the current patch geometry in a grid."""
        patch_col_edges = np.empty(grid.shape + (2,), dtype=int)
        patch_row_edges = np.empty(grid.shape + (2,), dtype=int)

        for index, patch in np.ndenumerate(grid.patches):
            patch_col_edges[index] = patch.col_edges
            patch_row_edges[index] = patch.row_edges

        horizontal_peaks = grid.horizontal_peaks if keep_peaks else None
        vertical_peaks = grid.vertical_peaks if keep_peaks else None

        return cls(
            horizontal_peaks=horizontal_peaks,
            vertical_peaks=vertical_peaks,
            patch_col_edges=patch_col_edges,
            patch_row_edges=patch_row_edges,
        )

    def _init_patch_edges_from_troughs(self):
        if self.patch_col_edges is not None and self.patch_row_edges is not None:
            return
        if self.vertical_troughs is None or self.horizontal_troughs is None:
            return
        if len(self.vertical_troughs) < 2 or len(self.horizontal_troughs) < 2:
            return

        col_edges = np.asarray(
            [
                (int(self.vertical_troughs[i]), int(self.vertical_troughs[i + 1]))
                for i in range(len(self.vertical_troughs) - 1)
            ],
            dtype=int,
        )
        row_edges = np.asarray(
            [
                (int(self.horizontal_troughs[i]), int(self.horizontal_troughs[i + 1]))
                for i in range(len(self.horizontal_troughs) - 1)
            ],
            dtype=int,
        )
        n_cols = len(col_edges)
        n_rows = len(row_edges)

        if self.patch_col_edges is None:
            self.patch_col_edges = np.repeat(col_edges[:, np.newaxis, :], n_rows, axis=1)
        if self.patch_row_edges is None:
            self.patch_row_edges = np.repeat(row_edges[np.newaxis, :, :], n_cols, axis=0)

    def _normalize_patch_edges(self, edges):
        if edges is None:
            return None
        edges = np.asarray(edges, dtype=int)
        if edges.ndim != 3 or edges.shape[-1] != 2:
            raise ValueError("Patch edge arrays must have shape grid.shape + (2,).")
        return edges

    def _validate_patch_edges(self):
        if self.patch_col_edges is None or self.patch_row_edges is None:
            return
        if self.patch_col_edges.shape != self.patch_row_edges.shape:
            raise ValueError("Patch column and row edge arrays must have the same shape.")


class PatchGrid:
    """Store a processed 2D patch grid and grid-level metadata."""

    def __init__(
        self,
        patches,
        layout=None,
        atom_types=None,
        source_signal=None,
    ):
        self.patches = np.asarray(patches, dtype=object)
        self.layout = layout
        self.atom_types = self._init_atom_types(atom_types)
        self.source_signal = source_signal
        self._attach_patches()

    @property
    def horizontal_peaks(self):
        """Return horizontal peak positions from the layout."""
        return None if self.layout is None else self.layout.horizontal_peaks

    @property
    def horizontal_troughs(self):
        """Return horizontal trough positions from the layout."""
        return None if self.layout is None else self.layout.horizontal_troughs

    @property
    def vertical_peaks(self):
        """Return vertical peak positions from the layout."""
        return None if self.layout is None else self.layout.vertical_peaks

    @property
    def vertical_troughs(self):
        """Return vertical trough positions from the layout."""
        return None if self.layout is None else self.layout.vertical_troughs

    @property
    def shape(self):
        """Return the patch grid shape."""
        return self.patches.shape

    @property
    def flat(self):
        """Return a flat iterator over patches."""
        return self.patches.flat

    @property
    def size(self):
        """Return the number of patches in the grid."""
        return self.patches.size

    def __getitem__(self, index):
        """Return a patch by grid index."""
        return self.patches[index]

    def __setitem__(self, index, value):
        """Set a patch by grid index."""
        atom_type = value.atom_type
        if atom_type is not None:
            self.atom_types[index] = atom_type
        value._attach_grid(self, index)
        self.patches[index] = value

    def iter_patches(self):
        """Iterate over patches in flat order."""
        return self.patches.flat

    def values(self, metric="mean_intensity"):
        """Return a 2D array of a patch metric or attribute."""
        values = np.full(self.shape, np.nan, dtype=float)
        missing = object()

        for index, patch in np.ndenumerate(self.patches):
            if patch is None:
                continue
            value = getattr(patch, metric, missing)
            if value is missing:
                raise ValueError(f"Patch at index {index} has no metric '{metric}'.")
            if value is not None:
                values[index] = value
        return values

    def atom_type_mask(self, atom_type):
        """Return a boolean mask for patches with the selected atom type."""
        if atom_type == "all":
            return np.ones(self.shape, dtype=bool)
        return self.atom_types == atom_type

    def _init_atom_types(self, atom_types):
        if atom_types is not None:
            atom_types = np.asarray(atom_types, dtype=object)
            if atom_types.shape != self.shape:
                raise ValueError("atom_types must have the same shape as patches.")
            return atom_types.copy()

        inferred = np.full(self.shape, None, dtype=object)
        for index, patch in np.ndenumerate(self.patches):
            if patch is not None:
                inferred[index] = patch.atom_type
        return inferred

    def _attach_patches(self):
        for index, patch in np.ndenumerate(self.patches):
            if patch is not None:
                patch._attach_grid(self, index)
