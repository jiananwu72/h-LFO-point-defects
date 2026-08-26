"""Image processing and analysis tools for LuFeO3 patch grids."""

import numpy as np
import scipy.signal
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

from .patch import Patch, PatchGrid, PatchLayout


def _signal_data(signal):
    if hasattr(signal, "data"):
        return np.asarray(signal.data)
    return np.asarray(signal)


def _crop_signal(signal, left, right, start, end):
    if hasattr(signal, "isig"):
        return signal.isig[left:right, start:end]
    return np.asarray(signal)[start:end, left:right]


class GridCreator:
    """Convert a HyperSpy signal or image array into a LuFeO3 PatchGrid."""

    def __init__(
        self,
        detect_height=1,
        detect_width=1,
        min_sep=3,
        prom_coeff=0.1,
        tol=0.0,
    ):
        self.detect_height = detect_height
        self.detect_width = detect_width
        self.min_sep = min_sep
        self.prom_coeff = prom_coeff
        self.tol = tol

    def process(self, signal, layout=None):
        """Build and classify a patch grid."""
        if layout is None:
            layout = self.detect_layout(signal)

        grid = self.build_grid(signal, layout=layout)
        self._assign_lufe_atom_types(grid)
        return grid

    def detect_layout(self, signal):
        """Detect patch boundaries from a signal and return a reusable layout."""
        data = _signal_data(signal)
        if data.ndim != 2:
            raise ValueError("GridCreator requires a 2D signal.")

        h_peaks, h_troughs, _ = self._find_horizontal_peaks(signal)
        v_peaks, v_troughs, _ = self._find_vertical_peaks(signal)

        if len(v_troughs) < 2 or len(h_troughs) < 2:
            raise ValueError("Need at least two vertical and two horizontal troughs.")

        return PatchLayout(
            horizontal_peaks=h_peaks,
            horizontal_troughs=h_troughs,
            vertical_peaks=v_peaks,
            vertical_troughs=v_troughs,
        )

    def build_grid(self, signal, layout):
        """Build an image-specific PatchGrid from an existing layout."""
        data = _signal_data(signal)
        if data.ndim != 2:
            raise ValueError("GridCreator requires a 2D signal.")

        patches = np.empty(layout.shape, dtype=object)
        for (col_idx, row_idx), (x0, x1), (y0, y1) in layout.iter_patch_edges():
            image = _crop_signal(signal, x0, x1, y0, y1)
            patches[col_idx, row_idx] = Patch(
                image,
                (x0, x1),
                (y0, y1),
            )

        return PatchGrid(
            patches,
            layout=layout,
            source_signal=signal,
        )

    def _find_horizontal_peaks(self, signal):
        data = _signal_data(signal)
        strengths = []
        for y in range(data.shape[0]):
            strip = _crop_signal(signal, 0, data.shape[1], y, y + self.detect_height)
            strengths.append(np.asarray(_signal_data(strip)).sum())
        return self._find_peaks_and_troughs(np.asarray(strengths))

    def _find_vertical_peaks(self, signal):
        data = _signal_data(signal)
        strengths = []
        for x in range(data.shape[1]):
            strip = _crop_signal(signal, x, x + self.detect_width, 0, data.shape[0])
            strengths.append(np.asarray(_signal_data(strip)).sum())
        return self._find_peaks_and_troughs(np.asarray(strengths))

    def _find_peaks_and_troughs(self, strengths):
        prom = self.prom_coeff * (strengths.max() - strengths.min())
        peaks, _ = scipy.signal.find_peaks(
            strengths,
            distance=self.min_sep,
            prominence=prom,
        )
        troughs, _ = scipy.signal.find_peaks(
            -strengths,
            distance=self.min_sep,
            prominence=prom,
        )
        return peaks.astype(int), troughs.astype(int), strengths

    def _assign_lufe_atom_types(self, grid):
        n_cols, n_layers = grid.shape
        layer_intensities = np.zeros(n_layers, dtype=float)

        for i in range(n_cols):
            for j in range(n_layers):
                layer_intensities[j] += grid[i, j].mean_intensity

        is_lu_layer = np.zeros(n_layers, dtype=bool)
        for j in range(n_layers):
            left = layer_intensities[j - 1] if j > 0 else -np.inf
            right = layer_intensities[j + 1] if j + 1 < n_layers else -np.inf
            is_lu_layer[j] = layer_intensities[j] > max(left, right) + self.tol

        for i in range(n_cols):
            for j in range(n_layers):
                grid.atom_types[i, j] = "Lu" if is_lu_layer[j] else "Fe"


def equalize_patch_sizes(grid, signal=None):
    """Redistribute interior edges without changing the grid's outer bounds."""
    if grid.size == 0:
        raise ValueError("Cannot equalize an empty patch grid.")

    signal = getattr(grid, "source_signal", None) if signal is None else signal
    if signal is None:
        raise ValueError("The grid does not have a source signal.")

    signal_data = _signal_data(signal)
    if signal_data.ndim != 2:
        raise ValueError("The source signal must be two-dimensional.")
    grid.source_signal = signal

    n_cols, n_layers = grid.shape
    layer_types = np.asarray(grid.atom_types[0], dtype=object)
    if not np.all(grid.atom_types == layer_types):
        raise ValueError("Each grid layer must contain one atom type.")
    if not set(layer_types) <= {"Lu", "Fe"}:
        raise ValueError("Grid atom types must be 'Lu' or 'Fe'.")

    layer_heights = np.array(
        [grid[0, layer].height for layer in range(n_layers)],
        dtype=float,
    )
    mean_heights = {
        atom_type: layer_heights[layer_types == atom_type].mean()
        for atom_type in set(layer_types)
    }
    if any(height <= 0 for height in mean_heights.values()):
        raise ValueError("Patch dimensions must be positive.")

    start_y = int(grid[0, 0].row_edges[0])
    end_y = int(grid[0, -1].row_edges[1])
    start_x = int(grid[0, 0].col_edges[0])
    end_x = int(grid[-1, 0].col_edges[1])

    x_edges = np.rint(np.linspace(start_x, end_x, n_cols + 1)).astype(int)
    y_steps = [mean_heights[atom_type] for atom_type in layer_types]
    y_edges = np.rint(start_y + np.r_[0, np.cumsum(y_steps)]).astype(int)
    x_edges[[0, -1]] = (start_x, end_x)
    y_edges[[0, -1]] = (start_y, end_y)

    if np.any(np.diff(x_edges) <= 0) or np.any(np.diff(y_edges) <= 0):
        raise ValueError("Equalized patch dimensions must be positive.")

    height, width = signal_data.shape
    if not (0 <= start_x < end_x <= width and 0 <= start_y < end_y <= height):
        raise ValueError("Equalized patch edges exceed the source signal bounds.")

    for i, j in np.ndindex(grid.shape):
        x0, x1 = x_edges[i : i + 2]
        y0, y1 = y_edges[j : j + 2]
        image = _crop_signal(signal, x0, x1, y0, y1)
        grid[i, j] = Patch(image, (x0, x1), (y0, y1))

    grid.layout = PatchLayout.from_patch_grid(grid)
    return grid


class IntensityAnalyzer:
    """Compute patch intensity arrays and vicinity metrics."""

    def values(self, grid, metric="mean_intensity"):
        """Return a 2D array of patch metric values."""
        return grid.values(metric)

    def vicinity(self, grid, offsets, metric="mean_intensity", values=None):
        """Return mean neighbor values, optionally from precomputed values."""
        offsets = self._validate_offsets(offsets)
        n_rows, n_cols = grid.shape
        values = self._resolve_values(grid, metric, values)
        vicinity = np.full(grid.shape, np.nan, dtype=float)

        for i in range(n_rows):
            for j in range(n_cols):
                neighbor_values = []
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_rows and 0 <= nj < n_cols:
                        neighbor_values.append(values[ni, nj])
                    else:
                        neighbor_values = []
                        break
                if neighbor_values:
                    vicinity[i, j] = float(np.mean(neighbor_values))

        return vicinity

    def relative_vicinity(self, grid, offsets, metric="mean_intensity", values=None):
        """Return relative neighbor values, optionally using precomputed values."""
        values = self._resolve_values(grid, metric, values)
        vicinity = self.vicinity(
            grid,
            offsets=offsets,
            metric=metric,
            values=values,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            return values / vicinity

    def _resolve_values(self, grid, metric, values):
        if values is None:
            return self.values(grid, metric=metric)

        values = np.asarray(values, dtype=float)
        if values.shape != grid.shape:
            raise ValueError("values must have the same shape as the grid.")
        return values

    def _validate_offsets(self, offsets):
        if offsets is None:
            raise ValueError("offsets must be provided.")
        clean_offsets = [tuple(offset) for offset in offsets]
        if not all(len(offset) == 2 for offset in clean_offsets):
            raise ValueError("Each offset must be a pair: (di, dj).")
        return clean_offsets


class PositionAnalyzer:
    """Fit atom centers for patches in a PatchGrid."""

    def __init__(self, maxfev=1000):
        self.maxfev = maxfev

    def fit(self, grid, method="circular", return_parameters=False):
        """Fit all patches, optionally returning per-patch parameters."""
        self._validate_method(method)
        positions = np.full(grid.shape + (2,), np.nan, dtype=float)
        parameters = (
            self._empty_parameter_arrays(grid.shape, method)
            if return_parameters
            else None
        )

        for index, patch in np.ndenumerate(grid.patches):
            fit_result = self.fit_patch(
                patch,
                method=method,
                return_parameters=return_parameters,
            )

            if return_parameters:
                fit_result, patch_parameters = fit_result
                for name, value in patch_parameters.items():
                    parameters[name][index] = value

            if fit_result is None:
                continue

            local_x, local_y = fit_result
            image_x = patch.col_edges[0] + local_x
            image_y = patch.row_edges[0] + local_y

            positions[index] = np.array([image_x, image_y], dtype=float)

        if return_parameters:
            return positions, parameters
        return positions

    def get_Lu_phases(self, grid, method="circular"):
        """Return Lu phase distances and nearest-state classifications."""
        positions = self.fit(grid, method=method)
        lu_mask = grid.atom_type_mask("Lu")
        singlet_distances = np.full(grid.shape, np.nan)
        doublet_distances = np.full(grid.shape, np.nan)
        class_map = np.full(grid.shape, np.nan)

        theta = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        singlet_phases = np.mod(theta[1] + np.array([0, np.pi]), 2 * np.pi)
        doublet_phases = np.mod(
            np.concatenate((theta[[0, 2]], theta[[0, 2]] + np.pi)),
            2 * np.pi,
        )

        for column in range(1, grid.shape[0] - 1):
            for row in range(grid.shape[1]):
                if not np.all(lu_mask[column - 1 : column + 2, row]):
                    continue

                z = positions[column - 1 : column + 2, row, 1]
                if not np.all(np.isfinite(z)):
                    continue

                u = z - np.mean(z)
                a = 2 / 3 * np.sum(u * np.cos(theta))
                b = 2 / 3 * np.sum(u * np.sin(theta))
                phi = np.mod(np.arctan2(b, a), 2 * np.pi)

                singlet_distances[column, row] = np.min(
                    np.abs(np.angle(np.exp(1j * (phi - singlet_phases))))
                )
                doublet_distances[column, row] = np.min(
                    np.abs(np.angle(np.exp(1j * (phi - doublet_phases))))
                )

                singlet_distance = singlet_distances[column, row]
                doublet_distance = doublet_distances[column, row]
                if np.isclose(singlet_distance, doublet_distance):
                    continue
                if singlet_distance < doublet_distance:
                    class_map[column, row] = 1
                else:
                    class_map[column, row] = -1

        return singlet_distances, doublet_distances, class_map

    def fit_patch(self, patch, method="circular", return_parameters=False):
        """Fit one patch, optionally returning its fitted parameters."""
        self._validate_method(method)
        failed_result = (
            ((np.nan, np.nan), self._empty_patch_parameters(method))
            if return_parameters
            else None
        )
        data = np.asarray(patch.data, dtype=float)
        if data.ndim != 2 or not np.isfinite(data).all():
            return failed_result

        height, width = data.shape
        if height == 0 or width == 0:
            return failed_result

        center_y, center_x = height // 2, width // 2
        y_grid, x_grid = np.mgrid[-center_y : height - center_y, -center_x : width - center_x]

        bg = float(np.min(data))
        amp_guess = float(np.max(data) - bg)
        if amp_guess <= 0:
            return failed_result

        cy_guess, cx_guess = center_of_mass(data - bg)
        if not np.isfinite(cx_guess) or not np.isfinite(cy_guess):
            return failed_result

        if method == "circular":
            model = self._gaussian_2d
            p0 = [amp_guess, cx_guess - center_x, cy_guess - center_y, 2.0, bg]
        else:
            model = self._elliptical_gaussian_2d
            p0 = [amp_guess, cx_guess - center_x, cy_guess - center_y, 2.0, 2.0, 0.0, bg]

        try:
            popt, _ = curve_fit(
                model,
                (x_grid, y_grid),
                data.ravel(),
                p0=p0,
                maxfev=self.maxfev,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            return failed_result

        if not np.isfinite(popt).all():
            return failed_result

        dx, dy = popt[1:3]
        x_fit = center_x + dx
        y_fit = center_y + dy

        if not (0 <= x_fit < width and 0 <= y_fit < height):
            return failed_result

        local_center = np.array([x_fit, y_fit], dtype=float)
        if not return_parameters:
            return float(x_fit), float(y_fit)

        global_center = local_center + np.array(
            [patch.col_edges[0], patch.row_edges[0]],
            dtype=float,
        )
        parameters = self._empty_patch_parameters(method)
        parameters.update(
            amplitude=float(popt[0]),
            local_center=local_center,
            global_center=global_center,
            theta=0.0 if method == "circular" else float(popt[5]),
            offset=float(popt[-1]),
            fit_success=True,
        )
        if method == "circular":
            parameters["sigma"] = abs(float(popt[3]))
        else:
            parameters["sigma_x"] = abs(float(popt[3]))
            parameters["sigma_y"] = abs(float(popt[4]))

        return (float(x_fit), float(y_fit)), parameters

    @staticmethod
    def _empty_patch_parameters(method):
        parameters = {
            "amplitude": np.nan,
            "local_center": np.full(2, np.nan),
            "global_center": np.full(2, np.nan),
            "theta": np.nan,
            "offset": np.nan,
            "fit_success": False,
        }
        if method == "circular":
            parameters["sigma"] = np.nan
        else:
            parameters["sigma_x"] = np.nan
            parameters["sigma_y"] = np.nan
        return parameters

    @staticmethod
    def _empty_parameter_arrays(shape, method):
        parameters = {
            "amplitude": np.full(shape, np.nan),
            "local_center": np.full(shape + (2,), np.nan),
            "global_center": np.full(shape + (2,), np.nan),
            "theta": np.full(shape, np.nan),
            "offset": np.full(shape, np.nan),
            "fit_success": np.zeros(shape, dtype=bool),
        }
        if method == "circular":
            parameters["sigma"] = np.full(shape, np.nan)
        else:
            parameters["sigma_x"] = np.full(shape, np.nan)
            parameters["sigma_y"] = np.full(shape, np.nan)
        return parameters

    @staticmethod
    def _gaussian_2d(coords, amplitude, xo, yo, sigma, offset):
        x, y = coords
        g = offset + amplitude * np.exp(-((x - xo) ** 2 + (y - yo) ** 2) / (2 * sigma**2))
        return g.ravel()

    @staticmethod
    def _elliptical_gaussian_2d(
        coords,
        amplitude,
        xo,
        yo,
        sigma_x,
        sigma_y,
        theta,
        offset,
    ):
        x, y = coords
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_rotated = cos_theta * (x - xo) + sin_theta * (y - yo)
        y_rotated = -sin_theta * (x - xo) + cos_theta * (y - yo)
        g = offset + amplitude * np.exp(
            -0.5
            * (
                (x_rotated / sigma_x) ** 2
                + (y_rotated / sigma_y) ** 2
            )
        )
        return g.ravel()

    @staticmethod
    def _validate_method(method):
        if method not in {"circular", "elliptical"}:
            raise ValueError("method must be 'circular' or 'elliptical'.")


def subtract_background(grid):
    """Subtract fitted constant offsets and return them per patch."""
    _, parameters = PositionAnalyzer().fit(
        grid,
        method="circular",
        return_parameters=True,
    )
    backgrounds = parameters["offset"]
    failed = ~parameters["fit_success"]
    if np.any(failed):
        raise RuntimeError(
            f"Background fitting failed for {np.count_nonzero(failed)} patches."
        )

    for index, patch in np.ndenumerate(grid.patches):
        if hasattr(patch, "background"):
            del patch.background
        patch.image = patch.image - backgrounds[index]

    return backgrounds
