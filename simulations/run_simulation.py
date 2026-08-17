import argparse
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence, Tuple


script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


# ---------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationSettings:
    frozen_phonon_configs: int = 100
    frozen_phonon_sigma: float = 0.07
    frozen_phonon_seed: int | None = None
    ensemble_mean: bool = False
    potential_sampling: float = 0.05
    scan_start: Tuple[float, float] = (0, 0)
    scan_end: Tuple[float, float] = (1, 1)
    detector_inner: float = 80
    detector_outer: float = 200
    interpolation_sampling: float = 0.05
    dose_per_area: float = 1e7
    max_batch: str = "1 GB"


@dataclass(frozen=True)
class StructureBlockSpec:
    filename: str
    x_repeat: int
    y_repeat: int
    layers: int


# ---------------------------------------------------------------------
# Structure construction
# ---------------------------------------------------------------------

def repeat_structure(filename, x_repeat=1, y_repeat=1, z_layers=1):
    """
    Reads a structure file and repeats it in x, y, and z.

    The z repeat is the number of layers/cells contributed by this structure
    block to the final stacked simulation cell.
    """
    if min(x_repeat, y_repeat, z_layers) < 1:
        raise ValueError("x_repeat, y_repeat, and z_layers must all be >= 1")

    import ase.io

    atoms = ase.io.read(filename)
    atoms.wrap(eps=1e-4)
    return atoms * (x_repeat, y_repeat, z_layers)


def stack_structure_blocks(blocks):
    """
    Vertically stacks a sequence of ASE Atoms blocks along the z cell vector.

    The x/y cell vectors must already match after repeat_structure.
    """
    if not blocks:
        raise ValueError("At least one structure block is required")

    import numpy as np

    stacked = blocks[0].copy()
    total_z = stacked.cell[2].copy()

    for block in blocks[1:]:
        x_matches = np.allclose(block.cell[0], blocks[0].cell[0])
        y_matches = np.allclose(block.cell[1], blocks[0].cell[1])
        if not x_matches or not y_matches:
            raise ValueError(
                "All stacked blocks must have matching x/y cell vectors after repetition"
            )

        block_instance = block.copy()
        block_instance.translate(total_z)
        stacked += block_instance
        total_z = total_z + block.cell[2]

    new_cell = blocks[0].cell.copy()
    new_cell[2] = total_z
    stacked.set_cell(new_cell)
    stacked.wrap(eps=1e-4)

    return stacked


def build_layered_structure(structure_blocks: Sequence[StructureBlockSpec]):
    """
    Builds the final simulation structure from an ordered sequence of input
    files.

    Each file is independently repeated in x, y, and z/layers before all
    nonzero blocks are stacked along z.
    """
    blocks = []

    for spec in structure_blocks:
        if spec.layers > 0:
            blocks.append(
                repeat_structure(
                    filename=spec.filename,
                    x_repeat=spec.x_repeat,
                    y_repeat=spec.y_repeat,
                    z_layers=spec.layers,
                )
            )

    return stack_structure_blocks(blocks)


# ---------------------------------------------------------------------
# Simulation execution
# ---------------------------------------------------------------------

def run_simulation(atoms, energy, Cs, semiangle_cutoff, defocus, settings=None):
    """
    Runs one complete abTEM HAADF simulation for a fixed structure and
    microscope parameter combination.

    This function intentionally hides the abTEM setup sequence so the rest of
    the program can treat one simulation as a single operation.

    Returns:
        output:
            Clean simulated HAADF array.
        output_noisy:
            Poisson-noised simulated HAADF array.
    """
    settings = settings or SimulationSettings()
    atoms.set_pbc((True, True, False))

    import abtem

    frozen_phonons = abtem.FrozenPhonons(
        atoms,
        num_configs=settings.frozen_phonon_configs,
        sigmas=settings.frozen_phonon_sigma,
        ensemble_mean=settings.ensemble_mean,
        seed=settings.frozen_phonon_seed,
    )

    potential = abtem.Potential(
        frozen_phonons,
        sampling=settings.potential_sampling,
        plane="xy",
    )

    probe = abtem.Probe(
        energy=energy,
        semiangle_cutoff=semiangle_cutoff,
        Cs=Cs,
        defocus=defocus,
    )
    probe.grid.match(potential)

    grid_scan = abtem.GridScan(
        start=settings.scan_start,
        end=settings.scan_end,
        sampling=probe.aperture.nyquist_sampling,
        fractional=True,
        potential=potential,
    )

    detector = abtem.AnnularDetector(
        inner=settings.detector_inner,
        outer=settings.detector_outer,
    )

    measurement = probe.scan(
        potential,
        scan=grid_scan,
        detectors=detector,
        max_batch=settings.max_batch,
    )
    haadf = measurement.interpolate(
        settings.interpolation_sampling
    ).compute()

    output = haadf.array.astype("float32")
    output = output.transpose(0, 2, 1)

    return output


def save_simulation_result(save_dir, filename, data):
    """
    Saves a single simulation array as a .npy file.

    Creates the target directory if it does not already exist.
    """
    import numpy as np

    directory = Path(save_dir)
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / f"{filename}.npy"
    np.save(file_path, data)


# ---------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------

def parse_pair(value):
    parts = [float(item) for item in str(value).split(",") if item != ""]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated values, e.g. 0.25,0.2"
        )
    return tuple(parts)


def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected an integer >= 0")
    return number


def parse_structure_block(value):
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4 or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            "expected FILE,X_REPEAT,Y_REPEAT,LAYERS, "
            "e.g. data/structures/pristine.cif,5,1,12"
        )

    filename, x_repeat, y_repeat, layers = parts

    try:
        return StructureBlockSpec(
            filename=filename,
            x_repeat=positive_int(x_repeat),
            y_repeat=positive_int(y_repeat),
            layers=nonnegative_int(layers),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "X_REPEAT, Y_REPEAT, and LAYERS must be integers"
        ) from exc


def add_arguments(parser):
    parser.add_argument(
        "--iter",
        type=int,
        required=True,
        help="Iteration number used in output names",
    )
    parser.add_argument(
        "--output-filename",
        required=True,
        help="Base name for output files",
    )
    parser.add_argument(
        "--save-dir",
        default="data/simulations/layered",
        help="Directory for .npy outputs",
    )
    parser.add_argument(
        "--save-structure",
        default=None,
        help="Optional path to write the assembled structure",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        help="abTEM device, usually gpu or cpu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings without importing abTEM/ASE",
    )

    parser.add_argument(
        "--structure-block",
        action="append",
        type=parse_structure_block,
        required=True,
        metavar="FILE,X_REPEAT,Y_REPEAT,LAYERS",
        help=(
            "One z-stacked structure block. Repeat for each block in order, "
            "e.g. --structure-block=data/structures/pristine.cif,5,1,12"
        ),
    )

    parser.add_argument(
        "--energy",
        type=float,
        default=100000,
        help="Electron beam energy in eV",
    )
    parser.add_argument(
        "--cs",
        type=float,
        default=0,
        help="Spherical aberration in Angstroms",
    )
    parser.add_argument(
        "--semiangle-cutoff",
        type=float,
        default=30,
        help="Probe semiangle cutoff in mrad",
    )
    parser.add_argument(
        "--defocus",
        type=float,
        default=10,
        help="Defocus in Angstroms",
    )

    parser.add_argument("--frozen-phonon-configs", type=int, default=100)
    parser.add_argument("--frozen-phonon-sigma", type=float, default=0.07)
    parser.add_argument(
        "--frozen-phonon-seed",
        type=int,
        default=None,
        help="Random seed for frozen phonon displacements",
    )
    parser.add_argument(
        "--ensemble-mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average frozen phonon configurations into an ensemble mean",
    )
    parser.add_argument("--potential-sampling", type=float, default=0.05)
    parser.add_argument("--scan-start", type=parse_pair, default=(0.25, 0.2))
    parser.add_argument("--scan-end", type=parse_pair, default=(0.75, 0.8))
    parser.add_argument("--detector-inner", type=float, default=80)
    parser.add_argument("--detector-outer", type=float, default=200)
    parser.add_argument("--interpolation-sampling", type=float, default=0.05)
    parser.add_argument("--dose-per-area", type=float, default=1e7)
    parser.add_argument("--max-batch", default="1 GB")


def settings_from_args(args):
    return SimulationSettings(
        frozen_phonon_configs=args.frozen_phonon_configs,
        frozen_phonon_sigma=args.frozen_phonon_sigma,
        frozen_phonon_seed=args.frozen_phonon_seed,
        ensemble_mean=args.ensemble_mean,
        potential_sampling=args.potential_sampling,
        scan_start=args.scan_start,
        scan_end=args.scan_end,
        detector_inner=args.detector_inner,
        detector_outer=args.detector_outer,
        interpolation_sampling=args.interpolation_sampling,
        dose_per_area=args.dose_per_area,
        max_batch=args.max_batch,
    )


def print_job_summary(args):
    print("Layered simulation job")
    print("Structure blocks:")

    for index, spec in enumerate(args.structure_block, start=1):
        print(
            f"  {index}: file={project_path(spec.filename)}, "
            f"x={spec.x_repeat}, y={spec.y_repeat}, layers={spec.layers}"
        )

    print(f"Total layers: {sum(spec.layers for spec in args.structure_block)}")
    print(f"Save directory: {project_path(args.save_dir)}")
    print("Simulation count in this Python run: 1")
    print("Microscope parameters:")
    print(f"  energy = {args.energy} eV")
    print(f"  Cs = {args.cs} Angstroms")
    print(f"  semiangle_cutoff = {args.semiangle_cutoff} mrad")
    print(f"  defocus = {args.defocus} Angstroms")
    print(f"  frozen_phonon_configs = {args.frozen_phonon_configs}")
    print(f"  frozen_phonon_seed = {args.frozen_phonon_seed}")
    print(f"  ensemble_mean = {args.ensemble_mean}")


def validate_structure_paths(args, parser):
    for index, spec in enumerate(args.structure_block, start=1):
        path = project_path(spec.filename)
        if not path.exists():
            parser.error(f"--structure-block {index} file does not exist: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run one shell-configured layered abTEM simulation."
    )
    add_arguments(parser)
    args = parser.parse_args()

    if sum(spec.layers for spec in args.structure_block) < 1:
        parser.error("at least one --structure-block must have LAYERS > 0")

    print_job_summary(args)

    base_filename = args.output_filename

    if args.dry_run:
        print(f"Dry run output file: {base_filename}.npy")
        return

    validate_structure_paths(args, parser)

    import abtem
    import ase.io
    import dask

    dask.config.set(scheduler="single-threaded")
    abtem.config.set({"device": args.device})

    settings = settings_from_args(args)

    structure_blocks = [
        StructureBlockSpec(
            filename=project_path(spec.filename),
            x_repeat=spec.x_repeat,
            y_repeat=spec.y_repeat,
            layers=spec.layers,
        )
        for spec in args.structure_block
    ]

    atoms = build_layered_structure(structure_blocks)

    if args.save_structure:
        structure_path = project_path(args.save_structure)
        structure_path.parent.mkdir(parents=True, exist_ok=True)
        ase.io.write(structure_path, atoms)
        print(f"Assembled structure written to: {structure_path}")

    images = run_simulation(
        atoms=atoms,
        energy=args.energy,
        Cs=args.cs,
        semiangle_cutoff=args.semiangle_cutoff,
        defocus=args.defocus,
        settings=settings,
    )

    save_dir = project_path(args.save_dir)

    save_simulation_result(
        save_dir,
        base_filename,
        images,
    )

    print(f"Saved array with shape: {images.shape}")


if __name__ == "__main__":
    main()
