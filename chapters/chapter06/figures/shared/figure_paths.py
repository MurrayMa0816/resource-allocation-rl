"""Locate original figure fonts without loading plotting libraries."""
from pathlib import Path
import os


def fonts_root(anchor):
    explicit = os.environ.get("THESIS_FIGURE_FONT_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    for parent in Path(anchor).resolve().parents:
        candidate = parent / "assets/fonts"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Original figure fonts are required. Pass --font-dir /path/to/assets/fonts "
        "or set THESIS_FIGURE_FONT_DIR; the directory must contain windows/ and/or figure/."
    )


def configure_fonts(anchor, directory, required):
    if directory is not None:
        os.environ["THESIS_FIGURE_FONT_DIR"] = str(Path(directory).expanduser().resolve())
    root = fonts_root(anchor)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing original figure font files: " + ", ".join(str(root / name) for name in missing))
    return root


def validate_output_dir(anchor, directory, *, sources=(), font_directory=None, font_file=None):
    """Reject source overlap and nonempty outputs before loading plotting code."""
    output = Path(directory).expanduser().resolve()
    figure_root = next(p for p in Path(anchor).resolve().parents if p.name == "figures")
    project_root = figure_root.parents[2]
    protected = [project_root, figure_root / "shared/data"]
    protected.extend(Path(p).expanduser().resolve() for p in sources)
    font_value = font_directory or os.environ.get("THESIS_FIGURE_FONT_DIR")
    if font_value:
        protected.append(Path(font_value).expanduser().resolve())
    if font_file:
        protected.append(Path(font_file).expanduser().resolve().parent)
    for parent in Path(anchor).resolve().parents:
        candidate = parent / "assets/fonts"
        if candidate.is_dir():
            protected.append(candidate.resolve())
    for source in protected:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output directory must be separate from source, data and font paths: " + str(source))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must not exist or must be empty: " + str(output))
    return output
