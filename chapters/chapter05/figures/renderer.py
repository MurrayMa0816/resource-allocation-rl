"""Run the published current figure source with explicitly supplied observations.

The original plotting and layout sources remain byte-identical in _support.
Only filesystem staging is added. --check verifies files without plotting.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe figure package path: " + str(value))
    return path


def _copy_checked(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.is_symlink() or _digest(destination) != _digest(source):
        raise ValueError("Figure staging changed file bytes: " + str(source))


def _stage_files(stage, published, external, fonts):
    package = stage / "figure"
    package.mkdir()
    # The exact current sources use these historical, case-sensitive font names.
    font_names = {"simsun.ttc": "simsun.ttc", "simhei.ttf": "simhei.ttf",
        "times.ttf": "TIMES.TTF", "timesbd.ttf": "TIMESBD.TTF",
        "timesi.ttf": "TIMESI.TTF", "timesbi.ttf": "TIMESBI.TTF"}
    window_fonts = stage / "assets/fonts/windows"
    for name, canonical in font_names.items():
        # Native sources use both canonical Windows names and lowercase lookup names.
        for filename in dict.fromkeys((canonical, name)):
            _copy_checked(fonts[name], window_fonts / filename)
    for relative, source in published + external:
        _copy_checked(source, package / relative)
    if any(path.is_symlink() for path in stage.rglob("*")):
        raise ValueError("Figure staging must contain ordinary files and directories")
    return package, window_fonts


def _output_path(value, protected):
    destination = Path(value).expanduser().resolve()
    for root in protected:
        root = Path(root).resolve()
        if destination.is_relative_to(root) or root.is_relative_to(destination):
            raise ValueError("Output must be disjoint from the source, input and font directories")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Choose a new or empty output directory")
    return destination


def _materialize_outputs(destination):
    for path in destination.rglob("*"):
        if path.is_symlink():
            if not path.is_file():
                raise ValueError("Figure output link does not refer to an ordinary file: " + str(path))
            contents = path.read_bytes()
            path.unlink()
            path.write_bytes(contents)
    if any(path.is_symlink() for path in destination.rglob("*")):
        raise ValueError("Figure outputs must contain ordinary files and directories")


def main(argv=None, here=None):
    here = Path(here).resolve()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-dir", type=Path, required=True,
        help="Complete current figure package listed in inputs.json, including reference images")
    parser.add_argument("--font-dir", type=Path,
        default=os.environ.get("THESIS_FIGURE_FONT_DIR"),
        help="Original SimSun, SimHei and Times font files")
    parser.add_argument("--output-dir", type=Path, help="Required for plotting; use a separate new directory")
    parser.add_argument("--check", action="store_true", help="Check source/input/font files without plotting")
    args = parser.parse_args(argv)
    manifest = json.loads((here / "support.json").read_text())
    overlap = set(manifest["files"]) & set(manifest["external_files"])
    if overlap:
        raise ValueError("Published source and external input paths overlap: " + ", ".join(sorted(overlap)))
    data = args.data_dir.expanduser().resolve(strict=True)
    if args.font_dir is None:
        parser.error("Provide --font-dir or THESIS_FIGURE_FONT_DIR")
    fonts = args.font_dir.expanduser().resolve(strict=True)
    names = {}
    for path in sorted(fonts.iterdir()):
        if not path.is_file():
            continue
        key = path.name.lower()
        if key in names and _digest(names[key]) != _digest(path):
            parser.error("Ambiguous original font filenames: " + key)
        names[key] = path
    required_fonts = ("simsun.ttc", "simhei.ttf", "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")
    for name in required_fonts:
        if name not in names:
            parser.error("Required original font is missing: " + name)
    published = []
    for relative, record in manifest["files"].items():
        source = here.parent / "_support" / _relative(record["file"])
        if _digest(source) != record["sha256"]:
            raise ValueError("Published figure source differs: " + relative)
        published.append((_relative(relative), source))
    external = []
    for relative, record in manifest["external_files"].items():
        relative = _relative(relative)
        if relative.suffix.lower() in {".py", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".js", ".sh", ".bash", ".ipynb"}:
            raise ValueError("External attachments cannot supply executable figure code: " + str(relative))
        source = data / relative
        if not source.is_file():
            raise FileNotFoundError("Required figure attachment is missing: " + str(source))
        if _digest(source) != record["sha256"]:
            raise ValueError("Figure attachment differs: " + str(source))
        external.append((relative, source))
    result = {"figure": manifest["figure"], "source_files": len(published),
        "external_files": len(external), "source_and_input_hashes": "PASS",
        "plotting": False, "model_loading": 0, "environment_steps": 0}
    if args.check:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("Plotting requires an explicit --output-dir")
    try:
        destination = _output_path(args.output_dir, (here.parent, data, fonts))
    except ValueError as error:
        parser.error(str(error))
    with tempfile.TemporaryDirectory(prefix="thesis-current-figure-") as temporary:
        stage = Path(temporary)
        package, window_fonts = _stage_files(stage, published, external, names)
        entry = package / _relative(manifest["entry"])
        arguments = [sys.executable, "-B", str(entry), "--output-dir", str(destination)]
        if manifest.get("entry_accepts_font_dir"):
            arguments.extend(("--font-dir", str(window_fonts)))
        subprocess.run(arguments, check=True)
        _materialize_outputs(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit("Run a figure directory's plot.py")
