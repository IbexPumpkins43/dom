"""Build a release of dom for the current OS/CPU: a packaged app with llama.cpp included.

Steps:

1. Download (and SHA-256-verify) the llama.cpp CPU and Vulkan builds for this platform, using the
   same code as the app (``dom.downloads``). Archives are cached in ``build/downloads``.
2. Package dom with PyInstaller (one folder, no console window).
3. Put the llama.cpp builds in ``llama/`` next to the executable (where the app looks for a
   bundled engine) and collect third-party licence texts.
4. Linux: ``dist/dom-<version>-linux-<arch>.tar.gz`` with ``install.sh``/``uninstall.sh``.
   Windows: ``dist/dom-<version>-windows-<arch>-setup.exe`` via Inno Setup (``iscc``) if it is
   installed, otherwise a ``.zip`` of the app folder.

PyInstaller cannot cross-compile: run this on each target (see .github/workflows/release.yml).

Run with ``uv run python scripts/build_release.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import logging
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from dom.config.paths import BUNDLED_ENGINE_DIR
from dom.downloads import DownloadError, download_verified, make_client
from dom.downloads.http import sha256_of
from dom.downloads.llama_release import (
    LLAMA_LICENSE_FILE,
    Target,
    detect_target,
    find_releases,
    install_release,
)

logger = logging.getLogger("build_release")

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"
APP_NAME = "dom"
# Qt modules dom does not use; leaving them out keeps the download smaller.
EXCLUDED_MODULES = (
    "tkinter",
    "PySide6.QtNetwork",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtPdf",
    "PySide6.QtDBus",
    "PySide6.QtConcurrent",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtSvgWidgets",
    "PySide6.QtXml",
)
# Third-party packages whose licences must ship with the app.
BUNDLED_DISTRIBUTIONS = (
    "pyside6-essentials",
    "shiboken6",
    "qasync",
    "httpx",
    "httpcore",
    "h11",
    "anyio",
    "idna",
    "certifi",
    "sniffio",
    "platformdirs",
    "psutil",
)


async def fetch_engine(target: Target) -> Path:
    """Download and unpack every llama.cpp build for ``target`` into ``build/engine``."""
    engine = BUILD / "engine"
    async with make_client() as client:
        for release in await find_releases(client, target):
            archive = BUILD / "downloads" / release.asset.name
            if not (archive.is_file() and sha256_of(archive) == release.asset.sha256):
                logger.info("Downloading %s", release.asset.name)
                await download_verified(client, release.asset, archive)
            dest = await asyncio.to_thread(install_release, archive, engine, release)
            logger.info("llama.cpp %s %s -> %s", release.version_label, release.variant, dest)
    return engine


def run_pyinstaller() -> Path:
    """Package dom; returns the app folder."""
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--windowed",
        "--name", APP_NAME,
        "--paths", str(ROOT / "src"),
        "--collect-submodules", "dom",
        "--collect-data", "dom",  # the logo and other assets
        "--distpath", str(BUILD / "pyinstaller"),
        "--workpath", str(BUILD / "work"),
        "--specpath", str(BUILD),
    ]  # fmt: skip
    for module in EXCLUDED_MODULES:
        command += ["--exclude-module", module]
    command.append(str(ROOT / "src" / "dom" / "__main__.py"))
    logger.info("Running PyInstaller")
    subprocess.run(command, check=True, cwd=ROOT)
    return BUILD / "pyinstaller" / APP_NAME


def write_notices(app_dir: Path, engine: Path) -> None:
    """Licence texts for dom and everything bundled with it."""
    notices = app_dir / "licenses"
    notices.mkdir(exist_ok=True)
    shutil.copyfile(ROOT / "LICENSE", notices / "dom-LICENSE.txt")
    for variant_dir in sorted(p for p in engine.iterdir() if p.is_dir()):
        llama_license = variant_dir / LLAMA_LICENSE_FILE
        if llama_license.is_file():
            shutil.copyfile(llama_license, notices / "llama.cpp-LICENSE.txt")
            break
    for name in BUNDLED_DISTRIBUTIONS:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        files = [
            f for f in dist.files or [] if "licen" in f.name.lower() or "copying" in f.name.lower()
        ]
        for index, file in enumerate(files):
            suffix = f"-{index}" if index else ""
            shutil.copyfile(Path(str(file.locate())), notices / f"{name}-LICENSE{suffix}.txt")
    # PySide6/shiboken6 wheels ship no licence file; they are LGPL-3.0 (which builds on GPL-3.0).
    for text in ("LGPL-3.0.txt", "GPL-3.0.txt"):
        shutil.copyfile(ROOT / "packaging" / "licenses" / text, notices / f"PySide6-Qt-{text}")
    shutil.copyfile(
        ROOT / "src" / "dom" / "ui" / "assets" / "fonts" / "OFL.txt", notices / "Figtree-OFL.txt"
    )
    (notices / "README.txt").write_text(
        "dom is MIT-licensed. It bundles llama.cpp (MIT), Qt for Python / PySide6 (LGPLv3; the\n"
        "Qt libraries are shipped as separate, replaceable shared libraries), Python (PSF), the\n"
        "Figtree font (SIL Open Font License 1.1) and the Python packages whose licences are in\n"
        "this folder. The dom logo is drawn from the Climate Crisis font (SIL OFL 1.1).\n",
        encoding="utf-8",
    )


def package_linux(app_dir: Path, version: str, target: Target) -> Path:
    """``dom-<version>-<target>.tar.gz`` with install/uninstall scripts and a menu entry."""
    DIST.mkdir(exist_ok=True)
    out = DIST / f"{APP_NAME}-{version}-{target.value}.tar.gz"
    top = f"{APP_NAME}-{version}"
    with tarfile.open(out, "w:gz") as tf:
        tf.add(app_dir, arcname=f"{top}/{APP_NAME}")
        for extra in ("install.sh", "uninstall.sh", "dom.desktop"):
            tf.add(ROOT / "packaging" / "linux" / extra, arcname=f"{top}/{extra}")
    return out


def package_windows(app_dir: Path, version: str, target: Target) -> Path:
    """An Inno Setup installer if ``iscc`` is available, else a zip of the app folder."""
    DIST.mkdir(exist_ok=True)
    arch = "arm64" if target == Target.WINDOWS_ARM64 else "x64"
    iscc = shutil.which("iscc") or shutil.which("ISCC")
    if iscc is None:
        default = Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe")
        iscc = str(default) if default.is_file() else None
    if iscc is None:
        logger.warning("Inno Setup (iscc) not found: building a zip instead of an installer")
        archive = shutil.make_archive(
            str(DIST / f"{APP_NAME}-{version}-windows-{arch}"), "zip", app_dir.parent, app_dir.name
        )
        return Path(archive)
    subprocess.run(
        [
            iscc,
            f"/DAppVersion={version}",
            f"/DSourceDir={app_dir}",
            f"/DArch={arch}",
            f"/DOutputDir={DIST}",
            str(ROOT / "packaging" / "windows" / "dom.iss"),
        ],
        check=True,
    )
    return DIST / f"{APP_NAME}-{version}-windows-{arch}-setup.exe"


def main() -> int:
    """Build the release for this platform. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Build a dom release for this OS/CPU.")
    parser.add_argument("--skip-package", action="store_true", help="stop after the app folder")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    version = importlib.metadata.version(APP_NAME)
    try:
        target = detect_target()
        logger.info("Building dom %s for %s", version, target)
        engine = asyncio.run(fetch_engine(target))
    except DownloadError as exc:
        logger.error("Error: %s", exc)
        return 1
    app_dir = run_pyinstaller()
    bundled = app_dir / BUNDLED_ENGINE_DIR
    if bundled.exists():
        shutil.rmtree(bundled)
    shutil.copytree(engine, bundled, symlinks=True)
    write_notices(app_dir, engine)
    if args.skip_package:
        logger.info("App folder: %s", app_dir)
        return 0
    if target in (Target.WINDOWS_X64, Target.WINDOWS_ARM64):
        out = package_windows(app_dir, version, target)
    else:
        out = package_linux(app_dir, version, target)
    logger.info("Built %s (%.0f MB)", out, out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
