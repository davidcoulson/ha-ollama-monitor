"""Render the brand images of Ollama Monitor.

Usage: python3 assets/brand/render.py
Needs Google Chrome (headless rendering) and ImageMagick (`magick`).

The icons are Ollama's llama (ollama/icon@2x.png and ollama/dark_icon@2x.png, as
published for Home Assistant's core `ollama` integration on brands.home-assistant.io)
with the gauge badge from badge.svg in the bottom-right corner. The logos are the badged
icon followed by an "Ollama Monitor" wordmark. Output goes to
custom_components/ollama_monitor/brand/.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).parent
TARGET = HERE.parent.parent / "custom_components" / "ollama_monitor" / "brand"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WORDMARK = "Ollama Monitor"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"
ICON_SIZE = 512
# The badge spans 38 % of the icon width.
BADGE_SIZE = round(ICON_SIZE * 0.38)
OPTIMIZE = [
    "-strip",
    "-define",
    "png:compression-level=9",
    "-define",
    "png:compression-filter=5",
]


def chrome_screenshot(source: Path, width: int, height: int, out: Path) -> None:
    """Render a file with headless Chrome on a transparent background."""
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--default-background-color=00000000",
            "--force-device-scale-factor=1",
            f"--window-size={width},{height}",
            f"--screenshot={out}",
            source.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def magick(*args: str) -> None:
    """Run ImageMagick."""
    subprocess.run(["magick", *args], check=True)


def build_icons(tmp: Path) -> None:
    """Put the badge on the light and dark llama icons."""
    badge_raw = tmp / "badge_raw.png"
    chrome_screenshot(HERE / "badge.svg", 200, 200, badge_raw)
    badge = tmp / "badge.png"
    magick(
        str(badge_raw), "-filter", "Lanczos", "-resize", f"{BADGE_SIZE}x", str(badge)
    )
    for name in ("icon", "dark_icon"):
        large = TARGET / f"{name}@2x.png"
        magick(
            str(HERE / "ollama" / f"{name}@2x.png"),
            str(badge),
            "-gravity",
            "southeast",
            "-composite",
            *OPTIMIZE,
            f"PNG32:{large}",
        )
        magick(
            str(large),
            "-filter",
            "Lanczos",
            "-resize",
            f"{ICON_SIZE // 2}x{ICON_SIZE // 2}",
            *OPTIMIZE,
            f"PNG32:{TARGET / f'{name}.png'}",
        )


def build_logo(tmp: Path, icon: str, text_color: str, name: str) -> None:
    """Render the badged icon followed by the wordmark, at 256 and 512 px high."""
    shutil.copy(TARGET / f"{icon}@2x.png", tmp / f"{icon}.png")
    page = tmp / f"{name}.svg"
    page.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="2600" height="512" '
        'viewBox="0 0 2600 512">\n'
        f'  <image href="{icon}.png" x="0" y="0" width="512" height="512"/>\n'
        f'  <text x="576" y="256" dominant-baseline="central" font-family="{FONT}" '
        f'font-size="230" font-weight="700" fill="{text_color}">{WORDMARK}</text>\n'
        "</svg>\n"
    )
    raw = tmp / f"{name}_raw.png"
    chrome_screenshot(page, 2600, 512, raw)
    for suffix, height in (("", 256), ("@2x", 512)):
        magick(
            str(raw),
            "-trim",
            "+repage",
            "-filter",
            "Lanczos",
            "-resize",
            f"x{height}",
            *OPTIMIZE,
            f"PNG32:{TARGET / f'{name}{suffix}.png'}",
        )


def main() -> None:
    """Render every brand image."""
    TARGET.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        build_icons(tmp)
        build_logo(tmp, "icon", "#000000", "logo")
        build_logo(tmp, "dark_icon", "#FFFFFF", "dark_logo")


if __name__ == "__main__":
    main()
