"""Generate synthetic rapatriement document variants for OCR accuracy tests."""

import json
import random
from pathlib import Path

import fitz
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF_CANDIDATES = [
    ROOT / "Rapatriment_AVA.pdf",
    ROOT / "doc" / "Rapatriment_AVA.pdf",
    ROOT / "doc" / "Rapatriement_AVA_12960626 (1).pdf",
]
SOURCE_PDF = SOURCE_PDF_CANDIDATES[0]
OUTPUT_DIR = ROOT / "doc" / "test_documents" / "rapatriement_variants"
GROUND_TRUTH_PATH = OUTPUT_DIR / "ground_truth.json"

EXPECTED_DATA = {
    "dateDosRap": "2026-06-08",
    "mntRap": 250.0,
    "typePieceBenef": 1,
    "noPieceBenef": "0004545Y",
}


def render_source_pdf() -> Image.Image:
    doc = fitz.open(SOURCE_PDF)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return image


def add_noise(image: Image.Image, amount: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    noise = Image.new("RGB", image.size)
    pixels = []
    for _ in range(image.size[0] * image.size[1]):
        value = rng.randint(-amount, amount)
        pixels.append((128 + value, 128 + value, 128 + value))
    noise.putdata(pixels)
    return Image.blend(image, noise, min(0.35, amount / 120))


def add_shadow(image: Image.Image, opacity: int = 70) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    draw.ellipse((int(w * 0.08), int(h * 0.05), int(w * 0.8), int(h * 0.45)), fill=(0, 0, 0, opacity))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=90))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_stamp(image: Image.Image, text: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    color = rng.choice([(120, 30, 30), (30, 70, 130), (80, 80, 80)])
    x = rng.randint(int(w * 0.05), int(w * 0.35))
    y = rng.randint(int(h * 0.08), int(h * 0.2))
    draw.rectangle((x - 10, y - 8, x + 370, y + 34), outline=color, width=3)
    draw.text((x, y), text, fill=color)
    return img


def add_extra_data(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x = rng.randint(int(w * 0.52), int(w * 0.65))
    y = rng.randint(int(h * 0.12), int(h * 0.3))
    lines = [
        "REF TEST: AVA-2026-EXTRA",
        "Montant archive: 9999,000",
        "Controle interne: OK",
    ]
    draw.rectangle((x - 8, y - 8, x + 360, y + 72), fill=(250, 250, 245), outline=(170, 170, 170))
    for idx, line in enumerate(lines):
        draw.text((x, y + idx * 20), line, fill=(60, 60, 60))
    return img


def rotate_with_canvas(image: Image.Image, angle: float, fill=(255, 255, 255)) -> Image.Image:
    rotated = image.rotate(angle, expand=True, fillcolor=fill)
    canvas = Image.new("RGB", image.size, fill)
    x = (canvas.width - rotated.width) // 2
    y = (canvas.height - rotated.height) // 2
    canvas.paste(rotated, (x, y))
    return canvas


def crop_and_pad(image: Image.Image, crop_px: int) -> Image.Image:
    w, h = image.size
    cropped = image.crop((crop_px, crop_px, w - crop_px, h - crop_px))
    return ImageOps.pad(cropped, (w, h), color=(255, 255, 255))


def perspective_skew(image: Image.Image, seed: int, strength: float = 0.035) -> Image.Image:
    rng = random.Random(seed)
    w, h = image.size
    xshift = int(w * strength)
    yshift = int(h * strength)
    coeffs = (
        1,
        rng.uniform(-0.025, 0.025),
        rng.randint(-xshift, xshift),
        rng.uniform(-0.015, 0.015),
        1,
        rng.randint(-yshift, yshift),
    )
    return image.transform(image.size, Image.Transform.AFFINE, coeffs, fillcolor=(255, 255, 255))


def color_cast(image: Image.Image, color: tuple[int, int, int], opacity: float) -> Image.Image:
    overlay = Image.new("RGB", image.size, color)
    return Image.blend(image, overlay, opacity)


def threshold_bw(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    return gray.point(lambda p: 255 if p > 170 else 0).convert("RGB")


def save_pdf(image: Image.Image, output_path: Path) -> None:
    image.save(output_path, "PDF", resolution=160.0)


def variant_specs():
    return [
        ("01_clean_reference", ["stamp"]),
        ("02_light_rotation", ["rotate_light"]),
        ("03_heavy_rotation", ["rotate_heavy", "shadow"]),
        ("04_grayscale", ["grayscale"]),
        ("05_black_white_threshold", ["bw"]),
        ("06_low_contrast", ["low_contrast"]),
        ("07_high_contrast", ["high_contrast", "grayscale"]),
        ("08_blur", ["blur"]),
        ("09_noise_light", ["noise_light"]),
        ("10_noise_heavy", ["noise_heavy", "low_contrast"]),
        ("11_yellow_color_cast", ["yellow_cast", "shadow"]),
        ("12_blue_color_cast", ["blue_cast", "extra_data"]),
        ("13_dark_scan", ["dark", "noise_light"]),
        ("14_overexposed_scan", ["bright", "low_contrast"]),
        ("15_slight_crop", ["crop_light"]),
        ("16_skewed_capture", ["skew", "shadow"]),
        ("17_skewed_grayscale", ["skew", "grayscale", "noise_light"]),
        ("18_extra_data_amounts", ["extra_data", "stamp"]),
        ("19_blurry_color_low_quality", ["blur", "yellow_cast", "noise_light", "rotate_light"]),
        ("20_max_degraded", ["rotate_heavy", "skew", "noise_heavy", "shadow", "low_contrast", "extra_data"]),
    ]


def apply_operations(base: Image.Image, operations: list[str], seed: int) -> Image.Image:
    image = base.copy()
    rng = random.Random(seed)

    for operation in operations:
        if operation == "stamp":
            image = add_stamp(image, "COPIE TEST OCR", seed)
        elif operation == "extra_data":
            image = add_extra_data(image, seed)
        elif operation == "rotate_light":
            image = rotate_with_canvas(image, rng.choice([-1.2, 1.5, -2.0, 2.0]))
        elif operation == "rotate_heavy":
            image = rotate_with_canvas(image, rng.choice([-4.5, 4.8, -6.0, 5.5]))
        elif operation == "skew":
            image = perspective_skew(image, seed)
        elif operation == "shadow":
            image = add_shadow(image, opacity=rng.randint(45, 95))
        elif operation == "grayscale":
            image = ImageOps.grayscale(image).convert("RGB")
        elif operation == "bw":
            image = threshold_bw(image)
        elif operation == "low_contrast":
            image = ImageEnhance.Contrast(image).enhance(0.55)
        elif operation == "high_contrast":
            image = ImageEnhance.Contrast(image).enhance(1.9)
        elif operation == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.choice([1.1, 1.5, 2.0])))
        elif operation == "noise_light":
            image = add_noise(image, amount=18, seed=seed)
        elif operation == "noise_heavy":
            image = add_noise(image, amount=45, seed=seed)
        elif operation == "yellow_cast":
            image = color_cast(image, (255, 230, 150), 0.18)
        elif operation == "blue_cast":
            image = color_cast(image, (170, 205, 255), 0.16)
        elif operation == "dark":
            image = ImageEnhance.Brightness(image).enhance(0.62)
        elif operation == "bright":
            image = ImageEnhance.Brightness(image).enhance(1.45)
        elif operation == "crop_light":
            image = crop_and_pad(image, crop_px=35)

    return image


def main() -> None:
    source_pdf = next((path for path in SOURCE_PDF_CANDIDATES if path.exists()), None)
    if source_pdf is None:
        candidates = "\n".join(str(path) for path in SOURCE_PDF_CANDIDATES)
        raise FileNotFoundError(f"Source PDF not found. Checked:\n{candidates}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_pdf in OUTPUT_DIR.glob("*.pdf"):
        old_pdf.unlink()
    global SOURCE_PDF
    SOURCE_PDF = source_pdf
    base = render_source_pdf()
    manifest = []

    for index, (name, operations) in enumerate(variant_specs(), start=1):
        seed = 20260622 + index
        image = apply_operations(base, operations, seed)
        output_name = f"{name}.pdf"
        output_path = OUTPUT_DIR / output_name
        save_pdf(image, output_path)
        manifest.append(
            {
                "file": output_name,
                "source": SOURCE_PDF.name,
                "expected": EXPECTED_DATA,
                "issues": operations,
                "description": "Synthetic degraded variant for OCR/LLM accuracy testing.",
            }
        )

    GROUND_TRUTH_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(manifest)} documents in {OUTPUT_DIR}")
    print(f"Ground truth: {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
