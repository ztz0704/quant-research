"""Make review sheets of the exact ML report figures, without changing them."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    project = root / "nonstationary_financial_ml"
    manifest = json.loads((project / "outputs/tables/reproducibility_manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["figure_sha256"])
    output = Path(__file__).resolve().parent / "outputs/figure_review"
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    records = []
    for offset in range(0, len(names), 8):
        sheet = Image.new("RGB", (1280, 1680), "#e5e7eb")
        draw = ImageDraw.Draw(sheet)
        for slot, name in enumerate(names[offset:offset + 8]):
            source = project / "outputs/figures" / name
            with Image.open(source) as original:
                original.load()
                records.append({"name": name, "width": original.width, "height": original.height})
                preview = original.convert("RGB")
            preview.thumbnail((620, 370), Image.Resampling.LANCZOS)
            x, y = (slot % 2) * 640, (slot // 2) * 420
            sheet.paste(preview, (x + (640 - preview.width) // 2, y + 10))
            draw.text((x + 12, y + 385), f"{offset + slot + 1:02d} {name}", fill="#111827", font=font)
        target = output / f"review_{offset // 8 + 1}.jpg"
        sheet.save(target, quality=95)
        print(target)
    (output / "image_dimensions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
