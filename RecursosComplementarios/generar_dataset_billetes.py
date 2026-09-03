"""
Genera un dataset sintetico de deteccion (formato YOLO) a partir del dataset
de clasificacion de billetes peruanos (NicolasPCS/Dataset-Billetes-Peruanos).

Cada imagen fuente = un billete recortado al ras (sin fondo). Se pega 1-4
billetes por imagen sintetica sobre un fondo, con rotacion/escala aleatoria,
y se calcula automaticamente el bounding box en formato YOLO.

Uso:
    python generar_dataset_billetes.py --src training_set --out dataset_billetes_yolo --n 2000
"""
import argparse, os, random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# --- Mapeo de carpeta -> class_id y class_id -> valor en soles ---
CLASES = [
    "billete10_anverso_antiguo", "billete10_anverso_nuevo",
    "billete10_reverso_antiguo", "billete10_reverso_nuevo",
    "billete20_anverso_antiguo", "billete20_anverso_nuevo",
    "billete20_reverso_antiguo", "billete20_reverso_nuevo",
    "billete50_anverso_antiguo", "billete50_anverso_nuevo",
    "billete50_reverso_antiguo", "billete50_reverso_nuevo",
    "billete100_anverso_antiguo", "billete100_anverso_nuevo",
    "billete100_reverso_antiguo", "billete100_reverso_nuevo",
]
CLASE_A_VALOR = {i: int(c.split("billete")[1].split("_")[0]) for i, c in enumerate(CLASES)}


def fondo_aleatorio(w, h):
    """Fondo placeholder (madera/tela simple). Reemplazar despues con fotos
    reales de mesas/superficies vacias para mas realismo."""
    base = random.choice([
        (168, 130, 90), (120, 100, 80), (60, 60, 65),
        (200, 190, 170), (90, 70, 55),
    ])
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        arr[:, :, c] = base[c]
    ruido = np.random.randint(-15, 15, (h, w, 3))
    arr = np.clip(arr.astype(int) + ruido, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(2))
    return img


def pegar_billete(canvas, billete_img, ocupadas):
    cw, ch = canvas.size
    escala = random.uniform(0.28, 0.45) * cw / billete_img.width
    bw = int(billete_img.width * escala)
    bh = int(billete_img.height * escala)
    bimg = billete_img.resize((bw, bh))

    angulo = random.uniform(0, 360)
    brot = bimg.rotate(angulo, expand=True, resample=Image.BICUBIC)

    for _ in range(30):
        x = random.randint(0, max(1, cw - brot.width))
        y = random.randint(0, max(1, ch - brot.height))
        caja = (x, y, x + brot.width, y + brot.height)
        solapa = any(not (caja[2] < o[0] or caja[0] > o[2] or caja[3] < o[1] or caja[1] > o[3]) for o in ocupadas)
        if not solapa:
            canvas.paste(brot, (x, y), brot if brot.mode == "RGBA" else None)
            ocupadas.append(caja)
            return caja
    return None


def generar_una(imagenes_por_clase, canvas_size=640):
    canvas = fondo_aleatorio(canvas_size, canvas_size)
    n_billetes = random.randint(1, 4)
    ocupadas = []
    labels = []
    clases_disp = list(imagenes_por_clase.keys())
    for _ in range(n_billetes):
        cls = random.choice(clases_disp)
        ruta = random.choice(imagenes_por_clase[cls])
        billete = Image.open(ruta).convert("RGBA")
        caja = pegar_billete(canvas, billete, ocupadas)
        if caja:
            x0, y0, x1, y1 = caja
            xc = (x0 + x1) / 2 / canvas_size
            yc = (y0 + y1) / 2 / canvas_size
            w = (x1 - x0) / canvas_size
            h = (y1 - y0) / canvas_size
            labels.append(f"{cls_idx(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return canvas.convert("RGB"), labels


def cls_idx(nombre_carpeta):
    return CLASES.index(nombre_carpeta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="carpeta training_set del dataset original")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()

    imagenes_por_clase = {}
    for cls in CLASES:
        carpeta = Path(args.src) / cls
        if carpeta.exists():
            imagenes_por_clase[cls] = [str(p) for p in carpeta.glob("*.jp*g")]

    out_img = Path(args.out) / "images"
    out_lbl = Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        img, labels = generar_una(imagenes_por_clase, args.size)
        img.save(out_img / f"synth_{i:05d}.jpg", quality=90)
        (out_lbl / f"synth_{i:05d}.txt").write_text("\n".join(labels))

    with open(Path(args.out) / "classes.txt", "w") as f:
        f.write("\n".join(CLASES))

    print(f"Generadas {args.n} imagenes sinteticas en {args.out}")


if __name__ == "__main__":
    main()
