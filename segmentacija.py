"""
segmentacija.py — opcionalno izrezivanje pluća (lung bbox)
==========================================================

Slično "diarizaciji" kod zvuka: zaseban pretprocesni korak koji se uključuje/
isključuje prekidačem u main.py:

    SEGMENTACIJA = True   ili   False

Kad je True, prije svega ostalog svaka se snimka izreže na bounding-box pluća
(iz maske segmentatora), pa model ne gleda ramena/rubove/marker. Radi i u
treningu i u inferenciji (isti rez), s keširanjem po putanji.

Treba:  pip install torchxrayvision
        (skine težine pretreniranog segmentatora pluća — ChestX-Det PSPNet)

VAŽNO — prvo provjeri maske (segmentatori su trenirani na odraslima, a ovo je
pedijatrijski set, pa maske znaju zakazati):

    python segmentacija.py provjera Data/test/PNEUMONIA      # spremi par preklopa
    python segmentacija.py provjera Data/test/NORMAL  6

Ako su maske loše -> ostavi SEGMENTACIJA = False. Ako bilo što pukne tijekom
treniranja/inferencije, modul tiho padne natrag na "bez rezanja" (uz jedno
upozorenje), pa nikad ne ruši pipeline.
"""

import os
import numpy as np
from PIL import Image

import main as cfg

_segmenter = None
_cache = {}            # putanja -> bbox (l, g, d, dd) ili None
_UPOZOREN = False
_MARGINA = 0.08        # margina oko pluća (8% širine/visine bboxa)


def aktivna():
    return bool(getattr(cfg, "SEGMENTACIJA", False))


# ----------------------------------------------------------------
# SEGMENTATOR (lazy)
# ----------------------------------------------------------------
def _ucitaj_segmenter():
    global _segmenter
    if _segmenter is not None:
        return _segmenter
    import torch
    import torchxrayvision as xrv
    model = xrv.baseline_models.chestx_det.PSPNet()   # 14 struktura, uklj. lijevo/desno plućno krilo
    model.eval()
    _segmenter = model
    return _segmenter


def _maska_pluca(pil):
    """Vrati binarnu masku pluća (np.bool HxW na originalnoj veličini) ili None."""
    import torch
    import torchxrayvision as xrv
    model = _ucitaj_segmenter()

    arr = np.array(pil.convert("L"), dtype=np.float32)
    arr = xrv.datasets.normalize(arr, 255)            # -> raspon ~[-1024, 1024]
    t = torch.from_numpy(arr)[None, None]             # (1,1,H,W)
    t = torch.nn.functional.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)

    with torch.no_grad():
        out = model(t)                                # (1, C, 512, 512) logiti po strukturi
    mete = [n.lower() for n in getattr(model, "targets", [])]
    idx = [i for i, n in enumerate(mete) if "lung" in n]
    if not idx:
        idx = list(range(out.shape[1]))               # fallback: sve (neće biti idealno)

    plut = (out[0, idx].sigmoid() > 0.5).any(0).cpu().numpy()   # unija L+D krila
    if not plut.any():
        return None
    # natrag na originalnu veličinu
    maska = Image.fromarray((plut * 255).astype(np.uint8)).resize(pil.size, Image.NEAREST)
    return np.array(maska) > 127


def _bbox(putanja, pil):
    if putanja in _cache:
        return _cache[putanja]
    bbox = None
    try:
        maska = _maska_pluca(pil)
        if maska is not None and maska.any():
            ys, xs = np.where(maska)
            l, g, d, dd = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            w, h = pil.size
            mx = int(_MARGINA * max(1, d - l))
            my = int(_MARGINA * max(1, dd - g))
            bbox = (max(0, l - mx), max(0, g - my), min(w, d + mx), min(h, dd + my))
    except Exception as e:                            # txrv nije instaliran ili API drukčiji
        _upozori(e)
    _cache[putanja] = bbox
    return bbox


def mozda_izrezi(pil, putanja):
    """Glavni ulaz: vrati izrezanu sliku ako je segmentacija uključena i uspjela,
    inače originalnu (nikad ne ruši pipeline)."""
    if not aktivna():
        return pil
    bbox = _bbox(putanja, pil)
    if not bbox:
        return pil
    return pil.crop(bbox)


def _upozori(e):
    global _UPOZOREN
    if not _UPOZOREN:
        print(f"  [segmentacija] upozorenje: ne mogu segmentirati ({type(e).__name__}: {e}).")
        print(f"  [segmentacija] nastavljam BEZ izrezivanja. "
              f"(provjeri 'pip install torchxrayvision' ili stavi SEGMENTACIJA=False)")
        _UPOZOREN = True


# ----------------------------------------------------------------
# PROVJERA MASKI — preklopi masku/bbox preko par slika i spremi PNG
#   python segmentacija.py provjera <mapa_ili_slika> [n]
# ----------------------------------------------------------------
def provjeri(putanje, izlaz="provjera_maski", n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # skupi slike
    slike = []
    if isinstance(putanje, str):
        putanje = [putanje]
    for p in putanje:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    slike.append(os.path.join(p, f))
        elif os.path.isfile(p):
            slike.append(p)
    slike = slike[:n]
    if not slike:
        print("Nema slika za provjeru.")
        return

    os.makedirs(izlaz, exist_ok=True)
    for putanja in slike:
        pil = Image.open(putanja).convert("L")
        try:
            maska = _maska_pluca(pil)
        except Exception as e:
            print(f"GRESKA na {os.path.basename(putanja)}: {e}")
            return
        bbox = None
        if maska is not None and maska.any():
            ys, xs = np.where(maska)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(pil, cmap="gray"); ax[0].set_title("original"); ax[0].axis("off")
        ax[1].imshow(pil, cmap="gray")
        if maska is not None:
            ax[1].imshow(maska, alpha=0.35, cmap="autumn")
        if bbox:
            l, g, d, dd = bbox
            ax[1].add_patch(plt.Rectangle((l, g), d - l, dd - g, fill=False, edgecolor="lime", lw=2))
        ax[1].set_title("maska + bbox"); ax[1].axis("off")
        fig.tight_layout()
        izl = os.path.join(izlaz, "maska_" + os.path.splitext(os.path.basename(putanja))[0] + ".png")
        fig.savefig(izl, dpi=110); plt.close(fig)
        print(f"  spremljeno: {izl}")
    print(f"\nProvjeri PNG-ove u '{izlaz}/' — sjeda li maska na pluća. Ako da, SEGMENTACIJA=True ima smisla.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "provjera":
        n = int(sys.argv[3]) if len(sys.argv) >= 4 else 8
        provjeri(sys.argv[2], n=n)
    else:
        print("Koristenje: python segmentacija.py provjera <mapa_ili_slika> [n]")
