"""
pripremi_rsna.py — presloži RSNA u ImageFolder strukturu (kao stari pedijatrijski set)
=====================================================================================

Iz RSNA (slike + CSV) napravi:

    Data_split/
        train/ NORMAL/  PNEUMONIA/
        val/   NORMAL/  PNEUMONIA/
        test/  NORMAL/  PNEUMONIA/

Klase (iz stage_2_detailed_class_info.csv):
    "Normal"                       -> NORMAL
    "Lung Opacity"                 -> PNEUMONIA
    "No Lung Opacity / Not Normal" -> PRESKAČE SE (čišća binarna podjela)
Ako nema detailed_class_info, pada na stage_2_train_labels.csv (Target 0/1).

Pokretanje:
    python pripremi_rsna.py                 # default: 2500/1500/1000 PO KLASI (=5k/3k/2k)
    python pripremi_rsna.py 2500 1500 1000  # ručno: train val test (po klasi)

main.py s IZVOR_PODATAKA = "imagefolder" (default) koristi train, val i test
točno ovako kako su složeni, pa omjer biraš ovdje brojevima po klasi.
"""

import os
import csv
import glob
import random
import shutil
import sys

# ----------------------------------------------------------------
_PROJEKT = os.path.dirname(os.path.abspath(__file__))
IZVOR_DIR = os.path.join(_PROJEKT, "Data")           # gdje su RSNA slike + .csv
IZLAZ_DIR = os.path.join(_PROJEKT, "Data_split")     # kamo presložiti
SJEME = 42
PODRZANI = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
ISKLJUCI_SREDNJU = True      # preskoči "No Lung Opacity / Not Normal"
# koliko PO KLASI u svaki split (default => 5k train, 3k val, 2k test ukupno):
BROJ = {"train": 2500, "val": 1500, "test": 1000}
# ----------------------------------------------------------------


def _nadji_slike_dir():
    pod = [d for d in glob.glob(os.path.join(IZVOR_DIR, "*")) if os.path.isdir(d)]
    return next((d for d in pod if "train" in os.path.basename(d).lower()),
               pod[0] if pod else IZVOR_DIR)


def _mapa_labela():
    """patientId -> 'NORMAL' / 'PNEUMONIA' / None (preskoči)."""
    detailed = labels = None
    for c in glob.glob(os.path.join(IZVOR_DIR, "*.csv")):
        with open(c, newline="") as fh:
            head = fh.readline()
        if "class" in head.lower():
            detailed = c
        if "Target" in head:
            labels = c

    mapa = {}
    if detailed:
        with open(detailed, newline="") as fh:
            for row in csv.DictReader(fh):
                pid = row["patientId"]
                if pid in mapa:
                    continue
                kl = (row.get("class") or "").strip().lower()
                if kl == "normal":
                    mapa[pid] = "NORMAL"
                elif kl == "lung opacity":
                    mapa[pid] = "PNEUMONIA"
                else:
                    mapa[pid] = None if ISKLJUCI_SREDNJU else "NORMAL"
        print(f"Labele iz: {os.path.basename(detailed)} (3-klasni; srednja "
              f"{'preskočena' if ISKLJUCI_SREDNJU else 'kao NORMAL'})")
    elif labels:
        with open(labels, newline="") as fh:
            for row in csv.DictReader(fh):
                pid = row["patientId"]
                t = int(row.get("Target", 0))
                mapa[pid] = "PNEUMONIA" if (mapa.get(pid) == "PNEUMONIA" or t == 1) else "NORMAL"
        print(f"Labele iz: {os.path.basename(labels)} (Target 0/1)")
    else:
        sys.exit(f"Ne nalazim CSV (s 'class' ili 'Target') u '{IZVOR_DIR}'.")
    return mapa


def main():
    if len(sys.argv) == 4:
        BROJ["train"], BROJ["val"], BROJ["test"] = map(int, sys.argv[1:4])

    random.seed(SJEME)
    slike_dir = _nadji_slike_dir()
    po_stemu = {os.path.splitext(f)[0]: os.path.join(slike_dir, f)
                for f in os.listdir(slike_dir) if f.lower().endswith(PODRZANI)}
    print(f"Slike iz: {slike_dir}  ({len(po_stemu)} datoteka)")

    mapa = _mapa_labela()
    po_klasi = {"NORMAL": [], "PNEUMONIA": []}
    for pid, kl in mapa.items():
        if kl and pid in po_stemu:
            po_klasi[kl].append(po_stemu[pid])

    treba = sum(BROJ.values())
    for kl, lst in po_klasi.items():
        random.shuffle(lst)
        print(f"  {kl}: dostupno {len(lst)} (traženo {treba} po klasi)"
              + ("  ⚠ premalo, uzimam koliko ima" if len(lst) < treba else ""))

    idx = {"NORMAL": 0, "PNEUMONIA": 0}
    ukupno = {}
    for split in ("train", "val", "test"):
        for kl in ("NORMAL", "PNEUMONIA"):
            out = os.path.join(IZLAZ_DIR, split, kl)
            os.makedirs(out, exist_ok=True)
            dio = po_klasi[kl][idx[kl]: idx[kl] + BROJ[split]]
            idx[kl] += len(dio)
            for p in dio:
                shutil.copy2(p, os.path.join(out, os.path.basename(p)))
            ukupno[(split, kl)] = len(dio)

    print("\nGotovo. Raspodjela:")
    for split in ("train", "val", "test"):
        n = ukupno[(split, "NORMAL")]; pn = ukupno[(split, "PNEUMONIA")]
        print(f"  {split:5s}: NORMAL={n:5d}  PNEUMONIA={pn:5d}  (ukupno {n+pn})")
    print(f"\nmain.py s IZVOR_PODATAKA = \"imagefolder\" koristi ovu podjelu ({IZLAZ_DIR}).")


if __name__ == "__main__":
    main()
