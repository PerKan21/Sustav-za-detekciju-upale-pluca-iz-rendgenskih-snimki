"""
evaluacija.py — metrike, grafovi i usporedba modela
===================================================

Ovaj modul poziva treniranje.py tijekom i na kraju treniranja:
  • predvidi / metrike / nadji_optimalni_prag  — brojke,
  • crtaj_*                                     — grafovi (PNG),
  • spremi_izvjestaj_treniranja                 — sve odjednom u mapu
    "Rezultati treniranja/<MODEL>/".
Zasebno se koristi i za `python main.py eval` (usporedba spremljenih modela).

Sve brojke se računaju na test skupu pri zadanom (optimiranom) pragu odluke.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")               # bez GUI-a; samo spremanje u datoteku
import matplotlib.pyplot as plt

import main as cfg
import main as core

# paleta usklađena s GUI-em
_SIVA = "#4b5563"
_CRVENA = "#e02424"
_ZELENA = "#0e9f6e"
_PLAVA = "#2563eb"


# ================================================================
# OSNOVNE METRIKE
# ================================================================
@torch.no_grad()
def predvidi(model, loader, device=None):
    """Vrati (y_true, y_prob) — vjerojatnost pozitivne klase za cijeli loader."""
    device = device or cfg.DEVICE
    model.eval()
    sve_y, sve_p = [], []
    for x, y in loader:
        x = x.to(device)
        logiti = model(x)
        prob = F.softmax(logiti, dim=1)[:, cfg.POZITIVNA_KLASA]
        sve_p.append(prob.cpu().numpy())
        sve_y.append(y.numpy())
    return np.concatenate(sve_y), np.concatenate(sve_p)


def metrike(y_true, y_prob, prag=None):
    """Recall, specifičnost, precision, F1, accuracy, AUC + matrica konfuzije."""
    prag = cfg.PRAG_ODLUKE if prag is None else prag
    y_pred = (y_prob >= prag).astype(int)
    poz = cfg.POZITIVNA_KLASA
    tp = int(np.sum((y_pred == poz) & (y_true == poz)))
    tn = int(np.sum((y_pred != poz) & (y_true != poz)))
    fp = int(np.sum((y_pred == poz) & (y_true != poz)))
    fn = int(np.sum((y_pred != poz) & (y_true == poz)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificnost = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y_true), 1)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"prag": prag, "recall": recall, "specificnost": specificnost, "precision": precision,
            "f1": f1, "accuracy": accuracy, "auc": auc, "TP": tp, "TN": tn, "FP": fp, "FN": fn}


def nadji_optimalni_prag(y_true, y_prob, recall_pod=None):
    """
    Youdenova točka: prag koji maksimizira (tpr − fpr), tj. najbolji balans
    recall/specifičnost — bez fiksiranja proizvoljnog cilja.
    recall_pod: opcionalni donji prag osjetljivosti; Youden se tada bira samo
                među točkama gdje je recall ≥ recall_pod (medicinski kontekst).
    """
    recall_pod = getattr(cfg, "RECALL_POD", None) if recall_pod is None else recall_pod
    fpr, tpr, pragovi = roc_curve(y_true, y_prob, pos_label=cfg.POZITIVNA_KLASA)
    konacni = np.isfinite(pragovi)
    fpr, tpr, pragovi = fpr[konacni], tpr[konacni], pragovi[konacni]
    if len(pragovi) == 0:
        return cfg.PRAG_ODLUKE
    j = tpr - fpr
    if recall_pod:
        maska = tpr >= recall_pod
        if maska.any():
            j = np.where(maska, j, -np.inf)   # poštuj recall-pod ako je ostvariv
    return float(pragovi[int(np.argmax(j))])


# ================================================================
# GRAFOVI
# ================================================================
def crtaj_povijest(povijest, putanja, tip):
    """povijest: list dict-ova {epoha, train_loss, val_loss, recall, spec, train_recall}."""
    if not povijest:
        return
    epohe = [h["epoha"] for h in povijest]
    tl = [h["train_loss"] for h in povijest]
    vl = [h["val_loss"] for h in povijest]
    rec = [h["recall"] * 100 for h in povijest]
    spec = [h["spec"] * 100 for h in povijest]
    trec = [h.get("train_recall", float("nan")) * 100 for h in povijest]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.6), sharex=True)
    ax1.plot(epohe, tl, "-o", ms=3, color=_SIVA, label="train loss")
    ax1.plot(epohe, vl, "-o", ms=3, color=_CRVENA, label="val loss")
    ax1.set_ylabel("Gubitak")
    ax1.set_title(f"Tijek treniranja — {tip.upper()}")
    ax1.legend(); ax1.grid(alpha=.3)

    ax2.plot(epohe, rec, "-o", ms=3, color=_ZELENA, label="val recall")
    ax2.plot(epohe, trec, "--o", ms=3, color=_ZELENA, alpha=0.45, label="train recall")
    ax2.plot(epohe, spec, "-o", ms=3, color=_PLAVA, label="val specifičnost")
    ax2.set_ylabel("%"); ax2.set_xlabel("Epoha"); ax2.set_ylim(0, 100)
    ax2.legend(); ax2.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(putanja, dpi=130)
    plt.close(fig)


def crtaj_matricu_konfuzije(m, putanja, tip):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    klase = ["NORMAL", "PNEUMONIA"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(klase)
    ax.set_yticks([0, 1]); ax.set_yticklabels(klase, rotation=90, va="center")
    ax.set_xlabel("Predviđeno"); ax.set_ylabel("Stvarno")
    ax.set_title(f"Matrica konfuzije — {tip.upper()}")
    vmax = cm.max() if cm.max() else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if cm[i, j] > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(putanja, dpi=130)
    plt.close(fig)


def crtaj_roc(y_true, y_prob, putanja, tip):
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=cfg.POZITIVNA_KLASA)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot(fpr, tpr, color=_SIVA, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
    ax.set_xlabel("FPR (1 − specifičnost)")
    ax.set_ylabel("TPR (recall)")
    ax.set_title(f"ROC krivulja — {tip.upper()}")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(putanja, dpi=130)
    plt.close(fig)


def spremi_metrike_txt(m, putanja, tip, prag):
    with open(putanja, "w", encoding="utf-8") as f:
        f.write(f"METRIKE — {tip.upper()} (test skup)\n")
        f.write("=" * 48 + "\n")
        f.write(f"Prag odluke           : {prag:.3f}\n")
        f.write(f"Recall (osjetljivost) : {m['recall']*100:.2f} %\n")
        f.write(f"Specifičnost          : {m['specificnost']*100:.2f} %\n")
        f.write(f"Precision             : {m['precision']*100:.2f} %\n")
        f.write(f"F1                    : {m['f1']*100:.2f} %\n")
        f.write(f"Accuracy              : {m['accuracy']*100:.2f} %\n")
        f.write(f"ROC-AUC               : {m['auc']:.4f}\n\n")
        f.write("Matrica konfuzije (redak = stvarno, stupac = predviđeno):\n")
        f.write(f"             NORMAL   PNEUMONIA\n")
        f.write(f"  NORMAL     {m['TN']:>6}   {m['FP']:>9}\n")
        f.write(f"  PNEUMONIA  {m['FN']:>6}   {m['TP']:>9}\n")


# ================================================================
# IZVJEŠTAJ TRENIRANJA — poziva ga treniranje.py
# ================================================================
def spremi_izvjestaj_treniranja(tip, povijest, y_true_test, y_prob_test, prag, izlazna_mapa=None):
    """
    Generira i sprema: povijest_treniranja.png, matrica_konfuzije.png,
    roc_krivulja.png i metrike.txt u 'Rezultati treniranja/<MODEL>/'.
    Vraća (test_metrike, mapa).
    """
    baza = getattr(cfg, "DIR_REZULTATI_TRENIRANJA", "Rezultati treniranja")
    izlazna_mapa = izlazna_mapa or os.path.join(baza, tip.upper())
    os.makedirs(izlazna_mapa, exist_ok=True)

    m = metrike(y_true_test, y_prob_test, prag)
    crtaj_povijest(povijest, os.path.join(izlazna_mapa, "povijest_treniranja.png"), tip)
    crtaj_matricu_konfuzije(m, os.path.join(izlazna_mapa, "matrica_konfuzije.png"), tip)
    crtaj_roc(y_true_test, y_prob_test, os.path.join(izlazna_mapa, "roc_krivulja.png"), tip)
    spremi_metrike_txt(m, os.path.join(izlazna_mapa, "metrike.txt"), tip, prag)
    return m, izlazna_mapa


# ================================================================
# USPOREDBA SPREMLJENIH MODELA  (python main.py eval)
# ================================================================
def usporedi():
    import treniranje as ui
    ui.hint_rich()
    _, _, test_loader, _, _ = core.napravi_dataloadere(verbose=False)
    rezultati = {}
    ui.naslov("Usporedba modela na test skupu")
    for tip in ("cnn", "vgg16"):
        putanja = cfg.PUTANJA_MODEL[tip]
        if not os.path.exists(putanja):
            ui.upozorenje(f"preskačem {tip}: nema {putanja} (prvo istreniraj).")
            continue
        model, meta = core.ucitaj_model(tip, putanja)
        y_true, y_prob = predvidi(model, test_loader)
        prag = meta.get("prag", cfg.PRAG_ODLUKE)
        rezultati[tip] = metrike(y_true, y_prob, prag)
        # uz usporedbu spremimo i grafove svakog modela
        baza = getattr(cfg, "DIR_REZULTATI_TRENIRANJA", "Rezultati treniranja")
        mapa = os.path.join(baza, tip.upper())
        os.makedirs(mapa, exist_ok=True)
        crtaj_matricu_konfuzije(rezultati[tip], os.path.join(mapa, "matrica_konfuzije.png"), tip)
        crtaj_roc(y_true, y_prob, os.path.join(mapa, "roc_krivulja.png"), tip)

    if rezultati:
        redovi = []
        for tip, m in rezultati.items():
            redovi.append([tip.upper(), f"{m['recall']*100:.2f}%", f"{m['specificnost']*100:.2f}%",
                          f"{m['precision']*100:.2f}%", f"{m['f1']*100:.2f}%",
                          f"{m['accuracy']*100:.2f}%", f"{m['auc']:.4f}",
                          f"TP={m['TP']} FN={m['FN']} FP={m['FP']} TN={m['TN']}"])
        ui.tablica("Metrike", ["Model", "Recall", "Spec.", "Precision", "F1", "Accuracy", "AUC", "Matrica"], redovi)
    return rezultati
