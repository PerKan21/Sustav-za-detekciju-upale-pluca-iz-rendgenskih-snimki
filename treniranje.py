"""
treniranje.py — treniranje modela + terminalni prikaz
=====================================================

Ova datoteka spaja bivši cli_ui.py (moderan terminalni prikaz: spinneri,
progress trake, tablice) i logiku treniranja u jedno mjesto.

Pokretanje:
    python main.py train vgg16     # ili: cnn / oba
    python treniranje.py           # trenira oba (cnn + vgg16)

Animirani progress bar i u PyCharmu:
    PyCharm "Run" konzola nije pravi terminal (TTY), pa rich inače gasi
    animaciju. Ako se detektira PyCharm okruženje (env varijabla PYCHARM*),
    rich konzola se tjera u terminal način (force_terminal=True) i traka se
    uredno renderira. Ručni override:
        CLI_UI_FORCE=1   -> uvijek animirano
        CLI_UI_FORCE=0   -> nikad (npr. kad preusmjeravaš izlaz u datoteku)

Ostatak projekta:
    main.py / evaluacija.py uvoze prikaz preko `import treniranje as ui`.
"""

import os
import sys
import copy
import math
import time


# ================================================================
# RICH KONZOLA  (s PyCharm-friendly detekcijom)
# ================================================================
def _odluci_force_terminal():
    """True -> tjeraj animaciju; False -> nikad; None -> neka rich sam odluči."""
    izbor = os.environ.get("CLI_UI_FORCE")
    if izbor in ("0", "false", "False"):
        return False
    if izbor in ("1", "true", "True"):
        return True
    # PyCharm Run/Debug konzola postavlja PYCHARM* varijable, a nije TTY.
    if any(k.startswith("PYCHARM") for k in os.environ):
        return True
    return None


try:
    from rich.console import Console
    from rich.progress import (Progress, SpinnerColumn, BarColumn, TextColumn,
                               TimeElapsedColumn)
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    _RICH = True
    _console = Console(force_terminal=_odluci_force_terminal())
except Exception:                       # rich nije instaliran
    _RICH = False
    _console = None

_HINT_PRIKAZAN = False


def hint_rich():
    """Jednom ispiše prijedlog za instalaciju rich-a (samo ako nije prisutan)."""
    global _HINT_PRIKAZAN
    if not _RICH and not _HINT_PRIKAZAN:
        print("  (savjet: `pip install rich` za ljepši terminalni prikaz)\n")
        _HINT_PRIKAZAN = True


# ================================================================
# OSNOVNI ISPIS
# ================================================================
def naslov(tekst, podnaslov=""):
    """Veliki, uokvireni naslov sekcije."""
    if _RICH:
        sadrzaj = Text(tekst, style="bold white")
        if podnaslov:
            sadrzaj.append(f"\n{podnaslov}", style="dim")
        _console.print(Panel(sadrzaj, box=box.ROUNDED, border_style="cyan",
                             padding=(0, 2)))
    else:
        crta = "=" * 64
        print(f"\n{crta}\n  {tekst}")
        if podnaslov:
            print(f"  {podnaslov}")
        print(crta)


def info(tekst, stil="dim"):
    if _RICH:
        _console.print(f"  {tekst}", style=stil)
    else:
        print(f"  {tekst}")


def uspjeh(tekst):
    if _RICH:
        _console.print(f"  ✓ {tekst}", style="bold green")
    else:
        print(f"  [OK] {tekst}")


def upozorenje(tekst):
    if _RICH:
        _console.print(f"  ! {tekst}", style="bold yellow")
    else:
        print(f"  [!] {tekst}")


def greska(tekst):
    if _RICH:
        _console.print(f"  ✗ {tekst}", style="bold red")
    else:
        print(f"  [GRESKA] {tekst}")


# ================================================================
# PROGRESS TRAKA (jedna epoha / skup datoteka)
# ================================================================
class _RichTraka:
    def __init__(self, opis, ukupno):
        self.ukupno = ukupno
        self._opis = opis
        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=32, complete_style="cyan", finished_style="green"),
            TextColumn("{task.completed:>4}/{task.total:<4}"),
            TextColumn("[dim]{task.fields[info]}"),
            TimeElapsedColumn(),
            console=_console,
            refresh_per_second=8,
            transient=True,            # nestane nakon završetka -> čist ispis
        )

    def __enter__(self):
        self.progress.start()
        self.task = self.progress.add_task(self._opis, total=self.ukupno, info="")
        return self

    def korak(self, info=""):
        self.progress.update(self.task, advance=1, info=info)

    def __exit__(self, *a):
        self.progress.stop()
        return False


class _PlainTraka:
    """ASCII traka s \\r — za pravi terminal bez rich-a."""
    def __init__(self, opis, ukupno, duljina=32):
        self.opis = opis
        self.ukupno = max(1, ukupno)
        self.duljina = duljina
        self.i = 0
        self.t0 = time.time()

    def __enter__(self):
        return self

    def korak(self, info=""):
        self.i += 1
        udio = self.i / self.ukupno
        pun = int(self.duljina * udio)
        bar = "█" * pun + "░" * (self.duljina - pun)
        proteklo = time.time() - self.t0
        print(f"\r  {self.opis} |{bar}| {self.i:>4}/{self.ukupno:<4} "
              f"{info}  {proteklo:4.0f}s", end="", flush=True)

    def __exit__(self, *a):
        print("\r" + " " * 100, end="\r")    # očisti liniju
        return False


class _DumbTraka:
    """Newline ispis svakih ~10% — za konzole koje ne podnose \\r ni rich."""
    def __init__(self, opis, ukupno, koraka=10):
        self.opis = opis
        self.ukupno = max(1, ukupno)
        self.i = 0
        self.prag = max(1, self.ukupno // koraka)
        self.t0 = time.time()

    def __enter__(self):
        print(f"  {self.opis}: 0/{self.ukupno}", flush=True)
        return self

    def korak(self, info=""):
        self.i += 1
        if self.i % self.prag == 0 or self.i == self.ukupno:
            pct = 100 * self.i / self.ukupno
            print(f"  {self.opis}: {self.i}/{self.ukupno} ({pct:3.0f}%)  "
                  f"{info}  {time.time()-self.t0:4.0f}s", flush=True)

    def __exit__(self, *a):
        return False


def traka(opis, ukupno):
    """Kontekst-menadžer s metodom .korak(info='')."""
    if _RICH:
        return _RichTraka(opis, ukupno)
    if sys.stdout.isatty():
        return _PlainTraka(opis, ukupno)
    return _DumbTraka(opis, ukupno)


# ================================================================
# TABLICA (npr. sažetak metrika)
# ================================================================
def tablica(naziv, zaglavlja, redovi):
    """
    naziv: naslov tablice
    zaglavlja: list[str]
    redovi: list[list] (svaki red iste duljine kao zaglavlja)
    """
    if _RICH:
        t = Table(title=naziv, box=box.SIMPLE_HEAVY, title_style="bold white",
                  header_style="bold cyan")
        for i, z in enumerate(zaglavlja):
            t.add_column(z, justify="left" if i == 0 else "right")
        for r in redovi:
            t.add_row(*[str(c) for c in r])
        _console.print(t)
    else:
        print(f"\n{naziv}")
        sirine = [max(len(str(zaglavlja[i])),
                      *(len(str(r[i])) for r in redovi)) if redovi else len(str(zaglavlja[i]))
                  for i in range(len(zaglavlja))]
        linija = "  ".join(str(zaglavlja[i]).ljust(sirine[i]) for i in range(len(zaglavlja)))
        print("  " + linija)
        print("  " + "-" * len(linija))
        for r in redovi:
            print("  " + "  ".join(str(r[i]).ljust(sirine[i]) for i in range(len(r))))


# ================================================================
# TRENIRANJE — ovisi o core (podaci/modeli) i evaluacija (metrike/grafovi)
# ================================================================
import torch
import torch.nn as nn

import main as cfg
import main as core
import evaluacija as ev


def _epoha_trening(model, loader, kriterij, optimizer, scaler, device, prefiks=""):
    model.train()
    ukupni_gubitak = 0.0
    vidjeno = 0
    with traka(prefiks, len(loader)) as bar:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    izlaz = model(x)
                    gubitak = kriterij(izlaz, y)
                scaler.scale(gubitak).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                izlaz = model(x)
                gubitak = kriterij(izlaz, y)
                gubitak.backward()
                optimizer.step()
            ukupni_gubitak += gubitak.item() * x.size(0)
            vidjeno += x.size(0)
            bar.korak(info=f"loss={ukupni_gubitak / max(vidjeno, 1):.4f}")
    return ukupni_gubitak / len(loader.dataset)


@torch.no_grad()
def _epoha_validacija(model, loader, kriterij, device):
    model.eval()
    ukupni_gubitak = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        izlaz = model(x)
        ukupni_gubitak += kriterij(izlaz, y).item() * x.size(0)
    return ukupni_gubitak / len(loader.dataset)


def _odaberi_index(n):
    """Pitaj korisnika broj 1..n (Enter ili nevažeće -> 1). Radi i ne-interaktivno."""
    try:
        s = input(f"  Odaberi model 1-{n} (Enter = #1): ").strip()
    except (EOFError, KeyboardInterrupt):
        s = ""
    if not s:
        return 0
    try:
        v = int(s)
        if 1 <= v <= n:
            return v - 1
    except ValueError:
        pass
    info("nevažeći unos -> uzimam #1")
    return 0


def _novi_optimizer(model, lr):
    parametri = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(parametri, lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    return opt, sched


def treniraj(tip, train_loader, val_loader, test_loader, tezine_klasa, train_eval_loader=None):
    device = cfg.DEVICE
    naslov(f"Treniranje modela: {tip.upper()}", f"uređaj: {device}")

    model = core.napravi_model(tip).to(device)
    tez = tezine_klasa.to(device) if tezine_klasa is not None else None   # sampler vec balansira
    kriterij = nn.CrossEntropyLoss(weight=tez)
    lr = cfg.LR_VGG if tip == "vgg16" else cfg.LR_CNN
    optimizer, scheduler = _novi_optimizer(model, lr)
    scaler = torch.amp.GradScaler("cuda") if (cfg.KORISTI_AMP and device == "cuda") else None

    # VGG dvofazno: faza 2 odmrzava dio baze na zadanoj epohi
    faza2 = (tip == "vgg16" and getattr(cfg, "VGG_FINE_TUNE", False))
    odmrznuto = False

    najbolji_val = float("inf")
    bez_napretka = 0
    povijest = []
    top = []                            # kandidati: {epoha, val_loss, tezine, y_true, y_prob}

    for epoha in range(1, cfg.EPOCHS + 1):
        # --- faza 2 (samo VGG): odmrzni i spusti LR ---
        if faza2 and not odmrznuto and epoha >= getattr(cfg, "VGG_ODMRZNI_NA", 6):
            n = core.odmrzni_vgg(model, getattr(cfg, "VGG_ODMRZNI_BLOKOVA", 1))
            optimizer, scheduler = _novi_optimizer(model, getattr(cfg, "FINE_TUNE_LR", 1e-5))
            odmrznuto = True
            info(f"FAZA 2: odmrznuto zadnjih {getattr(cfg,'VGG_ODMRZNI_BLOKOVA',1)} blok(ova) "
                 f"({n} modula) · LR={getattr(cfg,'FINE_TUNE_LR',1e-5):.0e}", stil="cyan")

        t0 = time.time()
        train_gubitak = _epoha_trening(model, train_loader, kriterij, optimizer, scaler, device,
                                       prefiks=f"epoha {epoha:02d}/{cfg.EPOCHS}")
        val_gubitak = _epoha_validacija(model, val_loader, kriterij, device)
        y_true, y_prob = ev.predvidi(model, val_loader, device)
        m = ev.metrike(y_true, y_prob, cfg.PRAG_ODLUKE)
        if train_eval_loader is not None:
            yt_tr, yp_tr = ev.predvidi(model, train_eval_loader, device)
            mtr = ev.metrike(yt_tr, yp_tr, cfg.PRAG_ODLUKE)
        else:
            mtr = {"recall": float("nan"), "auc": float("nan")}
        scheduler.step(val_gubitak)

        # top-N po val gubitku (čuvamo i val predikcije za kasniji prag/metrike)
        top.append({"epoha": epoha, "val_loss": val_gubitak,
                    "tezine": copy.deepcopy(model.state_dict()),
                    "y_true": y_true, "y_prob": y_prob,
                    "faza2": odmrznuto})
        top.sort(key=lambda d: d["val_loss"])
        del top[getattr(cfg, "TOP_N_MODELA", 5):]
        je_najbolji = top[0]["epoha"] == epoha

        info(f"epoha {epoha:02d}/{cfg.EPOCHS} · train={train_gubitak:.4f} · val={val_gubitak:.4f} · "
             f"recall(tr/val)={mtr['recall']*100:4.1f}/{m['recall']*100:4.1f}% · "
             f"spec={m['specificnost']*100:5.1f}% · AUC(tr/val)={mtr['auc']:.3f}/{m['auc']:.3f} "
             f"({time.time()-t0:.0f}s) {'✓ top' if je_najbolji else ''}",
             stil="green" if je_najbolji else "dim")
        povijest.append({"epoha": epoha, "train_loss": train_gubitak, "val_loss": val_gubitak,
                         "recall": m["recall"], "spec": m["specificnost"],
                         "train_recall": mtr["recall"]})

        if val_gubitak < najbolji_val - 1e-4:
            najbolji_val = val_gubitak
            bez_napretka = 0
        else:
            bez_napretka += 1
            if bez_napretka >= cfg.EARLY_STOP_STRPLJENJE:
                upozorenje(f"early stopping (nema napretka {bez_napretka} epoha).")
                break

    # --- za svaki top kandidat: Youdenov prag + val metrike (iz spremljenih predikcija) ---
    for k in top:
        k["prag"] = ev.nadji_optimalni_prag(k["y_true"], k["y_prob"])
        k["mval"] = ev.metrike(k["y_true"], k["y_prob"], k["prag"])

    tablica(f"Top {len(top)} modela (validacija) — {tip.upper()}",
            ["#", "Epoha", "Val loss", "Prag", "Val recall", "Val spec.", "Val AUC", "Faza"],
            [[i + 1, k["epoha"], f"{k['val_loss']:.4f}", f"{k['prag']:.3f}",
              f"{k['mval']['recall']*100:.1f}%", f"{k['mval']['specificnost']*100:.1f}%",
              f"{k['mval']['auc']:.3f}", "2" if k.get("faza2") else "1"]
             for i, k in enumerate(top)])

    # (opcionalno) spremi svih top-N na disk — VGG je velik (~0.5GB/komad), pa je default OFF.
    # Top-5 i bez toga ostaju u memoriji za izbor; na disk ide samo odabrani.
    if getattr(cfg, "SPREMI_SVE_TOP", False):
        try:
            for i, k in enumerate(top, 1):
                p = cfg.PUTANJA_MODEL[tip].replace("_best", f"_top{i}")
                model.load_state_dict(k["tezine"])
                core.spremi_model(model, p, meta={"tip": tip, "prag": k["prag"],
                                                  "val_loss": k["val_loss"], "epoha": k["epoha"]})
            info(f"Spremljeno top-{len(top)} ({tip}_top1..{len(top)}.pth)")
        except Exception as e:
            upozorenje(f"ne mogu spremiti sve top modele ({type(e).__name__}: {e}) — "
                       f"spremam samo odabrani. (provjeri slobodan prostor na disku)")

    # izbor (ručno ili automatski #1)
    izbor = _odaberi_index(len(top)) if getattr(cfg, "IZBOR_MODELA", True) else 0
    odabran = top[izbor]
    model.load_state_dict(odabran["tezine"])
    prag = odabran["prag"]
    try:
        core.spremi_model(model, cfg.PUTANJA_MODEL[tip],
                          meta={"tip": tip, "prag": prag, "val_loss": odabran["val_loss"],
                                "epoha": odabran["epoha"]})
        uspjeh(f"Odabran #{izbor+1} (epoha {odabran['epoha']}) -> spremljen kao {cfg.PUTANJA_MODEL[tip]}")
    except Exception as e:
        upozorenje(f"spremanje modela NIJE uspjelo ({type(e).__name__}: {e}). "
                   f"Provjeri slobodan prostor na disku / da projekt nije u OneDrive sync mapi. "
                   f"Nastavljam s evaluacijom iz memorije.")

    mval = odabran["mval"]
    pod = getattr(cfg, "RECALL_POD", None)
    info(f"Youdenov prag (val{f', recall ≥ {pod}' if pod else ''}): {prag:.3f} · "
         f"val recall={mval['recall']*100:.1f}% spec={mval['specificnost']*100:.1f}%")

    # test skup + grafovi/metrike u "Rezultati treniranja/<MODEL>/"
    yt, yp = ev.predvidi(model, test_loader, device)
    mtest, mapa = ev.spremi_izvjestaj_treniranja(tip, povijest, yt, yp, prag)

    redovi_pov = [[h["epoha"], f"{h['train_loss']:.4f}", f"{h['val_loss']:.4f}",
                   f"{h.get('train_recall', float('nan'))*100:.1f}%",
                   f"{h['recall']*100:.1f}%", f"{h['spec']*100:.1f}%"] for h in povijest]
    tablica(f"Povijest treniranja — {tip.upper()}",
            ["Epoha", "Train loss", "Val loss", "Train recall", "Val recall", "Spec."], redovi_pov)
    tablica(f"Test metrika — {tip.upper()}",
            ["Recall", "Spec.", "Precision", "F1", "Accuracy", "ROC-AUC", "Matrica"], [[
                f"{mtest['recall']*100:.2f}%", f"{mtest['specificnost']*100:.2f}%",
                f"{mtest['precision']*100:.2f}%", f"{mtest['f1']*100:.2f}%",
                f"{mtest['accuracy']*100:.2f}%", f"{mtest['auc']:.4f}",
                f"TP={mtest['TP']} FN={mtest['FN']} FP={mtest['FP']} TN={mtest['TN']}"]])

    raz_auc = (mval["auc"] - mtest["auc"]) if math.isfinite(mtest["auc"]) else float("nan")
    info(f"Rascjep val→test: AUC {mval['auc']:.3f}→{mtest['auc']:.3f} (Δ{raz_auc:+.3f}) · "
         f"spec {mval['specificnost']*100:.1f}%→{mtest['specificnost']*100:.1f}% "
         f"(veliki Δ = overfitting / pomak distribucije)")
    uspjeh(f"Grafovi i metrike spremljeni: {mapa}")
    return model, prag, mtest


def treniraj_iz_cli(tip="oba"):
    hint_rich()
    train_loader, val_loader, test_loader, tezine_klasa, train_eval_loader = \
        core.napravi_dataloadere(verbose=True)
    tipovi = ["cnn", "vgg16"] if tip == "oba" else [tip]
    rezultati = {}
    for t in tipovi:
        rezultati[t] = treniraj(t, train_loader, val_loader, test_loader,
                                tezine_klasa, train_eval_loader)
    return rezultati


if __name__ == "__main__":
    treniraj_iz_cli("oba")
