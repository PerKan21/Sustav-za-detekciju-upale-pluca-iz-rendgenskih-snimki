"""
gui.py — grafičko sučelje za detekciju upale pluća
===================================================

Popravljeno prema bilješkama:
  • nema vertikalnog scrollanja u osnovnom rasporedu;
  • početna i minimalna veličina su 1360x920;
  • radio izbori zamijenjeni su modernim kvadratićima s oštrim X znakom;
  • rezultati se pregledavaju horizontalno pomoću strelica i brojčanih oznaka;
  • gumb Očisti briše i ulazne snimke i rezultate;
  • originalna i Grad-CAM snimka imaju deblji zeleni/crveni okvir ovisno o nalazu;
  • detalji ispod dijagnoze su formatirani u kartice;
  • raspored se pravilno širi pri maksimiziranju/fullscreen prikazu.
"""

import os
import threading
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

from PIL import Image, ImageTk, ImageOps, ImageDraw

import main as cfg
import main as core

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_DOSTUPAN = True
except ImportError:
    DND_DOSTUPAN = False

# ================================================================
# PALETA
# ================================================================
BG          = "#f4f5f7"
CARD        = "#ffffff"
CARD_ALT    = "#f8f9fb"
BORDER      = "#e3e6ea"
TEXT        = "#1f2630"
TEXT_MUTE   = "#6b7280"
PRIMARY     = "#4b5563"
PRIMARY_HOV = "#3a414c"
PRIMARY_FG  = "#ffffff"
INFO        = PRIMARY          # neutralna siva kao kod glavnih gumba
INFO_BG     = "#eef0f3"
INFO_FG     = PRIMARY_HOV
OK          = "#10b981"       # svjetlija moderna zelena
OK_BG       = "#e6f6ee"
OK_FG       = "#0e6245"
DANGER      = "#f43f5e"       # moderna crveno-pink nijansa
DANGER_BG   = "#ffe4ea"
DANGER_FG   = "#9f1239"
DND_BORDER  = "#c2c9d2"
DND_HOVER   = "#eef0f3"
TRACK       = "#e8eaed"
FONT        = "Segoe UI"

# ================================================================
# LOGO / LINK
# ================================================================
# Stavi svoj logo u istu mapu kao gui.py, ili u podmapu assets/.
# Podržani nazivi su navedeni dolje; najjednostavnije je koristiti: assets/logo.png
LOGO_PUTANJE = (
    os.path.join("assets", "logo.png"),
    "logo.png",
    "moj_logo.png",
    "tp_logo.png",
)
LOGO_SIZE = 64
AKCIJSKI_GUMB_PADY = 13
DISABLED_BTN_BG = "#eef0f3"
DISABLED_BTN_FG = "#c3c8d0"
GITHUB_URL = "https://github.com/PerKan21"  # promijeni u svoj pravi GitHub link

# Uključi samo dok tražiš idealnu veličinu prozora.
DEBUG_REZOLUCIJA = False


def label(roditelj, tekst="", boja=TEXT, vel=11, bold=False, bg=CARD, **kw):
    tezina = "bold" if bold else "normal"
    return tk.Label(roditelj, text=tekst, fg=boja, bg=bg, font=(FONT, vel, tezina), **kw)


def gumb(roditelj, tekst, naredba, primarni=False, **kw):
    """Kreiraj gumb s ručnim hover efektom.

    Tkinterovi activebackground/activeforeground rade tek tijekom klika,
    pa ovdje dodatno bindamo Enter/Leave da se boja promijeni čim miš prijeđe
    preko gumba. Disabled gumbi ostaju vizualno mirni i ne dobivaju hover.
    """
    if primarni:
        normal_bg = PRIMARY
        hover_bg = PRIMARY_HOV
        normal_fg = PRIMARY_FG
        hover_fg = PRIMARY_FG
        btn = tk.Button(roditelj, text=tekst, command=naredba, bg=normal_bg, fg=normal_fg,
                        activebackground=hover_bg, activeforeground=hover_fg,
                        relief="flat", cursor="hand2", font=(FONT, 10, "bold"),
                        padx=14, pady=7, bd=0, **kw)
    else:
        normal_bg = CARD
        hover_bg = CARD_ALT
        normal_fg = TEXT
        hover_fg = PRIMARY
        btn = tk.Button(roditelj, text=tekst, command=naredba, bg=normal_bg, fg=normal_fg,
                        activebackground=hover_bg, activeforeground=hover_fg,
                        relief="flat", cursor="hand2", font=(FONT, 10), padx=14, pady=7,
                        bd=0, highlightbackground=BORDER, highlightthickness=1, **kw)

    def _enter(event=None):
        if str(btn.cget("state")) != "disabled":
            btn.configure(bg=hover_bg, fg=hover_fg)

    def _leave(event=None):
        if str(btn.cget("state")) != "disabled":
            btn.configure(bg=normal_bg, fg=normal_fg)

    btn.bind("<Enter>", _enter)
    btn.bind("<Leave>", _leave)
    return btn


def unos(roditelj, varijabla, **kw):
    return tk.Entry(roditelj, textvariable=varijabla, bg=CARD_ALT, fg=TEXT,
                    insertbackground=TEXT, relief="flat", font=(FONT, 10),
                    highlightbackground=BORDER, highlightcolor=PRIMARY,
                    highlightthickness=1, **kw)


def radio(roditelj, tekst, var, val, naredba=None):
    return tk.Radiobutton(roditelj, text=tekst, variable=var, value=val, command=naredba,
                          bg=CARD, fg=TEXT, selectcolor=CARD, activebackground=CARD,
                          activeforeground=TEXT, font=(FONT, 10), cursor="hand2",
                          bd=0, highlightthickness=0)


class ChoicePill:
    """Moderni izbor kao mali checkbox kvadratić, bez default Windows radio izgleda."""
    BOX = 18

    def __init__(self, roditelj, varijabla, opcije, naredba=None, bg=CARD, sirine=None):
        self.var = varijabla
        self.opcije = opcije
        self.naredba = naredba
        self.bg = bg
        self.sirine = sirine or [None] * len(opcije)
        self.frame = tk.Frame(roditelj, bg=bg)
        self._items = {}

        for i, (tekst, vrijednost) in enumerate(opcije):
            sirina = self.sirine[i] if i < len(self.sirine) else None
            item = tk.Frame(self.frame, bg=bg, cursor="hand2",
                            width=sirina if sirina else 1, height=24)
            item.pack(side="left", padx=(0 if i == 0 else 10, 0))
            if sirina:
                item.pack_propagate(False)

            box = tk.Canvas(item, width=self.BOX, height=self.BOX,
                            bg=bg, highlightthickness=0, cursor="hand2")
            box.pack(side="left", padx=(0, 6), pady=3)

            lbl = tk.Label(item, text=tekst, bg=bg, fg=TEXT,
                           font=(FONT, 10), cursor="hand2", anchor="w")
            lbl.pack(side="left", fill="x", expand=True)

            for w in (item, box, lbl):
                w.bind("<Button-1>", lambda e, v=vrijednost: self._odaberi(v))
                w.bind("<Enter>", lambda e, v=vrijednost: self._hover(v, True))
                w.bind("<Leave>", lambda e, v=vrijednost: self._hover(v, False))

            self._items[vrijednost] = {"frame": item, "box": box, "label": lbl, "tekst": tekst}

        self.azuriraj()

    def pack(self, **kw):
        self.frame.pack(**kw)

    def grid(self, **kw):
        self.frame.grid(**kw)

    def _odaberi(self, vrijednost):
        if self.var.get() != vrijednost:
            self.var.set(vrijednost)
            self.azuriraj()
            if self.naredba:
                self.naredba()

    def _hover(self, vrijednost, aktivno):
        item = self._items.get(vrijednost)
        if not item or self.var.get() == vrijednost:
            return
        boja = DND_HOVER if aktivno else self.bg
        item["frame"].configure(bg=boja)
        item["label"].configure(bg=boja)
        item["box"].configure(bg=boja)
        self._nacrtaj_box(vrijednost, odabrano=False, hover=aktivno)

    def _nacrtaj_box(self, vrijednost, odabrano, hover=False):
        c = self._items[vrijednost]["box"]
        c.delete("all")
        pad = 2
        fill = PRIMARY if odabrano else (CARD if not hover else DND_HOVER)
        outline = PRIMARY if odabrano else BORDER
        c.create_rectangle(pad, pad, self.BOX - pad, self.BOX - pad,
                           fill=fill, outline=outline, width=2)
        if odabrano:
            # Oštri X je čišći u Tkinteru od male tekstualne kvačice
            # koja na Windowsu zna izgledati mutno.
            c.create_line(5, 5, self.BOX - 5, self.BOX - 5,
                          fill=PRIMARY_FG, width=2.2,
                          capstyle="butt")
            c.create_line(self.BOX - 5, 5, 5, self.BOX - 5,
                          fill=PRIMARY_FG, width=2.2,
                          capstyle="butt")

    def azuriraj(self):
        trenutno = self.var.get()
        for vrijednost, item in self._items.items():
            odabrano = vrijednost == trenutno
            item["frame"].configure(bg=self.bg)
            item["box"].configure(bg=self.bg)
            item["label"].configure(
                bg=self.bg,
                fg=TEXT,
                font=(FONT, 10, "bold" if odabrano else "normal")
            )
            self._nacrtaj_box(vrijednost, odabrano=odabrano)


class ProgressBar:
    def __init__(self, roditelj, boja=PRIMARY):
        self._boja_default = boja
        self.canvas = tk.Canvas(roditelj, height=8, bg=TRACK, highlightthickness=0)
        self.rect = self.canvas.create_rectangle(0, 0, 0, 8, fill=boja, outline="")
        self.canvas.bind("<Configure>", lambda e: self.postavi(self._vrijednost))
        self._vrijednost = 0.0

    def pack(self, **kw):
        self.canvas.pack(**kw)

    def postavi(self, napredak, boja=None):
        self._vrijednost = max(0.0, min(1.0, napredak))
        if boja:
            self.canvas.itemconfig(self.rect, fill=boja)
        self.canvas.update_idletasks()
        sirina = self.canvas.winfo_width()
        self.canvas.coords(self.rect, 0, 0, sirina * self._vrijednost, 8)

    def reset(self):
        self.canvas.itemconfig(self.rect, fill=self._boja_default)
        self.postavi(0.0)

    def zavrseno(self):
        self.postavi(1.0, boja=INFO)


class SlikaPanel:
    _PLACEHOLDER = "🖼\n\nSnimka će se prikazati ovdje"

    def __init__(self, roditelj, naslov, visina=250):
        self.frame = tk.Frame(roditelj, bg=CARD)
        self._naslov = label(self.frame, naslov, boja=TEXT_MUTE, vel=10, bold=True, anchor="center")
        self._naslov.pack(fill="x", pady=(0, 6))
        self.okvir = tk.Frame(self.frame, bg=CARD_ALT, highlightbackground=BORDER,
                              highlightthickness=2, height=visina)
        self.okvir.pack(fill="both", expand=True)
        self.okvir.pack_propagate(False)
        self.lbl = tk.Label(self.okvir, bg=CARD_ALT, fg=TEXT_MUTE, text=self._PLACEHOLDER,
                            font=(FONT, 11), justify="center", bd=0)
        self.lbl.place(relx=0.5, rely=0.5, anchor="center")

    def grid(self, **kw):
        self.frame.grid(**kw)

    def postavi_sliku(self, photo, akcent=BORDER):
        self.lbl.configure(image=photo, text="")
        self.okvir.configure(highlightbackground=akcent, highlightthickness=4)

    def reset(self):
        self.lbl.configure(image="", text=self._PLACEHOLDER)
        self.okvir.configure(highlightbackground=BORDER, highlightthickness=2)


class SnimkeZona:
    _TEKST_PRAZAN = "Klikni za odabir snimki ili mape"
    _TEKST_HOVER = "Ispusti datoteke ovdje"

    def __init__(self, roditelj, callback_promjena):
        self._callback = callback_promjena
        self._datoteke = []
        self._popup = None

        # Fiksna visina bez širenja. Donja traka ima zaseban, fiksan red
        # kako naziv snimke nikad ne bi završio ispod ruba ili ispod gumba.
        self.frame = tk.Frame(roditelj, bg=CARD_ALT, highlightbackground=DND_BORDER,
                              highlightthickness=2, cursor="hand2", height=170)
        self.frame.pack_propagate(False)

        self._zona = tk.Frame(self.frame, bg=CARD_ALT, cursor="hand2", height=116)
        self._zona.pack(fill="x")
        self._zona.pack_propagate(False)

        self._ikona = tk.Label(self._zona, text="📥", bg=CARD_ALT, fg=PRIMARY,
                               font=(FONT, 18), pady=0, cursor="hand2")
        self._ikona.pack(pady=(2, 0))
        self._lbl_glavni = tk.Label(self._zona, text=self._TEKST_PRAZAN, bg=CARD_ALT,
                                    fg=TEXT, font=(FONT, 10, "bold"), justify="center",
                                    pady=0, cursor="hand2")
        self._lbl_glavni.pack(pady=(3, 0))
        hint = "klikni  ·  ili povuci datoteke / mapu" if DND_DOSTUPAN else "klikni za odabir snimki"
        self._lbl_hint = tk.Label(self._zona, text=hint, bg=CARD_ALT, fg=TEXT_MUTE,
                                  font=(FONT, 9), pady=0, cursor="hand2")
        self._lbl_hint.pack(pady=(3, 0))

        tk.Frame(self.frame, bg=BORDER, height=1).pack(fill="x")
        self._dno = tk.Frame(self.frame, bg=CARD_ALT, height=48)
        self._dno.pack(fill="x", side="bottom")
        self._dno.pack_propagate(False)
        self._dno.grid_columnconfigure(0, weight=1)

        self._lbl_info = tk.Label(self._dno, text="Nema odabranih snimki.", bg=CARD_ALT,
                                  fg=TEXT_MUTE, font=(FONT, 9), anchor="w", justify="left")
        self._lbl_info.grid(row=0, column=0, sticky="ew", padx=(12, 8), pady=(12, 0))

        # Diskretan gumb za čišćenje u donjem desnom kutu zone.
        # Bez bijelog okvira i bez zauzimanja prostora za naziv snimke.
        self._btn_ocisti = tk.Button(
            self._dno, text="🗑  Očisti", command=self.ocisti,
            bg=CARD_ALT, fg=TEXT_MUTE, activebackground=CARD_ALT,
            activeforeground=DANGER_FG, relief="flat", cursor="hand2",
            font=(FONT, 9, "bold"), padx=4, pady=0, bd=0,
            highlightthickness=0
        )
        self._btn_ocisti.bind("<Enter>", lambda e: self._btn_ocisti.configure(fg=DANGER_FG))
        self._btn_ocisti.bind("<Leave>", lambda e: self._btn_ocisti.configure(fg=TEXT_MUTE))
        self._btn_ocisti.grid(row=0, column=1, sticky="e", padx=(6, 12), pady=(13, 0))
        self._btn_ocisti.grid_remove()

        self._klikabilni = [self._zona, self._ikona, self._lbl_glavni, self._lbl_hint]
        for w in self._klikabilni:
            w.bind("<Button-1>", self._otvori_popup)
            w.bind("<Enter>", self._hover_enter)
            w.bind("<Leave>", self._hover_leave)

        if DND_DOSTUPAN:
            for w in self._klikabilni + [self.frame]:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<DropEnter>>", self._dnd_enter)
                w.dnd_bind("<<DropLeave>>", self._dnd_leave)
                w.dnd_bind("<<Drop>>", self._dnd_drop)

    def pack(self, **kw):
        self.frame.pack(**kw)

    @property
    def datoteke(self):
        return self._datoteke[:]

    @property
    def ima_datoteke(self):
        return bool(self._datoteke)

    def _otvori_popup(self, event=None):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
            self._popup = None
            return
        popup = tk.Toplevel(self.frame)
        popup.overrideredirect(True)
        popup.configure(bg=BORDER)
        popup.attributes("-topmost", True)
        self._popup = popup
        self.frame.update_idletasks()
        popup.geometry(f"+{self.frame.winfo_rootx()}+{self.frame.winfo_rooty() + self.frame.winfo_height() + 2}")

        def stil(tekst, naredba):
            b = tk.Button(popup, text=tekst, command=naredba, bg=CARD, fg=TEXT,
                          activebackground=DND_HOVER, activeforeground=PRIMARY,
                          relief="flat", cursor="hand2", font=(FONT, 10), anchor="w",
                          padx=16, pady=8, width=28, bd=0)
            b.bind("<Enter>", lambda e, btn=b: btn.configure(bg=DND_HOVER, fg=PRIMARY))
            b.bind("<Leave>", lambda e, btn=b: btn.configure(bg=CARD, fg=TEXT))
            b.pack(fill="x", padx=1, pady=1)

        stil("🖼   Odaberi snimke", self._popup_datoteke)
        stil("📁   Odaberi mapu", self._popup_mapa)
        popup.bind("<FocusOut>", lambda e: self._zatvori_popup())
        popup.focus_set()

    def _zatvori_popup(self):
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
        self._popup = None

    def _popup_datoteke(self):
        self._zatvori_popup()
        tipovi = [("Slike", " ".join(f"*{e}" for e in cfg.PODRZANI_FORMATI)), ("Sve datoteke", "*.*")]
        odabrano = filedialog.askopenfilenames(title="Odaberi rendgenske snimke", filetypes=tipovi)
        if odabrano:
            self._dodaj_datoteke(list(odabrano))

    def _popup_mapa(self):
        self._zatvori_popup()
        mapa = filedialog.askdirectory(title="Odaberi mapu sa snimkama")
        if mapa:
            self._dodaj_iz_mape(mapa)

    def _hover_enter(self, event=None):
        self.frame.configure(highlightbackground=PRIMARY)

    def _hover_leave(self, event=None):
        self.frame.configure(highlightbackground=DND_BORDER)

    def _dnd_enter(self, event=None):
        self.frame.configure(bg=DND_HOVER, highlightbackground=PRIMARY)
        for w in (self._zona, self._ikona, self._lbl_hint):
            w.configure(bg=DND_HOVER)
        self._lbl_glavni.configure(bg=DND_HOVER, fg=PRIMARY, text=self._TEKST_HOVER)

    def _dnd_leave(self, event=None):
        self._resetiraj_izgled()

    def _dnd_drop(self, event):
        self._resetiraj_izgled()
        putanje = self._parsiraj_putanje(event.data)
        if not putanje:
            return
        slike = []
        for p in putanje:
            if os.path.isdir(p):
                slike.extend(self._slike_iz_mape(p))
            elif os.path.isfile(p) and p.lower().endswith(cfg.PODRZANI_FORMATI):
                slike.append(p)
        if slike:
            self._dodaj_datoteke(slike)
        else:
            messagebox.showwarning("Drag & Drop", "Nisu pronađene podržane slike.")

    def _resetiraj_izgled(self):
        self.frame.configure(bg=CARD_ALT, highlightbackground=DND_BORDER)
        for w in (self._zona, self._ikona, self._lbl_hint):
            w.configure(bg=CARD_ALT)
        self._lbl_glavni.configure(bg=CARD_ALT, fg=TEXT, text=self._TEKST_PRAZAN)

    def _slike_iz_mape(self, mapa):
        return [os.path.join(mapa, f) for f in sorted(os.listdir(mapa))
                if f.lower().endswith(cfg.PODRZANI_FORMATI)]

    def _dodaj_iz_mape(self, mapa):
        nove = self._slike_iz_mape(mapa)
        if not nove:
            messagebox.showwarning("Odabir mape", f"Mapa ne sadrži slike:\n{mapa}")
            return
        self._dodaj_datoteke(nove)

    def _dodaj_datoteke(self, putanje):
        for p in putanje:
            if p not in self._datoteke:
                self._datoteke.append(p)
        self._datoteke.sort()
        self._azuriraj_prikaz()
        self._callback()

    def ocisti(self):
        self._datoteke = []
        self._azuriraj_prikaz()
        self._callback()

    def _azuriraj_prikaz(self):
        n = len(self._datoteke)
        if n == 0:
            self._lbl_info.configure(text="Nema odabranih snimki.", fg=TEXT_MUTE, font=(FONT, 9))
            self._lbl_glavni.configure(text=self._TEKST_PRAZAN)
            self._ikona.configure(text="📥", fg=PRIMARY, font=(FONT, 22))
            self._btn_ocisti.grid_remove()
        else:
            nazivi = [os.path.basename(p) for p in self._datoteke]
            prvi = nazivi[0]
            if len(prvi) > 22:
                prvi = prvi[:19] + "…"
            if n == 1:
                prikaz = f"1 snimka:  {prvi}"
            else:
                prikaz = f"{n} snimki  ·  {prvi}  (+{n-1} još)"
            self._lbl_info.configure(text=prikaz, fg=INFO, font=(FONT, 9, "bold"))
            self._lbl_glavni.configure(text=f"{n} {'snimka odabrana' if n == 1 else 'snimki odabrano'}")
            self._ikona.configure(text=f"🖼 {n}", fg=INFO, font=(FONT, 22, "bold"))
            self._btn_ocisti.grid()

    @staticmethod
    def _parsiraj_putanje(data):
        putanje, data, i = [], data.strip(), 0
        while i < len(data):
            if data[i] == "{":
                kraj = data.find("}", i)
                if kraj == -1:
                    putanje.append(data[i + 1:].strip())
                    break
                putanje.append(data[i + 1:kraj])
                i = kraj + 1
            elif data[i] == " ":
                i += 1
            else:
                kraj = data.find(" ", i)
                if kraj == -1:
                    putanje.append(data[i:])
                    break
                putanje.append(data[i:kraj])
                i = kraj + 1
        return [p.strip() for p in putanje if p.strip()]


_BaseClass = TkinterDnD.Tk if DND_DOSTUPAN else tk.Tk


class App(_BaseClass):
    IDEAL_W = 1360
    IDEAL_H = 920
    MIN_W = 1360
    MIN_H = 920
    THUMB = 62
    THUMB_BOX = 70

    def __init__(self):
        super().__init__()
        self.title("Sustav za detekciju upale pluća")
        self.geometry(f"{self.IDEAL_W}x{self.IDEAL_H}")
        self.minsize(self.MIN_W, self.MIN_H)
        self.configure(bg=BG)

        self.rezultati = []
        self.odabrani = 0
        self.timestamp_analize = None
        self.var_tip = tk.StringVar(value="vgg16")
        self.var_model_putanja = tk.StringVar(value=cfg.PUTANJA_MODEL["vgg16"])
        self.var_format = tk.StringVar(value="txt")
        self._thumb_imgs = []
        self._img_orig = None
        self._img_overlay = None
        self._fullscreen = False

        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._izlaz_fullscreen)
        self.bind("<Configure>", self._on_configure)
        self._debug_var = tk.StringVar(value=f"Prozor: {self.IDEAL_W} × {self.IDEAL_H}")
        self._izgradnja_ui()

    def _pronadji_logo_putanju(self):
        """Vrati prvu postojeću putanju loga iz LOGO_PUTANJE ili None."""
        baza = os.path.dirname(os.path.abspath(__file__))
        for putanja in LOGO_PUTANJE:
            kandidati = [putanja]
            if not os.path.isabs(putanja):
                kandidati.append(os.path.join(baza, putanja))
            for kandidat in kandidati:
                if os.path.exists(kandidat):
                    return kandidat
        return None

    @staticmethod
    def _hex_u_rgb(hex_boja):
        hex_boja = hex_boja.lstrip("#")
        return tuple(int(hex_boja[i:i + 2], 16) for i in (0, 2, 4))

    def _ucitaj_logo(self, boja):
        """
        Učita logo i oboji ga u zadanu sivu boju.
        Radi i ako je logo PNG s prozirnom pozadinom i ako je logo
        crn/taman na bijeloj pozadini: svijetla pozadina se uklanja.
        """
        putanja = self._pronadji_logo_putanju()
        if not putanja:
            return None
        try:
            img = Image.open(putanja).convert("RGBA")
            img = ImageOps.contain(img, (LOGO_SIZE, LOGO_SIZE), method=Image.LANCZOS)

            # Ako PNG već ima prozirnost, koristi se postojeći alpha kanal.
            # Ako je pozadina neprozirna bijela/svijetla, pretvara se u prozirnu
            # i zadržavaju se samo tamni dijelovi loga. Time se izbjegne sivi kvadrat.
            pix = img.load()
            alpha_img = Image.new("L", img.size, 0)
            alpha_pix = alpha_img.load()
            for y in range(img.height):
                for x in range(img.width):
                    rr, gg, bb, aa = pix[x, y]
                    svjetlina = (rr + gg + bb) / 3
                    maxc = max(rr, gg, bb)
                    minc = min(rr, gg, bb)
                    saturacija = maxc - minc

                    # Prozirni pikseli ostaju prozirni.
                    if aa < 10:
                        alpha_pix[x, y] = 0
                    # Skoro bijela/svijetla i slabo zasićena pozadina se uklanja.
                    elif svjetlina > 225 and saturacija < 35:
                        alpha_pix[x, y] = 0
                    else:
                        # Taman logo dobije punu neprozirnost, a rubovi mekši prijelaz.
                        alpha_pix[x, y] = max(0, min(255, int(aa * (255 - svjetlina) / 170)))

            # Ako je maska slučajno ispala prazna, vrati se na originalni alpha.
            if not alpha_img.getbbox():
                alpha_img = img.getchannel("A")

            r, g, b = self._hex_u_rgb(boja)
            obojano = Image.new("RGBA", img.size, (r, g, b, 0))
            obojano.putalpha(alpha_img)
            return ImageTk.PhotoImage(obojano)
        except Exception:
            return None

    def _logo_hover(self, aktivno=False):
        if not hasattr(self, "logo_lbl"):
            return
        if self._logo_img and self._logo_img_hover:
            self.logo_lbl.configure(image=self._logo_img_hover if aktivno else self._logo_img)
        else:
            self.logo_lbl.configure(fg=PRIMARY_HOV if aktivno else PRIMARY)

    def _otvori_github(self, event=None):
        webbrowser.open(GITHUB_URL)

    def _izgradnja_ui(self):
        zaglavlje = tk.Frame(self, bg=BG)
        zaglavlje.pack(fill="x", padx=24, pady=(10, 8))
        zaglavlje.grid_columnconfigure(0, weight=1)
        zaglavlje.grid_columnconfigure(1, weight=0)

        # Naslov je lijevo, a logo je desno kao diskretan klikabilni brand element.
        tk.Label(
            zaglavlje,
            text="Sustav za detekciju upale pluća iz rendgenskih snimki",
            bg=BG,
            fg=TEXT,
            font=(FONT, 19, "bold"),
            anchor="w",
            justify="left"
        ).grid(row=0, column=0, sticky="w", padx=(0, 16))

        self._logo_img = self._ucitaj_logo(PRIMARY)
        self._logo_img_hover = self._ucitaj_logo(PRIMARY_HOV)
        if self._logo_img:
            self.logo_lbl = tk.Label(
                zaglavlje,
                image=self._logo_img,
                bg=BG,
                cursor="hand2",
                bd=0,
                padx=0,
                pady=0
            )
        else:
            # Fallback ako logo.png još nije dodan u mapu projekta.
            # Čim dodaš logo, automatski će se koristiti prava slika.
            self.logo_lbl = tk.Label(
                zaglavlje,
                text="TP",
                bg=BG,
                fg=PRIMARY,
                cursor="hand2",
                font=(FONT, 19, "bold"),
                bd=0,
                padx=10,
                pady=6
            )
        self.logo_lbl.grid(row=0, column=1, sticky="e", padx=(16, 8), pady=(4, 0))
        self.logo_lbl.bind("<Button-1>", self._otvori_github)
        self.logo_lbl.bind("<Enter>", lambda e: self._logo_hover(True))
        self.logo_lbl.bind("<Leave>", lambda e: self._logo_hover(False))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24)

        tijelo = tk.Frame(self, bg=BG)
        tijelo.pack(fill="both", expand=True, padx=24, pady=14)
        tijelo.grid_columnconfigure(0, minsize=500)
        tijelo.grid_columnconfigure(1, weight=1)
        tijelo.grid_rowconfigure(0, weight=1)

        self.lijevo = tk.Frame(tijelo, bg=BG, width=500)
        self.lijevo.grid(row=0, column=0, sticky="nsew")
        self.lijevo.grid_propagate(False)

        self.desno = tk.Frame(tijelo, bg=BG)
        self.desno.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        self.desno.grid_columnconfigure(0, weight=1)
        self.desno.grid_rowconfigure(0, weight=1)

        self._sekcija_model(self.lijevo)
        self._sekcija_snimke(self.lijevo)
        self._sekcija_akcije(self.lijevo)
        self._sekcija_rezultati(self.desno)

        self.lbl_debug_rez = None
        if DEBUG_REZOLUCIJA:
            self.lbl_debug_rez = tk.Label(
                self, textvariable=self._debug_var, bg=BG, fg=TEXT_MUTE,
                font=(FONT, 8), anchor="w"
            )
            self.lbl_debug_rez.place(x=8, rely=1.0, y=-8, anchor="sw")

    def _kartica(self, roditelj, naslov=None, expand=False):
        k = tk.Frame(roditelj, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        k.pack(fill="both" if expand else "x", expand=expand, pady=(0, 12))
        unutar = tk.Frame(k, bg=CARD)
        unutar.pack(fill="both", expand=True, padx=16, pady=12)
        if naslov:
            label(unutar, naslov, bold=True, vel=12).pack(anchor="w", pady=(0, 10))
        return unutar

    def _sekcija_model(self, roditelj):
        u = self._kartica(roditelj, naslov="🧠  Model")
        red = tk.Frame(u, bg=CARD)
        red.pack(fill="x")
        self.tip_choice = ChoicePill(
            red, self.var_tip,
            [("VGG16 (transfer)", "vgg16"), ("CNN (od nule)", "cnn")],
            naredba=self._tip_promijenjen,
            sirine=[170, 145]
        )
        self.tip_choice.pack(anchor="w")
        red2 = tk.Frame(u, bg=CARD)
        red2.pack(fill="x", pady=(10, 0))
        unos(red2, self.var_model_putanja).pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=3)
        gumb(red2, "Odaberi", self._odaberi_model, primarni=True).pack(side="left", padx=(0, 6))
        gumb(red2, "Učitaj", self._ucitaj_model, primarni=True).pack(side="left")
        self.lbl_model_status = label(u, "Model nije učitan.", boja=TEXT_MUTE, vel=10)
        self.lbl_model_status.pack(anchor="w", pady=(10, 0))

    def _sekcija_snimke(self, roditelj):
        u = self._kartica(roditelj, naslov="🖼  Ulazne snimke")
        self.snimke_zona = SnimkeZona(u, callback_promjena=self._ulaz_promijenjen)
        self.snimke_zona.pack(fill="x")

    def _sekcija_akcije(self, roditelj):
        u = self._kartica(roditelj, naslov="▶  Analiza")
        # Oba glavna akcijska gumba koriste isti font i isti vertikalni padding,
        # pa "Pokreni analizu" i "Spremi rezultate" ostaju jednake visine.
        self.btn_run = gumb(u, "▶   Pokreni analizu", self._pokreni, primarni=True, state="disabled")
        self.btn_run.config(font=(FONT, 12, "bold"), pady=AKCIJSKI_GUMB_PADY)
        self.btn_run.pack(fill="x")

        # Kartica spremanja se rasteže do dna lijevog stupca. Ne povećavamo gumb,
        # nego dodajemo prazan prostor iznad njega kako bi se vanjski donji rubovi
        # lijevog i desnog panela vizualno poravnali.
        u2 = self._kartica(roditelj, naslov="💾  Spremanje", expand=True)
        red = tk.Frame(u2, bg=CARD)
        red.pack(fill="x", pady=(0, 2))
        label(red, "Format:", vel=10, boja=TEXT_MUTE).pack(side="left", padx=(0, 8))
        self.format_choice = ChoicePill(
            red, self.var_format,
            [("Tekstualna datoteka", "txt"), ("Excel datoteka", "xlsx")],
            sirine=[220, 170]
        )
        self.format_choice.pack(side="left")

        # Elastični razmak drži gumb pri dnu kartice bez mijenjanja njegove visine.
        tk.Frame(u2, bg=CARD).pack(fill="both", expand=True)

        self.btn_spremi = gumb(u2, "💾   Spremi rezultate", self._spremi, primarni=True)
        self.btn_spremi.config(font=(FONT, 12, "bold"), pady=AKCIJSKI_GUMB_PADY)
        self.btn_spremi.pack(fill="x", side="bottom", pady=(12, 0))

        # Početna stanja: Pokreni analizu i Spremi rezultate izgledaju kao
        # neaktivne strelice dok ne postoje preduvjeti za njihovu akciju.
        self._postavi_btn_run_aktivan(False)
        self._postavi_btn_spremi_aktivan(False)

    def _postavi_btn_run_aktivan(self, aktivan):
        """Vizualno uskladi gumb Pokreni analizu s neaktivnim strelicama.

        Kad model nije učitan ili analiza radi, gumb dobije istu svijetlu
        disabled paletu kao i strelice. Kad je spreman za klik, vraća se na
        standardni primarni stil.
        """
        if aktivan:
            self.btn_run.configure(
                state="normal",
                bg=PRIMARY,
                fg=PRIMARY_FG,
                activebackground=PRIMARY_HOV,
                activeforeground=PRIMARY_FG,
                disabledforeground=DISABLED_BTN_FG,
                cursor="hand2"
            )
        else:
            self.btn_run.configure(
                state="disabled",
                bg=DISABLED_BTN_BG,
                fg=DISABLED_BTN_FG,
                activebackground=DISABLED_BTN_BG,
                activeforeground=DISABLED_BTN_FG,
                disabledforeground=DISABLED_BTN_FG,
                cursor="arrow"
            )

    def _postavi_btn_spremi_aktivan(self, aktivan):
        """Vizualno uskladi gumb Spremi rezultate s neaktivnim strelicama.

        Kad nema analiziranih podataka (početno stanje, nakon Očisti ili nakon
        resetiranja rezultata), gumb izgleda kao disabled strelice. Kad analiza
        uspješno završi i postoje rezultati, vraća se na standardni primarni stil.
        """
        if aktivan:
            self.btn_spremi.configure(
                state="normal",
                bg=PRIMARY,
                fg=PRIMARY_FG,
                activebackground=PRIMARY_HOV,
                activeforeground=PRIMARY_FG,
                disabledforeground=DISABLED_BTN_FG,
                cursor="hand2"
            )
        else:
            self.btn_spremi.configure(
                state="disabled",
                bg=DISABLED_BTN_BG,
                fg=DISABLED_BTN_FG,
                activebackground=DISABLED_BTN_BG,
                activeforeground=DISABLED_BTN_FG,
                disabledforeground=DISABLED_BTN_FG,
                cursor="arrow"
            )

    def _sekcija_rezultati(self, roditelj):
        # Gornji dio desnog panela sada je rezerviran za pregled analiziranih
        # snimki, a kompletan blok dijagnostike nalazi se ispod njega.
        u = self._kartica(roditelj, naslov="🖼  Analizirane snimke", expand=True)
        u.grid_columnconfigure(0, weight=1)
        u.grid_rowconfigure(5, weight=1)

        # 1) Sličice i strelice — gore.
        self._napravi_thumb_strip(u)

        tk.Frame(u, bg=BORDER, height=1).pack(fill="x", pady=12)

        # 2) Rezultat dijagnostike + progress/status — ispod sličica.
        label(u, "📊  Rezultat analize", bold=True, vel=12).pack(anchor="w", pady=(0, 8))

        self.prog = ProgressBar(u)
        self.prog.pack(fill="x", pady=(0, 7))
        # Status poruka je u posebnom wrapperu kako bi završna poruka mogla
        # imati djelomično boldane brojeve, npr. 13 i 14.
        self.status_wrap = tk.Frame(u, bg=CARD)
        self.status_wrap.pack(fill="x")
        self.lbl_status = label(self.status_wrap, "Učitaj model, zatim pokreni analizu.",
                                boja=TEXT_MUTE, vel=10, bg=CARD)
        self.lbl_status.pack(anchor="w")

        # Završna poruka je Text widget u jednom retku kako ne bi nastajali
        # čudni razmaci između zasebnih Label widgeta. Brojevi su i dalje boldani.
        self.status_final_text = tk.Text(
            self.status_wrap, height=1, bg=CARD, fg=INFO,
            font=(FONT, 10), bd=0, highlightthickness=0,
            padx=0, pady=0, wrap="none"
        )
        self.status_final_text.tag_configure("bold", font=(FONT, 10, "bold"))
        self.status_final_text.configure(state="disabled", cursor="arrow")

        top = tk.Frame(u, bg=CARD)
        top.pack(fill="x", pady=(10, 0))
        self.lbl_presuda = tk.Label(top, text="—", bg=CARD, fg=TEXT_MUTE, font=(FONT, 25, "bold"), anchor="w")
        self.lbl_presuda.pack(anchor="w")

        self.snimka_redak = tk.Frame(top, bg=CARD)
        self.snimka_redak.pack(fill="x", pady=(6, 0))
        self.lbl_snimka_prefix = label(self.snimka_redak, "Snimka:", boja=PRIMARY, vel=10, bold=True, bg=CARD)
        self.lbl_snimka_prefix.pack(side="left")
        self.lbl_snimka_naziv = label(self.snimka_redak, "—", boja=TEXT, vel=10, bold=True, bg=CARD, anchor="w")
        self.lbl_snimka_naziv.configure(width=72)
        self.lbl_snimka_naziv.pack(side="left", padx=(6, 0))

        self.detalji_frame = tk.Frame(top, bg=CARD)
        self.detalji_frame.pack(fill="x", pady=(8, 0))
        self._detail_titles = []
        self._detail_values = []
        for i in range(4):
            card = tk.Frame(self.detalji_frame, bg=CARD_ALT, highlightbackground=BORDER,
                            highlightthickness=1, height=62)
            card.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 10, 0))
            card.grid_propagate(False)
            self.detalji_frame.grid_columnconfigure(i, weight=1, uniform="detalji")
            ttl = label(card, "—", boja=TEXT_MUTE, vel=8, bold=True, bg=CARD_ALT)
            ttl.pack(anchor="w", padx=8, pady=(5, 0))
            val = label(card, "—", boja=TEXT, vel=10, bold=True, bg=CARD_ALT, justify="left")
            val.pack(anchor="w", padx=8, pady=(1, 6))
            self._detail_titles.append(ttl)
            self._detail_values.append(val)

        tk.Frame(u, bg=BORDER, height=1).pack(fill="x", pady=12)

        slike = tk.Frame(u, bg=CARD)
        slike.pack(fill="both", expand=True, pady=(0, 0))
        slike.grid_columnconfigure(0, weight=1, uniform="slike")
        slike.grid_columnconfigure(1, weight=1, uniform="slike")
        slike.grid_rowconfigure(0, weight=1)
        self.panel_orig = SlikaPanel(slike, "Originalna snimka", visina=260)
        self.panel_overlay = SlikaPanel(slike, "Grad-CAM toplinska karta", visina=260)
        self.panel_orig.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.panel_overlay.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

    def _napravi_strelicu_canvas(self, roditelj, smjer, naredba):
        """Canvas gumb za strelice sa čistim, oštrim rubovima.

        Ovdje namjerno ne koristimo PIL resize/antialiasing za pozadinski kvadrat,
        jer to može dati blagi blurry dojam na rubovima. Kvadrat i trokut crtaju
        se direktno u Canvasu, pa su rubovi čisti i oštri.
        """
        size = 36
        c = tk.Canvas(
            roditelj,
            width=size,
            height=size,
            bg=CARD,
            highlightthickness=0,
            bd=0,
            cursor="arrow"
        )
        c._strelica_aktivna = False
        c._strelica_hover = False
        c._strelica_naredba = naredba
        c._size = size

        # Pozadina gumba: čisti kvadrat.
        c._rect = c.create_rectangle(0, 0, size, size, fill="#eef0f3", outline="#eef0f3")

        # Unutarnja strelica kao oštar chevron umjesto punog trokuta.
        # To na Windows/Tkinter prikazu često djeluje čišće i manje blurry.
        cx = size // 2
        cy = size // 2
        if smjer == "lijevo":
            points = (
                cx + 3, cy - 6,
                cx - 3, cy,
                cx + 3, cy + 6,
            )
        else:
            points = (
                cx - 3, cy - 6,
                cx + 3, cy,
                cx - 3, cy + 6,
            )
        c._trokut = c.create_line(
            *points,
            fill="#c3c8d0",
            width=3,
            joinstyle="miter",
            capstyle="projecting",
            smooth=False
        )

        def enter(event=None):
            c._strelica_hover = True
            self._postavi_strelicu(c, c._strelica_aktivna)

        def leave(event=None):
            c._strelica_hover = False
            self._postavi_strelicu(c, c._strelica_aktivna)

        def click(event=None):
            if c._strelica_aktivna:
                c._strelica_naredba()

        c.bind("<Enter>", enter)
        c.bind("<Leave>", leave)
        c.bind("<Button-1>", click)
        return c

    def _postavi_strelicu(self, canvas, aktivna):
        """Postavi stanje Canvas strelice i njezinu hover boju."""
        canvas._strelica_aktivna = bool(aktivna)
        hover = bool(getattr(canvas, "_strelica_hover", False))

        if canvas._strelica_aktivna:
            fill = PRIMARY_HOV if hover else PRIMARY
            fg = PRIMARY_FG
            cursor = "hand2"
        else:
            fill = "#eef0f3"
            fg = "#c3c8d0"
            cursor = "arrow"

        canvas.itemconfig(canvas._rect, fill=fill, outline=fill)
        canvas.itemconfig(canvas._trokut, fill=fg)
        canvas.configure(cursor=cursor)

    def _napravi_thumb_strip(self, roditelj):
        wrap = tk.Frame(roditelj, bg=CARD)
        wrap.pack(fill="x")

        # Vizualni centar prati sredinu same rendgenske sličice unutar thumba.
        self._thumb_strip_h = 120
        self._thumb_visual_center_y = 58
        self._thumb_arrow_slot_w = 44

        lijevi_slot = tk.Frame(wrap, bg=CARD, width=self._thumb_arrow_slot_w, height=self._thumb_strip_h)
        lijevi_slot.pack(side="left", padx=(0, 8))
        lijevi_slot.pack_propagate(False)

        self.btn_prev = self._napravi_strelicu_canvas(
            lijevi_slot,
            "lijevo",
            lambda: self._thumb_scroll(-1)
        )
        self.btn_prev.place(relx=0.5, y=self._thumb_visual_center_y, anchor="center")
        self._postavi_strelicu(self.btn_prev, False)

        self.thumb_viewport = tk.Frame(wrap, bg=CARD, height=self._thumb_strip_h)
        self.thumb_viewport.pack(side="left", fill="x", expand=True)
        self.thumb_viewport.pack_propagate(False)

        self.thumb_canvas = tk.Canvas(self.thumb_viewport, height=self._thumb_strip_h, bg=CARD, highlightthickness=0)
        self.thumb_canvas.pack(fill="both", expand=True)

        self.thumb_placeholder_lbl = tk.Label(
            self.thumb_viewport,
            text="Nakon analize ovdje se pojavljuju sličice snimki.",
            bg=CARD,
            fg=TEXT_MUTE,
            font=(FONT, 10),
            justify="center",
            anchor="center"
        )
        self.thumb_placeholder_lbl.place(relx=0.5, y=self._thumb_visual_center_y, anchor="center")
        self.thumb_placeholder_lbl.lift()

        desni_slot = tk.Frame(wrap, bg=CARD, width=self._thumb_arrow_slot_w, height=self._thumb_strip_h)
        desni_slot.pack(side="left", padx=(8, 0))
        desni_slot.pack_propagate(False)

        self.btn_next = self._napravi_strelicu_canvas(
            desni_slot,
            "desno",
            lambda: self._thumb_scroll(1)
        )
        self.btn_next.place(relx=0.5, y=self._thumb_visual_center_y, anchor="center")
        self._postavi_strelicu(self.btn_next, False)

        self.thumb_inner = tk.Frame(self.thumb_canvas, bg=CARD)
        self.thumb_win = self.thumb_canvas.create_window((0, 0), window=self.thumb_inner, anchor="nw")
        self.thumb_inner.bind("<Configure>", self._thumb_inner_resize)
        self.thumb_canvas.bind("<Configure>", self._thumb_canvas_resize)

    def _thumb_canvas_resize(self, event):
        self.thumb_canvas.itemconfigure(self.thumb_win, height=event.height)
        if hasattr(self, "thumb_placeholder_lbl"):
            self.thumb_placeholder_lbl.place(relx=0.5, y=self._thumb_visual_center_y, anchor="center")
            self.thumb_placeholder_lbl.lift()
        self._azuriraj_thumb_strelice()

    def _thumb_inner_resize(self, event=None):
        self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox(self.thumb_win))
        self._azuriraj_thumb_strelice()

    def _thumb_ima_overflow(self):
        bbox = self.thumb_canvas.bbox(self.thumb_win)
        if not bbox or not self.rezultati:
            return False
        sirina_sadrzaja = bbox[2] - bbox[0]
        sirina_canvasa = self.thumb_canvas.winfo_width()
        return sirina_sadrzaja > sirina_canvasa + 4

    def _azuriraj_thumb_strelice(self):
        if not hasattr(self, "btn_prev") or not hasattr(self, "btn_next"):
            return
        overflow = self._thumb_ima_overflow()
        self._postavi_strelicu(self.btn_prev, overflow)
        self._postavi_strelicu(self.btn_next, overflow)
        if not overflow:
            self.thumb_canvas.xview_moveto(0)

    def _thumb_scroll(self, smjer):
        if not self._thumb_ima_overflow():
            self.thumb_canvas.xview_moveto(0)
            return
        self.thumb_canvas.xview_scroll(smjer * 5, "units")

    def _tip_promijenjen(self):
        self.var_model_putanja.set(cfg.PUTANJA_MODEL[self.var_tip.get()])

    def _odaberi_model(self):
        p = filedialog.askopenfilename(title="Odaberi .pth model", filetypes=[("PyTorch model", "*.pth"), ("Sve", "*.*")])
        if p:
            self.var_model_putanja.set(p)

    def _ucitaj_model(self):
        tip = self.var_tip.get()
        putanja = self.var_model_putanja.get()
        if not os.path.exists(putanja):
            messagebox.showerror("Greška", f"Model ne postoji:\n{putanja}\n\nPrvo istreniraj: python main.py train {tip}")
            return
        self.lbl_model_status.configure(text="⏳  Učitavanje modela...", fg=TEXT_MUTE)
        self._postavi_btn_run_aktivan(False)

        def rad():
            try:
                _, prag = core.ucitaj(tip, putanja)
                self.after(0, self._model_spreman, tip, prag)
            except Exception as e:
                self.after(0, self.lbl_model_status.configure, {"text": "✗  Model nije učitan", "fg": DANGER})
                self.after(0, messagebox.showerror, "Greška", f"Učitavanje nije uspjelo:\n{e}")
        threading.Thread(target=rad, daemon=True).start()

    def _model_spreman(self, tip, prag):
        self.lbl_model_status.configure(text=f"✓  {tip.upper()} učitan · prag odluke: {self._fmt_broj(prag, 3)}", fg=INFO)
        self._postavi_btn_run_aktivan(True)
        self.lbl_status.configure(text="Odaberi snimke i pokreni analizu.", fg=TEXT_MUTE)

    def _ulaz_promijenjen(self):
        self._resetiraj_rezultate()
        if core.model_ucitan():
            self._postavi_btn_run_aktivan(True)

    def _resetiraj_rezultate(self):
        self.rezultati = []
        self.odabrani = 0
        self.timestamp_analize = None
        self.prog.reset()
        self.lbl_status.configure(text="Rezultati su očišćeni. Pokreni novu analizu.", fg=TEXT_MUTE)
        self.lbl_presuda.configure(text="—", fg=TEXT_MUTE)
        self.lbl_snimka_prefix.configure(text="Snimka:")
        self.lbl_snimka_naziv.configure(text="—")
        for ttl, val in zip(self._detail_titles, self._detail_values):
            ttl.configure(text="—")
            val.configure(text="—")
        self.panel_orig.reset()
        self.panel_overlay.reset()
        for w in self.thumb_inner.winfo_children():
            w.destroy()
        if hasattr(self, "thumb_placeholder_lbl"):
            self.thumb_placeholder_lbl.place(relx=0.5, y=self._thumb_visual_center_y, anchor="center")
            self.thumb_placeholder_lbl.lift()
        self.thumb_canvas.xview_moveto(0)
        self._postavi_strelicu(self.btn_prev, False)
        self._postavi_strelicu(self.btn_next, False)
        self._postavi_btn_spremi_aktivan(False)
        self._thumb_imgs = []
        self._img_orig = None
        self._img_overlay = None

    def _status(self, poruka, napredak=None, success=False):
        # Obične status poruke idu kao jedan label.
        self.status_final_text.pack_forget()
        self.lbl_status.pack(anchor="w")
        self.lbl_status.configure(text=poruka, fg=INFO if success else TEXT_MUTE)
        if napredak is not None:
            # Kada je analiza završena, traka ostaje 100 %, ali boju ne
            # prebacujemo na neutralnu. Boja se kasnije veže uz trenutno
            # promatranu snimku: zelena za normalno, crvena za upalu.
            if success:
                self.prog.postavi(1.0)
            else:
                self.prog.postavi(napredak)
        self.update_idletasks()

    def _status_analiza_zavrsena(self, broj_upala, ukupno):
        # Završna poruka s djelomično boldanim brojevima bez dodatnog razmaka.
        self.lbl_status.pack_forget()
        self.status_final_text.configure(state="normal")
        self.status_final_text.delete("1.0", "end")
        self.status_final_text.insert("end", "Analiza završena: upala pluća detektirana je na ")
        self.status_final_text.insert("end", str(broj_upala), "bold")
        self.status_final_text.insert("end", " od ukupno ")
        self.status_final_text.insert("end", str(ukupno), "bold")
        self.status_final_text.insert("end", " snimki")
        self.status_final_text.configure(state="disabled")
        self.status_final_text.pack(anchor="w")
        self.prog.postavi(1.0)
        self.update_idletasks()

    def _pokreni(self):
        if not core.model_ucitan():
            messagebox.showwarning("Upozorenje", "Prvo učitaj model!")
            return
        if not self.snimke_zona.ima_datoteke:
            messagebox.showwarning("Upozorenje", "Nema odabranih snimki!")
            return
        putanje = self.snimke_zona.datoteke
        self._postavi_btn_run_aktivan(False)
        self._resetiraj_rezultate()
        self._status("Pokretanje analize...", 0.0)

        def rad():
            try:
                rez = []
                ukupno = len(putanje)
                for i, p in enumerate(putanje):
                    rez.append(core.obradi_snimku(p))
                    self.after(0, self._status, f"{os.path.basename(p)} ({i+1}/{ukupno})", (i + 1) / ukupno, False)
                self.after(0, self._analiza_gotova, rez)
            except Exception as e:
                self.after(0, messagebox.showerror, "Greška", f"Analiza nije uspjela:\n{e}")
                self.after(0, self._postavi_btn_run_aktivan, True)
        threading.Thread(target=rad, daemon=True).start()

    def _analiza_gotova(self, rez):
        self.rezultati = rez
        self.odabrani = 0
        self.timestamp_analize = datetime.now().strftime("%d.%m.%Y. u %H:%M:%S")
        self._izgradi_thumbs()
        if rez:
            self._prikazi_detalj(0)
        n_pneu = sum(1 for r in rez if r["oznaka"] == cfg.POZITIVNA_KLASA)
        self._status_analiza_zavrsena(n_pneu, len(rez))
        self._postavi_btn_spremi_aktivan(bool(rez))
        self._postavi_btn_run_aktivan(True)

    def _izgradi_thumbs(self):
        for w in self.thumb_inner.winfo_children():
            w.destroy()
        self.thumb_placeholder_lbl.place_forget()
        self._thumb_imgs = []
        for i, r in enumerate(self.rezultati):
            upala = r["oznaka"] == cfg.POZITIVNA_KLASA
            rub = DANGER if upala else OK
            cell = tk.Frame(self.thumb_inner, bg=CARD, width=88, height=112)
            cell.pack(side="left", padx=4, pady=2)
            cell.pack_propagate(False)
            oznaka = tk.Label(cell, text=f"#{i+1}", bg=CARD, fg=TEXT_MUTE, font=(FONT, 8, "bold"))
            oznaka.pack(anchor="center")
            okvir = tk.Frame(cell, bg=rub, highlightbackground=PRIMARY if i == self.odabrani else rub,
                             highlightthickness=3 if i == self.odabrani else 2, width=self.THUMB_BOX, height=self.THUMB_BOX)
            okvir.pack()
            okvir.pack_propagate(False)
            mini = ImageOps.contain(r["original"].copy(), (self.THUMB, self.THUMB), method=Image.BILINEAR)
            podloga = Image.new("RGB", (self.THUMB_BOX - 6, self.THUMB_BOX - 6), CARD_ALT)
            x = (podloga.width - mini.width) // 2
            y = (podloga.height - mini.height) // 2
            podloga.paste(mini, (x, y))
            foto = ImageTk.PhotoImage(podloga)
            self._thumb_imgs.append(foto)
            b = tk.Label(okvir, image=foto, bg=CARD_ALT, cursor="hand2", bd=0)
            b.pack(expand=True)
            b.bind("<Button-1>", lambda e, idx=i: self._prikazi_detalj(idx))
            pred = "UPALA" if upala else "NORMALNO"
            tk.Label(cell, text=pred, bg=CARD, fg=rub, font=(FONT, 8, "bold")).pack(anchor="center")
        self.thumb_canvas.xview_moveto(0)
        self._azuriraj_thumb_strelice()

    def _maks_dimenzije_slike(self):
        # Sigurna unutarnja mjera okvira: slika se nikad ne smije širiti preko ruba.
        w = max(260, self.panel_orig.okvir.winfo_width() - 22)
        h = max(200, self.panel_orig.okvir.winfo_height() - 22)
        return w, h

    def _skaliraj(self, pil):
        max_w, max_h = self._maks_dimenzije_slike()
        return ImageOps.contain(pil.convert("RGB"), (max_w, max_h), method=Image.BILINEAR)


    @staticmethod
    def _skrati_sredina(tekst, max_len=72):
        """Skrati dugačak tekst po sredini da ne mijenja geometriju GUI-ja."""
        if len(tekst) <= max_len:
            return tekst
        lijevo = max_len // 2 - 2
        desno = max_len - lijevo - 1
        return tekst[:lijevo] + "…" + tekst[-desno:]

    @staticmethod
    def _fmt_postotak(vrijednost):
        return f"{vrijednost * 100:.1f}".replace(".", ",") + " %"

    @staticmethod
    def _fmt_broj(vrijednost, decimale=3):
        return f"{vrijednost:.{decimale}f}".replace(".", ",")

    @staticmethod
    def _skrati_naziv(naziv, max_len=18):
        """Skrati naziv datoteke da kartica detalja ostane jednake veličine."""
        if len(naziv) <= max_len:
            return naziv
        if "." in naziv:
            baza, ekstenzija = naziv.rsplit(".", 1)
            ekstenzija = "." + ekstenzija
        else:
            baza, ekstenzija = naziv, ""
        prostor = max_len - len(ekstenzija) - 1
        if prostor < 6:
            return naziv[:max_len-1] + "…"
        return baza[:prostor] + "…" + ekstenzija

    def _prikazi_detalj(self, index):
        if not (0 <= index < len(self.rezultati)):
            return
        self.odabrani = index
        r = self.rezultati[index]
        upala = r["oznaka"] == cfg.POZITIVNA_KLASA
        akcent = DANGER if upala else OK

        # Progress traka nakon završene analize prati trenutno promatranu snimku:
        # crveno za nalaz upale, zeleno za normalan nalaz.
        if self.rezultati:
            self.prog.postavi(1.0, boja=akcent)

        self._img_orig = ImageTk.PhotoImage(self._skaliraj(r["original"]))
        self._img_overlay = ImageTk.PhotoImage(self._skaliraj(r["overlay"]))
        self.panel_orig.postavi_sliku(self._img_orig, akcent=akcent)
        self.panel_overlay.postavi_sliku(self._img_overlay, akcent=akcent)

        ikona = "⚠️  " if upala else "✅  "
        natpis = "UPALA PLUĆA" if upala else "NORMALNO"
        p_odluke = r["p_pneumonija"] if upala else r["p_normal"]
        self.lbl_presuda.configure(text=f"{ikona}{natpis} ({self._fmt_postotak(p_odluke)})", fg=akcent)
        self.lbl_snimka_prefix.configure(text=f"Snimka #{index+1}:")
        self.lbl_snimka_naziv.configure(text=self._skrati_sredina(r["naziv"], 72))

        if upala:
            detalji = [
                ("P (upala)", self._fmt_postotak(r["p_pneumonija"])),
                ("P (normalno)", self._fmt_postotak(r["p_normal"])),
                ("Prag", self._fmt_broj(r["prag"], 3)),
                ("Model", (core.trenutni_tip() or "?").upper()),
            ]
        else:
            detalji = [
                ("P (normalno)", self._fmt_postotak(r["p_normal"])),
                ("P (upala)", self._fmt_postotak(r["p_pneumonija"])),
                ("Prag", self._fmt_broj(r["prag"], 3)),
                ("Model", (core.trenutni_tip() or "?").upper()),
            ]

        for (naslov, vrijednost), ttl, val in zip(detalji, self._detail_titles, self._detail_values):
            ttl.configure(text=naslov)
            val.configure(text=vrijednost)

        for i, cell in enumerate(self.thumb_inner.winfo_children()):
            # cell children: label, okvir, label
            if len(cell.winfo_children()) >= 2:
                okvir = cell.winfo_children()[1]
                rub = DANGER if self.rezultati[i]["oznaka"] == cfg.POZITIVNA_KLASA else OK
                okvir.configure(highlightbackground=PRIMARY if i == index else rub,
                                highlightthickness=3 if i == index else 2)

    def _spremi(self):
        if not self.rezultati:
            messagebox.showwarning("Upozorenje", "Nema rezultata za spremanje!")
            return
        os.makedirs(cfg.DIR_REZULTATI, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fmt = self.var_format.get()
        if fmt == "xlsx":
            putanja = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")],
                                                   initialdir=os.path.join(os.getcwd(), cfg.DIR_REZULTATI),
                                                   initialfile=f"nalaz_{ts}.xlsx")
            if not putanja:
                return
            core.spremi_izvjestaj_xlsx(self.rezultati, putanja, self.timestamp_analize)
        else:
            putanja = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")],
                                                   initialdir=os.path.join(os.getcwd(), cfg.DIR_REZULTATI),
                                                   initialfile=f"nalaz_{ts}.txt")
            if not putanja:
                return
            core.spremi_izvjestaj_txt(self.rezultati, putanja, self.timestamp_analize)
        mapa_png = os.path.join(os.path.dirname(putanja), f"gradcam_{ts}")
        core.spremi_heatmape(self.rezultati, mapa_png)
        messagebox.showinfo("Spremljeno", f"Izvještaj:\n{putanja}\n\nGrad-CAM slike:\n{mapa_png}")

    def _toggle_fullscreen(self, event=None):
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)
        self.after(120, self._refresh_prikaz)

    def _izlaz_fullscreen(self, event=None):
        if self._fullscreen:
            self._fullscreen = False
            self.attributes("-fullscreen", False)
            self.after(120, self._refresh_prikaz)

    def _on_configure(self, event=None):
        # Debug prikaz trenutne rezolucije prozora + ponovno skaliranje odabrane snimke.
        if event is not None and event.widget is self:
            if DEBUG_REZOLUCIJA:
                self._debug_var.set(f"Prozor: {event.width} × {event.height}")
            self.after_idle(self._refresh_prikaz)

    def _refresh_prikaz(self):
        if self.rezultati:
            self._prikazi_detalj(self.odabrani)


if __name__ == "__main__":
    App().mainloop()