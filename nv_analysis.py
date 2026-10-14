#!/usr/bin/env python3
"""
NV-centre magnetometry (PHY405) - analysis script
-------------------------------------------------
1. Reads the exported LabVIEW ODMR spectra (xlsx).
2. Fits multi-Lorentzian dips  (same model as Origin "Multiple Peaks Fit").
3. Computes the coil field with the on-axis single-loop formula
        B = mu0 * N * I * R^2 / ( 2 (R^2 + z^2)^(3/2) )
   z = 0 mm for the z-axis coil (B1), z = 160 mm for the y-axis coil (B2).
4. Rotates the lab-frame field into the diamond frame and projects it on the
   four NV axes (Eq. 3a-3d of the manual); applies  B_i = dnu_i / (2 gamma).
5. Writes results.json and the figures used in the report.
"""
import json, os
import numpy as np
import openpyxl
from scipy.optimize import curve_fit, brentq
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/mnt/user-data/uploads/"
OUT  = "figures/"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "font.size": 9, "axes.grid": True, "grid.alpha": .25})

# ------------------------------------------------------------------ constants
MU0   = 4e-7 * np.pi          # T m / A
N     = 150                   # turns
R     = 0.200                 # m  (diameter 400 mm)
D_Z   = 0.000                 # m  z-axis coil  <-> diamond distance (B1)
D_Y   = 0.160                 # m  y-axis coil  <-> diamond distance (B2)
GAMMA = 28.0                  # MHz / mT
S2, S3 = np.sqrt(2), np.sqrt(3)

def coil_field(I, d):
    """On-axis field (mT) of an N-turn loop of radius R at axial distance d."""
    return 1e3 * MU0 * N * I * R**2 / (2.0 * (R**2 + d**2) ** 1.5)

K_Z = coil_field(1.0, D_Z)    # mT / A
K_Y = coil_field(1.0, D_Y)    # mT / A

def diamond_field(I_Z, I_Y):
    """Lab field (0, B_Y, B_Z) -> diamond frame (rotation by 45 deg about Z)."""
    B_Z, B_Y = K_Z * I_Z, K_Y * I_Y
    phi = np.deg2rad(45.0)
    Rm = np.array([[np.cos(phi), np.sin(phi), 0],
                   [-np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])
    return Rm @ np.array([0.0, B_Y, B_Z]), B_Y, B_Z

U = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]]) / S3   # NV1..NV4

def nv_projections(B):
    return U @ B

# ------------------------------------------------------------------ fitting
def load(fname):
    ws = openpyxl.load_workbook(DATA + fname, data_only=True).active
    a = np.array([(r[0], r[1]) for r in ws.iter_rows(min_row=2, max_col=2, values_only=True)
                  if r[0] is not None], float)
    return a[:, 0], a[:, 1]

def model(x, *p):
    y = p[0] * np.ones_like(x)
    for i in range((len(p) - 1) // 3):
        A, c, w = p[1 + 3 * i: 4 + 3 * i]
        y = y - A * (w / 2) ** 2 / ((x - c) ** 2 + (w / 2) ** 2)      # Lorentzian dip
    return y

def fit_peaks(x, y, centres):
    p0, lo, hi = [1.0], [0.95], [1.05]
    for c in centres:
        p0 += [0.03, c, 0.007]; lo += [0, c - 0.010, 0.002]; hi += [0.2, c + 0.010, 0.06]
    p, cov = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=50000)
    e = np.sqrt(np.diag(cov))
    rms = np.sqrt(np.mean((y - model(x, *p)) ** 2))
    pk = sorted([dict(c=p[2 + 3 * i], ce=e[2 + 3 * i], A=p[1 + 3 * i], w=p[3 + 3 * i],
                      we=e[3 + 3 * i]) for i in range(len(centres))], key=lambda d: d["c"])
    return dict(y0=p[0], peaks=pk, rms=rms, p=list(p))

# ------------------------------------------------------------------ runs
# (file, series, I_Z = current in B1 [A], I_Y = current in B2 [A], seeds [GHz])
RUNS = [
 ("B1_0_1", "A", 0, 1, [2.868, 2.873]), ("B1_0_2", "A", 0, 2, [2.868, 2.873]),
 ("B1_0_3", "A", 0, 3, [2.859, 2.868, 2.874, 2.883]),
 ("B1_0_4", "A", 0, 4, [2.855, 2.868, 2.874, 2.886]),
 ("B1_0_5", "A", 0, 5, [2.851, 2.867, 2.8745, 2.890]),
 ("B2_0_1", "B", 1, 0, [2.861, 2.881]), ("B2_0_2", "B", 2, 0, [2.852, 2.890]),
 ("B2_0_3", "B", 3, 0, [2.844, 2.899]), ("B2_0_4", "B", 4, 0, [2.836, 2.909]),
 ("B2_0_5", "B", 5, 0, [2.827, 2.919]),
 ("B1_B2_4",   "C", 4.0, 4.0, [2.822, 2.835, 2.849, 2.896, 2.909, 2.923]),
 ("B1_B2_5_1", "C", 5.0, 1.0, [2.827, 2.919]),
 ("B1_B2_1_5", "C", 1.0, 5.0, [2.841, 2.860, 2.882, 2.900]),
 ("B1_B2_1_4_1", "C", 1.4, 1.0, [2.857, 2.885]),
 ("B1_B2_1_1_4", "C", 1.0, 1.4, [2.860, 2.882]),
]

def predicted_dips(Bn, tol=0.5):
    """number of distinct ODMR dips for projections Bn (mT); tol in MHz of splitting"""
    s = sorted(round(2 * GAMMA * abs(b) / tol) for b in Bn)
    groups = sorted(set(s))
    return sum(1 if g == 0 else 2 for g in groups)

results, spectra = {}, {}
for f, ser, IZ, IY, seeds in RUNS:
    x, y = load(f + ".xlsx")
    spectra[f] = (x, y)
    r = fit_peaks(x, y, seeds)
    B, BY, BZ = diamond_field(IZ, IY)
    Bn = nv_projections(B)
    r.update(series=ser, IZ=IZ, IY=IY, BY=BY, BZ=BZ, B=list(B), Bn=list(Bn),
             split_pred=[2 * GAMMA * abs(b) for b in Bn], dips_pred=predicted_dips(Bn))
    results[f] = r

# ------------------------------------------------------------------ noise
xn, yn = load("Noise.xlsx")
noise = dict(mean=float(yn.mean()), std=float(yn.std(ddof=1)), ptp=float(np.ptp(yn)), n=len(yn),
             f0=float(xn[0]), f1=float(xn[-1]))

# ------------------------------------------------------------------ measured splittings
def pair(f, i, j):
    pk = results[f]["peaks"]; a, b = pk[i], pk[j]
    s = (b["c"] - a["c"]) * 1e3                      # MHz
    se = np.hypot(a["ce"], b["ce"]) * 1e3
    return dict(fm=a["c"], fp=b["c"], s=s, se=se, ctr=(a["c"] + b["c"]) / 2,
                B=s / (2 * GAMMA), Be=se / (2 * GAMMA))

meas = {}
for f in ["B2_0_1", "B2_0_2", "B2_0_3", "B2_0_4", "B2_0_5", "B1_B2_5_1", "B1_B2_1_4_1", "B1_B2_1_1_4"]:
    meas[f] = dict(pairs=[pair(f, 0, 1)])                               # one pair
for f in ["B1_0_3", "B1_0_4", "B1_0_5"]:
    meas[f] = dict(pairs=[pair(f, 0, 3)], strain=pair(f, 1, 2))          # outer pair + central doublet
for f in ["B1_0_1", "B1_0_2"]:
    meas[f] = dict(pairs=[], strain=pair(f, 0, 1))                       # Zeeman part unresolved
meas["B1_B2_4"] = dict(pairs=[pair("B1_B2_4", 0, 5), pair("B1_B2_4", 1, 4), pair("B1_B2_4", 2, 3)])
meas["B1_B2_1_5"] = dict(pairs=[pair("B1_B2_1_5", 0, 3), pair("B1_B2_1_5", 1, 2)])

# ------------------------------------------------------------------ calibration (linear fits)
def calib(files, which):
    I = np.array([results[f]["IZ" if which == "Z" else "IY"] for f in files], float)
    s = np.array([meas[f]["pairs"][0]["s"] for f in files])
    lr = linregress(I, s)
    k0 = np.sum(I * s) / np.sum(I ** 2)                                  # through origin
    k0e = np.sqrt(np.sum((s - k0 * I) ** 2) / (len(I) - 1) / np.sum(I ** 2))
    conv = S3 / (2 * GAMMA) if which == "Z" else np.sqrt(1.5) / (2 * GAMMA)   # MHz -> mT of lab field
    return dict(I=list(I), s=list(s), slope=lr.slope, slope_e=lr.stderr, icpt=lr.intercept,
                icpt_e=lr.intercept_stderr, r=lr.rvalue, k0=k0, k0e=k0e,
                kexp=k0 * conv, kexp_e=k0e * conv, conv=conv)

cal_Z = calib(["B2_0_1", "B2_0_2", "B2_0_3", "B2_0_4", "B2_0_5"], "Z")
cal_Y = calib(["B1_0_3", "B1_0_4", "B1_0_5"], "Y")
# effective geometry that would reproduce the measured constants
d_needed = brentq(lambda d: coil_field(1, d) - cal_Y["kexp"], 0.0, 0.6)
R_needed = MU0 * N / (2 * cal_Z["kexp"] * 1e-3)

# ------------------------------------------------------------------ full reconstruction of run (4 A, 4 A)
m = meas["B1_B2_4"]["pairs"]                       # splittings 102.1 / 74.5 / 47.0 MHz
Bn1, Bn23, Bn4 = m[0]["B"], m[1]["B"], m[2]["B"]
rec_Bz = S3 * Bn23                                 # NV2 = NV3 = -Bz/sqrt3
rec_b = (S3 * Bn1 - rec_Bz) / 2                    # NV1 = (2b + Bz)/sqrt3
chk = (Bn1 + Bn4) / 2                              # NV1 + NV4 = 2 NV2  ->  must equal Bn23
rec = dict(Bn1=Bn1, Bn23=Bn23, Bn4=Bn4, Bz=rec_Bz, b=rec_b, BY=S2 * rec_b, check=chk,
           kZ=rec_Bz / 4, kY=S2 * rec_b / 4, Bn4_pred=abs(2 * rec_b - rec_Bz) / S3)

# ------------------------------------------------------------------ case-4 current ratios
ratio_theory = S2 * K_Z / K_Y                     # I_Y / I_Z for Bx = By = Bz (corrected formula)
ratio_exp = S2 * cal_Z["kexp"] / cal_Y["kexp"]

json.dump(dict(K_Z=K_Z, K_Y=K_Y, results=results, meas=meas, noise=noise, cal_Z=cal_Z,
               cal_Y=cal_Y, d_needed=d_needed, R_needed=R_needed, rec=rec,
               ratio_theory=ratio_theory, ratio_exp=ratio_exp,
               naive_kZ_manual=1e3 * MU0 * N / (2 * np.pi) / R),
          open("results.json", "w"), indent=1, default=float)

# ================================================================== figures
def waterfall(files, fname, title, xlim, labels, step=0.045):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for k, f in enumerate(files):
        x, y = spectra[f]; r = results[f]; off = k * step
        ax.plot(x, y + off, ".", ms=2.5, color="C%d" % k)
        xs = np.linspace(x[0], x[-1], 1500)
        ax.plot(xs, model(xs, *r["p"]) + off, "-", lw=1, color="k")
        ax.text(xlim[1] - 0.001, 1.0 + off + 0.006, labels[k], ha="right", fontsize=8)
    ax.set_xlim(*xlim); ax.set_ylim(0.955, 1.0 + step * (len(files) - 1) + 0.016)
    ax.set_xlabel("Microwave frequency (GHz)"); ax.set_ylabel("Normalised PL (offset)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT + fname, dpi=200); plt.close(fig)

waterfall(["B1_0_%d" % i for i in range(1, 6)], "seriesA.png",
          "Series A: coil B1 off, coil B2 (y-axis) on", (2.82, 2.92),
          ["$I_{B2}=%d$ A" % i for i in range(1, 6)])
waterfall(["B2_0_%d" % i for i in range(1, 6)], "seriesB.png",
          "Series B: coil B2 off, coil B1 (z-axis) on", (2.81, 2.94),
          ["$I_{B1}=%d$ A" % i for i in range(1, 6)])
waterfall(["B1_B2_5_1", "B1_B2_1_4_1", "B1_B2_1_1_4", "B1_B2_1_5", "B1_B2_4"], "seriesC.png",
          "Series C: both coils on", (2.81, 2.935),
          ["$(I_{B1},I_{B2})=(5,1)$ A", "(1.4, 1) A", "(1, 1.4) A", "(1, 5) A", "(4, 4) A"], step=0.04)

# individual annotated spectra (Case 3 example, Case 4, and 6-dip)
def single(f, fname, labels, xlim, ttl):
    x, y = spectra[f]; r = results[f]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.plot(x, y, ".", ms=3, label="data"); xs = np.linspace(x[0], x[-1], 2000)
    ax.plot(xs, model(xs, *r["p"]), "r-", lw=1.2, label="Lorentzian fit")
    for pk, lab in zip(r["peaks"], labels):
        ax.annotate(lab, (pk["c"], 1 - pk["A"]), xytext=(pk["c"], 0.9915 - 0.0), ha="center", fontsize=7,
                    arrowprops=dict(arrowstyle="-", lw=.5, color="gray"))
    ax.set_xlim(*xlim); ax.set_ylim(0.955, 1.004)
    ax.set_xlabel("Microwave frequency (GHz)"); ax.set_ylabel("Normalised PL")
    ax.set_title(ttl, fontsize=9); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT + fname, dpi=200); plt.close(fig)

single("B1_B2_4", "case_44.png", [r"$f_{1-}$", r"$f_{2-},f_{3-}$", r"$f_{4-}$", r"$f_{4+}$", r"$f_{2+},f_{3+}$", r"$f_{1+}$"],
       (2.81, 2.935), r"$(I_{B1},I_{B2})=(4,4)$ A  - six dips: three distinct NV projections")
single("B1_B2_1_5", "case_15.png", [r"$f_{1-}$", r"$f_{2-},f_{3-},f_{4-}$", r"$f_{2+},f_{3+},f_{4+}$", r"$f_{1+}$"],
       (2.81, 2.93), r"$(I_{B1},I_{B2})=(1,5)$ A  - four dips (Case 4: $B_x\approx B_y\approx B_z$)")

# splitting vs current
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
for a, cal, k, lab, col in [(ax[0], cal_Z, K_Z, "Series B (z-axis coil B1)", "C0"),
                            (ax[1], cal_Y, K_Y, "Series A (y-axis coil B2)", "C1")]:
    I = np.array(cal["I"]); s = np.array(cal["s"])
    se = [meas[f]["pairs"][0]["se"] for f in (["B2_0_%d" % i for i in range(1, 6)] if col == "C0"
                                                else ["B1_0_%d" % i for i in (3, 4, 5)])]
    a.errorbar(I, s, yerr=se, fmt="o", color=col, label="measured")
    xx = np.linspace(0, 5.3, 20)
    a.plot(xx, cal["k0"] * xx, "-", color=col, label="fit (through origin)")
    conv = 2 * GAMMA / S3 if col == "C0" else 2 * GAMMA * np.sqrt(2 / 3)
    a.plot(xx, conv * k * xx, "k--", label="corrected coil formula")
    a.set_xlabel("Current (A)"); a.set_ylabel(r"Splitting $\Delta\nu$ (MHz)"); a.set_title(lab, fontsize=8)
    a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT + "splitting_vs_I.png", dpi=200); plt.close(fig)

# noise
fig, ax = plt.subplots(figsize=(6.0, 2.2))
ax.plot(xn, yn, ".-", ms=3, lw=.6); ax.set_ylim(0.9985, 1.0015)
ax.axhline(noise["mean"], color="r", lw=.8)
ax.set_xlabel("Microwave frequency (GHz)"); ax.set_ylabel("Normalised PL")
fig.tight_layout(); fig.savefig(OUT + "noise.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    print("K_Z = %.5f mT/A, K_Y = %.5f mT/A" % (K_Z, K_Y))
    for f, r in results.items():
        print(f, [round(p["c"], 4) for p in r["peaks"]], "rms %.4f" % r["rms"])
    print(json.dumps(dict(cal_Z=cal_Z, cal_Y=cal_Y, rec=rec, d_needed=d_needed, R_needed=R_needed,
                          ratio_theory=ratio_theory, ratio_exp=ratio_exp, noise=noise), indent=1, default=float))
