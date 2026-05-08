"""
Sheet Metal Transition  –  Streamlit Web App
============================================
Run:   streamlit run sheet_metal_app.py
Then open  http://localhost:8501  in any browser.

On Android / iPhone on the SAME WiFi:
  1. Find your PC's local IP  (ipconfig → IPv4 address, e.g. 192.168.1.5)
  2. Open  http://192.168.1.5:8501  in Chrome / Safari
  3. Chrome menu → "Add to Home screen"  → works like a native app

Cloud deploy (free, shareable link):
  https://streamlit.io/cloud  → connect your GitHub repo → done
"""

import io
import os
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Import geometry engine from the tool file ─────────────────────
# Add both the file's directory and cwd to path (handles Streamlit Cloud)
_here = os.path.dirname(os.path.abspath(__file__))
for _p in [_here, os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from sheet_metal_transition import (
        build_transition,
        unfold_triangles,
        get_all_boundaries,
        map_2d_pts,
        surface_area,
        get_reference_points,
        export_dxf,
        _make_preview,
        _make_workshop,
        calculate_bend_data,
        thickness_report,
    )
except Exception as _import_err:
    import traceback
    st.error("**Import error — could not load geometry engine:**")
    st.code(traceback.format_exc())
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  Page configuration  (must be FIRST Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Sheet Metal Transition",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help":    None,
        "Report a bug": None,
        "About": (
            "### Sheet Metal Transition Tool\n"
            "Surface development for rectangle-to-circle transitions.\n\n"
            "Generates flat patterns, workshop drawings, coordinates and DXF files."
        ),
    },
)

# ── Mobile-friendly CSS ────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Hide Streamlit branding, footer, GitHub badge ── */
  footer                              { visibility: hidden !important; height: 0; }
  #MainMenu                           { visibility: hidden !important; }
  header[data-testid="stHeader"]      { background: transparent; }
  div[data-testid="stDecoration"]     { display: none !important; }
  div[data-testid="stToolbar"]        { display: none !important; }
  div[data-testid="stToolbarActions"] { display: none !important; }
  #stDecoration                       { display: none !important; }
  /* Badge selectors — all known Streamlit Cloud versions */
  .viewerBadge_container__1QSob  { display: none !important; }
  .styles_viewerBadge__1yB5_     { display: none !important; }
  [class*="viewerBadge"]         { display: none !important; }
  [class*="badge_container"]     { display: none !important; }
  [class*="BadgeContainer"]      { display: none !important; }
  /* Nuclear option — white overlay covering bottom-right corner */
  body::after {
    content: "";
    position: fixed;
    bottom: 0; right: 0;
    width: 260px; height: 52px;
    background: white;
    z-index: 99999;
  }

  /* Larger touch targets on mobile */
  div[data-testid="stNumberInput"] input  { font-size: 17px !important; height: 44px; }
  div[data-testid="stSlider"]             { padding-top: 6px; }
  div[data-testid="stButton"] > button    { height: 52px; font-size: 17px; font-weight: 600; }
  div[data-testid="stDownloadButton"] > button {
        height: 48px; font-size: 15px; border-radius: 8px; }
  /* Compact metric cards */
  div[data-testid="metric-container"]     { background: #f0f4ff;
        border-radius: 8px; padding: 10px 14px; }
  /* Header */
  .app-header { text-align: center; padding: 8px 0 4px 0; }
  .app-header h1 { font-size: 1.6rem; margin-bottom: 2px; }
  .app-header p  { color: #666; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  Top navigation
st.divider()

# ══════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <h1>⚙ Sheet Metal Transition</h1>
  <p>Rectangle → Circle  |  Concentric &amp; Eccentric  |  Surface Development</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
#  Input panel
# ══════════════════════════════════════════════════════════════════
with st.container():
    st.subheader("Transition Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Rectangle (base)**")
        rw = st.number_input("Width  (mm)",  min_value=10.0, max_value=5000.0,
                             value=400.0, step=10.0, key="rw")
        rh = st.number_input("Height (mm)", min_value=10.0, max_value=5000.0,
                             value=300.0, step=10.0, key="rh")

    with col2:
        st.markdown("**Circle (top)**")
        cd = st.number_input("Diameter (mm)", min_value=10.0, max_value=5000.0,
                             value=200.0, step=10.0, key="cd")
        h  = st.number_input("Transition Height (mm)", min_value=10.0, max_value=5000.0,
                             value=250.0, step=10.0, key="h")

    with col3:
        st.markdown("**Eccentricity**  *(set 0 for concentric)*")
        ox = st.number_input("Circle X Offset (mm)", min_value=-2000.0, max_value=2000.0,
                             value=0.0, step=5.0, key="ox",
                             help="Positive = circle shifts right of rect centre")
        oy = st.number_input("Circle Y Offset (mm)", min_value=-2000.0, max_value=2000.0,
                             value=0.0, step=5.0, key="oy",
                             help="Positive = circle shifts up from rect centre")

    n = st.select_slider(
        "Circle Segments  (more = smoother pattern, slower)",
        options=[8, 12, 16, 20, 24, 32, 36, 48, 60, 72],
        value=24,
        key="n",
    )

    ec_label = ("Concentric" if not (ox or oy)
                else "Eccentric  (offset {:.0f}, {:.0f} mm)".format(ox, oy))
    st.caption("Type: **{}**".format(ec_label))

st.divider()

# ── Material / thickness parameters ──────────────────────────────
with st.container():
    st.subheader("Material & Bending Parameters")
    st.caption(
        "These affect bend allowance at the 4 corner fold lines (K1–K4), "
        "seam edge preparation, and blank size correction. "
        "The flat pattern geometry above is always at the **mid-surface (neutral axis)**."
    )

    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        t_mm = st.number_input(
            "Sheet Thickness  (mm)", min_value=0.3, max_value=50.0,
            value=1.5, step=0.5, key="t_mm",
            help="Thickness of the sheet metal you will use"
        )
    with tc2:
        r_bend_mm = st.number_input(
            "Inside Bend Radius  (mm)", min_value=0.0, max_value=50.0,
            value=round(1.5 * 1.0, 1), step=0.5, key="r_bend",
            help="Inside radius at K-line folds. Rule of thumb: 1× thickness for mild steel"
        )
    with tc3:
        k_factor = st.selectbox(
            "K-Factor  (bend method)",
            options=[0.33, 0.38, 0.44, 0.50],
            index=2,
            key="kfactor",
            format_func=lambda x: {
                0.33: "0.33 — Coining / bottoming",
                0.38: "0.38 — Bottom bending",
                0.44: "0.44 — Air bending  (most common)",
                0.50: "0.50 — Theoretical / soft material",
            }[x],
            help="K-factor sets where the neutral axis sits inside the thickness"
        )
    with tc4:
        material = st.selectbox(
            "Material",
            ["Mild Steel (MS)", "Stainless Steel (SS)", "Aluminium",
             "Galvanised Steel", "Copper / Brass"],
            key="material",
            help="Used for edge-prep and minimum-radius guidance only"
        )

st.divider()

# ══════════════════════════════════════════════════════════════════
#  Generate button
# ══════════════════════════════════════════════════════════════════
gen = st.button("⚙  Generate Development", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  Compute  (on Generate click, or re-use cached result)
# ══════════════════════════════════════════════════════════════════

def _compute(rw, rh, cd, h, ox, oy, n, t_mm, r_bend_mm, k_factor):
    tris, rc, cp, rpp, cl3d, sp3d = build_transition(rw, rh, cd, h, ox, oy, n)
    unfolded, placed = unfold_triangles(tris)
    boundaries       = get_all_boundaries(unfolded)
    seam_2d, c2d     = map_2d_pts(sp3d, cl3d, placed)
    area             = surface_area(tris)
    c_refs, r_refs, k_refs = get_reference_points(cp, rpp, rc, placed)

    all2d  = [p for tri in unfolded for p in tri]
    xs     = [float(p[0]) for p in all2d]
    ys     = [float(p[1]) for p in all2d]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)

    bend_data = calculate_bend_data(tris, cl3d, t_mm, r_bend_mm, k_factor)
    thick_rep = thickness_report(t_mm, r_bend_mm, k_factor,
                                 bend_data, rw, rh, cd, h)

    return dict(
        tris=tris, rc=rc, cp=cp, rpp=rpp, cl3d=cl3d,
        unfolded=unfolded, boundaries=boundaries,
        seam_2d=seam_2d, c2d=c2d, area=area,
        c_refs=c_refs, r_refs=r_refs, k_refs=k_refs,
        span_x=span_x, span_y=span_y,
        thick_rep=thick_rep,
    )


def _fig_to_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _dxf_to_bytes(r, rw, rh, cd, h, ox, oy, n):
    try:
        import ezdxf  # noqa: F401
    except ImportError:
        return None, "ezdxf not installed"
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        tmpname = tmp.name
    try:
        export_dxf(
            r["unfolded"], r["boundaries"], r["seam_2d"], r["c2d"],
            r["c_refs"], r["r_refs"], r["k_refs"],
            r["span_x"], r["span_y"],
            rw, rh, cd, h, ox, oy, n,
            tmpname,
        )
        with open(tmpname, "rb") as f:
            data = f.read()
        return data, None
    except Exception as exc:
        return None, str(exc)
    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)


def _csv_bytes(c_refs, r_refs, k_refs, rw, rh, cd, h, ox, oy):
    lines = [
        "Sheet Metal Transition -- Reference Coordinates",
        "Rectangle,{} x {} mm".format(rw, rh),
        "Circle D,{} mm".format(cd),
        "Height,{} mm".format(h),
        ("Type,Concentric" if not (ox or oy)
         else "Offset,({}, {}) mm".format(ox, oy)),
        "",
        "Label,X (mm),Y (mm),Type",
    ]
    for lbl, q in c_refs:
        lines.append("{},{:.3f},{:.3f},Circle arc".format(lbl, q[0], q[1]))
    for lbl, q in r_refs:
        lines.append("{},{:.3f},{:.3f},Rect perimeter".format(lbl, q[0], q[1]))
    for lbl, q in k_refs:
        lines.append("{},{:.3f},{:.3f},Corner fold".format(lbl, q[0], q[1]))
    return "\n".join(lines).encode("utf-8")


# ══════════════════════════════════════════════════════════════════
#  Fabrication guide  (plain-language, layman-friendly)
# ══════════════════════════════════════════════════════════════════

def _show_thickness_tab(tr, t, r_bend, k_factor, material, cd, span_x, span_y):
    """Thickness corrections, bend allowance table, and seam edge prep."""

    import pandas as pd

    st.markdown("## Thickness & Bending Corrections")
    st.caption(
        "The flat pattern coordinates are computed at the **mid-surface** "
        "(neutral axis). The corrections below tell you how to adjust for "
        "real material thickness."
    )

    if not tr:
        st.warning("Generate the development first.")
        return

    # ── Key facts ────────────────────────────────────────────────
    st.markdown("### Your material parameters")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Thickness",         "{} mm".format(t))
    f2.metric("Inside Bend Radius","{} mm".format(r_bend))
    f3.metric("K-Factor",          str(k_factor))
    f4.metric("Material",          material)

    st.divider()

    # ── What mid-surface means ────────────────────────────────────
    st.markdown("### What the flat pattern coordinates represent")
    st.info(
        "All X, Y coordinates in the pattern are measured at the **mid-surface** "
        "(halfway through the thickness). This is the neutral axis — the line that "
        "neither stretches nor compresses during bending or rolling.\n\n"
        "- If you **mark and cut from the OUTSIDE face**: add **{:.2f} mm** to all "
        "outward dimensions (= t/2 = {:.1f}/2).\n"
        "- If you **mark and cut from the INSIDE face**: subtract **{:.2f} mm** from "
        "all dimensions.\n"
        "- For thin sheet (t < 1.5 mm) this difference is negligible — ignore it.".format(
            t / 2, t, t / 2)
    )

    st.divider()

    # ── Bend allowance at each K-line ────────────────────────────
    st.markdown("### Bend allowance at each corner fold line (K1–K4)")
    st.markdown(
        "When you bend the metal at a K-line, the material on the **outside** "
        "of the bend stretches slightly. This means the flat blank needs to be "
        "**slightly larger** than what the pattern shows. The amount extra is called "
        "the **Bend Allowance (BA)**. The **Bend Deduction (BD)** tells you how much "
        "to subtract from each flat leg if you are dimensioning from the outside faces."
    )

    bend_data = tr.get("bend_data", [])
    if bend_data:
        rows = []
        for d in bend_data:
            if d.get("bend_angle_deg") is not None:
                rows.append({
                    "Corner": d["label"],
                    "Bend Angle (°)":   d["bend_angle_deg"],
                    "Bend Allowance (mm)": d["bend_allowance"],
                    "Bend Deduction (mm)": d["bend_deduction"],
                    "Meaning": (
                        "Add {:.2f} mm to blank at this fold".format(d["bend_allowance"])
                        if d["bend_allowance"] else "—"
                    ),
                })
            else:
                rows.append({
                    "Corner": d["label"],
                    "Bend Angle (°)":      "—",
                    "Bend Allowance (mm)": "—",
                    "Bend Deduction (mm)": "—",
                    "Meaning": d.get("note", ""),
                })

        df_bend = pd.DataFrame(rows)
        st.dataframe(df_bend, use_container_width=True, hide_index=True)

        total_ba = tr.get("total_bend_allowance", 0)
        total_bd = tr.get("total_bend_deduction", 0)

        st.success(
            "**Total extra material across all 4 bends:**  "
            "Bend Allowance = **{:.2f} mm**   |   "
            "Bend Deduction = **{:.2f} mm**\n\n"
            "In practice: cut the flat blank **{:.1f} mm longer** than the "
            "pattern dimensions to have enough material for all bends.".format(
                total_ba, total_bd, total_ba)
        )
    else:
        st.warning("Bend data not available. Re-generate the pattern.")

    st.divider()

    # ── Circle edge rolling ───────────────────────────────────────
    st.markdown("### Circle edge — rolling allowance")
    st.markdown(
        "Rolling does not significantly change the flat length of the material "
        "(unlike a sharp bend). However the **diameter you set on the roller** "
        "depends on which face is the reference:"
    )

    circ_n = tr.get("circle_circ_neutral", 0)
    circ_i = tr.get("circle_circ_inside",  0)
    circ_o = tr.get("circle_circ_outside", 0)

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Neutral axis circumference",
               "{:.1f} mm".format(circ_n),
               help="= π × D. Pattern is developed at this length.")
    cc2.metric("Inside face circumference",
               "{:.1f} mm".format(circ_i),
               "−{:.1f} mm vs neutral".format(circ_n - circ_i),
               help="= π × (D − t). Set roller to this if measuring from inside.")
    cc3.metric("Outside face circumference",
               "{:.1f} mm".format(circ_o),
               "+{:.1f} mm vs neutral".format(circ_o - circ_n),
               help="= π × (D + t). Set roller to this if measuring from outside.")

    st.info(
        "**Rule of thumb:** Roll to the **inside diameter** = (Circle D − thickness) = "
        "**{:.1f} mm**.  The outside face will then equal the specified circle diameter.".format(
            cd - t)
    )

    st.divider()

    # ── Seam edge preparation ─────────────────────────────────────
    st.markdown("### Seam edge preparation for welding")

    seam_prep = tr.get("seam_edge_prep", "")

    prep_details = {
        "Square butt — no bevel needed": dict(
            icon="✅",
            detail=(
                "For t ≤ 1.5 mm — leave the cut edges square (90°).\n"
                "Clean with a file or flap disc to remove burrs.\n"
                "Gap between edges: 0 – 0.5 mm for MIG/TIG."
            ),
            diagram="Edge: |  | (square, no angle)"),
        "Light chamfer 15-20° each edge (half-V)": dict(
            icon="⚠️",
            detail=(
                "For t 1.5–4 mm — grind a 15–20° chamfer on each seam edge "
                "(both pieces).\n"
                "Total included angle ≈ 30–40°.\n"
                "Root gap: 1–2 mm. Add backing strip if available."
            ),
            diagram="Edge: /  \\ (light chamfer each side)"),
        "Single-V bevel 30-35° each edge": dict(
            icon="⚠️",
            detail=(
                "For t 4–8 mm — grind a 30–35° bevel on each seam edge.\n"
                "Total included angle ≈ 60–70°  (classic single-V groove).\n"
                "Root gap: 2–3 mm. Root face: 1–1.5 mm flat at root.\n"
                "Weld in two or more passes. Back-gouge and weld back side."
            ),
            diagram="Edge:  /    \\  (V-groove)"),
        "Full V or double-V bevel 35-45° each edge": dict(
            icon="🔴",
            detail=(
                "For t > 8 mm — 35–45° bevel each edge, double-V if accessible "
                "from both sides.\n"
                "Root face 2 mm. Root gap 3 mm. Multi-pass weld.\n"
                "Preheat to 100–150 °C for MS plate > 12 mm.\n"
                "Post-weld check with dye-penetrant or UT if structural."
            ),
            diagram="Edge:  /    \\  with root land (heavy V)"),
    }

    info = prep_details.get(seam_prep, dict(icon="ℹ️", detail=seam_prep, diagram=""))
    st.markdown("**{} Recommendation for {} mm {}:  {}**".format(
        info["icon"], t, material, seam_prep))
    st.markdown(info["detail"])
    st.code(info["diagram"], language=None)

    st.divider()

    # ── Minimum bend radius ───────────────────────────────────────
    st.markdown("### Minimum inside bend radius")
    r_min = tr.get("min_inside_radius", t * 0.5)

    mat_factors = {
        "Mild Steel (MS)":       (0.5, 1.0),
        "Stainless Steel (SS)":  (1.0, 2.0),
        "Aluminium":             (0.5, 1.5),
        "Galvanised Steel":      (0.5, 1.2),
        "Copper / Brass":        (0.3, 0.8),
    }
    lo_f, hi_f = mat_factors.get(material, (0.5, 1.0))

    r_lo = round(lo_f * t, 1)
    r_hi = round(hi_f * t, 1)

    if r_bend < r_lo:
        st.error(
            "Your inside bend radius ({} mm) is **below the minimum recommended** "
            "for {} at {} mm thickness ({} mm). "
            "Risk of cracking at the bend. Increase to at least {:.1f} mm.".format(
                r_bend, material, t, r_lo, r_lo)
        )
    elif r_bend > r_hi * 3:
        st.warning(
            "Your inside bend radius ({} mm) is quite large. "
            "The bend will be very gradual — this may be intentional for large-radius forming.".format(
                r_bend)
        )
    else:
        st.success(
            "Inside bend radius {} mm is within the recommended range "
            "({} – {} mm) for {} at {} mm thickness.".format(
                r_bend, r_lo, r_hi, material, t)
        )

    st.divider()

    # ── Blank size summary ────────────────────────────────────────
    st.markdown("### Recommended blank (sheet) size to order")

    handle   = 50   # handling margin each side
    blank_w  = round(span_x + tr.get("total_bend_allowance", 0) + handle, 0)
    blank_h  = round(span_y + tr.get("total_bend_allowance", 0) + handle, 0)

    bc1, bc2 = st.columns(2)
    bc1.metric("Minimum blank width",  "{:.0f} mm".format(blank_w))
    bc2.metric("Minimum blank height", "{:.0f} mm".format(blank_h))
    st.caption(
        "Includes {:.1f} mm total bend allowance + {} mm handling margin each side. "
        "Always order slightly larger — trim to fit is easier than piecing.".format(
            tr.get("total_bend_allowance", 0), handle // 2)
    )


def _show_fabrication_guide(r, rw, rh, cd, h, ox, oy, n):
    """Render the full step-by-step fabrication guide."""

    n_c  = len(r["c_refs"])
    n_r  = len(r["r_refs"])
    seam = "right side" if r["seam_2d"][0] is not None else "edge of pattern"
    ec   = "concentric" if not (ox or oy) else "eccentric"

    st.markdown("## Step-by-Step Fabrication Guide")
    st.caption(
        "Rect {}x{}mm  →  Circle D{}mm  |  Height {}mm  |  {} transition".format(
            rw, rh, cd, h, ec.title())
    )

    # ── TOOLS ────────────────────────────────────────────────────
    with st.expander("TOOLS YOU WILL NEED  (click to expand)", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**Marking out**
- Steel rule (longer than {:.0f} mm)
- Try square or engineer's square
- Scriber or sharp pencil
- Centre punch + hammer
- Flexible batten / thin steel strip (for curves)
- Dividers or trammel
- Chalk line (optional)

**Cutting**
- Metal snips (for thin sheet up to ~1.2 mm)
- Angle grinder with cutting disc (thicker)
- Plasma cutter or laser (if available)
- File for dressing edges
            """.format(max(r["span_x"], r["span_y"])))

        with col_b:
            st.markdown("""
**Forming**
- Bending brake or folder (for straight bends)
- Hammer + backing bar (manual bending)
- Rolling machine (for circle edge)
- Mallet / body hammer

**Welding & finishing**
- MIG / TIG / arc welder
- Clamps, tacking magnets
- Angle grinder for weld dressing
- PPE: gloves, eye protection, apron

**Optional but helpful**
- Plywood template board
- Self-adhesive paper (to stick pattern to sheet)
            """)

    st.divider()

    # ── PHASE 1 ──────────────────────────────────────────────────
    st.markdown("## PHASE 1 — Prepare Your Sheet Metal")

    st.markdown("""
### Step 1 – Choose the right sheet size
You need a flat sheet at least **{:.0f} mm wide** and **{:.0f} mm tall**.
Add 50 mm extra on each side for clamping and handling.

> **Tip:** The pattern is a curved fan shape — it is NOT a simple rectangle.
> The flat blank you need is roughly {:.0f} mm × {:.0f} mm.

### Step 2 – Lay the sheet flat
Place the sheet on a clean, flat workbench or on the floor.
Make sure it cannot slide while you mark it.
Clean off any oil, rust scale or paint from the surface so your scriber lines show clearly.

### Step 3 – Mark the datum corner  *(your starting point)*
Pick a corner of your sheet metal near the bottom-left area.
This will be point **C01 = (0, 0)** — your reference for all measurements.

Using a centre punch and hammer, make a small dent at this corner.
Label it **C01** with a marker pen.

> All other measurements are made FROM this single point.
> Positive X goes to the RIGHT.  Positive Y goes UPWARD.
> Negative Y goes DOWNWARD from C01.
    """.format(
        r["span_x"] + 100, r["span_y"] + 100,
        r["span_x"] + 100, r["span_y"] + 100,
    ))

    st.divider()

    # ── PHASE 2 ──────────────────────────────────────────────────
    st.markdown("## PHASE 2 — Mark All Reference Points on the Sheet")

    st.info(
        "Open the **Coordinates** tab and download the CSV.  "
        "It lists every point label, X value and Y value.  "
        "Work through the table row by row as described below."
    )

    st.markdown("""
### Step 4 – Understand what you are marking
The CSV has three groups of points:

| Colour in table | Label | What it is | What you do with it |
|---|---|---|---|
| Pink  | **C01 – C{nc}** | Circle arc edge (inner) | Join smoothly — this becomes the circular opening |
| Orange | **R01 – R{nr}** | Rect perimeter edge (outer) | Join with straight lines — this becomes the rectangular base |
| Blue | **K1 – K4** | Corner fold lines | You bend the metal here to form the 4 walls |

### Step 5 – Mark the C points  *(circle arc edge)*

1. Look at the CSV row for **C01** — it reads X=0, Y=0. That is already punched in Step 3.
2. For **C02**: Read its X value (e.g. 26.1 mm) and Y value (e.g. 0.0 mm).
   - From your C01 punch mark, measure **26.1 mm to the RIGHT** along the sheet.
   - Then measure **0.0 mm UP** (no movement vertically).
   - Make a small punch mark there. Write "C02" next to it.
3. Repeat for **C03, C04 … C{nc}**:
   - Positive X → measure that many mm to the **RIGHT** of C01.
   - Positive Y → measure that many mm **UP** from C01.
   - Negative Y → measure that many mm **DOWN** from C01.
4. After all C points are punched, place a flexible steel strip or thin batten
   so it bends smoothly through all the punch marks.
   Scribe a smooth curve through all C points.
   **This curved line is the circle edge — the inner cut line.**

### Step 6 – Mark the R points  *(rect perimeter edge)*

Do the same thing for **R01 – R{nr}** using the CSV X and Y values.
After punching all R points:
- Connect adjacent R points with **straight lines** using your steel rule.
- Where R points are close together, one straight line covers several points.
  **These straight lines form the outer edge — the rect perimeter cut line.**

### Step 7 – Mark the K points  *(corner fold lines)*

Mark **K1, K2, K3, K4** from the CSV exactly the same way.
After punching all 4 K points:
- Rule a **straight line** from each K point all the way across the piece
  from the circle arc edge to the rect perimeter edge.
- These 4 lines show where the metal will be bent (folded) later.
- Mark these lines clearly with a different colour marker or a deeper scribed line.

### Step 8 – Mark the seam line

The **seam line** runs from one end of the circle arc edge to the nearby point
on the rect perimeter edge.
It appears as a **yellow dash-dot line** on the workshop drawing.
Mark it on the sheet — this is where you will cut the pattern open
and later weld it closed.
    """.format(nc=n_c, nr=n_r))

    st.divider()

    # ── PHASE 3 ──────────────────────────────────────────────────
    st.markdown("## PHASE 3 — Cut Out the Flat Pattern")

    st.markdown("""
### Step 9 – Double-check before cutting  *(important!)*
Before you cut anything, stand back and look at the shape you have drawn.
It should look like a **curved fan or sector** (similar to a slice of pie, but irregular).
- The inner curved line (C points) = shorter arc
- The outer jagged boundary (R points) = longer, larger arc
- 4 straight fold lines (K1–K4) crossing the piece
- 1 seam line at one edge

If it looks very wrong (lines crossing, points out of order), re-check
a few points against the CSV before cutting.

### Step 10 – Cut along the C arc (inner edge)
Cut carefully along the smooth curve you scribed through the C points.
This curved cut will eventually become the circular opening of your transition.

> **Snips:** Cut just outside the line, then file back to the line.
> **Grinder:** Use a cutting disc, cut slightly wide and clean up with a flap disc.
> **Plasma:** Follow the scribed line directly.

### Step 11 – Cut along the R boundary (outer edge)
Cut along the straight lines you ruled through the R points.
These are straight cuts, so they are easier.

### Step 12 – Cut the seam line
Cut along the seam line.
This separates the two ends of the flat pattern —
you now have one flat piece that can be formed into the transition shape.

> **Do NOT throw away any off-cuts yet** — they may be useful for testing bends.

### Step 13 – Dress all edges
Use a file or flap disc to remove any burrs from the cut edges.
Sharp burrs will cut your hands and prevent clean welding later.
    """)

    st.divider()

    # ── PHASE 4 ──────────────────────────────────────────────────
    st.markdown("## PHASE 4 — Bend the Four Corner Fold Lines")

    st.markdown("""
The transition piece has **4 flat triangular walls** — one for each side of the rectangle.
These are formed by bending the flat sheet at the 4 fold lines (K1–K4).

### Step 14 – Understand what the bends look like
Each fold line (K1–K4) runs from a point on the circle arc edge to a corner of the
rectangle base. When you bend along each fold line, that triangular section of metal
lifts up and forms one wall of the transition.

The 4 bends do NOT need to be sharp 90° folds.
They are gentle bends — the angle depends on your transition height and rectangle size.
The metal will naturally find the right angle when you form the circular opening in the next step.

### Step 15 – Make the first bend (K1)
1. Place the flat sheet on a bending brake so the fold line (K1) sits exactly
   at the bend edge of the brake.
2. Clamp it in place.
3. Lift the brake handle gently — you only need a small bend angle to start.
   Do not over-bend. Aim for about 20–30° as a starting point.
4. Remove the sheet, hold it up and look at it — it should start to look
   like a folded piece of cardboard.

> **No bending brake?** Clamp the sheet to a sturdy steel bar along the fold line.
> Use a mallet to gently tap the metal down over the edge of the bar.
> Work slowly, moving the mallet along the fold line evenly.

### Step 16 – Repeat for K2, K3, K4
Make the same gentle bend at each of the other 3 fold lines.

> After all 4 bends, the piece should start to look like a funnel or pyramid
> with a curved opening at the top.

### Step 17 – Adjust the bends gradually
This is a trial-and-error step.
- Hold the formed piece upright and check whether the circle arc edge
  is forming a round opening.
- If some walls are too flat, increase those bends slightly.
- Work progressively — small adjustments each pass.
    """)

    st.divider()

    # ── PHASE 5 ──────────────────────────────────────────────────
    st.markdown("## PHASE 5 — Form the Circle Edge")

    st.markdown("""
The inner curved edge (C points) needs to be **rolled / curved** so it forms a perfect circle.

### Step 18 – Roll the circular opening
**Using a rolling machine (pipe roller / section bender):**
1. Set the rollers to a diameter slightly larger than your circle diameter
   ({:.0f} mm).
2. Feed the circle edge of the sheet through the rollers.
3. Roll in small passes, gradually increasing the curve.
4. Check against a cardboard or plywood template of the correct circle
   diameter as you go.

**No rolling machine — manual method:**
1. Make a round former from thick plywood or pipe of the correct diameter ({:.0f} mm).
2. Clamp the circle edge of your sheet against the former.
3. Use a rubber mallet to tap the metal around the curve, working in small
   sections along the full arc.
4. Unclamp, check the curve, re-clamp and tap again until it matches the former.

### Step 19 – Check the opening is round
Hold the formed piece up and look down through the circle opening.
It should appear round (or oval for eccentric transitions).
If it is egg-shaped or uneven, tap the high spots with a mallet until it rounds out.

> A piece of the correct-size pipe or a round template is very helpful here
> as a checking gauge.
    """.format(cd, cd))

    st.divider()

    # ── PHASE 6 ──────────────────────────────────────────────────
    st.markdown("## PHASE 6 — Close the Seam and Weld")

    st.markdown("""
### Step 20 – Bring the seam edges together
The two edges that were created when you cut the seam line in Step 12
now need to be pulled together and tacked.

1. Hold the transition piece in its 3-D shape (funnel / cone shape).
2. The two seam edges should now sit next to each other with a small gap.
3. Use clamps or welding magnets to hold them aligned.
4. Check that the rectangular base is sitting flat and the circle opening
   looks round before tacking.

### Step 21 – Tack weld the seam
Put small tack welds (5–10 mm long) every 50–80 mm along the seam.
This holds the shape without distorting the metal.
Let each tack cool before adding the next.

After tacking, stand back and check:
- Does the rectangular base look like a proper rectangle?
- Does the circle opening look round?
- Does the transition look symmetrical (or correctly offset for eccentric)?

Adjust by gently tapping if anything is out of shape, then re-tack.

### Step 22 – Fully weld the seam
Once you are happy with the shape, run a full weld along the seam.
Weld in short runs (50 mm) and let it cool between runs to minimise distortion.
**Weld from the inside if possible** — it gives a cleaner outside finish.

### Step 23 – Check and dress the weld
- Grind the external weld flush with the surface.
- File or disc-grind any burn-through or rough areas.
- Check the piece fits its intended opening by offering it up.
    """)

    st.divider()

    # ── PHASE 7 ──────────────────────────────────────────────────
    st.markdown("## PHASE 7 — Final Checks")

    st.markdown("""
### Step 24 – Measure the rectangular opening
Using a tape measure, check:
- Width of the rectangular base = **{:.0f} mm** ± 2 mm
- Height of the rectangular base = **{:.0f} mm** ± 2 mm
- The base sits flat on a surface (no twist)

### Step 25 – Measure the circular opening
Using a ruler or calliper, check the diameter of the circular opening
= **{:.0f} mm** ± 2 mm.
It should be round, not oval.

### Step 26 – Check the transition height
Measure from the rectangular base to the circular opening
= **{:.0f} mm** ± 3 mm.

### Step 27 – Test fit
Offer the transition piece up to its mating duct or equipment.
Mark any areas that need trimming.
Trim with snips or a grinder, dress the edge, and re-check.

---

## Common Problems and Fixes

| Problem | Likely cause | Fix |
|---|---|---|
| Circle opening is oval / not round | Bends at K lines uneven | Tap the high spots with a mallet |
| Rectangular base is twisted | Seam welded before checking flat | Clamp to a flat surface and re-tack |
| Seam gap too large | Pattern points mis-marked | Add a filler strip and weld |
| Transition height too short or tall | Bends too sharp or too flat | Adjust K-line bend angles |
| Pattern did not fit on sheet | Sheet too small | Use a bigger sheet or split into two halves and weld a centreline joint |
| Weld distortion | Welded too fast without cooling | Grind and re-form; use stitch welding next time |
    """.format(rw, rh, cd, h))

    st.divider()

    # ── Safety note ──────────────────────────────────────────────
    with st.expander("SAFETY REMINDERS"):
        st.warning("""
- Always wear **eye protection** when cutting, grinding or welding.
- Wear **leather gloves** when handling cut sheet metal — the edges are razor sharp.
- Wear a **welding mask** and **welding gloves** when welding.
- Keep a **fire extinguisher** nearby when grinding and welding.
- Work in a **ventilated area** — metal cutting and welding produce fumes.
- Sheet metal edges can cause deep cuts — always dress (file) cut edges before handling.
        """)


# ── Run computation ────────────────────────────────────────────────
if gen:
    with st.spinner("Calculating triangulation and flat pattern..."):
        try:
            result = _compute(rw, rh, cd, h, ox, oy, n, t_mm, r_bend_mm, k_factor)
            st.session_state["result"] = result
            st.session_state["params"] = (rw, rh, cd, h, ox, oy, n,
                                          t_mm, r_bend_mm, k_factor, material)
        except Exception as exc:
            st.error("Computation error: {}".format(exc))
            st.stop()

# ══════════════════════════════════════════════════════════════════
#  Results
# ══════════════════════════════════════════════════════════════════
if "result" in st.session_state:
    r  = st.session_state["result"]
    pw, ph, cd_, hh, ox_, oy_, nn, t_, rb_, kf_, mat_ = st.session_state["params"]
    tr = r.get("thick_rep", {})

    st.divider()

    # ── Quick stats ───────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pattern Width",  "{:.0f} mm".format(r["span_x"]))
    m2.metric("Pattern Height", "{:.0f} mm".format(r["span_y"]))
    m3.metric("Surface Area",   "{:.0f} cm²".format(r["area"] / 100))
    m4.metric("Triangles",      str(len(r["tris"])))
    m5.metric("Ref. Points",    str(len(r["c_refs"]) + len(r["r_refs"]) + len(r["k_refs"])))

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    tab_prev, tab_ws, tab_coord, tab_dl, tab_thick, tab_guide = st.tabs([
        "📐 Preview", "🔧 Workshop Drawing", "📋 Coordinates", "⬇ Downloads",
        "📏 Thickness & Bending", "📖 How to Fabricate"
    ])

    # ── Tab 1: Preview ────────────────────────────────────────────
    with tab_prev:
        st.caption("3-D shape + flat pattern overview.  "
                   "Download the Workshop Drawing tab for full dimensions.")
        with st.spinner("Rendering preview..."):
            fig_prev = _make_preview(
                pw, ph, cd_, hh, ox_, oy_, nn,
                r["tris"], r["rc"], r["cp"],
                r["unfolded"], r["boundaries"],
                r["seam_2d"], r["c2d"], r["area"],
            )
            st.pyplot(fig_prev, use_container_width=True)
            prev_png = _fig_to_bytes(fig_prev)
            plt.close(fig_prev)

        st.download_button(
            "⬇ Download Preview PNG",
            data=prev_png,
            file_name="transition_preview.png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Tab 2: Workshop Drawing ───────────────────────────────────
    with tab_ws:
        st.caption("Full fabrication drawing with dimension lines, "
                   "numbered reference points and scale bar.")
        with st.spinner("Rendering workshop drawing..."):
            fig_ws = _make_workshop(
                pw, ph, cd_, hh, ox_, oy_, nn,
                r["unfolded"], r["boundaries"],
                r["seam_2d"], r["c2d"],
                r["c_refs"], r["r_refs"], r["k_refs"],
                r["area"],
            )
            st.pyplot(fig_ws, use_container_width=True)
            ws_png = _fig_to_bytes(fig_ws, dpi=150)
            plt.close(fig_ws)

        st.download_button(
            "⬇ Download Workshop Drawing PNG",
            data=ws_png,
            file_name="transition_workshop.png",
            mime="image/png",
            use_container_width=True,
        )

    # ── Tab 3: Coordinate Table ───────────────────────────────────
    with tab_coord:
        st.markdown("""
**How to use this table for manual layout:**
1. Mark your sheet metal datum corner as **C01 = (0 , 0)**
2. Scribe each point using a square / dividers
3. Join **C** points with a smooth flexible curve (circle edge)
4. Join **R** points with straight lines (rect perimeter)
5. **K** points mark the corner fold/break lines — rule a line through each
        """)

        import pandas as pd
        all_refs = r["c_refs"] + r["r_refs"] + r["k_refs"]
        rows = []
        for lbl, q in all_refs:
            rows.append({
                "Point": lbl,
                "X (mm)": round(float(q[0]), 2),
                "Y (mm)": round(float(q[1]), 2),
                "Type": ("Circle arc"    if lbl.startswith("C") else
                          "Rect perimeter" if lbl.startswith("R") else
                          "Corner fold"),
            })
        df = pd.DataFrame(rows)

        # Colour rows by type
        def _colour(row):
            if row["Type"] == "Circle arc":
                return ["background-color: #ffe8e8"] * 4
            if row["Type"] == "Rect perimeter":
                return ["background-color: #fff4e8"] * 4
            return ["background-color: #e8eeff"] * 4

        st.dataframe(
            df.style.apply(_colour, axis=1),
            use_container_width=True,
            height=420,
        )

        csv_data = _csv_bytes(
            r["c_refs"], r["r_refs"], r["k_refs"],
            pw, ph, cd_, hh, ox_, oy_
        )
        st.download_button(
            "⬇ Download Coordinates CSV",
            data=csv_data,
            file_name="transition_coordinates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Tab 4: All downloads ──────────────────────────────────────
    with tab_dl:
        st.subheader("Download All Files")

        st.markdown("""
| File | Use for |
|---|---|
| **Preview PNG** | Quick reference / share via WhatsApp |
| **Workshop PNG** | Print and take to the workshop |
| **Coordinates CSV** | Open in Excel, scribe points manually |
| **DXF** | CNC plasma / laser / waterjet cutting |
        """)

        c_a, c_b = st.columns(2)

        with c_a:
            st.download_button(
                "⬇ Preview PNG",
                data=prev_png,
                file_name="transition_preview.png",
                mime="image/png",
                use_container_width=True,
            )
            st.download_button(
                "⬇ Coordinates CSV",
                data=csv_data,
                file_name="transition_coordinates.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with c_b:
            st.download_button(
                "⬇ Workshop Drawing PNG",
                data=ws_png,
                file_name="transition_workshop.png",
                mime="image/png",
                use_container_width=True,
            )

            dxf_data, dxf_err = _dxf_to_bytes(r, pw, ph, cd_, hh, ox_, oy_, nn)
            if dxf_data:
                st.download_button(
                    "⬇ DXF  (CAD / CNC)",
                    data=dxf_data,
                    file_name="transition_development.dxf",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                st.info("DXF not available: {}".format(dxf_err or "install ezdxf"))

        st.divider()
        st.markdown("""
#### Quick Reference
| File | Use for |
|---|---|
| Workshop PNG | Print and pin to the bench while working |
| Coordinates CSV | Scribe every X, Y point with a square and rule |
| DXF | CNC plasma / laser — plug in directly |
        """)

    # ── Tab 5: How to Fabricate ───────────────────────────────────
    # ── Tab 5: Thickness & Bending ────────────────────────────────
    with tab_thick:
        _show_thickness_tab(tr, t_, rb_, kf_, mat_, cd_, pw, ph)

    # ── Tab 6: How to Fabricate ───────────────────────────────────
    with tab_guide:
        _show_fabrication_guide(r, pw, ph, cd_, hh, ox_, oy_, nn)


# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Sheet Metal Transition Tool  |  "
    "Triangulation method  |  "
    "Geometry is exact — use DXF for CNC, PNG for manual layout"
)
