"""
Transition Calculator  —  Page 2
Traditional true-length development: C, J, F, L0..Ln
"""

import io
import sys
import os
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import streamlit as st

# ── path so we can import shared geometry ────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in [_root, os.getcwd()]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ============================================================
#  Calculations  (identical to transition_calc.py)
# ============================================================

def angle_to_rect_pt(angle, rw, rh):
    rx, ry = rw / 2.0, rh / 2.0
    dx, dy = np.cos(angle), np.sin(angle)
    ts = []
    if abs(dx) > 1e-12: ts += [rx/dx, -rx/dx]
    if abs(dy) > 1e-12: ts += [ry/dy, -ry/dy]
    if not ts: return np.array([0.0, 0.0])
    t = min(v for v in ts if v > 1e-12)
    return np.array([np.clip(dx*t, -rx, rx), np.clip(dy*t, -ry, ry)])


def calc_rect_to_round(L, W, D, H, N, ox=0.0, oy=0.0):
    R      = D / 2.0
    n_quad = N // 4
    angles = np.linspace(0, np.pi/2, n_quad + 1)
    C      = np.pi * D / N
    chord  = 2 * R * np.sin(np.pi / N)
    corner = np.array([L/2 + ox, W/2 + oy, 0.0])

    L_vals, circle_pts_3d, rect_pts_3d, slants = [], [], [], []
    for a in angles:
        cp = np.array([ox + R*np.cos(a), oy + R*np.sin(a), H])
        rp2 = angle_to_rect_pt(a, L, W)
        rp  = np.array([rp2[0], rp2[1], 0.0])
        circle_pts_3d.append(cp)
        rect_pts_3d.append(rp)
        L_vals.append(float(np.linalg.norm(cp - corner)))
        slants.append(float(np.linalg.norm(cp - rp)))

    J = max(slants)
    F = W

    # Surface area (triangulation, 4 quadrants)
    area = 0.0
    for i in range(n_quad):
        c0,c1 = circle_pts_3d[i], circle_pts_3d[i+1]
        r0,r1 = rect_pts_3d[i],   rect_pts_3d[i+1]
        area += 0.5 * float(np.linalg.norm(np.cross(
            np.asarray(c1)-np.asarray(c0), np.asarray(r0)-np.asarray(c0))))
        area += 0.5 * float(np.linalg.norm(np.cross(
            np.asarray(r1)-np.asarray(c1), np.asarray(r0)-np.asarray(c1))))
    area *= 4

    return dict(L=L, W=W, D=D, H=H, N=N, R=R, n_quad=n_quad,
                angles_deg=[np.degrees(a) for a in angles],
                C=C, chord=chord, J=J, F=F, L_vals=L_vals,
                slants=slants, s_min=min(slants), s_max=max(slants),
                circle_pts_3d=circle_pts_3d, rect_pts_3d=rect_pts_3d,
                corner=corner, area=area, ox=ox, oy=oy)


def calc_pyramid(L, W, H):
    sl = math.sqrt((L/2)**2 + H**2)
    ss = math.sqrt((W/2)**2 + H**2)
    sc = math.sqrt((L/2)**2 + (W/2)**2 + H**2)
    return dict(type='pyramid', L=L, W=W, H=H,
                slant_long=sl, slant_short=ss, slant_corner=sc,
                base_perim=2*(L+W), area=L*ss + W*sl)


def calc_trunc_pyramid(Lb, Wb, Lt, Wt, H):
    sl = math.sqrt(((Lb-Lt)/2)**2 + H**2)
    ss = math.sqrt(((Wb-Wt)/2)**2 + H**2)
    sc = math.sqrt(((Lb-Lt)/2)**2 + ((Wb-Wt)/2)**2 + H**2)
    return dict(type='trunc', Lb=Lb, Wb=Wb, Lt=Lt, Wt=Wt, H=H,
                slant_long=sl, slant_short=ss, slant_corner=sc)


# ============================================================
#  Diagrams
# ============================================================

def draw_quarter_development(ax, res):
    ax.clear(); ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('#f8f8f0')

    W, n_quad = res['W'], res['n_quad']
    L_vals, chord = res['L_vals'], res['chord']
    J, F = res['J'], res['F']

    corner_2d = np.array([0.0, 0.0])
    rect_mid  = np.array([-W/2, 0.0])

    # Build circle points by successive triangle placement
    c_pts = [corner_2d + np.array([0.0, L_vals[-1]])]
    for i in range(n_quad - 1, -1, -1):
        prev  = c_pts[-1]
        L_cur = L_vals[i]
        d     = float(np.linalg.norm(prev - corner_2d))
        if d < 1e-10:
            c_pts.append(corner_2d + np.array([-L_cur*0.7, L_cur*0.7]))
            continue
        a_c  = (L_cur**2 - chord**2 + d**2) / (2*d)
        h2   = max(0.0, L_cur**2 - a_c**2)
        hh   = math.sqrt(h2)
        dh   = (prev - corner_2d) / d
        perp = np.array([-dh[1], dh[0]])
        foot = corner_2d + a_c * dh
        c1, c2 = foot + hh*perp, foot - hh*perp
        c_pts.append(c1 if c1[0] <= c2[0] else c2)
    c_pts = c_pts[::-1]

    # Outer rect edge points
    r_pts = []
    for i, cp in enumerate(c_pts):
        slant = res['slants'][i]
        d_out = cp - corner_2d
        dn    = float(np.linalg.norm(d_out))
        r_pts.append(cp + (slant * d_out / dn) if dn > 1e-10
                     else cp + np.array([0, slant]))

    # Scale
    all_pts = c_pts + r_pts + [corner_2d, rect_mid]
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    pad  = span * 0.18
    mirror_x = c_pts[0][0]

    # Draw
    ax.plot([p[0] for p in r_pts], [p[1] for p in r_pts],
            color='#cc5500', lw=2.5, label='Rect edge')
    ax.plot([rect_mid[0], corner_2d[0]], [0, 0],
            color='#0044aa', lw=2.5, label='F (rect side)')
    for cp in c_pts:
        ax.plot([corner_2d[0], cp[0]], [corner_2d[1], cp[1]],
                'k-', lw=1.0, alpha=0.65)
    ax.plot([p[0] for p in c_pts], [p[1] for p in c_pts],
            'r-', lw=2.5, label='Circle arc')
    ax.axvline(mirror_x, color='#888', ls='--', lw=1.0)
    ax.text(mirror_x + span*0.01, max(ys) + pad*0.25,
            'Mirror Line', fontsize=7, color='#666')

    fs = 8
    # F
    ax.annotate('', xy=corner_2d, xytext=rect_mid,
                arrowprops=dict(arrowstyle='<->', color='#0044aa', lw=1.2))
    ax.text((rect_mid[0]+corner_2d[0])/2, -span*0.05,
            'F = {:.0f}'.format(F), ha='center', va='top',
            fontsize=fs, color='#0044aa', fontweight='bold')

    # C chord label
    if len(c_pts) >= 2:
        mid_c = (c_pts[-1] + c_pts[-2]) / 2
        ax.text(mid_c[0]+span*0.02, mid_c[1], 'C',
                ha='left', va='center', fontsize=fs, color='red', fontweight='bold')

    # J label
    j_idx = res['slants'].index(max(res['slants']))
    if j_idx < len(r_pts) and j_idx < len(c_pts):
        mj = (c_pts[j_idx] + r_pts[j_idx]) / 2
        ax.text(mj[0]+span*0.02, mj[1], 'J',
                ha='left', va='center', fontsize=fs,
                color='#cc5500', fontweight='bold')

    # L labels
    for i, (cp, lv) in enumerate(zip(c_pts, L_vals)):
        mid = (corner_2d + cp) / 2
        ax.text(mid[0]-span*0.04, mid[1],
                'L{}'.format(n_quad - i),
                ha='right', va='center', fontsize=fs-1, color='#333')

    ax.plot(*corner_2d, 'ko', ms=5, zorder=5)
    ax.set_xlim(min(xs)-pad, max(xs)+pad*1.8)
    ax.set_ylim(min(ys)-pad*0.5, max(ys)+pad*0.5)
    ax.set_title('Quarter Development  (mirror for full half)',
                 fontsize=9, pad=4)
    ax.legend(fontsize=7, loc='lower right')


def draw_3d_sketch(ax, L, W, D, H, ox=0.0, oy=0.0):
    ax.clear(); ax.axis('off'); ax.set_facecolor('#f0f4ff')

    def iso(x, y, z):
        px = (x-y) * np.cos(np.radians(30))
        py = (x+y) * np.sin(np.radians(30)) + z*0.9
        return px, py

    rx, ry = L/2, W/2
    rect = [(-rx,-ry,0),(rx,-ry,0),(rx,ry,0),(-rx,ry,0),(-rx,-ry,0)]
    xs,ys = zip(*[iso(*c) for c in rect])
    ax.plot(xs, ys, 'b-', lw=2, label='Rectangle')

    ang = np.linspace(0, 2*np.pi, 60)
    cp  = [iso(ox+D/2*np.cos(a), oy+D/2*np.sin(a), H) for a in ang]
    ax.plot([p[0] for p in cp], [p[1] for p in cp], 'r-', lw=2, label='Circle')

    for cx,cy,_ in rect[:4]:
        na = np.arctan2(cy, cx)
        ax.plot([iso(cx,cy,0)[0], iso(ox+D/2*np.cos(na), oy+D/2*np.sin(na), H)[0]],
                [iso(cx,cy,0)[1], iso(ox+D/2*np.cos(na), oy+D/2*np.sin(na), H)[1]],
                'k-', lw=0.8, alpha=0.45)

    ax.set_aspect('equal')
    ax.set_title('3-D View', fontsize=9, pad=3)
    ax.legend(fontsize=7, loc='upper left')


def make_figure(res):
    """Build the full output figure."""
    fig = plt.Figure(figsize=(13, 5.5), facecolor='#f8f8f0')
    gs  = GridSpec(1, 2, figure=fig,
                   left=0.03, right=0.98, bottom=0.05, top=0.90,
                   wspace=0.10)
    ax_dev = fig.add_subplot(gs[0])
    ax_3d  = fig.add_subplot(gs[1])

    draw_quarter_development(ax_dev, res)
    draw_3d_sketch(ax_3d, res['L'], res['W'], res['D'], res['H'],
                   res.get('ox', 0), res.get('oy', 0))

    ec = ("Concentric" if not (res.get('ox') or res.get('oy'))
          else "Eccentric ({},{})".format(res['ox'], res['oy']))
    fig.suptitle(
        "Rect {}×{}  →  Ø{}  |  H={}  |  N={} lines  |  {}".format(
            int(res['L']), int(res['W']), int(res['D']),
            int(res['H']), res['N'], ec),
        fontsize=10, fontweight='bold')
    return fig


def make_pyramid_figure(res):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    L, W, H = res['L'], res['W'], res['H']
    # Draw simple pyramid
    ax.plot([-L/2,-L/2, L/2, L/2,-L/2], [-W/2, W/2, W/2,-W/2,-W/2],
            'b-', lw=2)
    ax.plot([0],[0], 'r^', ms=12)
    ax.set_aspect('equal')
    ax.set_title('Pyramid  {}×{}  H={}'.format(L, W, H))
    return fig


# ============================================================
#  CSV export
# ============================================================

def results_csv(res):
    lines = [
        "Transition Calculator — True Length Values",
        "Rectangle,{} x {} mm".format(res['L'], res['W']),
        "Circle D,{} mm".format(res['D']),
        "Height,{} mm".format(res['H']),
        "Segments N,{}".format(res['N']),
        "",
        "Value,mm,Description",
        "C,{:.3f},Arc length per segment".format(res['C']),
        "Chord,{:.3f},Straight chord (mark on sheet)".format(res['chord']),
        "J,{:.3f},Maximum slant height".format(res['J']),
        "F,{:.3f},Rectangle short side".format(res['F']),
    ]
    n_quad = res['n_quad']
    for i, lv in enumerate(res['L_vals']):
        lbl = "L{}".format(n_quad - i)
        lines.append("{},{:.3f},True length corner to {}deg circle pt".format(
            lbl, lv, int(res['angles_deg'][i])))
    lines += [
        "Slant min,{:.3f},Shortest wall slant".format(res['s_min']),
        "Slant max,{:.3f},Longest wall slant".format(res['s_max']),
        "Surface area,{:.1f},mm2 (approx)".format(res['area']),
    ]
    return "\n".join(lines).encode("utf-8")


# ============================================================
#  Page layout
# ============================================================

st.set_page_config(
    page_title="Transition Calculator",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
<style>
div[data-testid="stNumberInput"] input { font-size:16px; height:40px; }
div[data-testid="stButton"]>button   { height:46px; font-size:15px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📐 Transition Calculator")
st.caption("True-length development values: C · J · F · L0..Ln  |  Quarter-development diagram")

st.divider()

# ── Transition type ───────────────────────────────────────────────
trans_type = st.radio(
    "Transition Type",
    ["Rectangular to Round", "Round to Rectangular",
     "Pyramid", "Truncated Pyramid"],
    horizontal=True,
)

st.divider()

# ── Inputs ────────────────────────────────────────────────────────
if trans_type in ("Rectangular to Round", "Round to Rectangular"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Rectangle**")
        L  = st.number_input("Length  L (mm)", min_value=10.0, value=600.0, step=10.0)
        W  = st.number_input("Width   W (mm)", min_value=10.0, value=300.0, step=10.0)
    with c2:
        st.markdown("**Circle**")
        D  = st.number_input("Diameter D (mm)", min_value=10.0, value=550.0, step=10.0)
        H  = st.number_input("Height  H (mm)",  min_value=10.0, value=400.0, step=10.0)
    with c3:
        st.markdown("**Eccentricity** *(0 = concentric)*")
        ox = st.number_input("X Offset (mm)", value=0.0, step=5.0)
        oy = st.number_input("Y Offset (mm)", value=0.0, step=5.0)
    with c4:
        st.markdown("**Development Lines N**")
        N  = st.select_slider("N", options=[12, 24, 36, 48], value=12)
        st.caption("N must be divisible by 4")

elif trans_type == "Pyramid":
    c1, c2 = st.columns(2)
    with c1:
        L = st.number_input("Base Length L (mm)", min_value=10.0, value=600.0, step=10.0)
        W = st.number_input("Base Width  W (mm)", min_value=10.0, value=300.0, step=10.0)
    with c2:
        H = st.number_input("Height H (mm)", min_value=10.0, value=400.0, step=10.0)

elif trans_type == "Truncated Pyramid":
    c1, c2, c3 = st.columns(3)
    with c1:
        Lb = st.number_input("Base Length L (mm)", min_value=10.0, value=600.0, step=10.0)
        Wb = st.number_input("Base Width  W (mm)", min_value=10.0, value=300.0, step=10.0)
    with c2:
        Lt = st.number_input("Top Length  l (mm)", min_value=1.0,  value=300.0, step=10.0)
        Wt = st.number_input("Top Width   w (mm)", min_value=1.0,  value=150.0, step=10.0)
    with c3:
        H  = st.number_input("Height H (mm)", min_value=10.0, value=400.0, step=10.0)

calc = st.button("⚙  Calculate", type="primary", use_container_width=False)

# ── Compute ───────────────────────────────────────────────────────
if calc:
    try:
        if trans_type in ("Rectangular to Round", "Round to Rectangular"):
            res = calc_rect_to_round(L, W, D, H, N, ox, oy)
            st.session_state["calc_result"] = res
            st.session_state["calc_type"]   = "rtr"
        elif trans_type == "Pyramid":
            res = calc_pyramid(L, W, H)
            st.session_state["calc_result"] = res
            st.session_state["calc_type"]   = "pyr"
        elif trans_type == "Truncated Pyramid":
            res = calc_trunc_pyramid(Lb, Wb, Lt, Wt, H)
            st.session_state["calc_result"] = res
            st.session_state["calc_type"]   = "tpyr"
        st.success("Calculated successfully.")
    except Exception as e:
        st.error("Error: {}".format(e))

# ── Results ───────────────────────────────────────────────────────
if "calc_result" in st.session_state:
    res  = st.session_state["calc_result"]
    ctyp = st.session_state.get("calc_type", "rtr")

    st.divider()

    if ctyp == "rtr":
        # ── Key metric cards ─────────────────────────────────────
        n_quad = res['n_quad']
        cols   = st.columns(4 + n_quad + 1)
        cols[0].metric("C  (arc)",    "{:.2f} mm".format(res['C']),
                       help="Arc length per segment = π×D/N")
        cols[1].metric("chord",       "{:.2f} mm".format(res['chord']),
                       help="Straight chord — mark this on sheet")
        cols[2].metric("J  (max slant)", "{:.2f} mm".format(res['J']))
        cols[3].metric("F  (rect side)", "{:.2f} mm".format(res['F']))
        for i, lv in enumerate(res['L_vals']):
            lbl = "L{}".format(n_quad - i)
            if 4+i < len(cols):
                cols[4+i].metric(lbl, "{:.2f} mm".format(lv))

        # ── Diagram ──────────────────────────────────────────────
        st.markdown("### Development Diagram")
        with st.spinner("Drawing..."):
            fig = make_figure(res)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig)

        # ── Full values table ─────────────────────────────────────
        st.markdown("### True Length Values")
        tab_data = {
            "Symbol": ["C (arc)", "chord", "J", "F"] +
                      ["L{}".format(n_quad-i) for i in range(len(res['L_vals']))] +
                      ["Slant min", "Slant max", "Surface area"],
            "Value (mm)": [
                "{:.3f}".format(res['C']),
                "{:.3f}".format(res['chord']),
                "{:.3f}".format(res['J']),
                "{:.3f}".format(res['F']),
            ] + ["{:.3f}".format(lv) for lv in res['L_vals']] + [
                "{:.3f}".format(res['s_min']),
                "{:.3f}".format(res['s_max']),
                "{:.0f} mm²".format(res['area']),
            ],
            "Description": [
                "Arc length per segment (π×D/N)",
                "Straight chord — mark this on sheet",
                "Maximum slant height",
                "Rectangle short side",
            ] + ["True length: corner → {}° circle pt".format(int(a))
                 for a in res['angles_deg']] + [
                "Shortest wall line",
                "Longest wall line",
                "Approx lateral surface area",
            ],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(tab_data), use_container_width=True,
                     hide_index=True)

        # ── Cross-check vs reference ──────────────────────────────
        with st.expander("How to use these values (traditional method)"):
            st.markdown("""
**To manually construct the flat pattern using compass and dividers:**

1. **Draw the base line** — length **F = {:.0f} mm** (rect short side, horizontal)
2. **Mark the corner** at one end of F
3. **From the corner, swing arc L3** ({:.2f} mm) — this locates the 0° circle point above
4. **Swing arc L2** ({:.2f} mm) from corner, AND swing arc **chord C** ({:.2f} mm) from the previous circle point — intersection = next circle point
5. **Repeat** for L1, L0 — each time: arc from corner (L value) + arc from previous circle point (chord)
6. **Connect circle points** with a smooth curve (the circle edge)
7. **For outer rect edge**: from each circle point, swing slant distance outward
8. **Mirror** the whole pattern about the end of F
9. The resulting shape × 4 (or use the full flat pattern from the other tool's DXF)

> **Note:** C = **{:.2f} mm** (arc) is used to verify. Chord = **{:.3f} mm** is what you physically mark with dividers.
            """.format(res['F'],
                       res['L_vals'][-1], res['L_vals'][-2], res['chord'],
                       res['C'], res['chord']))

        # ── Downloads ─────────────────────────────────────────────
        st.divider()
        st.markdown("### Downloads")
        dl1, dl2, dl3 = st.columns(3)

        # PNG
        with dl1:
            with st.spinner("PNG..."):
                fig2 = make_figure(res)
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
                plt.close(fig2)
            st.download_button("⬇ Development PNG",
                               data=buf2.getvalue(),
                               file_name="transition_calc.png",
                               mime="image/png",
                               use_container_width=True)

        # CSV
        with dl2:
            csv_bytes = results_csv(res)
            st.download_button("⬇ Values CSV",
                               data=csv_bytes,
                               file_name="transition_values.csv",
                               mime="text/csv",
                               use_container_width=True)

        # DXF (full flat pattern via sheet_metal_transition)
        with dl3:
            try:
                import ezdxf, tempfile
                from sheet_metal_transition import (
                    build_transition, unfold_triangles, get_all_boundaries,
                    map_2d_pts, get_reference_points, export_dxf,
                )
                tris,rc,cp2,rpp,cl3d,sp3d = build_transition(
                    res['L'], res['W'], res['D'], res['H'],
                    res.get('ox',0), res.get('oy',0), res['N'])
                unfolded, placed = unfold_triangles(tris)
                bounds  = get_all_boundaries(unfolded)
                s2d,c2d = map_2d_pts(sp3d, cl3d, placed)
                cr, rr, kr = get_reference_points(cp2, rpp, rc, placed)
                all2d = [p for t in unfolded for p in t]
                sw = max(p[0] for p in all2d)-min(p[0] for p in all2d)
                sh = max(p[1] for p in all2d)-min(p[1] for p in all2d)
                with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
                    fname = f.name
                export_dxf(unfolded, bounds, s2d, c2d, cr, rr, kr,
                            sw, sh, res['L'], res['W'], res['D'],
                            res['H'], res.get('ox',0), res.get('oy',0),
                            res['N'], fname)
                with open(fname,'rb') as f:
                    dxf_bytes = f.read()
                os.unlink(fname)
                st.download_button("⬇ Full Pattern DXF",
                                   data=dxf_bytes,
                                   file_name="transition_pattern.dxf",
                                   mime="application/octet-stream",
                                   use_container_width=True)
            except Exception as e:
                st.info("DXF: {}".format(e))

    elif ctyp == "pyr":
        st.markdown("### Pyramid True Lengths")
        c1,c2,c3 = st.columns(3)
        c1.metric("Slant — long side",  "{:.2f} mm".format(res['slant_long']))
        c2.metric("Slant — short side", "{:.2f} mm".format(res['slant_short']))
        c3.metric("Slant — corner",     "{:.2f} mm".format(res['slant_corner']))
        st.metric("Base perimeter", "{:.1f} mm".format(res['base_perim']))
        st.metric("Lateral area",   "{:.0f} mm²".format(res['area']))

    elif ctyp == "tpyr":
        st.markdown("### Truncated Pyramid True Lengths")
        c1,c2,c3 = st.columns(3)
        c1.metric("Slant — long side",  "{:.2f} mm".format(res['slant_long']))
        c2.metric("Slant — short side", "{:.2f} mm".format(res['slant_short']))
        c3.metric("Slant — corner",     "{:.2f} mm".format(res['slant_corner']))

st.divider()
st.caption("Transition Calculator  |  True-length method  |  "
           "C = arc per segment  |  L0..Ln = corner diagonals")
