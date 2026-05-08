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
    """
    Quarter development diagram matching the reference tool style:
    - Corner (apex) at BOTTOM-RIGHT of solid shape
    - F base goes LEFT from corner
    - L-lines fan UP-LEFT from corner
    - Circle arc connects L-line endpoints (inner edge)
    - Outer rect boundary connects F-end to outermost rect point
    - Dashed MIRROR on the RIGHT side of the corner
    - Mirror Line vertical through the corner
    """
    ax.clear()
    ax.set_facecolor('#f8f8f8')
    ax.axis('off')

    n_quad  = res['n_quad']
    L_vals  = res['L_vals']   # L_vals[0]=L0 (longest,α=90°) .. L_vals[-1]=Ln (α=0°)
    chord   = res['chord']
    J       = res['J']
    F       = res['F']
    slants  = res['slants']

    # ── Step 1: place L-line endpoints using fan angles ──────────────
    # Fan from ~80° (Ln, nearly vertical) to ~155° (L0, upper-left)
    # measured from +x axis.  Equal angular spacing gives a clean visual.
    angle_Ln = math.radians(80)    # Ln at 0° circle → nearly vertical
    angle_L0 = math.radians(155)   # L0 at 90° circle → upper-left

    angles = np.linspace(angle_Ln, angle_L0, n_quad + 1)
    # angles[0]  → Ln (rightmost, shortest angle from vertical)
    # angles[-1] → L0 (leftmost, longest)
    # But L_vals[0]=L0 (longest) and L_vals[-1]=Ln
    # So: L_vals[i] goes with angles[n_quad - i]

    c_pts = []   # circle arc endpoints in 2D
    for i in range(n_quad + 1):
        a   = angles[n_quad - i]      # L0 gets leftmost angle
        L   = L_vals[i]
        c_pts.append(np.array([L * math.cos(a), L * math.sin(a)]))

    corner = np.array([0.0, 0.0])

    # ── Step 2: outer rect boundary points ───────────────────────────
    # Each outer point = circle arc point extended by its slant distance
    # outward (away from corner)
    r_pts = []
    for i, cp in enumerate(c_pts):
        norm = float(np.linalg.norm(cp))
        if norm > 1e-6:
            r_pts.append(cp + slants[i] * cp / norm)
        else:
            r_pts.append(cp + np.array([0.0, slants[i]]))

    # ── Step 3: F base end ────────────────────────────────────────────
    # F goes LEFT from corner.  Length = F (full short side).
    F_end = np.array([-F, 0.0])

    # Outer boundary path: from F_end → along rect perimeter → to r_pts[0]
    # Simplified as two segments: F_end → r_pts[-1] (Ln side) for left wall,
    # then connect r_pts in order.  Draw as: F_end → r_pts[-1] → ... → r_pts[0]
    outer_path_x = [F_end[0]] + [r[0] for r in reversed(r_pts)]
    outer_path_y = [F_end[1]] + [r[1] for r in reversed(r_pts)]

    # ── Compute bounding box for axis limits ─────────────────────────
    all_pts = c_pts + r_pts + [corner, F_end]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin)
    pad  = span * 0.15

    # ── Draw dashed MIRROR first (behind solid) ───────────────────────
    # Mirror reflects x: (x,y) → (-x, y)
    # F mirror: goes RIGHT from corner
    ax.plot([0, F], [0, 0], color='#0044aa', lw=1.8, ls='--', alpha=0.45)
    # L-line mirrors
    for cp in c_pts:
        ax.plot([0, -cp[0]], [0, cp[1]], 'k--', lw=0.7, alpha=0.35)
    # Circle arc mirror
    ax.plot([-cp[0] for cp in c_pts], [cp[1] for cp in c_pts],
            'r--', lw=1.8, alpha=0.45)
    # Outer boundary mirror
    ax.plot([-x for x in outer_path_x], outer_path_y,
            color='#cc5500', lw=1.5, ls='--', alpha=0.4)

    # ── Mirror Line (vertical dashed through corner) ──────────────────
    ax.axvline(0, color='#888888', ls='--', lw=1.0, alpha=0.8)
    ax.text(span * 0.04, ymax + pad * 0.5, 'Mirror Line',
            fontsize=7.5, color='#666666', ha='left', va='bottom')

    # ── Draw solid quarter development ────────────────────────────────
    # Outer rect boundary (orange solid)
    ax.plot(outer_path_x, outer_path_y,
            color='#cc5500', lw=2.2, solid_capstyle='round',
            label='Rect edge (outer)')

    # F base (blue solid, with arrow)
    ax.annotate('', xy=(F_end[0], 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='#0044aa',
                                lw=1.4, mutation_scale=10))
    ax.plot([F_end[0], 0], [0, 0], color='#0044aa', lw=2.2)
    ax.text((F_end[0]) / 2, -span * 0.055,
            'F = {:.0f} mm'.format(F),
            ha='center', va='top', fontsize=8.5,
            color='#0044aa', fontweight='bold')

    # L-lines (black solid)
    for i, cp in enumerate(c_pts):
        ax.plot([0, cp[0]], [0, cp[1]], 'k-', lw=1.1, alpha=0.75, zorder=3)

    # Circle arc (red solid)
    ax.plot([cp[0] for cp in c_pts], [cp[1] for cp in c_pts],
            'r-', lw=2.5, solid_capstyle='round',
            label='Circle arc (inner)', zorder=4)

    # ── Labels ────────────────────────────────────────────────────────
    fs = 8.5

    # L labels along each radial line (mid-point)
    for i, (cp, lv) in enumerate(zip(c_pts, L_vals)):
        mid   = cp * 0.52
        lbl   = 'L{}'.format(n_quad - i)   # L0 = longest
        # offset label slightly to avoid overlapping the line
        perp  = np.array([-cp[1], cp[0]])
        pn    = float(np.linalg.norm(perp))
        offset = (perp / pn * span * 0.035) if pn > 1e-6 else np.zeros(2)
        ax.text(mid[0] + offset[0], mid[1] + offset[1], lbl,
                ha='center', va='center', fontsize=fs - 0.5,
                color='#333333', fontweight='bold')

    # C label between last two circle arc points (rightmost segment)
    if len(c_pts) >= 2:
        m = (c_pts[-1] + c_pts[-2]) / 2
        ax.text(m[0] + span*0.03, m[1] + span*0.02, 'C',
                ha='left', va='bottom', fontsize=fs,
                color='red', fontweight='bold')

    # J label on the longest slant (outer boundary at max-slant index)
    j_idx = slants.index(max(slants))
    if j_idx < len(c_pts) and j_idx < len(r_pts):
        mj = (c_pts[j_idx] + r_pts[j_idx]) / 2
        ax.text(mj[0] - span*0.04, mj[1], 'J',
                ha='right', va='center', fontsize=fs,
                color='#cc5500', fontweight='bold')

    # Corner dot
    ax.plot(0, 0, 'ko', ms=6, zorder=6)

    # ── Axis limits ───────────────────────────────────────────────────
    # Include mirror extent (positive x up to F)
    ax.set_xlim(xmin - pad, max(xmax, F) + pad * 1.5)
    ax.set_ylim(ymin - pad * 1.2, ymax + pad * 0.8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Quarter Development   (solid = left half, dashed = mirror)',
                 fontsize=9, pad=5)

    handles = [
        plt.Line2D([0],[0], color='red',     lw=2, label='Circle arc (inner)'),
        plt.Line2D([0],[0], color='#cc5500', lw=2, label='Rect edge (outer)'),
        plt.Line2D([0],[0], color='#0044aa', lw=2, label='F  (rect short side)'),
        plt.Line2D([0],[0], color='black',   lw=1, label='L0..Ln  (true lengths)'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='lower right',
              framealpha=0.9, edgecolor='#aaaaaa')


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
    menu_items={
        "Get Help":     None,
        "Report a bug": None,
        "About": (
            "### Transition Calculator\n"
            "True-length development values: C · J · F · L0..Ln\n\n"
            "Quarter-development diagram with mirror line."
        ),
    },
)

# ── Hide Streamlit branding ───────────────────────────────────────
st.markdown("""
<style>
  footer                              { visibility: hidden !important; height: 0; }
  #MainMenu                           { visibility: hidden !important; }
  div[data-testid="stDecoration"]     { display: none !important; }
  div[data-testid="stToolbar"]        { display: none !important; }
  div[data-testid="stToolbarActions"] { display: none !important; }
  #stDecoration                       { display: none !important; }
  [class*="viewerBadge"]             { display: none !important; }
  [class*="badge_container"]         { display: none !important; }
  [class*="BadgeContainer"]          { display: none !important; }
  body::after {
    content: "";
    position: fixed;
    bottom: 0; right: 0;
    width: 260px; height: 52px;
    background: white;
    z-index: 99999;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stNumberInput"] input { font-size:16px; height:40px; }
div[data-testid="stButton"]>button   { height:46px; font-size:15px; }
</style>
""", unsafe_allow_html=True)

# ── Navigation ───────────────────────────────────────────────────
nav1, nav2 = st.columns(2)
with nav1:
    st.page_link("sheet_metal_app.py",
                 label="⚙ Sheet Metal Development",
                 icon="⚙", use_container_width=True)
with nav2:
    st.page_link("pages/2_Transition_Calculator.py",
                 label="📐 Transition Calculator",
                 icon="📐", use_container_width=True)

st.divider()

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
