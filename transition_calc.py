"""
Transition Calculator
=====================
Traditional true-length development tool for sheet metal transitions.
Produces C, J, F, L0..Ln values + quarter-development diagram.

Transition types:
  1. Rectangular to Round   (concentric & eccentric)
  2. Round to Rectangular   (same geometry, reversed)
  3. Pyramid                (4-sided, flat faces)
  4. Truncated Pyramid      (pyramid cut at height)

Usage:  python transition_calc.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import math

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

# ============================================================
#  Geometry helpers
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


# ============================================================
#  Core calculations
# ============================================================

def calc_rect_to_round(L, W, D, H, N, ox=0.0, oy=0.0):
    """
    L  = rectangle length (long side)
    W  = rectangle width  (short side)
    D  = circle diameter
    H  = transition height
    N  = number of development lines (must be divisible by 4)
    ox,oy = circle offset from rect centre (eccentric)

    Returns dict with all calculated values.
    """
    R      = D / 2.0
    n_quad = N // 4                          # segments per quadrant
    angles = np.linspace(0, np.pi/2, n_quad + 1)  # 0 to 90 deg

    # Circle arc length and chord per segment
    C     = np.pi * D / N                    # arc length
    chord = 2 * R * np.sin(np.pi / N)       # chord (straight line)

    # Rectangle corner (concentric: L/2, W/2)
    corner = np.array([L/2 + ox, W/2 + oy, 0.0])

    # L values: true length from rect corner to each quadrant circle point
    L_vals = []
    circle_pts_3d = []
    for a in angles:
        cp = np.array([ox + R*np.cos(a), oy + R*np.sin(a), H])
        circle_pts_3d.append(cp)
        L_vals.append(float(np.linalg.norm(cp - corner)))

    # Slant heights: circle point to corresponding rect perimeter point
    slants = []
    rect_pts_3d = []
    for a in angles:
        rp2 = angle_to_rect_pt(a, L, W)
        rp  = np.array([rp2[0], rp2[1], 0.0])
        rect_pts_3d.append(rp)
        cp  = np.array([ox + R*np.cos(a), oy + R*np.sin(a), H])
        slants.append(float(np.linalg.norm(cp - rp)))

    # J = maximum slant height in the development (longest wall line)
    J = max(slants)

    # F = short side of rectangle
    F = W

    # Minimum and maximum wall slant
    s_min = min(slants)
    s_max = max(slants)

    # Surface area (quarter × 4, triangulation approximation)
    # For each quadrant triangle pair
    area = np.pi * R * np.sqrt(R**2 + H**2)   # cone approximation
    # Better: use actual triangulation geometry
    area_tri = 0.0
    for i in range(n_quad):
        # Two triangles per strip
        c0 = circle_pts_3d[i];   c1 = circle_pts_3d[i+1]
        r0 = rect_pts_3d[i];     r1 = rect_pts_3d[i+1]
        # Insert corners if needed — simplified for surface area
        v1 = np.asarray(c1) - np.asarray(c0)
        v2 = np.asarray(r0) - np.asarray(c0)
        area_tri += 0.5 * np.linalg.norm(np.cross(v1, v2))
        v1 = np.asarray(r1) - np.asarray(c1)
        v2 = np.asarray(r0) - np.asarray(c1)
        area_tri += 0.5 * np.linalg.norm(np.cross(v1, v2))
    area_total = area_tri * 4   # 4 quadrants

    return dict(
        L=L, W=W, D=D, H=H, N=N, R=R,
        n_quad=n_quad, angles_deg=[np.degrees(a) for a in angles],
        C=C, chord=chord, J=J, F=F,
        L_vals=L_vals,
        slants=slants, s_min=s_min, s_max=s_max,
        circle_pts_3d=circle_pts_3d,
        rect_pts_3d=rect_pts_3d,
        corner=corner,
        area_total=area_total,
        ox=ox, oy=oy,
    )


def calc_pyramid(L, W, H):
    """4-sided square/rectangular pyramid."""
    s_long  = math.sqrt((L/2)**2 + H**2)   # slant along long side centre
    s_short = math.sqrt((W/2)**2 + H**2)   # slant along short side centre
    s_corner = math.sqrt((L/2)**2 + (W/2)**2 + H**2)  # slant to corner
    base_perim = 2*(L+W)
    area = (L*W) + L*s_short + W*s_long   # approx lateral area
    return dict(L=L, W=W, H=H,
                slant_long=s_long, slant_short=s_short,
                slant_corner=s_corner, base_perim=base_perim, area=area)


def calc_trunc_pyramid(L_bot, W_bot, L_top, W_top, H):
    """Truncated pyramid (frustum with rectangular cross-section)."""
    s_long  = math.sqrt(((L_bot-L_top)/2)**2 + H**2)
    s_short = math.sqrt(((W_bot-W_top)/2)**2 + H**2)
    s_corner = math.sqrt(((L_bot-L_top)/2)**2 + ((W_bot-W_top)/2)**2 + H**2)
    return dict(L_bot=L_bot, W_bot=W_bot, L_top=L_top, W_top=W_top, H=H,
                slant_long=s_long, slant_short=s_short, slant_corner=s_corner)


# ============================================================
#  Diagram drawing
# ============================================================

def draw_quarter_development(ax, res):
    """
    Quarter-development diagram matching the reference tool style:
    - Corner (apex) at BOTTOM-RIGHT of solid shape
    - F base goes LEFT from corner (full short side)
    - L-lines fan UP-LEFT from corner (equal angular spacing for clarity)
    - Circle arc connects L-line endpoints (inner/red edge)
    - Outer rect boundary from F-end to outer rect points (orange)
    - Dashed MIRROR on the RIGHT side of corner
    - Mirror Line: vertical dashed through the corner
    """
    ax.clear()
    ax.set_facecolor('#f8f8f0')
    ax.axis('off')

    n_quad  = res['n_quad']
    L_vals  = res['L_vals']   # L_vals[0]=L0 longest, L_vals[-1]=Ln shortest
    chord   = res['chord']
    J       = res['J']
    F       = res['F']
    slants  = res['slants']

    # ── Fan angles: Ln nearly vertical, L0 upper-left ───────────────
    angle_Ln = math.radians(91)
    angle_L0 = math.radians(155)
    angles   = np.linspace(angle_Ln, angle_L0, n_quad + 1)

    c_pts = []
    for i in range(n_quad + 1):
        a = angles[n_quad - i]        # L0 → leftmost angle
        L = L_vals[i]
        c_pts.append(np.array([L * math.cos(a), L * math.sin(a)]))

    corner = np.array([0.0, 0.0])
    F_end  = np.array([-F, 0.0])

    # Outer rect boundary points (each circle pt extended by its slant)
    r_pts = []
    for i, cp in enumerate(c_pts):
        norm = float(np.linalg.norm(cp))
        r_pts.append(cp + slants[i] * cp / norm if norm > 1e-6
                     else cp + np.array([0.0, slants[i]]))

    # Outer boundary path: F_end → Ln outer pt → ... → L0 outer pt
    outer_x = [F_end[0]] + [r[0] for r in reversed(r_pts)]
    outer_y = [F_end[1]] + [r[1] for r in reversed(r_pts)]

    # Bounding box
    all_pts = c_pts + r_pts + [corner, F_end]
    xs = [p[0] for p in all_pts]; ys = [p[1] for p in all_pts]
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    pad  = span * 0.15

    # ── Dashed mirror ─────────────────────────────────────────────────
    ax.plot([0, F], [0, 0], color='#0044aa', lw=1.8, ls='--', alpha=0.4)
    for cp in c_pts:
        ax.plot([0, -cp[0]], [0, cp[1]], 'k--', lw=0.7, alpha=0.3)
    ax.plot([-cp[0] for cp in c_pts], [cp[1] for cp in c_pts],
            'r--', lw=1.8, alpha=0.4)
    ax.plot([-x for x in outer_x], outer_y,
            color='#cc5500', lw=1.5, ls='--', alpha=0.35)

    # ── Mirror line ───────────────────────────────────────────────────
    ax.axvline(0, color='#888', ls='--', lw=1.0)
    ax.text(span*0.04, max(ys)+pad*0.55, 'Mirror Line',
            fontsize=7, color='#666', ha='left', va='bottom')

    # ── Solid quarter ─────────────────────────────────────────────────
    ax.plot(outer_x, outer_y, color='#cc5500', lw=2.2,
            solid_capstyle='round', label='Rect edge (outer)')
    ax.annotate('', xy=(F_end[0], 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='#0044aa',
                                lw=1.4, mutation_scale=10))
    ax.plot([F_end[0], 0], [0, 0], color='#0044aa', lw=2.2)
    ax.text(F_end[0]/2, -span*0.055, 'F = {:.0f}'.format(F),
            ha='center', va='top', fontsize=8.5,
            color='#0044aa', fontweight='bold')
    for cp in c_pts:
        ax.plot([0, cp[0]], [0, cp[1]], 'k-', lw=1.1, alpha=0.75, zorder=3)
    ax.plot([cp[0] for cp in c_pts], [cp[1] for cp in c_pts],
            'r-', lw=2.5, solid_capstyle='round',
            label='Circle arc (inner)', zorder=4)

    # ── Labels ────────────────────────────────────────────────────────
    fs = 8.5
    for i, (cp, lv) in enumerate(zip(c_pts, L_vals)):
        mid  = cp * 0.52
        perp = np.array([-cp[1], cp[0]])
        pn   = float(np.linalg.norm(perp))
        off  = (perp/pn * span*0.035) if pn > 1e-6 else np.zeros(2)
        ax.text(mid[0]+off[0], mid[1]+off[1], 'L{}'.format(n_quad-i),
                ha='center', va='center', fontsize=fs-0.5,
                color='#333333', fontweight='bold')

    if len(c_pts) >= 2:
        m = (c_pts[-1] + c_pts[-2]) / 2
        ax.text(m[0]+span*0.03, m[1]+span*0.02, 'C',
                ha='left', va='bottom', fontsize=fs, color='red', fontweight='bold')

    j_idx = slants.index(max(slants))
    if j_idx < len(c_pts) and j_idx < len(r_pts):
        mj = (c_pts[j_idx] + r_pts[j_idx]) / 2
        ax.text(mj[0]-span*0.04, mj[1], 'J',
                ha='right', va='center', fontsize=fs,
                color='#cc5500', fontweight='bold')

    ax.plot(0, 0, 'ko', ms=6, zorder=6)
    ax.set_xlim(min(xs)-pad, max(max(xs), F)+pad*1.5)
    ax.set_ylim(min(ys)-pad*1.2, max(ys)+pad*0.8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Quarter Development  (mirror for full half)',
                 fontsize=9, pad=4)


def draw_3d_sketch(ax, L, W, D, H, ox=0.0, oy=0.0):
    """Isometric 3D sketch of the transition."""
    ax.clear()
    ax.set_facecolor('#f0f4ff')

    def iso(x, y, z):
        px = (x - y) * np.cos(np.radians(30))
        py = (x + y) * np.sin(np.radians(30)) + z * 0.9
        return px, py

    # Rectangle base
    rx, ry = L/2, W/2
    corners = [(-rx,-ry,0),(rx,-ry,0),(rx,ry,0),(-rx,ry,0),(-rx,-ry,0)]
    xs,ys = zip(*[iso(*c) for c in corners])
    ax.plot(xs, ys, 'b-', lw=2)

    # Circle top
    angles = np.linspace(0, 2*np.pi, 60)
    cxs = [iso(ox+D/2*np.cos(a), oy+D/2*np.sin(a), H) for a in angles]
    ax.plot([p[0] for p in cxs], [p[1] for p in cxs], 'r-', lw=2)

    # 4 corner lines
    for cx,cy,_ in corners[:4]:
        near_angle = np.arctan2(cy, cx)
        cpx = ox + D/2*np.cos(near_angle)
        cpy = oy + D/2*np.sin(near_angle)
        ax.plot([iso(cx,cy,0)[0], iso(cpx,cpy,H)[0]],
                [iso(cx,cy,0)[1], iso(cpx,cpy,H)[1]],
                'k-', lw=0.8, alpha=0.5)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('3-D View', fontsize=9, pad=3)


# ============================================================
#  Main GUI
# ============================================================

class TransitionCalculator:

    TYPE_RECT_ROUND = 0
    TYPE_ROUND_RECT = 1
    TYPE_PYRAMID    = 2
    TYPE_TRUNC_PYR  = 3

    def __init__(self, root):
        self.root = root
        self.root.title("Transition Calculator")
        self.root.resizable(True, True)
        self._result = None
        self._build_ui()

    # ── UI layout ─────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        root.configure(bg='#ececec')

        # ── Top bar ──────────────────────────────────────────────
        topbar = tk.Frame(root, bg='#2244aa', height=36)
        topbar.pack(fill='x')
        tk.Label(topbar, text="  Transition Calculator",
                 bg='#2244aa', fg='white',
                 font=('Arial', 13, 'bold')).pack(side='left', pady=6)

        # ── Main 3-column frame ───────────────────────────────────
        main = tk.Frame(root, bg='#ececec')
        main.pack(fill='both', expand=True, padx=6, pady=6)

        # Column 1: type selector
        col1 = tk.LabelFrame(main, text="Select Transition Type",
                              bg='#ececec', font=('Arial', 9, 'bold'))
        col1.pack(side='left', fill='y', padx=(0,4), pady=0)
        self._build_type_panel(col1)

        # Column 2: inputs
        col2 = tk.LabelFrame(main, text="Input Data",
                              bg='#ececec', font=('Arial', 9, 'bold'))
        col2.pack(side='left', fill='y', padx=(0,4), pady=0)
        self._build_input_panel(col2)

        # Column 3: output
        col3 = tk.LabelFrame(main, text="Output Data",
                              bg='#ececec', font=('Arial', 9, 'bold'))
        col3.pack(side='left', fill='both', expand=True, pady=0)
        self._build_output_panel(col3)

    # ── Type selector ─────────────────────────────────────────────

    def _build_type_panel(self, parent):
        self._type_var = tk.IntVar(value=self.TYPE_RECT_ROUND)

        types = [
            (self.TYPE_RECT_ROUND, "Rectangular\nTo Round",   self._icon_rect_round),
            (self.TYPE_ROUND_RECT, "Round To\nRectangular",   self._icon_round_rect),
            (self.TYPE_PYRAMID,    "Pyramid",                  self._icon_pyramid),
            (self.TYPE_TRUNC_PYR,  "Truncated\nPyramid",      self._icon_trunc),
        ]
        for val, label, icon_fn in types:
            frm = tk.Frame(parent, bg='#ececec', bd=1, relief='groove')
            frm.pack(fill='x', padx=6, pady=4)

            # Mini icon canvas
            c = tk.Canvas(frm, width=80, height=60, bg='white',
                          highlightthickness=0)
            c.pack(side='left', padx=4, pady=4)
            icon_fn(c)

            rb = tk.Radiobutton(frm, text=label, variable=self._type_var,
                                value=val, bg='#ececec',
                                font=('Arial', 9), justify='left',
                                command=self._on_type_change)
            rb.pack(side='left', padx=4)

    def _icon_rect_round(self, canvas):
        canvas.create_rectangle(10, 35, 70, 55, outline='#0044aa', width=2)
        canvas.create_oval(25, 5, 55, 30, outline='red', width=2)
        for x,y in [(15,35),(65,35),(40,35)]:
            canvas.create_line(x,y, 40,17, fill='gray', width=1)

    def _icon_round_rect(self, canvas):
        canvas.create_oval(25, 5, 55, 30, outline='red', width=2)
        canvas.create_rectangle(10, 35, 70, 55, outline='#0044aa', width=2)
        for x,y in [(15,55),(65,55),(40,55)]:
            canvas.create_line(x,y, 40,17, fill='gray', width=1)

    def _icon_pyramid(self, canvas):
        canvas.create_polygon(40,5, 10,55, 70,55, fill='', outline='#666', width=2)
        canvas.create_line(40,5, 40,55, fill='gray', width=1, dash=(3,2))

    def _icon_trunc(self, canvas):
        canvas.create_polygon(20,10, 60,10, 70,55, 10,55,
                               fill='', outline='#666', width=2)
        canvas.create_line(20,10, 60,10, fill='gray')

    # ── Input panel ───────────────────────────────────────────────

    def _build_input_panel(self, parent):
        self._inputs = {}

        # Input diagram canvas
        self._diag_canvas = tk.Canvas(parent, width=230, height=130,
                                       bg='white', highlightthickness=1,
                                       highlightbackground='#aaaaaa')
        self._diag_canvas.pack(padx=8, pady=8)

        # Unit selector
        uf = tk.Frame(parent, bg='#ececec')
        uf.pack(fill='x', padx=8)
        tk.Label(uf, text="Unit:", bg='#ececec',
                 font=('Arial', 9)).pack(side='left')
        self._unit_var = tk.StringVar(value='mm')
        for u in ['mm', 'inches']:
            tk.Radiobutton(uf, text=u, variable=self._unit_var, value=u,
                           bg='#ececec', font=('Arial', 9)).pack(side='left', padx=4)

        # Field definitions per type
        self._field_defs = {
            self.TYPE_RECT_ROUND: [
                ('D', 'Round Diameter (D):',     '550'),
                ('L', 'Length (L):',             '600'),
                ('W', 'Width (W):',              '300'),
                ('H', 'Height (H):',             '400'),
                ('ox','X Offset (eccentric):',   '0'),
                ('oy','Y Offset (eccentric):',   '0'),
            ],
            self.TYPE_ROUND_RECT: [
                ('D', 'Round Diameter (D):',     '550'),
                ('L', 'Length (L):',             '600'),
                ('W', 'Width (W):',              '300'),
                ('H', 'Height (H):',             '400'),
                ('ox','X Offset (eccentric):',   '0'),
                ('oy','Y Offset (eccentric):',   '0'),
            ],
            self.TYPE_PYRAMID: [
                ('L', 'Base Length (L):',        '600'),
                ('W', 'Base Width (W):',         '300'),
                ('H', 'Height (H):',             '400'),
            ],
            self.TYPE_TRUNC_PYR: [
                ('Lb','Base Length (L):',        '600'),
                ('Wb','Base Width (W):',         '300'),
                ('Lt','Top Length (l):',         '300'),
                ('Wt','Top Width (w):',          '150'),
                ('H', 'Height (H):',             '400'),
            ],
        }

        self._fields_frame = tk.Frame(parent, bg='#ececec')
        self._fields_frame.pack(fill='x', padx=8, pady=4)
        self._build_fields(self.TYPE_RECT_ROUND)

        # Development lines
        nf = tk.Frame(parent, bg='#ececec')
        nf.pack(fill='x', padx=8, pady=4)
        tk.Label(nf, text="Development Lines (N):",
                 bg='#ececec', font=('Arial', 9)).pack(side='left')
        self._N_var = tk.IntVar(value=12)
        for n in [12, 24, 36, 48]:
            tk.Radiobutton(nf, text=str(n), variable=self._N_var, value=n,
                           bg='#ececec', font=('Arial', 9)).pack(side='left', padx=3)

        # Buttons
        bf = tk.Frame(parent, bg='#ececec')
        bf.pack(fill='x', padx=8, pady=8)
        tk.Button(bf, text="Calculate", command=self._calculate,
                  bg='#2244aa', fg='white',
                  font=('Arial', 10, 'bold'), width=12).pack(side='left', padx=4)
        tk.Button(bf, text="Clear", command=self._clear,
                  font=('Arial', 10), width=8).pack(side='left', padx=4)
        tk.Button(bf, text="Export DXF", command=self._export_dxf,
                  font=('Arial', 10), width=10).pack(side='left', padx=4)

        self._draw_input_diagram()

    def _build_fields(self, type_id):
        for w in self._fields_frame.winfo_children():
            w.destroy()
        self._inputs.clear()
        defs = self._field_defs.get(type_id, [])
        for key, label, default in defs:
            row = tk.Frame(self._fields_frame, bg='#ececec')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg='#ececec',
                     font=('Arial', 9), width=22, anchor='w').pack(side='left')
            var = tk.StringVar(value=default)
            tk.Entry(row, textvariable=var, width=10,
                     font=('Arial', 9)).pack(side='left')
            self._inputs[key] = var

    def _draw_input_diagram(self):
        c = self._diag_canvas
        c.delete('all')
        t = self._type_var.get()
        w, h = 230, 130

        if t in (self.TYPE_RECT_ROUND, self.TYPE_ROUND_RECT):
            # Front view
            c.create_rectangle(15, 60, 110, 120, outline='#0044aa', width=2)
            c.create_oval(38, 65, 87, 115, outline='red', width=2)
            c.create_text(62, 90, text='D', font=('Arial', 8), fill='red')
            # Side view
            c.create_line(125, 20, 115, 120, fill='#333', width=2)
            c.create_line(210, 20, 220, 120, fill='#333', width=2)
            c.create_line(115, 120, 220, 120, fill='#0044aa', width=2)
            c.create_line(125, 20,  210, 20, fill='red', width=2)
            # Dims
            c.create_text(168, 10, text='D', font=('Arial', 8,'bold'), fill='red')
            c.create_text(230, 70, text='H', font=('Arial', 8,'bold'))
            c.create_text(168, 128, text='L×W', font=('Arial', 8,'bold'), fill='#0044aa')
        elif t == self.TYPE_PYRAMID:
            c.create_polygon(115,10, 15,120, 215,120, fill='', outline='#666', width=2)
            c.create_line(115,10, 115,120, fill='gray', dash=(4,2))
            c.create_text(115, 5, text='Apex', font=('Arial', 7))
            c.create_text(115, 128, text='L × W', font=('Arial', 8, 'bold'))
            c.create_text(228, 65, text='H', font=('Arial', 8, 'bold'))
        elif t == self.TYPE_TRUNC_PYR:
            c.create_polygon(75,15, 155,15, 220,120, 10,120,
                              fill='', outline='#666', width=2)
            c.create_text(115, 8, text='l × w', font=('Arial', 8, 'bold'))
            c.create_text(115, 128, text='L × W', font=('Arial', 8, 'bold'))

    # ── Output panel ──────────────────────────────────────────────

    def _build_output_panel(self, parent):
        # Diagram (top)
        self._fig = plt.Figure(figsize=(9, 5), facecolor='#f8f8f0')
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)
        toolbar_frame = tk.Frame(parent)
        toolbar_frame.pack(fill='x')
        NavigationToolbar2Tk(self._canvas, toolbar_frame)

        # Values text box (bottom)
        vf = tk.Frame(parent, bg='#f8f8f0')
        vf.pack(fill='x', padx=4, pady=4)
        self._val_text = tk.Text(vf, height=9, width=48,
                                  font=('Courier', 9), bg='#f8f8f0',
                                  relief='flat', state='disabled')
        self._val_text.pack(side='left', fill='x', expand=True)

        # Status bar
        self._status = tk.StringVar(value="Enter dimensions and click Calculate.")
        tk.Label(parent, textvariable=self._status, bg='#dde8ff',
                 font=('Arial', 8), anchor='w', relief='sunken').pack(
                     fill='x', padx=2, pady=(0,2))

    def _update_val_text(self, text):
        self._val_text.configure(state='normal')
        self._val_text.delete('1.0', 'end')
        self._val_text.insert('end', text)
        self._val_text.configure(state='disabled')

    # ── Event handlers ────────────────────────────────────────────

    def _on_type_change(self):
        t = self._type_var.get()
        self._build_fields(t)
        self._draw_input_diagram()
        self._fig.clear()
        self._canvas.draw()
        self._update_val_text("")
        self._result = None

    def _get_float(self, key, label):
        try:
            return float(self._inputs[key].get())
        except (ValueError, KeyError):
            raise ValueError("Invalid value for '{}'".format(label))

    def _calculate(self):
        t = self._type_var.get()
        N = self._N_var.get()

        try:
            if t in (self.TYPE_RECT_ROUND, self.TYPE_ROUND_RECT):
                D  = self._get_float('D', 'Diameter')
                L  = self._get_float('L', 'Length')
                W  = self._get_float('W', 'Width')
                H  = self._get_float('H', 'Height')
                ox = float(self._inputs.get('ox', tk.StringVar(value='0')).get() or 0)
                oy = float(self._inputs.get('oy', tk.StringVar(value='0')).get() or 0)

                if D <= 0 or L <= 0 or W <= 0 or H <= 0:
                    raise ValueError("All dimensions must be > 0")
                if N % 4 != 0:
                    raise ValueError("N must be divisible by 4")

                res = calc_rect_to_round(L, W, D, H, N, ox, oy)
                self._result = res
                self._show_rect_round(res)

            elif t == self.TYPE_PYRAMID:
                L = self._get_float('L', 'Length')
                W = self._get_float('W', 'Width')
                H = self._get_float('H', 'Height')
                res = calc_pyramid(L, W, H)
                self._result = res
                self._show_pyramid(res)

            elif t == self.TYPE_TRUNC_PYR:
                Lb = self._get_float('Lb', 'Base Length')
                Wb = self._get_float('Wb', 'Base Width')
                Lt = self._get_float('Lt', 'Top Length')
                Wt = self._get_float('Wt', 'Top Width')
                H  = self._get_float('H',  'Height')
                res = calc_trunc_pyramid(Lb, Wb, Lt, Wt, H)
                self._result = res
                self._show_trunc_pyramid(res)

            self._status.set("Calculated OK  |  N={}  |  Unit: {}".format(
                N, self._unit_var.get()))

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._status.set("Error: {}".format(e))

    def _show_rect_round(self, res):
        self._fig.clear()
        gs = GridSpec(1, 2, figure=self._fig,
                      left=0.04, right=0.98, bottom=0.04, top=0.94,
                      wspace=0.12)
        ax_dev = self._fig.add_subplot(gs[0])
        ax_3d  = self._fig.add_subplot(gs[1])

        draw_quarter_development(ax_dev, res)
        draw_3d_sketch(ax_3d, res['L'], res['W'], res['D'], res['H'],
                       res['ox'], res['oy'])

        self._fig.suptitle(
            "Rect {}×{}  →  Ø{}  |  H={}  |  N={} lines".format(
                int(res['L']), int(res['W']), int(res['D']),
                int(res['H']), res['N']),
            fontsize=10, fontweight='bold')
        self._canvas.draw()

        # Values text
        ec = "Concentric" if not (res['ox'] or res['oy']) else \
             "Eccentric ({},{})".format(res['ox'], res['oy'])
        lines = [
            "─" * 38,
            "  C  = {:.2f} mm   (arc per segment)".format(res['C']),
            "  chord = {:.3f} mm  (mark on sheet)".format(res['chord']),
            "  J  = {:.2f} mm   (max slant)".format(res['J']),
            "  F  = {:.2f} mm   (rect short side)".format(res['F']),
            "─" * 38,
        ]
        n_quad = res['n_quad']
        for i, lv in enumerate(res['L_vals']):
            lbl = "L{}".format(n_quad - i)
            lines.append("  {:4s} = {:.2f} mm".format(lbl, lv))
        lines += [
            "─" * 38,
            "  Slant min = {:.2f} mm".format(res['s_min']),
            "  Slant max = {:.2f} mm".format(res['s_max']),
            "  Surface   = {:.0f} mm²  ({:.1f} cm²)".format(
                res['area_total'], res['area_total']/100),
            "  Type: {}".format(ec),
            "─" * 38,
        ]
        self._update_val_text("\n".join(lines))

    def _show_pyramid(self, res):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.axis('off')
        ax.set_facecolor('#f8f8f0')
        ax.text(0.5, 0.6,
                "Pyramid\n\nSlant (long side) = {:.2f} mm\n"
                "Slant (short side) = {:.2f} mm\n"
                "Slant (corner)     = {:.2f} mm\n\n"
                "Base perimeter     = {:.1f} mm\n"
                "Lateral area       = {:.0f} mm²".format(
                    res['slant_long'], res['slant_short'],
                    res['slant_corner'], res['base_perim'], res['area']),
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))
        self._canvas.draw()
        self._update_val_text(
            "Slant long   = {:.2f} mm\n"
            "Slant short  = {:.2f} mm\n"
            "Slant corner = {:.2f} mm\n"
            "Base perim   = {:.1f} mm\n"
            "Area         = {:.0f} mm²".format(
                res['slant_long'], res['slant_short'], res['slant_corner'],
                res['base_perim'], res['area']))

    def _show_trunc_pyramid(self, res):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.axis('off')
        ax.set_facecolor('#f8f8f0')
        ax.text(0.5, 0.55,
                "Truncated Pyramid\n\n"
                "Slant (long side)  = {:.2f} mm\n"
                "Slant (short side) = {:.2f} mm\n"
                "Slant (corner)     = {:.2f} mm".format(
                    res['slant_long'], res['slant_short'], res['slant_corner']),
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))
        self._canvas.draw()
        self._update_val_text(
            "Slant long   = {:.2f} mm\n"
            "Slant short  = {:.2f} mm\n"
            "Slant corner = {:.2f} mm".format(
                res['slant_long'], res['slant_short'], res['slant_corner']))

    def _clear(self):
        for var in self._inputs.values():
            var.set('')
        self._fig.clear()
        self._canvas.draw()
        self._update_val_text("")
        self._result = None
        self._status.set("Cleared.")

    def _export_dxf(self):
        if self._result is None:
            messagebox.showinfo("Export", "Calculate first.")
            return
        try:
            import ezdxf
        except ImportError:
            messagebox.showerror("Export", "ezdxf not installed.\npip install ezdxf")
            return

        res = self._result
        if 'L_vals' not in res:
            messagebox.showinfo("Export", "DXF export only for Rect-to-Round.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".dxf",
            filetypes=[("DXF files","*.dxf"),("All","*.*")],
            title="Save DXF")
        if not path:
            return

        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from sheet_metal_transition import (
                build_transition, unfold_triangles, get_all_boundaries,
                map_2d_pts, get_reference_points, export_dxf
            )
            L, W, D, H, N = res['L'], res['W'], res['D'], res['H'], res['N']
            ox, oy = res['ox'], res['oy']
            tris, rc, cp, rpp, cl3d, sp3d = build_transition(L, W, D, H, ox, oy, N)
            unfolded, placed = unfold_triangles(tris)
            boundaries = get_all_boundaries(unfolded)
            seam_2d, c2d = map_2d_pts(sp3d, cl3d, placed)
            c_refs, r_refs, k_refs = get_reference_points(cp, rpp, rc, placed)
            all2d = [p for t in unfolded for p in t]
            xs = [float(p[0]) for p in all2d]
            ys = [float(p[1]) for p in all2d]
            export_dxf(unfolded, boundaries, seam_2d, c2d,
                       c_refs, r_refs, k_refs,
                       max(xs)-min(xs), max(ys)-min(ys),
                       L, W, D, H, ox, oy, N, path)
            self._status.set("DXF saved: " + path)
            messagebox.showinfo("Export", "DXF saved:\n" + path)
        except Exception as e:
            messagebox.showerror("Export error", str(e))


# ============================================================
#  Entry point
# ============================================================

def main():
    root = tk.Tk()
    root.geometry("1200x680")
    app = TransitionCalculator(root)
    root.mainloop()


if __name__ == '__main__':
    main()
