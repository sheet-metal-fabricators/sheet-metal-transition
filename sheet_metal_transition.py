"""
Sheet Metal Surface Development Tool
Rectangle-to-Circle Transition  (Concentric & Eccentric)
Method: Triangulation (True Length)

Usage
-----
  python sheet_metal_transition.py                             # concentric defaults
  python sheet_metal_transition.py --rect-w 400 --rect-h 300 --circle-d 200 --height 250
  python sheet_metal_transition.py --offset-x 80 --offset-y 50    # eccentric
  python sheet_metal_transition.py --gui                       # Tkinter GUI with live preview
  python sheet_metal_transition.py --no-show --output part1    # headless / batch

Output files
------------
  <prefix>_preview.png      – 3-D sketch + flat pattern overview
  <prefix>_workshop.png     – Full workshop drawing with dimensions, numbered points, table
  <prefix>_coordinates.csv  – All reference point X,Y coordinates for manual scribing / CNC
  <prefix>_development.dxf  – CAD-ready DXF with geometry + dimension lines + labels

Dependencies
------------
  pip install numpy matplotlib ezdxf
"""

import sys
import os
import csv
import argparse
import numpy as np

# ============================================================
#  Geometry helpers
# ============================================================

def angle_to_rect_pt(angle, rw, rh):
    """2-D point on rectangle boundary in direction `angle` from its centre."""
    rx, ry = rw / 2.0, rh / 2.0
    dx, dy = np.cos(angle), np.sin(angle)
    ts = []
    if abs(dx) > 1e-12:
        ts += [rx / dx, -rx / dx]
    if abs(dy) > 1e-12:
        ts += [ry / dy, -ry / dy]
    if not ts:
        return np.array([0.0, 0.0])
    t = min(v for v in ts if v > 1e-12)
    return np.array([np.clip(dx * t, -rx, rx), np.clip(dy * t, -ry, ry)])


def perim_param(pt, rx, ry):
    """Perimeter parameter (mm from bottom-left corner, CCW). Total = 2*(rw+rh)*2."""
    x, y = float(pt[0]), float(pt[1])
    if abs(y + ry) < 8e-4:
        return x + rx
    elif abs(x - rx) < 8e-4:
        return 2 * rx + (y + ry)
    elif abs(y - ry) < 8e-4:
        return 2 * rx + 2 * ry + (rx - x)
    else:
        return 4 * rx + 2 * ry + (ry - y)


def dist3(a, b):
    return float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))


# ============================================================
#  3-D triangulation
# ============================================================

def build_transition(rw, rh, cd, h, ox=0.0, oy=0.0, n=24):
    """
    Build 3-D triangulation for a rectangle-to-circle transition.

    Returns
    -------
    triangles        list[(p0,p1,p2)]   each vertex a (3,) ndarray
    rect_corners     (4,3)
    circle_pts       (n,3)
    rect_perim_pts   (n,3)   rect-boundary pts aligned to circle pts
    corner_lines     list[(top_pt, bot_pt)]   fold lines at rect corners
    seam_pts         (top_pt, bot_pt)
    """
    rx, ry, cr = rw / 2.0, rh / 2.0, cd / 2.0

    rect_corners = np.array([
        [-rx, -ry, 0.0], [rx, -ry, 0.0],
        [ rx,  ry, 0.0], [-rx,  ry, 0.0],
    ])

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    circle_pts = np.column_stack([
        ox + cr * np.cos(angles),
        oy + cr * np.sin(angles),
        np.full(n, float(h)),
    ])

    # Project each circle point onto the rect boundary from rect centre
    rect_perim_pts = []
    for cp in circle_pts:
        ang = np.arctan2(cp[1], cp[0])
        rp = angle_to_rect_pt(ang, rw, rh)
        rect_perim_pts.append(np.array([rp[0], rp[1], 0.0]))
    rect_perim_pts = np.array(rect_perim_pts)

    total_perim = 2.0 * (rw + rh)
    corner_params = [perim_param(c, rx, ry) for c in rect_corners]

    triangles = []

    for i in range(n):
        j = (i + 1) % n
        c0, c1 = circle_pts[i], circle_pts[j]
        r0, r1 = rect_perim_pts[i], rect_perim_pts[j]

        p0 = perim_param(r0, rx, ry)
        p1 = perim_param(r1, rx, ry)
        if p1 < p0 - 1e-6:
            p1 += total_perim

        inserted = []
        for ci, (cp, cpt) in enumerate(zip(corner_params, rect_corners)):
            cp_adj = cp
            if cp_adj < p0 - 1e-6:
                cp_adj += total_perim
            if p0 + 1e-6 < cp_adj < p1 - 1e-6:
                inserted.append((cp_adj, cpt.copy()))
        inserted.sort(key=lambda x: x[0])

        rect_seq = [r0] + [v[1] for v in inserted] + [r1]
        nr = len(rect_seq)

        ci_idx, ri_idx = 0, 0
        c_side = [c0, c1]
        while ci_idx < 1 or ri_idx < nr - 1:
            if ri_idx >= nr - 1:
                triangles.append((c_side[ci_idx], c_side[ci_idx + 1], rect_seq[ri_idx]))
                ci_idx += 1
            elif ci_idx >= 1:
                triangles.append((rect_seq[ri_idx], rect_seq[ri_idx + 1], c_side[ci_idx]))
                ri_idx += 1
            else:
                d1 = dist3(c_side[ci_idx + 1], rect_seq[ri_idx])
                d2 = dist3(c_side[ci_idx], rect_seq[ri_idx + 1])
                if d1 <= d2:
                    triangles.append((c_side[ci_idx], c_side[ci_idx + 1], rect_seq[ri_idx]))
                    ci_idx += 1
                else:
                    triangles.append((rect_seq[ri_idx], rect_seq[ri_idx + 1], c_side[ci_idx]))
                    ri_idx += 1

    # Fold lines at corners
    corner_lines = []
    for cpt in rect_corners:
        dists = [dist3(np.append(cpt[:2], 0.0), cp) for cp in circle_pts]
        nearest_cp = circle_pts[int(np.argmin(dists))]
        corner_lines.append((nearest_cp.copy(), cpt.copy()))

    seam_pts = (circle_pts[0].copy(), rect_perim_pts[0].copy())
    return triangles, rect_corners, circle_pts, rect_perim_pts, corner_lines, seam_pts


def surface_area(triangles):
    total = 0.0
    for p0, p1, p2 in triangles:
        v1 = np.asarray(p1, float) - np.asarray(p0, float)
        v2 = np.asarray(p2, float) - np.asarray(p0, float)
        total += 0.5 * float(np.linalg.norm(np.cross(v1, v2)))
    return total


# ============================================================
#  Thickness & bend allowance calculations
# ============================================================

def _tri_normal(tri):
    p0, p1, p2 = [np.asarray(v, float) for v in tri]
    n = np.cross(p1 - p0, p2 - p0)
    nrm = np.linalg.norm(n)
    return n / nrm if nrm > 1e-10 else n


def calculate_bend_data(triangles, corner_lines_3d, t, r_bend, k_factor=0.44):
    """
    Compute the dihedral angle, bend allowance, and bend deduction at each
    K-line (rectangle corner fold line).

    Strategy
    --------
    For each rectangle corner (bot_pt, z=0):
      1. Collect all triangles that contain that corner vertex.
      2. Among those triangles, find pairs that share a second vertex at z > 0
         (i.e., a circle point) — that shared edge IS the fold line.
      3. Compute the dihedral angle between the two triangle planes.
      4. Average the dihedral angles found (there may be more than one fold
         edge at a given corner for coarser segment counts).

    Parameters
    ----------
    triangles       : list[(p0,p1,p2)]  from build_transition
    corner_lines_3d : list[(top_pt, bot_pt)]  K-line endpoints in 3-D
    t               : material thickness  (mm)
    r_bend          : inside bend radius  (mm)
    k_factor        : 0.33 coining | 0.44 air bend | 0.50 theoretical

    Returns
    -------
    list of dicts – one per K-line
    """
    results = []

    def pkey(p, digits=1):
        return (round(float(p[0]), digits),
                round(float(p[1]), digits),
                round(float(p[2]), digits))

    # Pre-index: vertex key → list of triangles that contain it
    from collections import defaultdict
    vtx_to_tris = defaultdict(list)
    for tri in triangles:
        for v in tri:
            vtx_to_tris[pkey(v)].append(tri)

    for ki, (top_pt, bot_pt) in enumerate(corner_lines_3d):
        label  = "K{}".format(ki + 1)
        ck     = pkey(bot_pt)                      # corner key (z=0)
        c_tris = vtx_to_tris.get(ck, [])

        if len(c_tris) < 2:
            results.append(dict(label=label, bend_angle_deg=None,
                                bend_allowance=None, bend_deduction=None,
                                note="corner not found in triangulation"))
            continue

        # For every pair of triangles sharing the corner, check whether they
        # also share exactly one other vertex that is a circle point (z > 0).
        # That shared (corner, circle_pt) edge is the fold line.
        fold_angles = []
        checked_edges = set()

        for i in range(len(c_tris)):
            for j in range(i + 1, len(c_tris)):
                t1, t2 = c_tris[i], c_tris[j]
                vk1 = {pkey(v) for v in t1}
                vk2 = {pkey(v) for v in t2}
                shared_keys = vk1 & vk2 - {ck}    # shared vertices besides corner

                if len(shared_keys) != 1:
                    continue                        # not adjacent along one edge

                shared_k = next(iter(shared_keys))
                if shared_k[2] < 0.1:              # must be a circle point (z > 0)
                    continue

                edge_id = tuple(sorted([ck, shared_k]))
                if edge_id in checked_edges:
                    continue
                checked_edges.add(edge_id)

                n1 = _tri_normal(t1)
                n2 = _tri_normal(t2)
                cos_d = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
                dihedral   = float(np.arccos(cos_d))
                bend_angle = float(np.pi - dihedral)
                bend_angle = max(0.001, min(bend_angle, np.pi))
                fold_angles.append(bend_angle)

        if not fold_angles:
            results.append(dict(label=label, bend_angle_deg=None,
                                bend_allowance=None, bend_deduction=None,
                                note="fold edge not found — try more segments"))
            continue

        # Use the average bend angle across all fold edges at this corner
        bend_angle     = float(np.mean(fold_angles))
        bend_angle_deg = float(np.degrees(bend_angle))

        BA   = bend_angle * (r_bend + k_factor * t)
        OSSB = (r_bend + t) * float(np.tan(bend_angle / 2))
        BD   = 2.0 * OSSB - BA

        results.append(dict(
            label          = label,
            bend_angle_deg = round(bend_angle_deg, 1),
            bend_allowance = round(BA,   2),
            bend_deduction = round(BD,   2),
        ))

    return results


def thickness_report(t, r_bend, k_factor, bend_data, rw, rh, cd, h):
    """
    Return a plain-text / structured dict with all thickness-related guidance.
    """
    total_BA = sum(d["bend_allowance"] for d in bend_data
                   if d["bend_allowance"] is not None)
    total_BD = sum(d["bend_deduction"] for d in bend_data
                   if d["bend_deduction"] is not None)

    # Seam edge prep recommendation
    if t <= 1.5:
        seam_prep = "Square butt — no bevel needed"
    elif t <= 4.0:
        seam_prep = "Light chamfer 15-20° each edge (half-V)"
    elif t <= 8.0:
        seam_prep = "Single-V bevel 30-35° each edge"
    else:
        seam_prep = "Full V or double-V bevel 35-45° each edge"

    # Minimum recommended inside radius
    r_min = 0.5 * t   # mild steel annealed

    # Inside / outside circle circumference
    circ_neutral = np.pi * cd
    circ_inside  = np.pi * (cd - t)
    circ_outside = np.pi * (cd + t)

    return dict(
        t=t, r_bend=r_bend, k_factor=k_factor,
        total_bend_allowance = round(total_BA, 2),
        total_bend_deduction = round(total_BD, 2),
        seam_edge_prep       = seam_prep,
        min_inside_radius    = round(r_min, 2),
        circle_circ_neutral  = round(circ_neutral, 2),
        circle_circ_inside   = round(circ_inside, 2),
        circle_circ_outside  = round(circ_outside, 2),
        bend_data            = bend_data,
    )


# ============================================================
#  Flat-pattern unfolding
# ============================================================

def _pt_key(p):
    return (round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2))


def _place_opposite(qa, qb, da, db, ref):
    """2-D point at distances da,db from qa,qb, on opposite side of qa->qb from ref."""
    ab = qb - qa
    lab = float(np.linalg.norm(ab))
    if lab < 1e-10:
        return qa + np.array([0.0, da])
    t  = (da**2 - db**2 + lab**2) / (2.0 * lab)
    hh = float(np.sqrt(max(0.0, da**2 - t**2)))
    d_hat = ab / lab
    perp  = np.array([-d_hat[1], d_hat[0]])
    foot  = qa + t * d_hat
    cpos  = foot + hh * perp
    cneg  = foot - hh * perp
    cross_ref = d_hat[0]*(float(ref[1])-float(qa[1])) - d_hat[1]*(float(ref[0])-float(qa[0]))
    return cneg if cross_ref >= 0.0 else cpos


def unfold_triangles(triangles):
    placed      = {}
    edge_thirds = {}
    unfolded    = []

    def ekey(ka, kb):
        return (min(ka, kb), max(ka, kb))

    for idx, (p0, p1, p2) in enumerate(triangles):
        k0, k1, k2 = _pt_key(p0), _pt_key(p1), _pt_key(p2)
        keys = [k0, k1, k2]
        pts3 = [np.asarray(p0, float), np.asarray(p1, float), np.asarray(p2, float)]

        already = [k for k in keys if k in placed]

        if len(already) < 2:
            l01 = dist3(pts3[0], pts3[1])
            l02 = dist3(pts3[0], pts3[2])
            l12 = dist3(pts3[1], pts3[2])
            q0 = np.array([0.0, 0.0])
            q1 = np.array([l01 if l01 > 1e-10 else 1e-6, 0.0])
            if l01 > 1e-10:
                cos_a = np.clip((l01**2+l02**2-l12**2)/(2.0*l01*l02+1e-15), -1.0, 1.0)
                q2    = q0 + l02 * np.array([cos_a, float(np.sqrt(max(0.0,1-cos_a**2)))])
            else:
                q2 = np.array([0.0, l02])
            placed[k0] = q0; placed[k1] = q1; placed[k2] = q2
        else:
            new_keys = [k for k in keys if k not in placed]
            if new_keys:
                nk  = new_keys[0]
                np3 = pts3[keys.index(nk)]
                ok  = [k for k in keys if k in placed]
                qa, qb = placed[ok[0]], placed[ok[1]]
                da = dist3(np3, pts3[keys.index(ok[0])])
                db = dist3(np3, pts3[keys.index(ok[1])])
                ek  = ekey(ok[0], ok[1])
                ref = edge_thirds.get(ek)
                if ref is None:
                    ab = qb - qa; lab = float(np.linalg.norm(ab))
                    ref = (qa + np.array([-ab[1], ab[0]])/lab) if lab>1e-10 else qa+np.array([0,1])
                placed[nk] = _place_opposite(qa, qb, da, db, ref)

        q0r, q1r, q2r = placed[k0], placed[k1], placed[k2]
        unfolded.append((q0r.copy(), q1r.copy(), q2r.copy()))
        for i in range(3):
            edge_thirds[ekey(keys[i], keys[(i+1)%3])] = placed[keys[(i+2)%3]].copy()

    return unfolded, placed


# ============================================================
#  Boundary extraction
# ============================================================

def get_all_boundaries(unfolded, rnd=1):
    from collections import defaultdict

    def r2(p):
        return (round(float(p[0]), rnd), round(float(p[1]), rnd))

    ecnt, emap = defaultdict(int), {}
    for tri in unfolded:
        for i in range(3):
            a, b = r2(tri[i]), r2(tri[(i+1)%3])
            ek = (min(a,b), max(a,b))
            ecnt[ek] += 1; emap[ek] = (a,b)

    bedges = {k: v for k,v in emap.items() if ecnt[k]==1}
    if not bedges:
        return []

    adj = defaultdict(list)
    for a,b in bedges.values():
        adj[a].append(b); adj[b].append(a)

    visited, chains = set(), []
    for start in list(adj):
        if start in visited:
            continue
        chain = [start]; visited.add(start)
        prev, cur = None, start
        for _ in range(len(bedges)*2+4):
            nxt = [x for x in adj[cur] if x!=prev and x not in visited]
            if not nxt:
                break
            nxt = nxt[0]; visited.add(nxt)
            chain.append(nxt); prev, cur = cur, nxt
        if len(chain) > 2:
            chains.append(chain)
    return chains


# ============================================================
#  Reference-point extraction  (numbered for workshop use)
# ============================================================

def get_reference_points(circle_pts, rect_perim_pts, rect_corners, placed):
    """
    Return ordered lists of 2-D reference points with labels:
      C01..Cn   circle arc points
      R01..Rn   corresponding rect-perimeter points
      K1..K4    rectangle corner points (if they appear in placed)
    """
    circle_refs = []
    for i, cp3 in enumerate(circle_pts):
        k = _pt_key(cp3)
        q = placed.get(k)
        if q is not None:
            circle_refs.append(("C{:02d}".format(i+1), q.copy()))

    rect_refs = []
    for i, rp3 in enumerate(rect_perim_pts):
        k = _pt_key(rp3)
        q = placed.get(k)
        if q is not None:
            rect_refs.append(("R{:02d}".format(i+1), q.copy()))

    corner_refs = []
    for i, cpt in enumerate(rect_corners):
        k = _pt_key(cpt)
        q = placed.get(k)
        if q is not None:
            corner_refs.append(("K{}".format(i+1), q.copy()))

    return circle_refs, rect_refs, corner_refs


def map_2d_pts(seam_pts_3d, corner_lines_3d, placed):
    def lk(p3):
        return placed.get(_pt_key(p3))
    s = (lk(seam_pts_3d[0]), lk(seam_pts_3d[1]))
    cl = [(lk(t), lk(b)) for t,b in corner_lines_3d
          if lk(t) is not None and lk(b) is not None]
    return s, cl


# ============================================================
#  Coordinate CSV export
# ============================================================

def export_csv(circle_refs, rect_refs, corner_refs,
               rw, rh, cd, h, ox, oy, filename):
    """
    Write all reference point coordinates to a CSV.
    Origin (0,0) is the first seam point; all coordinates in mm.
    """
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sheet Metal Transition -- Reference Coordinates"])
        w.writerow(["Rectangle", "{} x {} mm".format(rw, rh)])
        w.writerow(["Circle D", "{} mm".format(cd)])
        w.writerow(["Height", "{} mm".format(h)])
        if ox or oy:
            w.writerow(["Offset (eccentric)", "({}, {}) mm".format(ox, oy)])
        else:
            w.writerow(["Type", "Concentric"])
        w.writerow([])
        w.writerow(["Label", "X (mm)", "Y (mm)", "Description"])
        w.writerow(["--- CIRCLE ARC POINTS (inner edge) ---", "", "", ""])
        for lbl, q in circle_refs:
            w.writerow([lbl, "{:.3f}".format(q[0]), "{:.3f}".format(q[1]),
                        "Circle arc - roll to form circle edge"])
        w.writerow([])
        w.writerow(["--- RECT PERIMETER POINTS (outer edge) ---", "", "", ""])
        for lbl, q in rect_refs:
            w.writerow([lbl, "{:.3f}".format(q[0]), "{:.3f}".format(q[1]),
                        "Rect perimeter - forms base rectangle"])
        w.writerow([])
        w.writerow(["--- RECTANGLE CORNER FOLD POINTS ---", "", "", ""])
        for lbl, q in corner_refs:
            w.writerow([lbl, "{:.3f}".format(q[0]), "{:.3f}".format(q[1]),
                        "Rect corner - fold/break line passes through here"])
    print("  CSV  saved : " + filename)


# ============================================================
#  DXF export  (with dimension lines and labels)
# ============================================================

def export_dxf(unfolded, boundaries, seam_2d, corner_2d,
               circle_refs, rect_refs, corner_refs,
               span_x, span_y, rw, rh, cd, h, ox, oy, n,
               filename):
    try:
        import ezdxf
        from ezdxf import units
    except ImportError:
        print("  ezdxf not installed -- DXF skipped.  pip install ezdxf")
        return

    doc = ezdxf.new("R2010")
    doc.header["$MEASUREMENT"] = 1      # metric
    doc.header["$INSUNITS"]    = 4      # mm
    msp = doc.modelspace()

    doc.layers.add("TRIANGULATION", color=9)
    doc.layers.add("OUTLINE_CIRCLE", color=1)    # red
    doc.layers.add("OUTLINE_RECT",   color=3)    # green
    doc.layers.add("FOLD_LINES",     color=5)    # blue
    doc.layers.add("SEAM",           color=2)    # yellow
    doc.layers.add("DIMENSIONS",     color=6)    # magenta
    doc.layers.add("LABELS",         color=7)    # white/black

    # Triangulation mesh
    for tri in unfolded:
        pts = [(float(p[0]), float(p[1])) for p in tri] + \
              [(float(tri[0][0]), float(tri[0][1]))]
        msp.add_lwpolyline(pts, dxfattribs={"layer": "TRIANGULATION"})

    # Boundaries
    BOUND_LAYERS = ["OUTLINE_CIRCLE", "OUTLINE_RECT"]
    for idx, chain in enumerate(boundaries):
        lyr = BOUND_LAYERS[idx] if idx < len(BOUND_LAYERS) else "OUTLINE_CIRCLE"
        bpts = [(float(p[0]), float(p[1])) for p in chain]
        msp.add_lwpolyline(bpts, close=True, dxfattribs={"layer": lyr})

    # Fold lines
    for qt, qb in corner_2d:
        msp.add_line((float(qt[0]), float(qt[1])),
                     (float(qb[0]), float(qb[1])),
                     dxfattribs={"layer": "FOLD_LINES"})

    # Seam
    if seam_2d[0] is not None and seam_2d[1] is not None:
        msp.add_line((float(seam_2d[0][0]), float(seam_2d[0][1])),
                     (float(seam_2d[1][0]), float(seam_2d[1][1])),
                     dxfattribs={"layer": "SEAM"})

    # ── Reference point labels ──
    txt_h = max(span_x, span_y) * 0.015
    for lbl, q in circle_refs + rect_refs + corner_refs:
        msp.add_text(lbl,
                     dxfattribs={"layer": "LABELS",
                                 "height": txt_h,
                                 "insert": (float(q[0]), float(q[1]))})
        # Small cross mark
        s = txt_h * 0.4
        msp.add_line((float(q[0])-s, float(q[1])),
                     (float(q[0])+s, float(q[1])),
                     dxfattribs={"layer": "LABELS"})
        msp.add_line((float(q[0]), float(q[1])-s),
                     (float(q[0]), float(q[1])+s),
                     dxfattribs={"layer": "LABELS"})

    # ── Dimension lines ──
    all2d  = [p for tri in unfolded for p in tri]
    xs     = [float(p[0]) for p in all2d]
    ys     = [float(p[1]) for p in all2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin = max(span_x, span_y) * 0.12

    dim_style = "STANDARD"
    # Horizontal overall width
    msp.add_linear_dim(
        base=(xmin, ymin - margin),
        p1=(xmin, ymin - margin * 0.5),
        p2=(xmax, ymin - margin * 0.5),
        dimstyle=dim_style,
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()

    # Vertical overall height
    msp.add_linear_dim(
        base=(xmax + margin, ymin),
        p1=(xmax + margin * 0.5, ymin),
        p2=(xmax + margin * 0.5, ymax),
        angle=90,
        dimstyle=dim_style,
        dxfattribs={"layer": "DIMENSIONS"},
    ).render()

    # ── Title block (simple text) ──
    tb_x = xmin
    tb_y = ymin - margin * 2.0
    title_lines = [
        "SHEET METAL TRANSITION -- SURFACE DEVELOPMENT",
        "Rectangle: {} x {} mm  |  Circle D: {} mm  |  Height: {} mm".format(rw, rh, cd, h),
        ("Concentric" if not (ox or oy) else
         "Eccentric offset ({}, {}) mm".format(ox, oy)),
        "Segments: {}  |  All dimensions in mm  |  Scale: 1:1".format(n),
    ]
    for i, line in enumerate(title_lines):
        msp.add_text(line, dxfattribs={
            "layer": "LABELS",
            "height": txt_h * 1.1,
            "insert": (tb_x, tb_y - i * txt_h * 1.8),
        })

    doc.saveas(filename)
    print("  DXF  saved : " + filename)


# ============================================================
#  Preview figure  (3-D + flat pattern overview)
# ============================================================

def _make_preview(rw, rh, cd, h, ox, oy, n,
                  triangles_3d, rect_corners, circle_pts,
                  unfolded, boundaries, seam_2d, corner_2d, surf_area):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection

    type_label = ("Concentric" if not (ox or oy)
                  else "Eccentric  ({:.1f}, {:.1f}) mm offset".format(ox, oy))

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle(
        "Sheet Metal Transition   Rect {}x{}  ->  D{}  |  H={}  |  {}".format(
            rw, rh, cd, h, type_label),
        fontsize=12, fontweight="bold")

    # 3-D
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.set_title("3-D View", fontsize=10)
    rc = np.vstack([rect_corners, rect_corners[0]])
    ax3.plot(rc[:,0], rc[:,1], rc[:,2], "b-", lw=2, label="Rectangle")
    cp_cl = np.vstack([circle_pts, circle_pts[0]])
    ax3.plot(cp_cl[:,0], cp_cl[:,1], cp_cl[:,2], "r-", lw=2, label="Circle")
    polys = [[list(t[j]) for j in range(3)] for t in triangles_3d]
    ax3.add_collection3d(Poly3DCollection(polys, alpha=0.18,
        facecolor="#88aacc", edgecolor="#336699", linewidths=0.25))
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.view_init(elev=28, azim=-50)

    # Flat pattern
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Flat Pattern (Surface Development)", fontsize=10)
    ax2.set_aspect("equal")

    patches = [MplPoly([t[0], t[1], t[2]], closed=True) for t in unfolded]
    ax2.add_collection(PatchCollection(patches, facecolor="#cce5ff",
        edgecolor="#336699", linewidths=0.35, alpha=0.80))

    BCOLS = ["#cc0000", "#cc6600"]
    BLBLS = ["Circle edge", "Rect edge"]
    for idx, chain in enumerate(boundaries):
        col = BCOLS[idx % len(BCOLS)]
        bx = [p[0] for p in chain] + [chain[0][0]]
        by = [p[1] for p in chain] + [chain[0][1]]
        ax2.plot(bx, by, color=col, lw=1.8, label=BLBLS[idx], zorder=5)

    for ii, (qt, qb) in enumerate(corner_2d):
        ax2.plot([qt[0], qb[0]], [qt[1], qb[1]], "g--", lw=1.0,
                 label=("Fold lines" if ii==0 else None), zorder=4, alpha=0.85)

    if seam_2d[0] is not None and seam_2d[1] is not None:
        ax2.plot([seam_2d[0][0], seam_2d[1][0]],
                 [seam_2d[0][1], seam_2d[1][1]],
                 color="#cc9900", lw=2.0, ls="-.", label="Seam/cut", zorder=6)

    all2d = [p for t in unfolded for p in t]
    xs = [float(p[0]) for p in all2d]
    ys = [float(p[1]) for p in all2d]
    m  = max(rw, rh, h) * 0.05
    ax2.set_xlim(min(xs)-m, max(xs)+m)
    ax2.set_ylim(min(ys)-m, max(ys)+m)
    ax2.grid(True, ls="--", alpha=0.25)
    ax2.set_xlabel("mm"); ax2.set_ylabel("mm")
    ax2.legend(fontsize=8, loc="lower right")

    span_x = max(xs)-min(xs); span_y = max(ys)-min(ys)
    info = "\n".join([
        "Rectangle : {} x {} mm".format(rw, rh),
        "Circle D  : {} mm".format(cd),
        "Height    : {} mm".format(h),
        "Type      : {}".format(type_label),
        "Segments  : {}".format(n),
        "-------------------",
        "Pattern W : {:.1f} mm".format(span_x),
        "Pattern H : {:.1f} mm".format(span_y),
        "Surf.Area : {:.0f} mm2".format(surf_area),
    ])
    ax2.text(0.02, 0.98, info, transform=ax2.transAxes, va="top",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.90))

    plt.tight_layout()
    return fig


# ============================================================
#  Workshop drawing  (large, with dim lines, numbered pts, coord table)
# ============================================================

def _dim_line(ax, x1, y1, x2, y2, text, side="below", offset=20, fontsize=8):
    """Draw a dimension line with arrows and text between two points."""
    import matplotlib.patches as mpa
    mx, my = (x1+x2)/2, (y1+y2)/2

    # Arrow line
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<->", color="#333333",
                                lw=0.9, mutation_scale=10))
    # Extension lines (short ticks)
    dx, dy = x2-x1, y2-y1
    length = np.hypot(dx, dy)
    if length > 1e-6:
        nx, ny = -dy/length, dx/length   # normal
        for px, py in [(x1,y1),(x2,y2)]:
            ax.plot([px, px+nx*offset*0.6], [py, py+ny*offset*0.6],
                    color="#333333", lw=0.7)
    # Text
    tx = mx + (nx * offset if length > 1e-6 else 0)
    ty = my + (ny * offset if length > 1e-6 else 0)
    ax.text(tx, ty, text, ha="center", va="center", fontsize=fontsize,
            bbox=dict(fc="white", ec="none", pad=1))


def _make_workshop(rw, rh, cd, h, ox, oy, n,
                   unfolded, boundaries, seam_2d, corner_2d,
                   circle_refs, rect_refs, corner_refs, surf_area):
    """
    Workshop drawing:
      Top    – flat pattern with dimension lines, numbered reference points, scale bar
      Bottom – coordinate reference table (all point X,Y values)
    Portrait layout sized to match the pattern's aspect ratio.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    import matplotlib.ticker as ticker

    type_label = ("Concentric" if not (ox or oy)
                  else "Eccentric  ({:.1f}, {:.1f}) mm offset".format(ox, oy))

    all2d = [p for t in unfolded for p in t]
    xs    = [float(p[0]) for p in all2d]
    ys    = [float(p[1]) for p in all2d]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    span_x = xmax - xmin
    span_y = ymax - ymin

    # ── Size figure to give the pattern ~11 inches of width ──────────
    dim_margin   = max(span_x, span_y) * 0.14    # space for dim lines
    content_w    = span_x + dim_margin            # mm
    content_h    = span_y + dim_margin            # mm
    pattern_inch = 11.0                           # target width for pattern panel
    mm_per_inch  = content_w / pattern_inch
    table_inch   = 6.0
    fig_w        = pattern_inch + table_inch + 0.8
    fig_h        = max(content_h / mm_per_inch + 1.5, 11.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#f5f5ee")

    fig.suptitle(
        "WORKSHOP DRAWING  --  Sheet Metal Transition Surface Development\n"
        "Rect {}x{}mm  ->  Circle D{}mm  |  Height {}mm  |  {}  |  All dimensions in mm".format(
            rw, rh, cd, h, type_label),
        fontsize=11, fontweight="bold", y=0.99, va="top")

    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[pattern_inch, table_inch],
        figure=fig,
        left=0.03, right=0.99, bottom=0.03, top=0.93,
        wspace=0.04,
    )

    # ════════════════════════════════════════════════════════════════
    #  LEFT panel  –  flat pattern drawing
    # ════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("#ffffff")
    ax.set_title("Flat Pattern   (geometry is 1:1 — use the DXF for CNC / plotter)",
                 fontsize=9, pad=5)
    ax.set_xlabel("mm", fontsize=8)
    ax.set_ylabel("mm", fontsize=8)
    ax.tick_params(labelsize=7)

    # 100 mm major grid, 50 mm minor
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))
    ax.grid(which="major", color="#c8c8c8", lw=0.55, zorder=0)
    ax.grid(which="minor", color="#e8e8e8", lw=0.28, zorder=0)

    # Filled triangulation
    patches = [MplPoly([t[0], t[1], t[2]], closed=True) for t in unfolded]
    ax.add_collection(PatchCollection(patches, facecolor="#ddeeff",
        edgecolor="#7799bb", linewidths=0.28, alpha=0.78, zorder=1))

    # Boundary curves
    BCOLS = ["#cc0000", "#cc5500"]
    BLBLS = ["Circle edge  (inner — roll)", "Rect edge  (outer — bend)"]
    for idx, chain in enumerate(boundaries[:2]):
        bx = [p[0] for p in chain] + [chain[0][0]]
        by = [p[1] for p in chain] + [chain[0][1]]
        ax.plot(bx, by, color=BCOLS[idx], lw=2.2, label=BLBLS[idx], zorder=4, solid_capstyle="round")

    # Corner fold lines
    for ii, (qt, qb) in enumerate(corner_2d):
        ax.plot([qt[0], qb[0]], [qt[1], qb[1]],
                color="#006600", lw=1.3, ls="--",
                label=("Corner fold lines" if ii == 0 else None), zorder=3)

    # Seam
    if seam_2d[0] is not None and seam_2d[1] is not None:
        ax.plot([seam_2d[0][0], seam_2d[1][0]],
                [seam_2d[0][1], seam_2d[1][1]],
                color="#aa7700", lw=2.2, ls="-.", label="Seam  (cut + weld)", zorder=5)

    # ── Reference points  (labelled every 3rd to keep readable) ──────
    step = max(1, len(circle_refs) // 12)   # max ~12 labels on circle
    for i, (lbl, q) in enumerate(circle_refs):
        ax.plot(q[0], q[1], "o", ms=3.5, color="#cc0000", zorder=6, markeredgewidth=0)
        if i % step == 0 or i == len(circle_refs)-1:
            ax.annotate(lbl, xy=(q[0], q[1]),
                        xytext=(q[0]+span_x*0.018, q[1]+span_y*0.010),
                        fontsize=6, color="#aa0000", zorder=7,
                        arrowprops=dict(arrowstyle="-", color="#cc0000", lw=0.5),
                        bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))

    rstep = max(1, len(rect_refs) // 12)
    for i, (lbl, q) in enumerate(rect_refs):
        ax.plot(q[0], q[1], "s", ms=3.5, color="#cc5500", zorder=6, markeredgewidth=0)
        if i % rstep == 0 or i == len(rect_refs)-1:
            ax.annotate(lbl, xy=(q[0], q[1]),
                        xytext=(q[0]-span_x*0.025, q[1]-span_y*0.012),
                        fontsize=6, color="#993300", zorder=7,
                        arrowprops=dict(arrowstyle="-", color="#cc5500", lw=0.5),
                        bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.8))

    for lbl, q in corner_refs:
        ax.plot(q[0], q[1], "D", ms=6, color="#0033cc",
                zorder=7, markeredgewidth=0.5, markeredgecolor="#ffffff")
        ax.annotate(lbl, xy=(q[0], q[1]),
                    xytext=(q[0]+span_x*0.022, q[1]+span_y*0.018),
                    fontsize=8, color="#0033cc", fontweight="bold", zorder=8,
                    arrowprops=dict(arrowstyle="-", color="#0033cc", lw=0.7),
                    bbox=dict(fc="#eeeeff", ec="#0033cc", pad=1.5, alpha=0.9,
                              boxstyle="round,pad=0.3"))

    # ── Dimension lines  (outside the pattern bounding box) ──────────
    d = max(span_x, span_y) * 0.065    # offset distance

    # Overall WIDTH  — below the pattern
    y_dim_h = ymin - d * 1.1
    ax.annotate("", xy=(xmax, y_dim_h), xytext=(xmin, y_dim_h),
                arrowprops=dict(arrowstyle="<->", color="#222222", lw=1.0,
                                mutation_scale=8))
    ax.plot([xmin, xmin], [ymin, y_dim_h], color="#444444", lw=0.6, ls=":")
    ax.plot([xmax, xmax], [ymin, y_dim_h], color="#444444", lw=0.6, ls=":")
    ax.text((xmin+xmax)/2, y_dim_h - d*0.15,
            "W = {:.1f} mm".format(span_x),
            ha="center", va="top", fontsize=9, fontweight="bold",
            bbox=dict(fc="white", ec="#888888", pad=2, alpha=0.9,
                      boxstyle="round,pad=0.3"))

    # Overall HEIGHT  — right of the pattern
    x_dim_v = xmax + d * 1.1
    ax.annotate("", xy=(x_dim_v, ymax), xytext=(x_dim_v, ymin),
                arrowprops=dict(arrowstyle="<->", color="#222222", lw=1.0,
                                mutation_scale=8))
    ax.plot([xmax, x_dim_v], [ymin, ymin], color="#444444", lw=0.6, ls=":")
    ax.plot([xmax, x_dim_v], [ymax, ymax], color="#444444", lw=0.6, ls=":")
    ax.text(x_dim_v + d*0.1, (ymin+ymax)/2,
            "H = {:.1f} mm".format(span_y),
            ha="left", va="center", fontsize=9, fontweight="bold",
            rotation=90,
            bbox=dict(fc="white", ec="#888888", pad=2, alpha=0.9,
                      boxstyle="round,pad=0.3"))

    # ── Scale bar  (100 mm physical) ──────────────────────────────────
    sb_x  = xmin
    sb_y  = y_dim_h - d * 1.25
    tick  = span_y * 0.008
    ax.plot([sb_x, sb_x+100], [sb_y, sb_y], color="#000000", lw=2.5, solid_capstyle="butt")
    for sx in [sb_x, sb_x+100]:
        ax.plot([sx, sx], [sb_y-tick, sb_y+tick], color="#000000", lw=1.5)
    ax.text(sb_x+50, sb_y-tick*1.5, "100 mm  (scale bar)",
            ha="center", va="top", fontsize=8, color="#000000")

    # ── Info box  (top-right of pattern panel) ───────────────────────
    ax.text(0.99, 0.99,
            "Surface Area : {:.0f} mm2\n"
            "Pattern W    : {:.1f} mm\n"
            "Pattern H    : {:.1f} mm\n"
            "----------------------------\n"
            "Seam  : cut here, weld last\n"
            "K pts : bend/break line\n"
            "C pts : circle arc edge\n"
            "R pts : rect perimeter edge".format(surf_area, span_x, span_y),
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                      ec="#999900", alpha=0.93))

    ax.legend(fontsize=8, loc="lower left", framealpha=0.92,
              edgecolor="#aaaaaa", fancybox=True)

    ax.set_xlim(xmin - d*0.4, x_dim_v + d*1.5)
    ax.set_ylim(sb_y - d*0.6,  ymax + d*0.5)

    # ════════════════════════════════════════════════════════════════
    #  RIGHT panel  –  coordinate reference table
    # ════════════════════════════════════════════════════════════════
    ax_t = fig.add_subplot(gs[1])
    ax_t.axis("off")
    ax_t.set_facecolor("#f5f5ee")

    # Build table data
    all_refs = circle_refs + rect_refs + corner_refs

    def role_short(lbl):
        if lbl.startswith("C"):  return "Circ"
        if lbl.startswith("R"):  return "Rect"
        return "Fold"

    col_hdr  = ["Pt", "X mm", "Y mm", "Type"]
    rows_all = [[lbl, "{:.1f}".format(q[0]), "{:.1f}".format(q[1]), role_short(lbl)]
                for lbl, q in all_refs]

    # Row colours
    C_COL = "#ffe8e8"; R_COL = "#fff4e8"; K_COL = "#e8eeff"

    def row_color(lbl):
        if lbl.startswith("C"): return C_COL
        if lbl.startswith("R"): return R_COL
        return K_COL

    cell_colors = [["#2244aa"]*4] + [[row_color(r[0])]*4 for r in rows_all]

    table_data = [col_hdr] + rows_all
    tbl = ax_t.table(
        cellText=table_data,
        cellLoc="center",
        loc="upper center",
        bbox=[0.01, 0.0, 0.99, 1.0],   # fill the axes
    )
    tbl.auto_set_font_size(False)

    # Dynamic font size based on row count
    nrows = len(table_data)
    fs = max(5, min(8, int(88 / nrows)))
    tbl.set_fontsize(fs)

    # Style header row
    for ci in range(4):
        cell = tbl[0, ci]
        cell.set_facecolor("#2244aa")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#ffffff")

    # Style data rows
    for ri, row in enumerate(rows_all):
        fc = row_color(row[0])
        for ci in range(4):
            cell = tbl[ri+1, ci]
            cell.set_facecolor(fc)
            cell.set_edgecolor("#cccccc")
            cell.set_text_props(color="#111111")

    # Column widths
    for ci, w in enumerate([0.18, 0.26, 0.26, 0.16]):
        for ri in range(len(table_data)):
            tbl[ri, ci].set_width(w)

    # ── Colour key ───────────────────────────────────────────────────
    ax_t.set_title("Reference Point Coordinates", fontsize=9, pad=4,
                   fontweight="bold")

    # Instructions text at bottom (axes fraction)
    instruct = (
        "HOW TO USE THIS DRAWING\n"
        "────────────────────────────────\n"
        "1. For CNC/plasma: import the DXF\n"
        "   file directly (geometry is 1:1)\n\n"
        "2. For manual layout:\n"
        "   a. Mark datum corner on sheet\n"
        "   b. Scribe each C, R, K point\n"
        "      using the X,Y table above\n"
        "   c. Join C points with a smooth\n"
        "      flexible curve (inner edge)\n"
        "   d. Join R points along the\n"
        "      outer rect boundary\n"
        "   e. Connect K1-K4 through the\n"
        "      piece (fold/break lines)\n\n"
        "3. Cut along the C + R boundary\n\n"
        "4. Bend at K1-K4 fold lines to\n"
        "   form the 4 rectangle sides\n\n"
        "5. Roll/form the circle edge\n\n"
        "6. Weld the seam line last\n"
        "────────────────────────────────\n"
        "C = circle arc edge points\n"
        "R = rect perimeter edge points\n"
        "K = corner fold/break points"
    )
    ax_t.text(0.5, -0.01, instruct, transform=ax_t.transAxes,
              ha="center", va="top", fontsize=7.5, family="monospace",
              linespacing=1.45,
              bbox=dict(boxstyle="round,pad=0.6", fc="#efffee",
                        ec="#44aa44", alpha=0.95))

    return fig


# ============================================================
#  High-level pipeline
# ============================================================

def run_development(rw, rh, cd, h, ox, oy, n, output_prefix, show=True):
    tris, rc, cp, rpp, cl3d, sp3d = build_transition(rw, rh, cd, h, ox, oy, n)
    unfolded, placed = unfold_triangles(tris)
    boundaries       = get_all_boundaries(unfolded)
    seam_2d, c2d     = map_2d_pts(sp3d, cl3d, placed)
    area             = surface_area(tris)
    c_refs, r_refs, k_refs = get_reference_points(cp, rpp, rc, placed)

    all2d  = [p for tri in unfolded for p in tri]
    xs     = [float(p[0]) for p in all2d]
    ys     = [float(p[1]) for p in all2d]
    span_x = max(xs)-min(xs)
    span_y = max(ys)-min(ys)

    import matplotlib.pyplot as plt

    # 1. Preview
    fig_prev = _make_preview(rw, rh, cd, h, ox, oy, n, tris, rc, cp,
                              unfolded, boundaries, seam_2d, c2d, area)
    out_prev = output_prefix + "_preview.png"
    fig_prev.savefig(out_prev, dpi=150, bbox_inches="tight")
    print("  Preview    : " + out_prev)
    if show:
        plt.show()
    plt.close(fig_prev)

    # 2. Workshop drawing
    fig_ws = _make_workshop(rw, rh, cd, h, ox, oy, n,
                             unfolded, boundaries, seam_2d, c2d,
                             c_refs, r_refs, k_refs, area)
    out_ws = output_prefix + "_workshop.png"
    fig_ws.savefig(out_ws, dpi=150, bbox_inches="tight")
    print("  Workshop   : " + out_ws)
    if show:
        plt.show()
    plt.close(fig_ws)

    # 3. CSV
    export_csv(c_refs, r_refs, k_refs, rw, rh, cd, h, ox, oy,
               output_prefix + "_coordinates.csv")

    # 4. DXF
    export_dxf(unfolded, boundaries, seam_2d, c2d,
               c_refs, r_refs, k_refs,
               span_x, span_y, rw, rh, cd, h, ox, oy, n,
               output_prefix + "_development.dxf")

    return dict(triangles=len(tris), boundary_loops=len(boundaries),
                pattern_w=span_x, pattern_h=span_y, surface_area=area,
                circle_pts=len(c_refs), rect_pts=len(r_refs), corner_pts=len(k_refs))


# ============================================================
#  Tkinter GUI  (embedded preview + workshop)
# ============================================================

def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title("Sheet Metal Transition -- Surface Development")

    PAD = dict(padx=8, pady=4)

    left = ttk.Frame(root, padding=8)
    left.grid(row=0, column=0, sticky="nsew")

    frm = ttk.LabelFrame(left, text="Transition Parameters", padding=10)
    frm.pack(fill="x")

    FIELDS = [
        ("Rectangle Width  (mm)",  "rect_w",   "400"),
        ("Rectangle Height (mm)",  "rect_h",   "300"),
        ("Circle Diameter  (mm)",  "circle_d", "200"),
        ("Transition Height (mm)", "height",   "250"),
        ("Circle X Offset  (mm)",  "offset_x", "0"),
        ("Circle Y Offset  (mm)",  "offset_y", "0"),
        ("Circle Segments",        "segments", "24"),
        ("Output Prefix",          "output",   "transition"),
    ]
    entries = {}
    for r, (lbl, key, default) in enumerate(FIELDS):
        ttk.Label(frm, text=lbl, width=24, anchor="w").grid(row=r, column=0, sticky="w", **PAD)
        var = tk.StringVar(value=default)
        ttk.Entry(frm, textvariable=var, width=14).grid(row=r, column=1, sticky="ew", **PAD)
        entries[key] = var

    # View toggle
    view_var = tk.StringVar(value="preview")
    vf = ttk.LabelFrame(left, text="View", padding=6)
    vf.pack(fill="x", pady=(6,0))
    ttk.Radiobutton(vf, text="Preview",          variable=view_var, value="preview").pack(side="left")
    ttk.Radiobutton(vf, text="Workshop Drawing", variable=view_var, value="workshop").pack(side="left")

    status_var = tk.StringVar(value="Enter parameters and click Generate.")
    ttk.Label(left, textvariable=status_var, relief="sunken", anchor="w",
              wraplength=310).pack(fill="x", pady=(8,0))

    right = ttk.Frame(root, padding=4)
    right.grid(row=0, column=1, sticky="nsew")

    # Placeholder figure
    ph_fig, ph_ax = plt.subplots(figsize=(11, 6))
    ph_ax.text(0.5, 0.5, "Preview will appear here.\nEnter parameters and click Generate.",
               ha="center", va="center", fontsize=12, color="gray")
    ph_ax.axis("off")
    canvas = FigureCanvasTkAgg(ph_fig, master=right)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    toolbar = NavigationToolbar2Tk(canvas, right)
    toolbar.update(); toolbar.pack(side="bottom", fill="x")

    # Store generated figures
    _figs = {}

    def show_fig(key):
        if key not in _figs:
            return
        canvas.figure = _figs[key]
        canvas.figure.set_canvas(canvas)
        canvas.draw(); toolbar.update()

    def on_view_change(*_):
        show_fig(view_var.get())

    view_var.trace_add("write", on_view_change)

    def run():
        try:
            rw  = float(entries["rect_w"].get())
            rh  = float(entries["rect_h"].get())
            cd  = float(entries["circle_d"].get())
            h   = float(entries["height"].get())
            ox  = float(entries["offset_x"].get())
            oy  = float(entries["offset_y"].get())
            n   = int(entries["segments"].get())
            pfx = entries["output"].get().strip() or "transition"
        except ValueError as exc:
            messagebox.showerror("Input Error", str(exc)); return

        if n < 8:
            messagebox.showwarning("Warning", "Segments < 8 gives low accuracy.")
        if any(v <= 0 for v in [rw, rh, cd, h]):
            messagebox.showerror("Input Error", "All dimensions must be > 0."); return

        status_var.set("Generating..."); root.update_idletasks()

        try:
            tris, rc, cp, rpp, cl3d, sp3d = build_transition(rw, rh, cd, h, ox, oy, n)
            unfolded, placed = unfold_triangles(tris)
            boundaries       = get_all_boundaries(unfolded)
            seam_2d, c2d     = map_2d_pts(sp3d, cl3d, placed)
            area             = surface_area(tris)
            c_refs, r_refs, k_refs = get_reference_points(cp, rpp, rc, placed)

            all2d  = [p for t in unfolded for p in t]
            xs     = [float(p[0]) for p in all2d]
            ys     = [float(p[1]) for p in all2d]
            span_x = max(xs)-min(xs); span_y = max(ys)-min(ys)

            _figs["preview"]  = _make_preview(rw, rh, cd, h, ox, oy, n,
                                              tris, rc, cp, unfolded, boundaries,
                                              seam_2d, c2d, area)
            _figs["workshop"] = _make_workshop(rw, rh, cd, h, ox, oy, n,
                                               unfolded, boundaries, seam_2d, c2d,
                                               c_refs, r_refs, k_refs, area)

            show_fig(view_var.get())

            # Save all files
            _figs["preview"].savefig(pfx+"_preview.png", dpi=150, bbox_inches="tight")
            _figs["workshop"].savefig(pfx+"_workshop.png", dpi=150, bbox_inches="tight")
            export_csv(c_refs, r_refs, k_refs, rw, rh, cd, h, ox, oy,
                       pfx+"_coordinates.csv")
            export_dxf(unfolded, boundaries, seam_2d, c2d,
                       c_refs, r_refs, k_refs,
                       span_x, span_y, rw, rh, cd, h, ox, oy, n,
                       pfx+"_development.dxf")

            status_var.set(
                "Done.  {:.0f}x{:.0f}mm  |  Area {:.0f}mm2  |  "
                "{} C-pts  {}  R-pts  {} K-pts  |  Saved {}_*".format(
                    span_x, span_y, area,
                    len(c_refs), len(r_refs), len(k_refs), pfx))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            status_var.set("Error -- check console."); raise

    btn_frm = ttk.Frame(left)
    btn_frm.pack(fill="x", pady=8)
    ttk.Button(btn_frm, text="Generate Development", command=run, width=32).pack()

    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)
    root.mainloop()


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sheet Metal Surface Development: Rectangle -> Circle Transition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gui",      action="store_true")
    parser.add_argument("--rect-w",   type=float, default=400,  metavar="mm")
    parser.add_argument("--rect-h",   type=float, default=300,  metavar="mm")
    parser.add_argument("--circle-d", type=float, default=200,  metavar="mm")
    parser.add_argument("--height",   type=float, default=250,  metavar="mm")
    parser.add_argument("--offset-x", type=float, default=0.0,  metavar="mm")
    parser.add_argument("--offset-y", type=float, default=0.0,  metavar="mm")
    parser.add_argument("--segments", type=int,   default=24)
    parser.add_argument("--output",   type=str,   default="transition")
    parser.add_argument("--no-show",  action="store_true")
    args = parser.parse_args()

    if args.no_show:
        import matplotlib; matplotlib.use("Agg")

    if args.gui:
        launch_gui(); return

    print("\n" + "="*52)
    print("  Sheet Metal Transition -- Surface Development")
    print("="*52)
    print("  Rectangle : {} x {} mm".format(args.rect_w, args.rect_h))
    print("  Circle D  : {} mm".format(args.circle_d))
    print("  Height    : {} mm".format(args.height))
    if args.offset_x or args.offset_y:
        print("  Offset    : ({}, {}) mm  [ECCENTRIC]".format(args.offset_x, args.offset_y))
    else:
        print("  Type      : Concentric")
    print("  Segments  : {}".format(args.segments))
    print()

    s = run_development(args.rect_w, args.rect_h, args.circle_d, args.height,
                        args.offset_x, args.offset_y, args.segments,
                        args.output, show=not args.no_show)

    print()
    print("  Triangles      : {}".format(s["triangles"]))
    print("  Bound. loops   : {}".format(s["boundary_loops"]))
    print("  Circle pts (C) : {}".format(s["circle_pts"]))
    print("  Rect pts   (R) : {}".format(s["rect_pts"]))
    print("  Corner pts (K) : {}".format(s["corner_pts"]))
    print("  Pattern W      : {:.2f} mm".format(s["pattern_w"]))
    print("  Pattern H      : {:.2f} mm".format(s["pattern_h"]))
    print("  Surf. Area     : {:.0f} mm2  ({:.1f} cm2)".format(
        s["surface_area"], s["surface_area"]/100))
    print("\nDone.")


if __name__ == "__main__":
    main()
