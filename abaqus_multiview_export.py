"""
abaqus_multiview_export.py
==========================
Companion to abaqus_camera_setup.py.

Generates and saves a configurable set of camera views (iso, +X, -Y, etc.)
from a single .inp + .odb pair in one shot.

Run:
    abaqus cae noGUI=abaqus_multiview_export.py
"""

import os
import sys
import math

# ── Re-use the pure-Python helpers from the main script ──────────────────────
# Either place both files in the same folder, or copy the functions below.
# Here we inline the minimal subset needed.

# ═════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

INP_FILE      = r"C:\models\mymodel.inp"
ODB_FILE      = r"C:\models\mymodel.odb"
TARGET_PART   = "PART-1"

FIELD_VARIABLE  = "S"
FIELD_COMPONENT = "Mises"
STEP_INDEX      = -1
FRAME_INDEX     = -1

# Dict of {view_label: (dx, dy, dz)} — camera-to-target direction vectors
# Add, remove, or rename entries freely.
VIEWS = {
    "iso_front_right" : ( 1.0,  0.8,  1.0),
    "iso_back_left"   : (-1.0,  0.8, -1.0),
    "front_XY"        : ( 0.0,  0.0,  1.0),
    "side_YZ"         : ( 1.0,  0.0,  0.0),
    "top_XZ"          : ( 0.0,  1.0,  0.0),
}

OUTPUT_DIR       = r"C:\models\views"
IMAGE_WIDTH_PX   = 1920
IMAGE_HEIGHT_PX  = 1080
STANDOFF_MULT    = 2.5
WORLD_UP         = (0.0, 1.0, 0.0)

# ═════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS  (inline)
# ═════════════════════════════════════════════════════════════════════════════

def _normalise(v):
    mag = math.sqrt(sum(c*c for c in v))
    if mag < 1e-12:
        raise ValueError("Zero-length vector: {}".format(v))
    return tuple(c / mag for c in v)

def _compute_camera(centre, radius, view_dir_raw, standoff_mult, world_up):
    view_dir = _normalise(view_dir_raw)
    standoff = radius * standoff_mult
    cam_pos  = tuple(centre[i] - view_dir[i]*standoff for i in range(3))

    up_hint = _normalise(world_up)
    dot = sum(up_hint[i]*view_dir[i] for i in range(3))
    if abs(dot) > 0.99:
        for fb in [(1,0,0),(0,0,1),(0,1,0)]:
            dot = sum(fb[i]*view_dir[i] for i in range(3))
            if abs(dot) < 0.99:
                up_hint = fb
                break
    up_ortho = tuple(up_hint[i] - dot*view_dir[i] for i in range(3))
    up_ortho = _normalise(up_ortho)
    return cam_pos, centre, up_ortho

def _parse_nodes(inp_path, part_name):
    nodes, inside_part, inside_nodes = [], False, False
    pn_up = part_name.strip().upper()
    with open(inp_path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("**"):
                continue
            up = line.upper()
            if up.startswith("*PART"):
                inside_nodes = False
                nv = None
                for tok in line.split(","):
                    tok = tok.strip()
                    if "=" in tok:
                        k, _, v = tok.partition("=")
                        if k.strip().upper() == "NAME":
                            nv = v.strip()
                inside_part = nv and nv.upper() == pn_up
                continue
            if up.startswith("*END PART"):
                inside_part = inside_nodes = False
                continue
            if not inside_part:
                continue
            if up.startswith("*NODE"):
                inside_nodes = True
                continue
            if line.startswith("*"):
                inside_nodes = False
                continue
            if inside_nodes:
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 3:
                    try:
                        nodes.append((float(p[1]), float(p[2]),
                                      float(p[3]) if len(p)>=4 else 0.0))
                    except (ValueError, IndexError):
                        pass
    return nodes

def _bbox_centre(nodes):
    xs=[n[0] for n in nodes]; ys=[n[1] for n in nodes]; zs=[n[2] for n in nodes]
    cx=(min(xs)+max(xs))*.5; cy=(min(ys)+max(ys))*.5; cz=(min(zs)+max(zs))*.5
    dx=max(xs)-min(xs); dy=max(ys)-min(ys); dz=max(zs)-min(zs)
    return (cx,cy,cz), .5*math.sqrt(dx*dx+dy*dy+dz*dz)

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    from abaqus import session
    from abaqusConstants import (
        CONTOURS_ON_DEF, ELEMENT_NODAL, INVARIANT, COMPONENT, PNG,
        MISES, MAX_PRINCIPAL, MAGNITUDE,
    )

    os.makedirs(OUTPUT_DIR) if not os.path.exists(OUTPUT_DIR) else None

    # ── Parse INP ────────────────────────────────────────────────────────
    print("Parsing INP for part '{}'…".format(TARGET_PART))
    nodes = _parse_nodes(INP_FILE, TARGET_PART)
    if not nodes:
        raise ValueError("Part '{}' not found in INP.".format(TARGET_PART))
    centre, radius = _bbox_centre(nodes)
    print("  Centre: ({:.4g}, {:.4g}, {:.4g})  radius: {:.4g}".format(*centre+(radius,)))

    # ── Open ODB once ────────────────────────────────────────────────────
    print("Opening ODB…")
    odb = session.openOdb(name=ODB_FILE)
    vp  = session.viewports["Viewport: 1"]
    vp.setValues(displayedObject=odb)
    vp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))

    step_keys = list(odb.steps.keys())
    res_step  = step_keys[STEP_INDEX]
    n_frames  = len(odb.steps[res_step].frames)
    res_frame = n_frames + FRAME_INDEX if FRAME_INDEX < 0 else FRAME_INDEX
    vp.odbDisplay.setFrame(step=step_keys.index(res_step), frame=res_frame)

    _INVARIANT_MAP = {
        "mises": MISES, "max. principal": MAX_PRINCIPAL, "magnitude": MAGNITUDE,
    }
    comp_lower = FIELD_COMPONENT.strip().lower()
    if comp_lower in _INVARIANT_MAP:
        vp.odbDisplay.setPrimaryVariable(
            variableLabel=FIELD_VARIABLE,
            outputPosition=ELEMENT_NODAL,
            refinement=(INVARIANT, FIELD_COMPONENT),
        )
    else:
        vp.odbDisplay.setPrimaryVariable(
            variableLabel=FIELD_VARIABLE,
            outputPosition=ELEMENT_NODAL,
            refinement=(COMPONENT, FIELD_COMPONENT),
        )

    # ── Iterate views ─────────────────────────────────────────────────────
    for view_label, view_vec in VIEWS.items():
        print("\nView: {}  direction={}".format(view_label, view_vec))
        cam_pos, cam_tgt, cam_up = _compute_camera(
            centre, radius, view_vec, STANDOFF_MULT, WORLD_UP
        )
        vp.view.setValues(
            cameraPosition  = cam_pos,
            cameraTarget    = cam_tgt,
            cameraUpVector  = cam_up,
        )

        img_path = os.path.join(OUTPUT_DIR, "{}_{}.png".format(
            os.path.splitext(os.path.basename(ODB_FILE))[0], view_label))

        session.pngOptions.setValues(imageSize=(IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
        session.printToFile(fileName=img_path, format=PNG, canvasObjects=(vp,))
        print("  Saved: {}".format(img_path))

        # Echo the setValues call for manual replication
        print("  session.views['Current Viewport'].setValues(")
        print("      cameraPosition  = ({:.5g}, {:.5g}, {:.5g}),".format(*cam_pos))
        print("      cameraTarget    = ({:.5g}, {:.5g}, {:.5g}),".format(*cam_tgt))
        print("      cameraUpVector  = ({:.5g}, {:.5g}, {:.5g}),".format(*cam_up))
        print("  )")

    print("\nAll views exported to: {}".format(OUTPUT_DIR))


if __name__ == "__main__" or True:
    main()
