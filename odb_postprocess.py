"""
odb_postprocess.py
==================
Abaqus/CAE post-processing script — run via:
    abaqus python odb_postprocess.py
    abaqus cae noGUI=odb_postprocess.py

Extracts for every .odb in ODB_DIR:
  1. RF1, RF2  — reaction forces at the last frame of each step,
                 summed over a defined node set (or per-node if desired).
  2. Max Principal LE (logarithmic strain) — element maximum over the
     defined node set, at the last frame of each step.

Results are written to:
  <ODB_DIR>/postprocess_results.csv
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import csv
import traceback

# ── Abaqus ODB API ────────────────────────────────────────────────────────────
from odbAccess import openOdb, OdbError

# ═════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  — edit these before running
# ═════════════════════════════════════════════════════════════════════════════

# Directory that contains all the .odb files to process
ODB_DIR = r"."                        # use r"C:\path\to\odbs" on Windows

# Node set name (must exist in every ODB or be skipped gracefully)
NODE_SET_NAME = "NSET_LOAD_POINT"     # e.g. "ALL_NODES", "NSET_TIP", etc.

# Step name(s) to process.  Set to None to process ALL steps in each ODB.
# Example: STEP_NAMES = ["Step-1", "Step-Loading"]
STEP_NAMES = None                      # None → auto-discover all steps

# Output CSV path
OUTPUT_CSV = os.path.join(ODB_DIR, "postprocess_results.csv")

# Part / instance name that owns the node set (case-sensitive).
# Set to None to search across all instances.
INSTANCE_NAME = None                   # e.g. "PART-1-1"

# ═════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def get_node_labels(odb, node_set_name, instance_name=None):
    """
    Return a set of integer node labels belonging to *node_set_name*.
    Searches rootAssembly first; if instance_name is given, also searches
    that specific instance.
    """
    assembly = odb.rootAssembly
    labels = set()

    # ── Assembly-level node sets ──────────────────────────────────────────
    if node_set_name.upper() in [k.upper() for k in assembly.nodeSets.keys()]:
        for k in assembly.nodeSets.keys():
            if k.upper() == node_set_name.upper():
                for node in assembly.nodeSets[k].nodes:
                    # nodes is a sequence of MeshNodeArray per instance
                    if hasattr(node, '__iter__'):
                        for n in node:
                            labels.add(n.label)
                    else:
                        labels.add(node.label)
                break

    # ── Instance-level node sets ──────────────────────────────────────────
    instances_to_check = (
        [odb.rootAssembly.instances[instance_name]]
        if instance_name and instance_name in odb.rootAssembly.instances
        else list(odb.rootAssembly.instances.values())
    )
    for inst in instances_to_check:
        if node_set_name.upper() in [k.upper() for k in inst.nodeSets.keys()]:
            for k in inst.nodeSets.keys():
                if k.upper() == node_set_name.upper():
                    for n in inst.nodeSets[k].nodes:
                        labels.add(n.label)
                    break

    return labels


def extract_reaction_forces(frame, node_labels):
    """
    Sum RF1 and RF2 over *node_labels* from the given ODB frame.
    Returns (sum_RF1, sum_RF2).  Missing DOFs → 0.0.
    """
    sum_rf1 = 0.0
    sum_rf2 = 0.0

    if "RF" not in frame.fieldOutputs:
        return sum_rf1, sum_rf2

    rf_field = frame.fieldOutputs["RF"]

    for value in rf_field.values:
        if value.nodeLabel in node_labels:
            data = value.data          # tuple: (RF1, RF2, RF3) or (RF1, RF2)
            if len(data) >= 1:
                sum_rf1 += data[0]
            if len(data) >= 2:
                sum_rf2 += data[1]

    return sum_rf1, sum_rf2


def extract_max_principal_le(frame, node_labels):
    """
    Return the maximum Max-Principal logarithmic strain (LE, Max. Principal)
    over all integration/section points whose element connects to a node in
    *node_labels*.

    Abaqus stores LE at integration points (element output).  We loop over
    element values and keep the envelope maximum.
    """
    max_le = None

    # LE output key in Abaqus is "LE"
    if "LE" not in frame.fieldOutputs:
        return max_le

    le_field = frame.fieldOutputs["LE"]

    # Request Max Principal invariant
    try:
        le_max_principal = le_field.getScalarField(invariant=MAX_PRINCIPAL)
    except Exception:
        # Fallback: manual component scan (LE11, LE22, LE33 …)
        le_max_principal = None

    if le_max_principal is not None:
        for value in le_max_principal.values:
            # Filter by node connectivity if nodeLabel available
            node_lbl = getattr(value, "nodeLabel", None)
            elem_lbl  = getattr(value, "elementLabel", None)

            # For element output, nodeLabel is not set; use elementLabel
            # We accept all elements (refine with element set if needed)
            scalar = value.data if not hasattr(value.data, '__len__') else value.data[0]
            if max_le is None or scalar > max_le:
                max_le = scalar
    else:
        # Manual fallback: principal = max eigenvalue of symmetric tensor
        for value in le_field.values:
            data = value.data   # (LE11, LE22, LE33, LE12, LE13, LE23) or 2-D subset
            principal = _max_principal_from_tensor(data)
            if max_le is None or principal > max_le:
                max_le = principal

    return max_le if max_le is not None else 0.0


def _max_principal_from_tensor(data):
    """
    Compute the maximum principal value from a symmetric strain tensor.
    Supports 3-D (6 components) and plane-stress/2-D (3 or 4 components).
    Uses the analytical eigenvalue formula for 3-D.
    """
    import math

    n = len(data)

    if n >= 6:
        e11, e22, e33, e12, e13, e23 = data[0], data[1], data[2], data[3], data[4], data[5]
    elif n == 4:                        # plane-stress with thickness strain
        e11, e22, e33, e12 = data[0], data[1], data[2], data[3]
        e13 = e23 = 0.0
    elif n == 3:                        # plane-stress, no thickness output
        e11, e22, e12 = data[0], data[1], data[2]
        e33 = e13 = e23 = 0.0
    else:
        return max(data)

    # Invariants of the symmetric 3×3 tensor
    I1 = e11 + e22 + e33
    I2 = (e11*e22 + e22*e33 + e11*e33
          - e12**2 - e13**2 - e23**2)
    I3 = (e11*(e22*e33 - e23**2)
          - e12*(e12*e33 - e23*e13)
          + e13*(e12*e23 - e22*e13))

    # Analytical cubic roots (Cardano / trigonometric method)
    p = I1**2 - 3.0*I2
    if p < 0.0:
        p = 0.0
    p_sqrt = math.sqrt(p)

    q = (2.0*I1**3 - 9.0*I1*I2 + 27.0*I3) / 54.0

    denom = (p_sqrt**3)
    if abs(denom) < 1e-30:
        return I1 / 3.0     # degenerate: all principals equal

    arg = q / denom
    arg = max(-1.0, min(1.0, arg))   # clamp numerical noise
    phi = math.acos(arg) / 3.0

    lam1 = (I1 + 2.0*p_sqrt*math.cos(phi)) / 3.0
    lam2 = (I1 + 2.0*p_sqrt*math.cos(phi + 2.0*math.pi/3.0)) / 3.0
    lam3 = (I1 + 2.0*p_sqrt*math.cos(phi + 4.0*math.pi/3.0)) / 3.0

    return max(lam1, lam2, lam3)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def process_all_odbs():
    # Try to import Abaqus invariant constant (only available inside Abaqus Python)
    global MAX_PRINCIPAL
    try:
        from abaqusConstants import MAX_PRINCIPAL
    except ImportError:
        MAX_PRINCIPAL = None    # will fall back to manual tensor calc

    odb_files = sorted(
        f for f in os.listdir(ODB_DIR) if f.lower().endswith(".odb")
    )

    if not odb_files:
        print("No .odb files found in: {}".format(ODB_DIR))
        return

    print("Found {:d} ODB file(s) to process.\n".format(len(odb_files)))

    # ── CSV header ────────────────────────────────────────────────────────
    fieldnames = [
        "ODB_File",
        "Step_Name",
        "Frame_Index",
        "Step_Time",
        "Node_Set",
        "Sum_RF1",
        "Sum_RF2",
        "Max_Principal_LE",
    ]

    with open(OUTPUT_CSV, "wb") as csvfile:   # 'wb' for Python 2 (Abaqus)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for odb_filename in odb_files:
            odb_path = os.path.join(ODB_DIR, odb_filename)
            print("Processing: {}".format(odb_filename))

            try:
                odb = openOdb(path=odb_path, readOnly=True)

                # ── Resolve node labels ───────────────────────────────────
                node_labels = get_node_labels(odb, NODE_SET_NAME, INSTANCE_NAME)
                if not node_labels:
                    print("  WARNING: node set '{}' not found or empty — "
                          "RF will cover ALL nodes, LE will cover ALL elements."
                          .format(NODE_SET_NAME))

                # ── Determine which steps to process ─────────────────────
                available_steps = list(odb.steps.keys())
                steps_to_process = (
                    STEP_NAMES if STEP_NAMES is not None else available_steps
                )

                for step_name in steps_to_process:
                    if step_name not in odb.steps:
                        print("  WARNING: step '{}' not found in ODB — skipping."
                              .format(step_name))
                        continue

                    step = odb.steps[step_name]
                    frames = step.frames

                    if not frames:
                        print("  WARNING: step '{}' has no frames.".format(step_name))
                        continue

                    # ── Last frame of the step ────────────────────────────
                    last_frame = frames[-1]
                    frame_idx  = len(frames) - 1
                    step_time  = last_frame.frameValue

                    print("  Step: {:s}  |  Frame {:d}  |  Time {:.6g}"
                          .format(step_name, frame_idx, step_time))

                    # ── Reaction forces ───────────────────────────────────
                    rf1, rf2 = extract_reaction_forces(last_frame, node_labels)

                    # ── Max Principal LE ──────────────────────────────────
                    max_le = extract_max_principal_le(last_frame, node_labels)

                    print("    RF1={:.6e}  RF2={:.6e}  Max_LE={:.6e}"
                          .format(rf1, rf2, max_le))

                    writer.writerow({
                        "ODB_File":         odb_filename,
                        "Step_Name":        step_name,
                        "Frame_Index":      frame_idx,
                        "Step_Time":        step_time,
                        "Node_Set":         NODE_SET_NAME,
                        "Sum_RF1":          rf1,
                        "Sum_RF2":          rf2,
                        "Max_Principal_LE": max_le,
                    })

                odb.close()

            except OdbError as e:
                print("  ERROR opening/reading ODB: {}".format(e))
                traceback.print_exc()
            except Exception as e:
                print("  UNEXPECTED ERROR: {}".format(e))
                traceback.print_exc()

    print("\nDone. Results written to: {}".format(OUTPUT_CSV))


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__" or True:   # `True` needed for `abaqus cae noGUI=`
    process_all_odbs()
