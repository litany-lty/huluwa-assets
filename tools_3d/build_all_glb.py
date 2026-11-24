#!/usr/bin/env python3
"""
Blender 5.3 Master 3D Asset Builder for Huluwa (All 48 Assets -> glTF 2.0 .glb).
Usage:
  blender --background --python tools_3d/build_all_glb.py [-- only_key1 only_key2 ...]

Outputs self-contained PBR .glb files to:
  ./Models_3D/
(Optional: set HULUWA3D_SYNC_DIR to mirror exported .glb files to an app repo.)
"""
import os
import sys
import json
import math
import shutil
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Euler

import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(REPO_ROOT, "Models_3D")
HULUWA3D_OUT_ROOT = os.environ.get("HULUWA3D_SYNC_DIR", "")
CHAR_CACHE = os.environ.get("HULUWA_3D_CACHE", os.path.join(tempfile.gettempdir(), "huluwa_3d_char_cache"))
TEX_CACHE = os.path.join(tempfile.gettempdir(), "huluwa_3d_proc_tex_cache")
os.makedirs(TEX_CACHE, exist_ok=True)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def save_rgba_png(name, rgba_float):
    h, w, _ = rgba_float.shape
    path = os.path.join(TEX_CACHE, f"{name}.png")
    bimg = bpy.data.images.new(name, width=w, height=h, alpha=True)
    # Note: bpy.data.images pixels are bottom-to-top! Flip vertically so image matches array top-to-bottom
    flipped = np.flipud(np.clip(rgba_float, 0.0, 1.0)).astype(np.float32)
    bimg.pixels.foreach_set(flipped.ravel())
    bimg.filepath_raw = path
    bimg.file_format = "PNG"
    bimg.save()
    bpy.data.images.remove(bimg)
    return path


def make_mat(
    name,
    base_color=(1.0, 1.0, 1.0, 1.0),
    roughness=0.52,
    metallic=0.0,
    albedo_path=None,
    normal_path=None,
    mr_path=None,
    normal_strength=0.60,
    repeat=False,
    emission_color=None,
    emission_strength=0.0,
    alpha=1.0,
    blend=False,
    double_sided=False,
    transmission=0.0,
):
    mat = bpy.data.materials.new(name=name)
    mat.use_backface_culling = not double_sided
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.38
    if alpha < 0.999 or blend:
        bsdf.inputs["Alpha"].default_value = alpha
        if hasattr(mat, " surface_render_method"):
            mat.surface_render_method = "BLENDED"
        elif hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
    if transmission > 0.01 and "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    ext_mode = "REPEAT" if repeat else "EXTEND"

    if albedo_path and os.path.exists(albedo_path):
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(albedo_path)
        tex.extension = ext_mode
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    if mr_path and os.path.exists(mr_path):
        mr_tex = nodes.new("ShaderNodeTexImage")
        mr_tex.image = bpy.data.images.load(mr_path)
        mr_tex.image.colorspace_settings.name = "Non-Color"
        mr_tex.extension = ext_mode
        sep = nodes.new("ShaderNodeSeparateColor")
        links.new(mr_tex.outputs["Color"], sep.inputs["Color"])
        links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    if normal_path and os.path.exists(normal_path):
        n_tex = nodes.new("ShaderNodeTexImage")
        n_tex.image = bpy.data.images.load(normal_path)
        n_tex.image.colorspace_settings.name = "Non-Color"
        n_tex.extension = ext_mode
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.inputs["Strength"].default_value = normal_strength
        links.new(n_tex.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    if emission_color and emission_strength > 0.0:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*emission_color, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission_strength

    return mat


def export_glb(rel_path):
    full_out = os.path.join(OUT_ROOT, rel_path)
    os.makedirs(os.path.dirname(full_out), exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=full_out,
        export_format="GLB",
        export_yup=True,
        export_apply=True,
    )
    if HULUWA3D_OUT_ROOT:
        mirror_out = os.path.join(HULUWA3D_OUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(mirror_out), exist_ok=True)
        shutil.copy2(full_out, mirror_out)
    print(f"[build_all_glb] Exported: {rel_path} ({os.path.getsize(full_out) // 1024} KB)")


def create_revolved_mesh(name, profile_rz, segs=40, smooth=True):
    """Revolves [(r, z), ...] around Blender Z axis (which exports as glTF +Y up!)."""
    bm = bmesh.new()
    rings = []
    for r, z in profile_rz:
        if r < 1e-4:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            ring = []
            for s in range(segs):
                a = (2.0 * math.pi * s) / segs
                ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
            rings.append(ring)

    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        if len(r0) == 1 and len(r1) > 1:
            for s in range(segs):
                bm.faces.new((r0[0], r1[(s + 1) % segs], r1[s]))
        elif len(r0) > 1 and len(r1) == 1:
            for s in range(segs):
                bm.faces.new((r0[s], r0[(s + 1) % segs], r1[0]))
        elif len(r0) > 1 and len(r1) > 1:
            for s in range(segs):
                sn = (s + 1) % segs
                bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    uv_lay = bm.loops.layers.uv.new("UVMap")
    z_vals = [p[1] for p in profile_rz]
    z_min, z_span = min(z_vals), max(1e-4, max(z_vals) - min(z_vals))
    for f in bm.faces:
        f.smooth = smooth
        for l in f.loops:
            co = l.vert.co
            ang = math.atan2(co.y, co.x) / (2.0 * math.pi) + 0.5
            l[uv_lay].uv = (ang, (co.z - z_min) / z_span)

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def make_gourd_mesh(name, height=0.65, radius_scale=1.0, segs=32):
    """Creates an authentic 1986 double-bulb Calabash gourd centered at X=0, Y=0, Z in [0..height]."""
    prof = [
        (0.000, 0.00),
        (0.18 * radius_scale, 0.02),
        (0.32 * radius_scale, 0.10),
        (0.36 * radius_scale, 0.22),
        (0.31 * radius_scale, 0.36),
        (0.16 * radius_scale, 0.47),  # Waist constriction
        (0.24 * radius_scale, 0.58),
        (0.25 * radius_scale, 0.70),  # Upper bulb
        (0.16 * radius_scale, 0.82),
        (0.055 * radius_scale, 0.90),  # Stem neck
        (0.045 * radius_scale, 0.98),
        (0.000, 1.00),
    ]
    scaled = [(r * height, z * height) for r, z in prof]
    return create_revolved_mesh(name, scaled, segs=segs, smooth=True)


def make_curve_tube(name, pts_xyzr, bevel_res=6, cyclic=False):
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 1.0
    curve.bevel_resolution = bevel_res
    sp = curve.splines.new("BEZIER")
    sp.use_cyclic_u = cyclic
    sp.bezier_points.add(len(pts_xyzr) - 1)
    for i, (x, y, z, r) in enumerate(pts_xyzr):
        bp = sp.bezier_points[i]
        bp.co = (x, y, z)
        bp.radius = r
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if not cyclic:
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = True
    bm.to_mesh(obj.data)
    bm.free()
    obj.select_set(False)
    return obj


# ==============================================================================
# 1. CHARACTERS & STAGE_04 (19 Assets from prep_characters.py cache)
# ==============================================================================
def build_character_from_npz(entry):
    clear_scene()
    data = np.load(entry["npz"], allow_pickle=True)
    verts = data["verts"]
    faces = data["faces"]
    G = float(data["G"])
    cx_px = float(data["cx_px"])
    feet_y_px = float(data["feet_y_px"])
    scale = float(data["scale"])
    albedo_path = str(data["albedo_path"])
    normal_path = str(data["normal_path"])
    mr_path = str(data["mr_path"])
    stealth = bool(entry.get("stealth", False))

    b_verts = [(float(v[0]), float(-v[2]), float(v[1])) for v in verts]
    b_faces = [tuple(int(i) for i in f) for f in faces]

    mesh = bpy.data.meshes.new(f"{entry['id']}_Mesh")
    mesh.from_pydata(b_verts, [], b_faces)
    mesh.update()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(entry["id"], mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    mod_smooth = obj.modifiers.new("Smooth", "SMOOTH")
    mod_smooth.factor = 0.52
    mod_smooth.iterations = 5
    bpy.ops.object.modifier_apply(modifier=mod_smooth.name)

    target_tris = 20000
    mod_dec = obj.modifiers.new("Decimate", "DECIMATE")
    mod_dec.ratio = min(1.0, target_tris / max(1, len(obj.data.polygons)))
    bpy.ops.object.modifier_apply(modifier=mod_dec.name)

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
    bm2.to_mesh(obj.data)
    bm2.free()

    for poly in obj.data.polygons:
        poly.use_smooth = True

    uv_layer = obj.data.uv_layers.new(name="UVMap")
    uv_data = uv_layer.data

    # Front faces within ~66° of forward (-Y in Blender) sample crisp front art;
    # steeper side & back faces sample 100% streak-free side/back pigment atlas!
    for poly in obj.data.polygons:
        is_front = poly.normal.y < -0.40
        u_offset = 0.0 if is_front else 0.5
        for loop_idx in poly.loop_indices:
            v_idx = obj.data.loops[loop_idx].vertex_index
            co = obj.data.vertices[v_idx].co
            ix = co.x / scale + cx_px
            iy = feet_y_px - co.z / scale
            u_norm = min(0.996, max(0.004, ix / G))
            v_norm = min(0.996, max(0.004, 1.0 - iy / G))
            uv_data[loop_idx].uv = (u_offset + u_norm * 0.5, v_norm)

    if stealth:
        mat = make_mat(
            f"{entry['id']}_StealthMat",
            albedo_path=albedo_path,
            normal_path=normal_path,
            mr_path=mr_path,
            alpha=0.52,
            blend=True,
            emission_color=(0.22, 0.72, 1.0),
            emission_strength=0.35,
        )
    else:
        mat = make_mat(
            f"{entry['id']}_Mat",
            albedo_path=albedo_path,
            normal_path=normal_path,
            mr_path=mr_path,
            normal_strength=0.58,
        )
    obj.data.materials.append(mat)
    export_glb(entry["out_rel_glb"])


# ==============================================================================
# 2. PROPS & TREASURES (6 Assets)
# ==============================================================================
def make_3d_leaf(name, length=0.55, width=0.45, palmate=False, mat=None):
    """Builds a curved 3D leaf blade with central rib crease & organic camber."""
    bm = bmesh.new()
    nx, ny = 9, 10
    grid = []
    for j in range(ny):
        v = j / (ny - 1)  # 0 at stem, 1 at tip
        row = []
        # Leaf width envelope
        if palmate:
            lobe = 1.0 + 0.28 * math.sin(v * math.pi * 3.0)
            w_env = math.sin(v * math.pi) ** 0.72 * lobe
        else:
            w_env = math.sin(v * math.pi) ** 0.78 * (1.0 - 0.18 * v)
        for i in range(nx):
            u = (i / (nx - 1)) * 2.0 - 1.0  # [-1..1]
            lx = u * (width * 0.5) * w_env
            ly = v * length
            # 3D arch along length + V-cup across width + subtle thickness
            lz = math.sin(v * math.pi) * 0.08 * length + abs(u) * 0.045 * width - (u * u) * 0.02 * width
            vert = bm.verts.new((lx, ly, lz))
            row.append(vert)
        grid.append(row)

    for j in range(ny - 1):
        for i in range(nx - 1):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))

    # Add real 3D leaf thickness via solidify extrude
    geom = bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.022 * length)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    uv_lay = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        f.smooth = True
        for l in f.loops:
            co = l.vert.co
            l[uv_lay].uv = (co.x / max(1e-4, width) + 0.5, co.y / max(1e-4, length))

    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    if mat:
        obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj


def build_prop_vine():
    """Prop_01_七色葫芦仙藤: 3D winding vines, curling tendrils, 11 3D palmate leaves, and 7 3D colored gourds!"""
    clear_scene()
    # Leaf texture with veins
    S = 256
    yy, xx = np.mgrid[0:S, 0:S]
    u = (xx / (S - 1)) * 2.0 - 1.0
    v = 1.0 - yy / (S - 1)
    dist_midrib = np.abs(u)
    diag_vein = np.abs(np.sin((v * 5.5 - np.abs(u) * 2.6) * math.pi))
    vein = np.exp(-((dist_midrib * 28.0) ** 2)) + 0.55 * np.exp(-((diag_vein * 7.0) ** 2)) * (dist_midrib > 0.03)
    vein = np.clip(vein, 0.0, 1.0)
    base_g = np.array([0.26, 0.56, 0.24])
    vein_g = np.array([0.46, 0.74, 0.34])
    rgba = np.ones((S, S, 4), dtype=np.float32)
    rgba[:, :, :3] = base_g[None, None, :] * (1.0 - 0.55 * vein[:, :, None]) + vein_g[None, None, :] * (0.55 * vein[:, :, None])
    leaf_tex = save_rgba_png("vine_leaf_tex", rgba)

    vine_mat = make_mat("VineStem", base_color=(0.30, 0.60, 0.25, 1.0), roughness=0.58)
    leaf_mat = make_mat("VineLeaf", albedo_path=leaf_tex, roughness=0.52, double_sided=True)

    # Winding 3D Vine Branches matching Prop_01 layout (X in [-2.2..2.2], Z height in [0.2..3.6], Y depth in [-0.35..0.35])
    branches = [
        [(-1.7, 0.10, 1.2, 0.055), (-1.3, -0.12, 2.3, 0.065), (-0.4, 0.14, 2.75, 0.070), (0.6, -0.10, 2.80, 0.065), (1.6, 0.12, 2.65, 0.050), (2.05, -0.05, 2.25, 0.030)],
        [(-1.95, -0.08, 1.45, 0.035), (-1.45, 0.18, 1.95, 0.052), (-0.85, -0.15, 1.45, 0.055), (-0.15, 0.12, 1.75, 0.060), (0.45, -0.14, 2.65, 0.052)],
        [(0.15, 0.10, 1.85, 0.052), (0.95, 0.18, 1.55, 0.052), (1.55, -0.12, 0.95, 0.048), (1.88, 0.10, 0.38, 0.025)],
        [(-1.35, 0.08, 1.35, 0.045), (-0.95, -0.16, 0.85, 0.040), (-1.65, 0.12, 0.52, 0.025)],
    ]
    for bi, bpts in enumerate(branches):
        v_obj = make_curve_tube(f"VineBranch_{bi}", bpts, bevel_res=6)
        v_obj.data.materials.append(vine_mat)

    # Helical 3D Curling Tendrils!
    for ti, (tx, ty, tz, sign) in enumerate([(-0.85, 0.15, 3.15, 1), (-0.02, -0.12, 2.35, -1), (1.45, 0.14, 0.65, 1), (-1.75, -0.10, 1.75, -1)]):
        t_pts = []
        for s in range(14):
            u_t = s / 13.0
            ang = u_t * math.pi * 3.5 * sign
            rad = 0.18 * (1.0 - 0.65 * u_t)
            t_pts.append((tx + rad * math.cos(ang), ty + rad * math.sin(ang) * 0.6, tz - u_t * 0.36, 0.018 * (1.0 - 0.5 * u_t)))
        tobj = make_curve_tube(f"Tendril_{ti}", t_pts, bevel_res=4)
        tobj.data.materials.append(vine_mat)

    # 7 Iconic 3D Calabashes in exact 1:1 positions & colors of Prop_01_七色葫芦仙藤.png!
    gourds_7 = [
        ("Gourd_1_Red",    (-1.26, -0.14, 1.15), 0.78, (0.86, 0.16, 0.15, 1.0)),
        ("Gourd_2_Orange", (-0.48, -0.12, 2.02), 0.76, (0.95, 0.52, 0.14, 1.0)),
        ("Gourd_3_Yellow", ( 0.68, -0.10, 1.98), 0.76, (0.98, 0.84, 0.20, 1.0)),
        ("Gourd_4_Green",  ( 1.36, -0.12, 2.02), 0.75, (0.22, 0.80, 0.38, 1.0)),
        ("Gourd_5_Cyan",   ( 0.05, -0.15, 0.86), 0.76, (0.16, 0.74, 0.90, 1.0)),
        ("Gourd_6_Blue",   ( 0.58, -0.12, 0.30), 0.78, (0.22, 0.34, 0.86, 1.0)),
        ("Gourd_7_Purple", (-0.98, -0.10, 0.42), 0.75, (0.62, 0.22, 0.84, 1.0)),
    ]
    for gname, (gx, gy, gz), gh, gcol in gourds_7:
        gobj = make_gourd_mesh(gname, height=gh, radius_scale=0.96, segs=32)
        gobj.location = (gx, gy, gz)
        gmat = make_mat(f"{gname}_Mat", base_color=gcol, roughness=0.26, metallic=0.04)
        gobj.data.materials.append(gmat)
        # Stem connecting gourd top to vine
        stem = make_curve_tube(f"{gname}_Stem", [(gx, gy, gz + gh * 0.96, 0.024), (gx, gy * 0.5, gz + gh + 0.16, 0.020)], bevel_res=4)
        stem.data.materials.append(vine_mat)

    # 10 Lush 3D Palmate Leaves distributed around the vine matching Prop_01
    leaf_placements = [
        ((-1.55, -0.05, 3.05), (0.35, -0.2, 0.6), 0.78),
        ((-1.42, -0.18, 2.35), (0.45, 0.15, 1.2), 0.85),
        ((-0.08, 0.08, 3.08), (0.30, 0.1, -0.2), 0.82),
        ((0.95, 0.10, 2.78), (0.40, -0.15, -0.7), 0.74),
        ((1.35, 0.05, 3.18), (0.32, 0.2, -0.4), 0.68),
        ((0.25, -0.18, 1.68), (0.50, -0.1, 2.4), 0.72),
        ((1.05, -0.20, 1.38), (0.45, 0.2, -1.8), 0.86),
        ((-0.52, 0.12, 0.95), (0.42, -0.15, 2.7), 0.68),
        ((-1.52, -0.10, 0.68), (0.48, 0.15, 1.9), 0.58),
        ((1.72, 0.10, 0.82), (0.40, -0.2, -1.2), 0.52),
    ]
    for li, (loc, rot, sz) in enumerate(leaf_placements):
        lf = make_3d_leaf(f"Leaf_{li}", length=sz, width=sz * 0.92, palmate=True, mat=leaf_mat)
        lf.location = loc
        lf.rotation_euler = rot

    export_glb("Props/Prop_01_七色葫芦仙藤.glb")


def build_prop_cauldron():
    """Prop_02_三足青铜炼丹炉: 360° Bronze Tripod Cauldron + Patina PBR + Swirling 3D Flames."""
    clear_scene()
    S = 512
    rng = np.random.default_rng(198602)
    yy, xx = np.mgrid[0:S, 0:S]
    patina = np.zeros((S, S), dtype=np.float32)
    for freq, amp in [(2, 0.45), (5, 0.30), (11, 0.16), (23, 0.09)]:
        phx, phy = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        patina += amp * np.sin(2 * np.pi * freq * xx / S + phx) * np.cos(2 * np.pi * freq * yy / S + phy)
    patina = (patina - patina.min()) / (patina.max() - patina.min() + 1e-6)

    teal = np.array([82, 138, 120], dtype=np.float32) / 255.0
    dark_teal = np.array([52, 88, 78], dtype=np.float32) / 255.0
    rust = np.array([148, 120, 72], dtype=np.float32) / 255.0
    col = teal[None, None, :] * (0.4 + 0.6 * patina[:, :, None]) + dark_teal[None, None, :] * (0.6 - 0.6 * patina[:, :, None])
    rust_mask = np.clip((patina - 0.58) * 3.2, 0.0, 1.0)[:, :, None]
    col = col * (1.0 - rust_mask * 0.78) + rust[None, None, :] * (rust_mask * 0.78)
    rgba = np.ones((S, S, 4), dtype=np.float32)
    rgba[:, :, :3] = col
    patina_path = save_rgba_png("cauldron_patina_tile", rgba)

    bronze_mat = make_mat("BronzePatina", albedo_path=patina_path, roughness=0.46, metallic=0.68, repeat=True)
    rim_mat = make_mat("BronzeRimGold", base_color=(0.68, 0.65, 0.44, 1.0), roughness=0.35, metallic=0.78)
    ash_mat = make_mat("HearthGlow", base_color=(0.30, 0.14, 0.05, 1.0), roughness=0.8, emission_color=(1.0, 0.45, 0.08), emission_strength=1.5)
    flame_outer_mat = make_mat("FlameOuter", base_color=(0.96, 0.48, 0.08, 1.0), roughness=0.3, emission_color=(0.98, 0.48, 0.06), emission_strength=1.4)
    flame_inner_mat = make_mat("FlameInner", base_color=(1.0, 0.80, 0.24, 1.0), roughness=0.24, emission_color=(1.0, 0.82, 0.22), emission_strength=2.2)

    profile = [
        (0.00, 0.38), (0.32, 0.39), (0.58, 0.46), (0.76, 0.60), (0.83, 0.78),
        (0.80, 0.96), (0.815, 0.99), (0.77, 1.03), (0.79, 1.06), (0.73, 1.15),
        (0.78, 1.24), (0.86, 1.27), (0.86, 1.35), (0.72, 1.35), (0.68, 1.24),
        (0.60, 0.95), (0.00, 0.92),
    ]
    body_obj = create_revolved_mesh("CauldronBody", profile, segs=48, smooth=True)
    body_obj.data.materials.append(bronze_mat)
    body_obj.data.materials.append(rim_mat)
    for p in body_obj.data.polygons:
        if p.center.z > 1.268 and math.hypot(p.center.x, p.center.y) > 0.71:
            p.material_index = 1

    for idx, ang in enumerate([-math.pi / 2.0, math.pi / 6.0, 5.0 * math.pi / 6.0]):
        pts = [
            (0.46 * math.cos(ang), 0.46 * math.sin(ang), 0.50, 0.165),
            (0.54 * math.cos(ang), 0.54 * math.sin(ang), 0.30, 0.135),
            (0.49 * math.cos(ang), 0.49 * math.sin(ang), 0.12, 0.110),
            (0.55 * math.cos(ang), 0.55 * math.sin(ang), 0.035, 0.145),
        ]
        leg = make_curve_tube(f"Leg_{idx}", pts, bevel_res=7)
        leg.data.materials.append(bronze_mat)

    for side in (-1, 1):
        h_pts = [
            (side * 0.74, 0.0, 0.98, 0.068),
            (side * 1.02, 0.0, 1.04, 0.068),
            (side * 1.08, 0.0, 1.48, 0.068),
            (side * 0.92, 0.0, 1.52, 0.064),
            (side * 0.85, 0.0, 1.30, 0.064),
        ]
        h_obj = make_curve_tube(f"Handle_{side}", h_pts, bevel_res=5)
        h_obj.scale.y = 1.65
        bpy.context.view_layer.objects.active = h_obj
        h_obj.select_set(True)
        bpy.ops.object.transform_apply(scale=True)
        h_obj.select_set(False)
        h_obj.data.materials.append(bronze_mat)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.64, location=(0, 0, 1.18))
    ember = bpy.context.active_object
    ember.scale = (1.0, 1.0, 0.25)
    bpy.ops.object.transform_apply(scale=True)
    ember.data.materials.append(ash_mat)

    flame_specs = [
        (0.00, 0.00, 0.96, 0.26, -0.05, 0.02, False),
        (-0.26, -0.06, 0.72, 0.19, -0.08, -0.03, False),
        (0.27, -0.05, 0.68, 0.18, 0.09, -0.02, False),
        (-0.44, 0.04, 0.46, 0.13, -0.06, 0.04, False),
        (0.45, 0.04, 0.48, 0.13, 0.07, 0.03, False),
        (0.08, 0.22, 0.62, 0.17, 0.03, 0.08, False),
        (-0.12, -0.18, 0.56, 0.16, -0.04, -0.05, True),
        (0.14, -0.16, 0.58, 0.16, 0.05, -0.04, True),
        (0.00, -0.10, 0.70, 0.18, -0.02, -0.04, True),
        (0.18, 0.02, 0.26, 0.065, 0.03, 0.0, False),
    ]
    for fi, (fx, fy, fh, fr, sx, sy, inner) in enumerate(flame_specs):
        z_s = 1.56 if fi == 9 else 1.22
        pts = [
            (fx, fy, z_s, fr * (0.5 if fi == 9 else 0.95)),
            (fx + sx * 0.5, fy + sy * 0.5, z_s + fh * 0.35, fr),
            (fx - sx * 0.4, fy + sy * 0.3, z_s + fh * 0.72, fr * 0.48),
            (fx + sx * 0.6, fy, z_s + fh, 0.008),
        ]
        f_obj = make_curve_tube(f"Flame_{fi}", pts, bevel_res=6)
        f_obj.data.materials.append(flame_inner_mat if inner else flame_outer_mat)

    export_glb("Props/Prop_02_三足青铜炼丹炉.glb")


def build_prop_lotus():
    """Prop_03_幽谷七色彩莲: 360° Notched Lily Pad + Porous Seed Pod + 3 Concentric Rainbow Petal Rings!"""
    clear_scene()
    pad_mat = make_mat("LotusPad", base_color=(0.24, 0.56, 0.27, 1.0), roughness=0.55)
    pod_mat = make_mat("SeedPod", base_color=(0.56, 0.80, 0.36, 1.0), roughness=0.48)
    seed_mat = make_mat("SeedHole", base_color=(0.22, 0.45, 0.18, 1.0), roughness=0.65)

    # 1. Notched 3D Lily Pad at Z=0.05
    bm_pad = bmesh.new()
    segs = 44
    notch_ang = -0.32 * math.pi  # Front-right notch matching Prop_03!
    center_v = bm_pad.verts.new((0.0, 0.0, 0.06))
    rim_verts = []
    for s in range(segs):
        a = -math.pi + (2.0 * math.pi * s) / segs
        da = abs((a - notch_ang + math.pi) % (2 * math.pi) - math.pi)
        r = 1.15 * (0.38 if da < 0.09 else (0.72 if da < 0.16 else 1.0))
        rim_verts.append(bm_pad.verts.new((r * math.cos(a), r * math.sin(a) * 0.88, 0.03 + 0.04 * (r / 1.15) ** 2)))
    for s in range(segs):
        bm_pad.faces.new((center_v, rim_verts[s], rim_verts[(s + 1) % segs]))
    bmesh.ops.solidify(bm_pad, geom=bm_pad.faces[:], thickness=0.035)
    bmesh.ops.recalc_face_normals(bm_pad, faces=bm_pad.faces)
    for f in bm_pad.faces:
        f.smooth = True
    pad_mesh = bpy.data.meshes.new("LilyPad")
    bm_pad.to_mesh(pad_mesh)
    bm_pad.free()
    pad_obj = bpy.data.objects.new("LilyPad", pad_mesh)
    pad_obj.data.materials.append(pad_mat)
    bpy.context.collection.objects.link(pad_obj)

    # 2. Central Conical Green Seed Pod (莲蓬) + 9 Seed Dimples
    pod = create_revolved_mesh("SeedPod", [(0.0, 0.08), (0.22, 0.12), (0.38, 0.42), (0.36, 0.46), (0.0, 0.46)], segs=32)
    pod.data.materials.append(pod_mat)
    seed_pts = [(0, 0)] + [(0.20 * math.cos(i * math.pi / 4), 0.20 * math.sin(i * math.pi / 4)) for i in range(8)]
    for si, (sx, sy) in enumerate(seed_pts):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.042, location=(sx, sy, 0.46))
        sd = bpy.context.active_object
        sd.scale = (1.0, 1.0, 0.45)
        bpy.ops.object.transform_apply(scale=True)
        sd.data.materials.append(seed_mat)

    # 3. Cupped 3D Rainbow Petals with white inner vein markings!
    def make_petal_mat(idx, rgb_col):
        S = 128
        yy, xx = np.mgrid[0:S, 0:S]
        u = (xx / (S - 1)) * 2.0 - 1.0
        v = 1.0 - yy / (S - 1)
        # Double concentric white/cream inner petal arches matching Prop_03!
        r1 = np.sqrt((u * 1.45) ** 2 + ((v - 0.35) * 1.15) ** 2)
        r2 = np.sqrt((u * 2.2) ** 2 + ((v - 0.28) * 1.45) ** 2)
        stripe = np.exp(-(((r1 - 0.52) * 18.0) ** 2)) + np.exp(-(((r2 - 0.46) * 18.0) ** 2))
        stripe = np.clip(stripe, 0.0, 1.0)
        c_base = np.array(rgb_col, dtype=np.float32)
        c_light = np.clip(c_base * 0.45 + 0.55, 0.0, 1.0)
        grad = c_base * (0.65 + 0.35 * v[:, :, None])
        final = grad * (1.0 - 0.72 * stripe[:, :, None]) + c_light[None, None, :] * (0.72 * stripe[:, :, None])
        rgba = np.ones((S, S, 4), dtype=np.float32)
        rgba[:, :, :3] = final
        p_path = save_rgba_png(f"lotus_petal_{idx}", rgba)
        return make_mat(f"PetalMat_{idx}", albedo_path=p_path, roughness=0.38, emission_color=rgb_col, emission_strength=0.15)

    rainbow_10 = [
        (0.96, 0.46, 0.64),  # Front rose-pink
        (0.95, 0.36, 0.32),  # Coral-red
        (0.96, 0.56, 0.22),  # Orange
        (0.98, 0.84, 0.26),  # Golden-yellow
        (0.42, 0.82, 0.38),  # Emerald-green (back-top)
        (0.24, 0.78, 0.88),  # Sky-cyan
        (0.36, 0.52, 0.92),  # Azure-blue
        (0.64, 0.38, 0.88),  # Amethyst-violet
        (0.82, 0.34, 0.76),  # Magenta-purple
        (0.56, 0.76, 0.42),  # Soft sage-pink lower sepal
    ]
    p_mats = [make_petal_mat(i, c) for i, c in enumerate(rainbow_10)]

    def add_cupped_petal(name, length, width, tilt_rad, yaw_rad, z_base, mat):
        bm = bmesh.new()
        nx, ny = 9, 11
        grid = []
        for j in range(ny):
            v = j / (ny - 1)
            w_env = math.sin(v * math.pi * 0.92) ** 0.72 * (1.0 - 0.25 * v ** 2)
            row = []
            for i in range(nx):
                u = (i / (nx - 1)) * 2.0 - 1.0
                px = u * (width * 0.5) * w_env
                py = v * length
                # Deep lotus petal spoon-cup
                pz = (u * u) * 0.14 * width + math.sin(v * math.pi) * 0.08 * length
                row.append(bm.verts.new((px, py, pz)))
            grid.append(row)
        for j in range(ny - 1):
            for i in range(nx - 1):
                bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
        bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.024)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        uv_l = bm.loops.layers.uv.new("UVMap")
        for f in bm.faces:
            f.smooth = True
            for l in f.loops:
                l[uv_l].uv = (l.vert.co.x / max(1e-4, width) + 0.5, l.vert.co.y / max(1e-4, length))
        m = bpy.data.meshes.new(name)
        bm.to_mesh(m)
        bm.free()
        o = bpy.data.objects.new(name, m)
        o.data.materials.append(mat)
        # Rotate pitch up by tilt_rad, then yaw around Z! (-Y is front in Blender)
        o.location = (0.14 * math.sin(yaw_rad), -0.14 * math.cos(yaw_rad), z_base)
        o.rotation_euler = Euler((math.pi * 0.5 - tilt_rad, 0.0, yaw_rad + math.pi), "XYZ")
        bpy.context.collection.objects.link(o)

    # 3 concentric tiers of petals (24 total 3D petals!)
    for ring_i, (n_petals, length, width, tilt, z_b, phase) in enumerate([
        (8, 0.88, 0.46, 1.18, 0.12, 0.0),      # Outer spreading tier
        (8, 0.82, 0.44, 0.78, 0.16, 0.39),     # Middle upright tier
        (8, 0.74, 0.40, 0.44, 0.20, 0.0),      # Inner crown tier
    ]):
        for k in range(n_petals):
            yaw = phase + (2.0 * math.pi * k) / n_petals
            c_idx = int(round(((yaw % (2 * math.pi)) / (2 * math.pi)) * 8)) % 9
            add_petal = add_cupped_petal(f"Petal_R{ring_i}_{k}", length, width, tilt, yaw, z_b, p_mats[c_idx])

    export_glb("Props/Prop_03_幽谷七色彩莲.glb")


def build_prop_mirror():
    """Prop_04_水晶魔镜: 3D Faceted Emerald-Cyan Crystal Gem + Carved Stone Crescent Cradle with Meander Relief!"""
    clear_scene()
    stone_mat = make_mat("MirrorStone", base_color=(0.30, 0.36, 0.39, 1.0), roughness=0.72, metallic=0.12)
    stone_dark = make_mat("MirrorCarve", base_color=(0.18, 0.22, 0.25, 1.0), roughness=0.82, metallic=0.08)
    crystal_mat = make_mat(
        "MagicCrystal",
        base_color=(0.18, 0.92, 0.78, 1.0),
        roughness=0.08,
        metallic=0.10,
        alpha=0.88,
        blend=True,
        transmission=0.45,
        emission_color=(0.12, 0.95, 0.68),
        emission_strength=0.65,
    )
    core_mat = make_mat("CrystalInnerGlow", base_color=(0.55, 1.0, 0.82, 1.0), roughness=0.15, emission_color=(0.45, 1.0, 0.78), emission_strength=1.6)

    # 1. Stepped Arched Stone Pedestal Base (Z in [0..0.38])
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.16))
    base = bpy.context.active_object
    base.scale = (1.62, 0.72, 0.32)
    bpy.ops.object.transform_apply(scale=True)
    bev = base.modifiers.new("Bev", "BEVEL")
    bev.width = 0.045
    bev.segments = 2
    bpy.ops.object.modifier_apply(modifier=bev.name)
    base.data.materials.append(stone_mat)

    # Upper plinth
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.42))
    plinth = bpy.context.active_object
    plinth.scale = (1.18, 0.56, 0.26)
    bpy.ops.object.transform_apply(scale=True)
    bev2 = plinth.modifiers.new("Bev", "BEVEL")
    bev2.width = 0.03
    bpy.ops.object.modifier_apply(modifier=bev2.name)
    plinth.data.materials.append(stone_mat)

    # 2. Crescent Stone Cradle Arms embracing lower crystal (Z in [0.48..1.25])
    cradle_pts = []
    for i in range(15):
        a = -math.pi * 0.86 + (math.pi * 0.72) * (i / 14.0)  # Bottom U-arc in XZ plane!
        # Wait: a = -pi/2 is bottom! Let angle phi go from -0.78*pi to -0.22*pi?
        phi = math.pi + 0.22 * math.pi + (0.56 * math.pi) * (i / 14.0)
        # Simpler: theta from -1.28 rad (left) through 0 (bottom) to +1.28 rad (right)
        th = -1.25 + 2.50 * (i / 14.0)
        cx = 0.88 * math.sin(th)
        cz = 1.28 - 0.76 * math.cos(th)
        cradle_pts.append((cx, 0.0, cz, 0.145))
    cradle = make_curve_tube("StoneCradle", cradle_pts, bevel_res=4)
    cradle.scale.y = 1.55
    bpy.context.view_layer.objects.active = cradle
    cradle.select_set(True)
    bpy.ops.object.transform_apply(scale=True)
    cradle.select_set(False)
    cradle.data.materials.append(stone_mat)

    # Decorative carved stone meander reliefs on front & back of cradle/plinth
    for side_y in (-0.29, 0.29):
        for sx in (-0.32, 0.32):
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, side_y, 0.42))
            rel = bpy.context.active_object
            rel.scale = (0.36, 0.03, 0.14)
            bpy.ops.object.transform_apply(scale=True)
            rel.data.materials.append(stone_dark)

    # 3. Faceted Elongated Hexagonal Crystal Gem (Z in [0.68..2.65]) matching Prop_04 1:1!
    bm_c = bmesh.new()
    # Outer silhouette ring (in XZ plane, at Y=0) + Front bevel table (Y=-0.22) + Back bevel table (Y=+0.22)
    outer_xz = [
        (0.00, 2.65),   # Top apex
        (0.56, 2.12),   # Upper-right shoulder
        (0.68, 1.38),   # Lower-right hip
        (0.46, 0.88),   # Bottom-right
        (0.00, 0.68),   # Bottom point
        (-0.46, 0.88),  # Bottom-left
        (-0.68, 1.38),  # Lower-left hip
        (-0.56, 2.12),  # Upper-left shoulder
    ]
    inner_xz = [(x * 0.72, 1.66 + (z - 1.66) * 0.76) for (x, z) in outer_xz]

    v_mid = [bm_c.verts.new((x, 0.0, z)) for x, z in outer_xz]
    v_front = [bm_c.verts.new((x, -0.22, z)) for x, z in inner_xz]
    v_back = [bm_c.verts.new((x, 0.22, z)) for x, z in inner_xz]

    n_pts = len(outer_xz)
    for i in range(n_pts):
        inext = (i + 1) % n_pts
        bm_c.faces.new((v_mid[i], v_mid[inext], v_front[inext], v_front[i]))
        bm_c.faces.new((v_mid[inext], v_mid[i], v_back[i], v_back[inext]))
    bm_c.faces.new(v_front)
    bm_c.faces.new(list(reversed(v_back)))
    bmesh.ops.recalc_face_normals(bm_c, faces=bm_c.faces)
    for f in bm_c.faces:
        f.smooth = False  # Crisp jewel facets!
    c_mesh = bpy.data.meshes.new("CrystalGem")
    bm_c.to_mesh(c_mesh)
    bm_c.free()
    c_obj = bpy.data.objects.new("CrystalGem", c_mesh)
    c_obj.data.materials.append(crystal_mat)
    bpy.context.collection.objects.link(c_obj)

    # Inner glowing emerald core
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.34, location=(0, 0, 1.62))
    core = bpy.context.active_object
    core.scale = (0.95, 0.35, 1.65)
    bpy.ops.object.transform_apply(scale=True)
    core.data.materials.append(core_mat)

    export_glb("Props/Prop_04_水晶魔镜.glb")


def build_prop_sword():
    """Prop_05_刚柔阴阳剑: Dual-Hue Emerald/Violet Diamond-Section Blade + Lotus Guard + 3D Yin-Yang Medallion!"""
    clear_scene()
    # Dual-hue blade gradient texture (left/lower emerald-cyan -> right/upper violet-amethyst)
    S = 256
    yy, xx = np.mgrid[0:S, 0:S]
    u = xx / (S - 1)
    v = 1.0 - yy / (S - 1)
    mix_t = np.clip(0.5 + (u - 0.5) * 1.6 + (v - 0.5) * 0.5, 0.0, 1.0)
    c_jade = np.array([0.32, 0.88, 0.68], dtype=np.float32)
    c_viol = np.array([0.74, 0.44, 0.92], dtype=np.float32)
    col = c_jade[None, None, :] * (1.0 - mix_t[:, :, None]) + c_viol[None, None, :] * mix_t[:, :, None]
    # Central fuller dark line
    fuller = np.exp(-(((u - 0.5) * 24.0) ** 2)) * (v < 0.86)
    col = col * (1.0 - 0.45 * fuller[:, :, None])
    rgba = np.ones((S, S, 4), dtype=np.float32)
    rgba[:, :, :3] = col
    blade_tex = save_rgba_png("yinyang_blade_tex", rgba)

    # Yin-Yang Taiji medallion texture
    yy_y, yy_x = np.mgrid[0:128, 0:128]
    dx = (yy_x - 63.5) / 58.0
    dy = (63.5 - yy_y) / 58.0
    ang = np.arctan2(dy, dx)
    r_top = np.hypot(dx, dy - 0.5)
    r_bot = np.hypot(dx, dy + 0.5)
    white_half = (dx < 0)
    white_half = np.where(r_top < 0.5, True, white_half)
    white_half = np.where(r_bot < 0.5, False, white_half)
    white_half = np.where(r_top < 0.14, False, white_half)
    white_half = np.where(r_bot < 0.14, True, white_half)
    yy_rgba = np.ones((128, 128, 4), dtype=np.float32)
    yy_rgba[:, :, :3] = np.where(white_half[:, :, None], np.array([0.95, 0.98, 1.0]), np.array([0.24, 0.16, 0.36]))
    taiji_tex = save_rgba_png("taiji_tex", yy_rgba)

    blade_mat = make_mat("YinYangBlade", albedo_path=blade_tex, roughness=0.18, metallic=0.72, emission_color=(0.45, 0.70, 0.85), emission_strength=0.25)
    purple_guard_mat = make_mat("PurpleLotusGuard", base_color=(0.52, 0.30, 0.72, 1.0), roughness=0.35, metallic=0.45)
    jade_guard_mat = make_mat("JadeCloudGuard", base_color=(0.30, 0.76, 0.58, 1.0), roughness=0.32, metallic=0.35)
    hilt_mat = make_mat("CordHilt", base_color=(0.38, 0.22, 0.54, 1.0), roughness=0.62, metallic=0.12)
    taiji_mat = make_mat("TaijiMedallion", albedo_path=taiji_tex, roughness=0.22, metallic=0.25, emission_color=(0.65, 0.92, 0.95), emission_strength=0.3)

    root = bpy.data.objects.new("YinYangSwordRoot", None)
    bpy.context.collection.objects.link(root)

    # Build vertical along Z (pommel at 0.0, guard at 0.82, tip at 2.85), then tilt 42° matching Prop_05!
    # 1. Pommel & Hilt
    pommel = create_revolved_mesh("Pommel", [(0.0, 0.0), (0.14, 0.06), (0.16, 0.14), (0.09, 0.20), (0.0, 0.21)], segs=24)
    pommel.data.materials.append(jade_guard_mat)
    pommel.parent = root

    hilt = create_revolved_mesh("Hilt", [(0.085, 0.20), (0.095, 0.35), (0.105, 0.52), (0.095, 0.68), (0.12, 0.74)], segs=24)
    hilt.data.materials.append(hilt_mat)
    hilt.parent = root

    # 2. Sculpted Lotus & Cloud Guard + 3D Yin-Yang Medallion
    for side in (-1, 1):
        wing = make_curve_tube(
            f"LotusWing_{side}",
            [(0.0, 0.0, 0.76, 0.09), (side * 0.26, 0.0, 0.80, 0.085), (side * 0.38, 0.0, 0.98, 0.045), (side * 0.32, 0.0, 1.08, 0.015)],
            bevel_res=5,
        )
        wing.scale.y = 0.65
        wing.data.materials.append(purple_guard_mat)
        wing.parent = root

    # Jade cloud collar above guard
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.24, radius2=0.04, depth=0.38, location=(0, 0, 0.98))
    collar = bpy.context.active_object
    collar.scale = (1.0, 0.42, 1.0)
    collar.data.materials.append(jade_guard_mat)
    collar.parent = root

    # 3D Yin-Yang Medallion Disc facing -Y and +Y
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.145, depth=0.13, location=(0, 0, 0.84), rotation=(math.pi * 0.5, 0, 0))
    med = bpy.context.active_object
    # Assign flat planar UV on cap faces
    bm_m = bmesh.new()
    bm_m.from_mesh(med.data)
    uv_m = bm_m.loops.layers.uv.verify()
    for f in bm_m.faces:
        f.smooth = True
        for l in f.loops:
            l[uv_m].uv = (l.vert.co.x / 0.29 + 0.5, l.vert.co.z / 0.29 + 0.5)
    bm_m.to_mesh(med.data)
    bm_m.free()
    med.data.materials.append(taiji_mat)
    med.parent = root

    # 3. Double-Edged Diamond-Cross-Section Sword Blade (Z in [0.88..2.86])
    bm_b = bmesh.new()
    stations = [
        (0.88, 0.165, 0.052),
        (1.45, 0.150, 0.048),
        (2.15, 0.132, 0.042),
        (2.62, 0.105, 0.032),
        (2.86, 0.002, 0.004),
    ]
    rings = []
    for z, hw, ht in stations:
        rings.append([
            bm_b.verts.new((-hw, 0.0, z)),  # Left sharp edge
            bm_b.verts.new((0.0, -ht, z)),  # Front central ridge
            bm_b.verts.new((hw, 0.0, z)),   # Right sharp edge
            bm_b.verts.new((0.0, ht, z)),   # Back central ridge
        ])
    for i in range(len(rings) - 1):
        r0, r1 = rings[i], rings[i + 1]
        for s in range(4):
            sn = (s + 1) % 4
            bm_b.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
    bm_b.faces.new(tuple(reversed(rings[0])))
    bm_b.faces.new(tuple(rings[-1]))
    bmesh.ops.recalc_face_normals(bm_b, faces=bm_b.faces)
    uv_b = bm_b.loops.layers.uv.new("UVMap")
    for f in bm_b.faces:
        f.smooth = False  # Crisp sword bevel facets!
        for l in f.loops:
            co = l.vert.co
            l[uv_b].uv = (co.x / 0.34 + 0.5, (co.z - 0.88) / 1.98)
    b_mesh = bpy.data.meshes.new("SwordBlade")
    bm_b.to_mesh(b_mesh)
    bm_b.free()
    b_obj = bpy.data.objects.new("SwordBlade", b_mesh)
    b_obj.data.materials.append(blade_mat)
    b_obj.parent = root
    bpy.context.collection.objects.link(b_obj)

    # Diagonal 42° heroic display tilt matching Prop_05_刚柔阴阳剑.png while centered cleanly above Z=0!
    root.rotation_euler = (0.0, math.radians(42.0), 0.0)
    root.location = (-0.95, 0.0, 0.15)
    export_glb("Props/Prop_05_刚柔阴阳剑.glb")


def build_prop_peach():
    """Prop_06_回血蟠桃: Plump 360° Heart-Cleft Immortal Peach with Rosy-Cream Gradient + 4 Veined 3D Leaves!"""
    clear_scene()
    # Smooth rosy-top to warm-cream-bottom peach skin gradient texture
    S = 256
    yy, xx = np.mgrid[0:S, 0:S]
    v = 1.0 - yy / (S - 1)  # 0 at bottom, 1 at top apex
    u = xx / (S - 1)
    c_bot = np.array([0.98, 0.86, 0.50], dtype=np.float32)
    c_mid = np.array([0.99, 0.72, 0.62], dtype=np.float32)
    c_top = np.array([0.93, 0.28, 0.44], dtype=np.float32)
    col = np.where(
        v[:, :, None] < 0.52,
        c_bot * (1.0 - v[:, :, None] / 0.52) + c_mid * (v[:, :, None] / 0.52),
        c_mid * (1.0 - (v[:, :, None] - 0.52) / 0.48) + c_top * ((v[:, :, None] - 0.52) / 0.48),
    )
    rgba = np.ones((S, S, 4), dtype=np.float32)
    rgba[:, :, :3] = col
    peach_tex = save_rgba_png("peach_skin_tex", rgba)

    # Leaf texture
    lu = (xx / (S - 1)) * 2.0 - 1.0
    vein = np.exp(-((np.abs(lu) * 26.0) ** 2)) + 0.45 * np.exp(-((np.abs(np.sin((v * 5.0 - np.abs(lu) * 2.2) * math.pi)) * 6.5) ** 2))
    l_rgba = np.ones((S, S, 4), dtype=np.float32)
    l_rgba[:, :, :3] = np.array([0.34, 0.68, 0.22]) * (1.0 - 0.35 * np.clip(vein, 0, 1)[:, :, None]) + np.array([0.58, 0.84, 0.32]) * (0.35 * np.clip(vein, 0, 1)[:, :, None])
    leaf_tex = save_rgba_png("peach_leaf_tex", l_rgba)

    peach_mat = make_mat("PeachSkin", albedo_path=peach_tex, roughness=0.44)
    leaf_mat = make_mat("PeachLeaf", albedo_path=leaf_tex, roughness=0.48, double_sided=True)

    # Sculpt 360° Plump Heart-Cleft Peach (Z in [0.12..1.48])
    bm = bmesh.new()
    n_u, n_v = 48, 32
    rings = []
    for j in range(n_v):
        tv = j / (n_v - 1)  # 0 bottom, 1 pointed top tip
        # Heart/peach silhouette profile: plump wide lower-mid belly tapering to a curved pointed tip!
        if tv == 0.0 or tv == 1.0:
            tip_x = 0.03 * (tv ** 2)
            rings.append([bm.verts.new((tip_x, 0.0, 0.14 + tv * 1.32))])
            continue
        r_base = 0.68 * (math.sin(tv * math.pi) ** 0.68) * (1.12 - 0.42 * tv ** 1.4)
        z = 0.14 + tv * 1.32
        ring = []
        for i in range(n_u):
            ang = (2.0 * math.pi * i) / n_u
            # Characteristic peach cleft groove on the front-left (-Y, slight -X at ang = -0.62*pi)!
            da = abs((ang - (-0.58 * math.pi) + math.pi) % (2 * math.pi) - math.pi)
            cleft = 1.0 - 0.16 * math.exp(-((da / 0.28) ** 2)) * math.sin(tv * math.pi)
            r = r_base * cleft
            bend_x = 0.04 * (tv ** 2.2)  # Subtle authentic curved peach tip!
            ring.append(bm.verts.new((r * math.cos(ang) + bend_x, r * math.sin(ang), z)))
        rings.append(ring)

    for j in range(n_v - 1):
        r0, r1 = rings[j], rings[j + 1]
        if len(r0) == 1:
            for i in range(n_u):
                bm.faces.new((r0[0], r1[(i + 1) % n_u], r1[i]))
        elif len(r1) == 1:
            for i in range(n_u):
                bm.faces.new((r0[i], r0[(i + 1) % n_u], r1[0]))
        else:
            for i in range(n_u):
                inext = (i + 1) % n_u
                bm.faces.new((r0[i], r0[inext], r1[inext], r1[i]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    uv_lay = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        f.smooth = True
        for l in f.loops:
            co = l.vert.co
            ang = math.atan2(co.y, co.x) / (2.0 * math.pi) + 0.5
            l[uv_lay].uv = (ang, np.clip((co.z - 0.14) / 1.32, 0.01, 0.99))

    p_mesh = bpy.data.meshes.new("PeachBody")
    bm.to_mesh(p_mesh)
    bm.free()
    p_obj = bpy.data.objects.new("PeachBody", p_mesh)
    p_obj.data.materials.append(peach_mat)
    bpy.context.collection.objects.link(p_obj)

    # 4 Curved 3D Peach Leaves cradling the base matching Prop_06!
    for li, (yaw, tilt, length, width) in enumerate([
        (-1.42, 0.32, 0.88, 0.44),  # Left-upper leaf
        (-1.05, 0.12, 0.72, 0.38),  # Left-front leaf
        (1.32, 0.24, 0.92, 0.46),   # Right sweeping leaf
        (0.25, 0.18, 0.76, 0.40),   # Rear support leaf
    ]):
        lf = make_3d_leaf(f"PeachLeaf_{li}", length=length, width=width, palmate=False, mat=leaf_mat)
        lf.location = (0.12 * math.sin(yaw), -0.12 * math.cos(yaw), 0.14)
        lf.rotation_euler = Euler((tilt, 0.0, yaw - math.pi * 0.5), "XYZ")

    export_glb("Props/Prop_06_回血蟠桃.glb")


# ==============================================================================
# 3. STAGE OBSTACLES (Stage_01, 02, 03, 05, 05b; Stage_04 is built from npz!)
# ==============================================================================
def make_chiseled_spire(name, x, y, height, rx, ry, lean_x=0.0, lean_y=0.0, sides=6, seed=1, mat=None):
    rng = np.random.default_rng(seed)
    bm = bmesh.new()
    tiers = [0.0, 0.38, 0.72, 1.0]
    scales = [1.0, 0.84, 0.52, 0.06]
    rings = []
    for ti, (tz, sc) in enumerate(zip(tiers, scales)):
        ring = []
        for s in range(sides):
            a = (2.0 * math.pi * s) / sides + (0.15 if ti == 2 else 0.0)
            jitter = 1.0 + (rng.uniform(-0.14, 0.14) if ti < 3 else 0.0)
            vx = x + lean_x * tz + rx * sc * jitter * math.cos(a)
            vy = y + lean_y * tz + ry * sc * jitter * math.sin(a)
            vz = tz * height
            ring.append(bm.verts.new((vx, vy, vz)))
        rings.append(ring)
    for ti in range(len(rings) - 1):
        r0, r1 = rings[ti], rings[ti + 1]
        for s in range(sides):
            sn = (s + 1) % sides
            bm.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
    bm.faces.new(tuple(reversed(rings[0])))
    bm.faces.new(tuple(rings[-1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = False  # Crisp Song-mineral / cavern crystal facets!
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    if mat:
        obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj


def build_stage_rock():
    """Stage_01_青绿剪纸奇石 -> 360° Chiseled Song-Mineral Blue-Green Crag Cluster!"""
    clear_scene()
    mat_main = make_mat("MineralJade", base_color=(0.24, 0.58, 0.53, 1.0), roughness=0.70, metallic=0.06)
    mat_light = make_mat("MineralHighlight", base_color=(0.36, 0.74, 0.65, 1.0), roughness=0.64, metallic=0.05)
    mat_deep = make_mat("MineralShadow", base_color=(0.16, 0.42, 0.42, 1.0), roughness=0.76, metallic=0.06)

    spires = [
        ("Peak_Main",   0.02,  0.08, 2.75, 0.78, 0.68, -0.06, 0.0, 7, 101, mat_main),
        ("Peak_Front", -0.05, -0.36, 2.02, 0.65, 0.55,  0.05, -0.05, 6, 102, mat_light),
        ("Peak_Left",  -0.78, -0.05, 1.68, 0.56, 0.52, -0.14, 0.0, 6, 103, mat_main),
        ("Peak_Right",  0.82,  0.02, 1.48, 0.58, 0.50,  0.16, 0.0, 6, 104, mat_deep),
        ("Peak_FL",    -0.62, -0.45, 1.05, 0.46, 0.42, -0.10, -0.08, 5, 105, mat_light),
        ("Peak_FR",     0.38, -0.52, 0.92, 0.42, 0.38,  0.08, -0.08, 5, 106, mat_light),
        ("Peak_Back",   0.15,  0.52, 1.85, 0.62, 0.52,  0.04, 0.10, 6, 107, mat_deep),
    ]
    for s in spires:
        make_chiseled_spire(*s)
    export_glb("Stages/Stage_01_青绿剪纸奇石.glb")


def build_stage_pillar():
    """Stage_02_妖洞钟乳石柱 -> 360° Towering Faceted Dark-Indigo Cavern Stalagmite Cluster!"""
    clear_scene()
    mat_core = make_mat("CavernSlate", base_color=(0.20, 0.19, 0.34, 1.0), roughness=0.62, metallic=0.14)
    mat_ridge = make_mat("CavernViolet", base_color=(0.34, 0.32, 0.52, 1.0), roughness=0.54, metallic=0.18)
    mat_dark = make_mat("CavernAbyss", base_color=(0.13, 0.12, 0.23, 1.0), roughness=0.70, metallic=0.10)

    spires = [
        ("Stalag_Center",  0.00,  0.00, 4.80, 0.72, 0.68,  0.04, 0.0, 6, 201, mat_core),
        ("Stalag_UpperL", -0.42, -0.18, 3.25, 0.48, 0.46, -0.08, -0.04, 5, 202, mat_ridge),
        ("Stalag_UpperR",  0.38, -0.15, 3.45, 0.46, 0.44,  0.08, -0.04, 5, 203, mat_ridge),
        ("Stalag_SideR",   0.78,  0.06, 2.25, 0.42, 0.40,  0.18, 0.0, 5, 204, mat_dark),
        ("Stalag_LowL",   -0.72, -0.22, 1.65, 0.48, 0.44, -0.15, -0.06, 6, 205, mat_ridge),
        ("Stalag_FrontL", -0.22, -0.48, 1.52, 0.40, 0.38, -0.05, -0.08, 5, 206, mat_ridge),
        ("Stalag_FrontR",  0.46, -0.44, 1.42, 0.38, 0.36,  0.06, -0.08, 5, 207, mat_core),
        ("Stalag_Back",   -0.12,  0.48, 2.85, 0.56, 0.50,  0.0,  0.10, 6, 208, mat_dark),
    ]
    for s in spires:
        make_chiseled_spire(*s)
    export_glb("Stages/Stage_02_妖洞钟乳石柱.glb")


def build_stage_mire():
    """Stage_03_迷魂泥潭陷阱 -> 360° Ring of 14 Beveled Stone Blocks + Toxic Purple-Olive Bubbling Mud Pool!"""
    clear_scene()
    stone_mat = make_mat("MireStone", base_color=(0.24, 0.25, 0.33, 1.0), roughness=0.78, metallic=0.08)
    mud_mat = make_mat("ToxicMud", base_color=(0.40, 0.30, 0.38, 1.0), roughness=0.28, metallic=0.05)
    rim_slime_mat = make_mat("ToxicSlime", base_color=(0.56, 0.58, 0.30, 1.0), roughness=0.32, emission_color=(0.45, 0.52, 0.18), emission_strength=0.25)

    # 14 Chiseled Stone Blocks around elliptical/circular basin (R = 2.05)
    rng = np.random.default_rng(198603)
    n_blocks = 14
    for i in range(n_blocks):
        a = (2.0 * math.pi * i) / n_blocks
        r = 1.98 + rng.uniform(-0.05, 0.05)
        bx, by = r * math.cos(a), r * math.sin(a)
        bh = 0.42 + rng.uniform(-0.05, 0.07)
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx, by, bh * 0.5))
        blk = bpy.context.active_object
        blk.scale = (0.48, 0.88, bh)
        blk.rotation_euler = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), a)
        bpy.ops.object.transform_apply(scale=True)
        bev = blk.modifiers.new("Bev", "BEVEL")
        bev.width = 0.045
        bpy.ops.object.modifier_apply(modifier=bev.name)
        blk.data.materials.append(stone_mat)

    # Outer olive slime ring & inner purple mud surface
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=1.86, depth=0.18, location=(0, 0, 0.10))
    slime_ring = bpy.context.active_object
    slime_ring.data.materials.append(rim_slime_mat)

    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=1.72, depth=0.20, location=(0, 0, 0.11))
    mud_disc = bpy.context.active_object
    mud_disc.data.materials.append(mud_mat)

    # 3D Hemispherical Toxic Mud Bubbles + Concentric Ripple Tori matching Stage_03!
    bubbles = [
        (-0.62, 0.18, 0.24),
        (0.58, 0.28, 0.22),
        (0.08, 0.65, 0.16),
        (0.35, -0.45, 0.15),
        (-0.28, -0.62, 0.12),
        (-0.98, -0.22, 0.09),
    ]
    for bi, (px, py, br) in enumerate(bubbles):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=18, ring_count=10, radius=br, location=(px, py, 0.19))
        b = bpy.context.active_object
        b.scale = (1.0, 1.0, 0.68)
        bpy.ops.object.transform_apply(scale=True)
        b.data.materials.append(rim_slime_mat)
        if br >= 0.14:
            bpy.ops.mesh.primitive_torus_add(major_radius=br * 1.45, minor_radius=0.022, location=(px, py, 0.21))
            rip = bpy.context.active_object
            rip.scale.z = 0.45
            rip.data.materials.append(rim_slime_mat)

    export_glb("Stages/Stage_03_迷魂泥潭陷阱.glb")


def _build_dungeon_archway(stone_mat):
    """Shared heavy 3D polygonal beveled cavern archway for Stage_05 (closed) & Stage_05b (open)."""
    # Outer arch polygon & inner portal cutout extruded with solid 3D depth Y in [-0.48..+0.48]
    outer = [
        (-1.92, 0.0), (-1.98, 2.35), (-1.52, 4.02), (-0.72, 4.72), (0.0, 4.86),
        (0.72, 4.72), (1.52, 4.02), (1.98, 2.35), (1.92, 0.0),
    ]
    inner = [
        (-1.16, 0.0), (-1.20, 2.25), (-0.88, 3.48), (0.0, 3.96),
        (0.88, 3.48), (1.20, 2.25), (1.16, 0.0),
    ]
    bm = bmesh.new()
    # Map inner indices (7 pts) to outer indices (9 pts) cleanly by interpolating 9 pts along inner arch!
    inner_9 = [
        (-1.16, 0.0), (-1.20, 2.25), (-0.92, 3.45), (-0.46, 3.72), (0.0, 3.96),
        (0.46, 3.72), (0.92, 3.45), (1.20, 2.25), (1.16, 0.0),
    ]
    yf, yb = -0.46, 0.46
    vo_f = [bm.verts.new((x * 0.96, yf, z)) for x, z in outer]
    vo_b = [bm.verts.new((x, yb, z)) for x, z in outer]
    vi_f = [bm.verts.new((x, yf, z)) for x, z in inner_9]
    vi_b = [bm.verts.new((x, yb, z)) for x, z in inner_9]

    for i in range(len(outer) - 1):
        # Front beveled stone face, back stone face, outer jambs, inner portal jambs
        bm.faces.new((vo_f[i], vi_f[i], vi_f[i + 1], vo_f[i + 1]))
        bm.faces.new((vo_b[i + 1], vi_b[i + 1], vi_b[i], vo_b[i]))
        bm.faces.new((vo_f[i], vo_f[i + 1], vo_b[i + 1], vo_b[i]))
        bm.faces.new((vi_b[i], vi_b[i + 1], vi_f[i + 1], vi_f[i]))
    # Bottom bases of left and right pillars
    bm.faces.new((vo_f[0], vo_b[0], vi_b[0], vi_f[0]))
    bm.faces.new((vi_f[-1], vi_b[-1], vo_b[-1], vo_f[-1]))

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
        f.smooth = False
    mesh = bpy.data.meshes.new("DungeonArchway")
    bm.to_mesh(mesh)
    bm.free()
    arch = bpy.data.objects.new("DungeonArchway", mesh)
    arch.data.materials.append(stone_mat)
    bpy.context.collection.objects.link(arch)
    return arch


def build_stage_doors():
    """Stage_05_妖洞通关石门_闭 & Stage_05b_妖洞通关石门_开!"""
    # --- CLOSED DOOR (Stage_05) ---
    clear_scene()
    stone_mat = make_mat("ArchStone", base_color=(0.25, 0.27, 0.36, 1.0), roughness=0.74, metallic=0.10)
    wood_mat = make_mat("DoorWalnut", base_color=(0.35, 0.21, 0.14, 1.0), roughness=0.68, metallic=0.05)
    seam_mat = make_mat("DoorSeam", base_color=(0.06, 0.05, 0.05, 1.0), roughness=0.9)
    brass_mat = make_mat("DoorBrass", base_color=(0.88, 0.68, 0.20, 1.0), roughness=0.26, metallic=0.82)

    _build_dungeon_archway(stone_mat)

    # Recessed twin wooden leaves (Y in [-0.12..+0.12])
    for side in (-1, 1):
        bm_d = bmesh.new()
        pts = [
            (side * 0.025, 0.0),
            (side * 1.20, 0.0),
            (side * 1.22, 2.28),
            (side * 0.92, 3.50),
            (side * 0.025, 3.98),
        ]
        vf = [bm_d.verts.new((x, -0.14, z)) for x, z in pts]
        vb = [bm_d.verts.new((x, 0.14, z)) for x, z in pts]
        bm_d.faces.new(vf if side > 0 else list(reversed(vf)))
        bm_d.faces.new(list(reversed(vb)) if side > 0 else vb)
        for i in range(len(pts)):
            inext = (i + 1) % len(pts)
            bm_d.faces.new((vf[i], vf[inext], vb[inext], vb[i]))
        bmesh.ops.recalc_face_normals(bm_d, faces=bm_d.faces)
        dm = bpy.data.meshes.new(f"DoorLeaf_{side}")
        bm_d.to_mesh(dm)
        bm_d.free()
        dobj = bpy.data.objects.new(f"DoorLeaf_{side}", dm)
        dobj.data.materials.append(wood_mat)
        bpy.context.collection.objects.link(dobj)

        # 8 Golden Boss Studs per door leaf (2x2 upper + 2x2 lower) on BOTH front and back!
        for y_face in (-0.15, 0.15):
            for sx in (0.38, 0.78):
                for sz in (0.55, 1.08, 2.58, 3.08):
                    bpy.ops.mesh.primitive_uv_sphere_add(segments=14, ring_count=8, radius=0.068, location=(side * sx, y_face, sz))
                    stud = bpy.context.active_object
                    stud.scale.y = 0.55
                    stud.data.materials.append(brass_mat)

            # Golden Escutcheon Disc + Hanging Torus Knocker Ring!
            kx, kz = side * 0.39, 1.92
            bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.145, depth=0.05, location=(kx, y_face, kz), rotation=(math.pi * 0.5, 0, 0))
            plate = bpy.context.active_object
            plate.data.materials.append(brass_mat)

            bpy.ops.mesh.primitive_torus_add(major_radius=0.195, minor_radius=0.038, location=(kx, y_face + (-0.04 if y_face < 0 else 0.04), kz - 0.16), rotation=(math.pi * 0.5, 0, 0))
            ring = bpy.context.active_object
            ring.data.materials.append(brass_mat)

    # Center dark iron/shadow seam bar
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 1.98))
    seam = bpy.context.active_object
    seam.scale = (0.055, 0.29, 3.96)
    seam.data.materials.append(seam_mat)

    export_glb("Stages/Stage_05_妖洞通关石门_闭.glb")

    # --- OPEN DOOR (Stage_05b) ---
    clear_scene()
    stone_mat2 = make_mat("ArchStoneOpen", base_color=(0.25, 0.27, 0.36, 1.0), roughness=0.74, metallic=0.10)
    abyss_mat = make_mat("AbyssPortal", base_color=(0.05, 0.04, 0.09, 1.0), roughness=0.95)
    step_mat = make_mat("PortalStep", base_color=(0.38, 0.28, 0.58, 1.0), roughness=0.52, emission_color=(0.34, 0.22, 0.58), emission_strength=0.45)

    _build_dungeon_archway(stone_mat2)

    # Deep dark recessed cavern chamber behind archway (Y in [-0.05..+1.35])
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.65, 2.02))
    chamber = bpy.context.active_object
    chamber.scale = (2.48, 1.25, 4.02)
    chamber.data.materials.append(abyss_mat)

    # 3 Ascending Glowing Purple Stone Steps inside the open portal matching Stage_05b!
    for si, (sw, sy, sz) in enumerate([
        (1.88, -0.10, 0.26),
        (1.56,  0.18, 0.68),
        (1.26,  0.46, 1.10),
    ]):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, sy, sz))
        st = bpy.context.active_object
        st.scale = (sw, 0.24, 0.13)
        st.data.materials.append(step_mat)

    export_glb("Stages/Stage_05b_妖洞通关石门_开.glb")


# ==============================================================================
# 4. SEVEN SPIRIT ORBS (Orb_01 .. Orb_07)
# ==============================================================================
def build_all_orbs():
    orb_specs = [
        ("Orb_01_赤红灵珠.glb", (0.92, 0.16, 0.14), (1.0, 0.45, 0.42)),
        ("Orb_02_橘橙灵珠.glb", (0.98, 0.50, 0.10), (1.0, 0.68, 0.35)),
        ("Orb_03_明黄灵珠.glb", (0.98, 0.82, 0.14), (1.0, 0.92, 0.48)),
        ("Orb_04_翠绿灵珠.glb", (0.18, 0.80, 0.36), (0.48, 0.95, 0.60)),
        ("Orb_05_青蓝灵珠.glb", (0.14, 0.74, 0.90), (0.48, 0.90, 1.00)),
        ("Orb_06_靛蓝灵珠.glb", (0.22, 0.38, 0.94), (0.50, 0.65, 1.00)),
        ("Orb_07_紫红灵珠.glb", (0.70, 0.22, 0.88), (0.88, 0.52, 1.00)),
    ]
    for fname, core_rgb, halo_rgb in orb_specs:
        clear_scene()
        core_mat = make_mat(
            "OrbGourdCore",
            base_color=(*core_rgb, 1.0),
            roughness=0.14,
            metallic=0.08,
            emission_color=core_rgb,
            emission_strength=0.85,
        )
        halo_mat = make_mat(
            "OrbAuraShell",
            base_color=(*halo_rgb, 1.0),
            roughness=0.12,
            alpha=0.32,
            blend=True,
            emission_color=halo_rgb,
            emission_strength=0.55,
        )
        # Plump 3D Spirit Gourd Core centered inside translucent glowing Aura Sphere (Z in [0..1.0], centered at Z=0.5)
        gourd = make_gourd_mesh("SpiritGourd", height=0.66, radius_scale=1.18, segs=32)
        gourd.location = (0.0, 0.0, 0.17)
        gourd.data.materials.append(core_mat)

        bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=16, radius=0.48, location=(0.0, 0.0, 0.50))
        halo = bpy.context.active_object
        for p in halo.data.polygons:
            p.use_smooth = True
        halo.data.materials.append(halo_mat)

        export_glb(f"Orbs/{fname}")


# ==============================================================================
# 5. EIGHT VOLUMETRIC 3D VFX (VFX_01 .. VFX_08)
# ==============================================================================
def build_all_vfx():
    # VFX_01: Dawa Shockwave Star Ring + Golden Lightning Rays (in XY ground plane -> glTF XZ / or upright XY?
    # Note: In Huluwa3D PowersAndTrapsFX, PlaneGeometry lies in XY and is rotated/billboarded!
    # Building flat/radial VFX centered in glTF XY (Blender XZ!) makes it a 1:1 drop-in replacement for PlaneGeometry!)

    # --- VFX_01: 大娃_裂地云雷震波 ---
    clear_scene()
    burst_mat = make_mat("ShockBurst", base_color=(0.88, 0.28, 0.16, 1.0), roughness=0.35, emission_color=(0.94, 0.32, 0.16), emission_strength=1.6)
    ray_mat = make_mat("ShockRay", base_color=(1.0, 0.92, 0.52, 1.0), roughness=0.2, emission_color=(1.0, 0.92, 0.48), emission_strength=2.4)
    bm = bmesh.new()
    n_pts = 56  # 28 outer spikes
    r_in_ring, r_out_ring = [], []
    for i in range(n_pts):
        a = (2.0 * math.pi * i) / n_pts
        ro = 1.95 if i % 2 == 0 else 1.52
        ri = 1.18 if i % 2 == 0 else 0.92
        r_out_ring.append(bm.verts.new((ro * math.cos(a), 0.0, ro * math.sin(a))))
        r_in_ring.append(bm.verts.new((ri * math.cos(a), 0.0, ri * math.sin(a))))
    for i in range(n_pts):
        inext = (i + 1) % n_pts
        bm.faces.new((r_in_ring[i], r_out_ring[i], r_out_ring[inext], r_in_ring[inext]))
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=0.14)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    m1 = bpy.data.meshes.new("VFX01_Star")
    bm.to_mesh(m1)
    bm.free()
    o1 = bpy.data.objects.new("VFX01_Star", m1)
    o1.location.y = 0.07
    o1.data.materials.append(burst_mat)
    bpy.context.collection.objects.link(o1)
    for k in range(12):
        a = (2.0 * math.pi * k) / 12.0 + 0.12
        ray = make_curve_tube(f"Ray_{k}", [(0.82 * math.cos(a), 0.0, 0.82 * math.sin(a), 0.032), (2.12 * math.cos(a), 0.0, 2.12 * math.sin(a), 0.014)], bevel_res=4)
        ray.data.materials.append(ray_mat)
    export_glb("VFX/VFX_01_大娃_裂地云雷震波.glb")

    # --- VFX_02: 二娃_千里眼暴击神光印 ---
    clear_scene()
    star_mat = make_mat("DivineGold", base_color=(1.0, 0.78, 0.22, 1.0), roughness=0.25, metallic=0.45, emission_color=(1.0, 0.76, 0.18), emission_strength=1.5)
    sclera_mat = make_mat("EyeSclera", base_color=(0.95, 1.0, 0.95, 1.0), roughness=0.18, emission_color=(0.9, 1.0, 0.92), emission_strength=0.9)
    iris_mat = make_mat("EyeIris", base_color=(0.12, 0.78, 0.46, 1.0), roughness=0.15, emission_color=(0.12, 0.85, 0.50), emission_strength=1.4)
    pupil_mat = make_mat("EyePupilGold", base_color=(1.0, 0.96, 0.68, 1.0), roughness=0.1, emission_color=(1.0, 0.96, 0.65), emission_strength=2.5)

    bm2 = bmesh.new()
    vc_f = bm2.verts.new((0, -0.16, 0))
    vc_b = bm2.verts.new((0, 0.16, 0))
    star_v = []
    for i in range(16):
        a = (2.0 * math.pi * i) / 16.0
        r = 2.02 if i % 2 == 0 else 0.78
        star_v.append(bm2.verts.new((r * math.cos(a), 0.0, r * math.sin(a))))
    for i in range(16):
        inext = (i + 1) % 16
        bm2.faces.new((vc_f, star_v[inext], star_v[i]))
        bm2.faces.new((vc_b, star_v[i], star_v[inext]))
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
    m2 = bpy.data.meshes.new("VFX02_Star")
    bm2.to_mesh(m2)
    bm2.free()
    o2 = bpy.data.objects.new("VFX02_Star", m2)
    o2.data.materials.append(star_mat)
    bpy.context.collection.objects.link(o2)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.58, location=(0, 0, 0))
    eye = bpy.context.active_object
    eye.scale = (1.12, 0.36, 0.64)
    eye.data.materials.append(sclera_mat)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.25, location=(0, 0, 0))
    iris = bpy.context.active_object
    iris.scale = (1.0, 0.92, 1.0)
    iris.data.materials.append(iris_mat)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.11, location=(0, 0, 0))
    pup = bpy.context.active_object
    pup.scale = (1.0, 1.12, 1.0)
    pup.data.materials.append(pupil_mat)
    export_glb("VFX/VFX_02_二娃_千里眼暴击神光印.glb")

    # --- VFX_03: 三娃_铜头铁臂金钟护盾 ---
    clear_scene()
    ring_gold = make_mat("BellGold", base_color=(1.0, 0.86, 0.28, 1.0), roughness=0.22, metallic=0.65, emission_color=(1.0, 0.84, 0.24), emission_strength=1.4)
    dome_gold = make_mat("BellShieldDome", base_color=(1.0, 0.93, 0.68, 1.0), roughness=0.18, alpha=0.38, blend=True, emission_color=(1.0, 0.88, 0.45), emission_strength=0.7)
    bpy.ops.mesh.primitive_torus_add(major_radius=1.72, minor_radius=0.075, location=(0, 0, 0), rotation=(math.pi * 0.5, 0, 0))
    t_out = bpy.context.active_object
    t_out.data.materials.append(ring_gold)
    bpy.ops.mesh.primitive_torus_add(major_radius=1.56, minor_radius=0.032, location=(0, 0, 0), rotation=(math.pi * 0.5, 0, 0))
    t_in = bpy.context.active_object
    t_in.data.materials.append(ring_gold)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.62, location=(0, 0, 0))
    dome = bpy.context.active_object
    dome.scale.y = 0.35
    dome.data.materials.append(dome_gold)
    for k in range(8):
        a = (2.0 * math.pi * k) / 8.0
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=0, radius=0.18, location=(1.72 * math.cos(a), 0, 1.72 * math.sin(a)))
        gem = bpy.context.active_object
        gem.data.materials.append(ring_gold)
    export_glb("VFX/VFX_03_三娃_铜头铁臂金钟护盾.glb")

    # --- VFX_04: 四娃_剪纸烈火舌喷射 -> 3D Multi-Layer Roaring Fire Cone (along +X matching sprite!) ---
    clear_scene()
    f_red = make_mat("FireRed", base_color=(0.90, 0.20, 0.10, 1.0), roughness=0.3, alpha=0.88, blend=True, emission_color=(0.95, 0.22, 0.08), emission_strength=1.8)
    f_org = make_mat("FireOrange", base_color=(0.98, 0.52, 0.10, 1.0), roughness=0.25, emission_color=(1.0, 0.55, 0.08), emission_strength=2.2)
    f_yel = make_mat("FireCore", base_color=(1.0, 0.90, 0.32, 1.0), roughness=0.2, emission_color=(1.0, 0.92, 0.35), emission_strength=2.8)
    for li, (x0, x1, r_end, n_lobes, mat) in enumerate([
        (-1.75, 1.65, 1.22, 9, f_red),
        (-1.72, 0.85, 0.92, 8, f_org),
        (-1.68, -0.05, 0.64, 7, f_yel),
    ]):
        bm_f = bmesh.new()
        segs, rings_n = 28, 12
        rings = []
        for j in range(rings_n):
            u = j / (rings_n - 1)
            xx = x0 + u * (x1 - x0)
            if j == 0:
                rings.append([bm_f.verts.new((xx, 0, 0))])
            else:
                ring = []
                for s in range(segs):
                    a = (2 * math.pi * s) / segs
                    jag = 1.0 + 0.18 * math.sin(a * n_lobes + u * 6.0)
                    rr = (0.06 + u * r_end) * jag
                    ring.append(bm_f.verts.new((xx, rr * math.cos(a) * 0.72, rr * math.sin(a))))
                rings.append(ring)
        for j in range(rings_n - 1):
            r0, r1 = rings[j], rings[j + 1]
            if len(r0) == 1:
                for s in range(segs):
                    bm_f.faces.new((r0[0], r1[(s + 1) % segs], r1[s]))
            else:
                for s in range(segs):
                    sn = (s + 1) % segs
                    bm_f.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
        bm_f.faces.new(tuple(rings[-1]))
        bmesh.ops.recalc_face_normals(bm_f, faces=bm_f.faces)
        for f in bm_f.faces:
            f.smooth = True
        mf = bpy.data.meshes.new(f"FireCone_{li}")
        bm_f.to_mesh(mf)
        bm_f.free()
        of = bpy.data.objects.new(f"FireCone_{li}", mf)
        of.data.materials.append(mat)
        bpy.context.collection.objects.link(of)
    export_glb("VFX/VFX_04_四娃_剪纸烈火舌喷射.glb")

    # --- VFX_05: 五娃_剪纸卷浪水柱 -> 3D Spiraling Water Torrent + 3D Foam Droplets ---
    clear_scene()
    w_deep = make_mat("WaterDeep", base_color=(0.10, 0.44, 0.82, 1.0), roughness=0.14, alpha=0.82, blend=True, emission_color=(0.10, 0.48, 0.88), emission_strength=0.9)
    w_mid = make_mat("WaterMid", base_color=(0.20, 0.68, 0.95, 1.0), roughness=0.12, alpha=0.86, blend=True, emission_color=(0.22, 0.72, 0.98), emission_strength=1.2)
    w_foam = make_mat("WaterFoam", base_color=(0.86, 0.98, 1.0, 1.0), roughness=0.18, emission_color=(0.85, 0.98, 1.0), emission_strength=1.8)
    for li, (x0, x1, r_end, mat) in enumerate([(-1.72, 1.58, 1.08, w_deep), (-1.70, 0.75, 0.78, w_mid), (-1.68, -0.18, 0.52, w_foam)]):
        bm_w = bmesh.new()
        segs, rings_n = 28, 14
        rings = []
        for j in range(rings_n):
            u = j / (rings_n - 1)
            xx = x0 + u * (x1 - x0)
            if j == 0:
                rings.append([bm_w.verts.new((xx, 0, 0))])
            else:
                ring = []
                for s in range(segs):
                    a = (2 * math.pi * s) / segs
                    wave = 1.0 + 0.14 * math.sin(a * 4.0 - u * 9.0)
                    rr = (0.08 + u * r_end) * wave
                    ring.append(bm_w.verts.new((xx, rr * math.cos(a) * 0.76, rr * math.sin(a))))
                rings.append(ring)
        for j in range(rings_n - 1):
            r0, r1 = rings[j], rings[j + 1]
            if len(r0) == 1:
                for s in range(segs):
                    bm_w.faces.new((r0[0], r1[(s + 1) % segs], r1[s]))
            else:
                for s in range(segs):
                    sn = (s + 1) % segs
                    bm_w.faces.new((r0[s], r0[sn], r1[sn], r1[s]))
        bm_w.faces.new(tuple(rings[-1]))
        bmesh.ops.recalc_face_normals(bm_w, faces=bm_w.faces)
        for f in bm_w.faces:
            f.smooth = True
        mw = bpy.data.meshes.new(f"WaterCone_{li}")
        bm_w.to_mesh(mw)
        bm_w.free()
        ow = bpy.data.objects.new(f"WaterCone_{li}", mw)
        ow.data.materials.append(mat)
        bpy.context.collection.objects.link(ow)
    for di, (dx, dy, dz, dr) in enumerate([(0.32, -0.58, 0.24, 0.14), (1.08, -0.62, 0.48, 0.18), (1.42, -0.72, -0.02, 0.22), (1.06, -0.58, -0.55, 0.18)]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10, radius=dr, location=(dx, dy, dz))
        sp = bpy.context.active_object
        sp.data.materials.append(w_foam)
    export_glb("VFX/VFX_05_五娃_剪纸卷浪水柱.glb")

    # --- VFX_06: 七娃_宝葫芦紫金旋涡 ---
    clear_scene()
    v_purp = make_mat("VortexPurple", base_color=(0.82, 0.30, 0.95, 1.0), roughness=0.22, emission_color=(0.82, 0.28, 0.96), emission_strength=1.8)
    v_gold = make_mat("VortexGold", base_color=(1.0, 0.84, 0.28, 1.0), roughness=0.20, emission_color=(1.0, 0.84, 0.26), emission_strength=2.2)
    v_void = make_mat("VortexSingularity", base_color=(0.06, 0.02, 0.12, 1.0), roughness=0.4)
    v_aura = make_mat("VortexAura", base_color=(0.76, 0.42, 0.92, 1.0), roughness=0.25, alpha=0.35, blend=True, emission_color=(0.72, 0.36, 0.90), emission_strength=0.8)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.24, location=(0, 0, 0))
    void_s = bpy.context.active_object
    void_s.data.materials.append(v_void)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.26, minor_radius=0.035, location=(0, 0, 0), rotation=(math.pi * 0.5, 0, 0))
    v_ring = bpy.context.active_object
    v_ring.data.materials.append(v_gold)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=14, radius=1.75, location=(0, 0, 0))
    aura_d = bpy.context.active_object
    aura_d.scale.y = 0.22
    aura_d.data.materials.append(v_aura)

    for arm in range(6):
        a0 = (2.0 * math.pi * arm) / 6.0
        pts = []
        for s in range(16):
            u = s / 15.0
            r = 0.25 + u * 1.65
            ang = a0 - u * 3.4
            pts.append((r * math.cos(ang), (1.0 - u) * -0.25 + u * 0.15, r * math.sin(ang), 0.055 * (1.0 - 0.65 * u)))
        a_obj = make_curve_tube(f"SpiralArm_{arm}", pts, bevel_res=5)
        a_obj.data.materials.append(v_gold if arm % 3 == 0 else v_purp)
    export_glb("VFX/VFX_06_七娃_宝葫芦紫金旋涡.glb")

    # --- VFX_07: 金蛇精_金如意冰凌寒风 ---
    clear_scene()
    ice_core_mat = make_mat("IceCore", base_color=(0.82, 0.97, 1.0, 1.0), roughness=0.10, metallic=0.12, alpha=0.88, blend=True, emission_color=(0.72, 0.95, 1.0), emission_strength=1.2)
    ice_spike_mat = make_mat("IceSpike", base_color=(0.55, 0.92, 1.0, 1.0), roughness=0.08, metallic=0.15, alpha=0.84, blend=True, emission_color=(0.45, 0.88, 1.0), emission_strength=1.5)
    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=0.44, depth=0.28, location=(0, 0, 0), rotation=(math.pi * 0.5, 0, 0))
    icore = bpy.context.active_object
    icore.data.materials.append(ice_core_mat)
    spike_lens = [1.85, 1.15, 1.92, 1.35, 1.88, 1.08, 1.72, 1.12, 1.86, 1.22, 1.94, 1.18]
    for k, slen in enumerate(spike_lens):
        a = (2.0 * math.pi * k) / 12.0
        bm_i = bmesh.new()
        v_base = [
            bm_i.verts.new((0.38 * math.cos(a) - 0.08 * math.sin(a), 0, 0.38 * math.sin(a) + 0.08 * math.cos(a))),
            bm_i.verts.new((0.38 * math.cos(a), -0.10, 0.38 * math.sin(a))),
            bm_i.verts.new((0.38 * math.cos(a) + 0.08 * math.sin(a), 0, 0.38 * math.sin(a) - 0.08 * math.cos(a))),
            bm_i.verts.new((0.38 * math.cos(a), 0.10, 0.38 * math.sin(a))),
        ]
        rmid = 0.38 + slen * 0.42
        v_mid = [
            bm_i.verts.new((rmid * math.cos(a) - 0.16 * math.sin(a), 0, rmid * math.sin(a) + 0.16 * math.cos(a))),
            bm_i.verts.new((rmid * math.cos(a), -0.14, rmid * math.sin(a))),
            bm_i.verts.new((rmid * math.cos(a) + 0.16 * math.sin(a), 0, rmid * math.sin(a) - 0.16 * math.cos(a))),
            bm_i.verts.new((rmid * math.cos(a), 0.14, rmid * math.sin(a))),
        ]
        rtip = 0.38 + slen
        v_tip = bm_i.verts.new((rtip * math.cos(a), 0, rtip * math.sin(a)))
        for s in range(4):
            sn = (s + 1) % 4
            bm_i.faces.new((v_base[s], v_base[sn], v_mid[sn], v_mid[s]))
            bm_i.faces.new((v_mid[s], v_mid[sn], v_tip))
        bm_i.faces.new(tuple(reversed(v_base)))
        bmesh.ops.recalc_face_normals(bm_i, faces=bm_i.faces)
        mi = bpy.data.meshes.new(f"IceSpike_{k}")
        bm_i.to_mesh(mi)
        bm_i.free()
        oi = bpy.data.objects.new(f"IceSpike_{k}", mi)
        oi.data.materials.append(ice_spike_mat)
        bpy.context.collection.objects.link(oi)
    export_glb("VFX/VFX_07_金蛇精_金如意冰凌寒风.glb")

    # --- VFX_08: 金蛇精_刚柔阴阳剑柔丝网 ---
    clear_scene()
    silk_mag = make_mat("SilkMagenta", base_color=(0.90, 0.32, 0.98, 1.0), roughness=0.22, emission_color=(0.92, 0.30, 1.0), emission_strength=1.8)
    silk_cya = make_mat("SilkCyan", base_color=(0.24, 0.96, 0.80, 1.0), roughness=0.22, emission_color=(0.22, 0.98, 0.82), emission_strength=1.8)
    # 5 horizontal wavy strands + 5 vertical wavy strands bowed forward on a 3D cocoon dome!
    for row in range(6):
        zc = -1.65 + (3.30 * row) / 5.0
        pts = []
        for s in range(13):
            xc = -1.75 + (3.50 * s) / 12.0
            zz = zc + 0.22 * math.sin(xc * 2.1 + row * 1.3)
            bow_y = -0.32 * max(0.0, 1.0 - (xc * xc + zz * zz) / 5.5) + (0.03 if row % 2 == 0 else -0.03)
            pts.append((xc, bow_y, zz, 0.038))
        sobj = make_curve_tube(f"SilkH_{row}", pts, bevel_res=4)
        sobj.data.materials.append(silk_mag if row % 2 == 0 else silk_cya)

    for col_i in range(6):
        xc = -1.65 + (3.30 * col_i) / 5.0
        pts = []
        for s in range(13):
            zc = -1.75 + (3.50 * s) / 12.0
            xx = xc + 0.22 * math.cos(zc * 2.1 + col_i * 1.2)
            bow_y = -0.32 * max(0.0, 1.0 - (xx * xx + zc * zc) / 5.5) + (-0.03 if col_i % 2 == 0 else 0.03)
            pts.append((xx, bow_y, zc, 0.038))
        sobj = make_curve_tube(f"SilkV_{col_i}", pts, bevel_res=4)
        sobj.data.materials.append(silk_cya if col_i % 2 == 0 else silk_mag)
    export_glb("VFX/VFX_08_金蛇精_刚柔阴阳剑柔丝网.glb")


# ==============================================================================
# 6. THREE DE-PAPERED SEAMLESS 3D TERRAIN MAPS (MapBG_01 .. MapBG_03)
# ==============================================================================
def build_all_maps():
    map_specs = [
        ("MapBG_01_葫芦山脚宣纸底.glb", "MapBG_01_clean.png", (0.58, 0.56, 0.44), (0.42, 0.48, 0.36), False, 198611),
        ("MapBG_02_妖洞前殿岩彩底.glb", "MapBG_02_clean.png", (0.20, 0.22, 0.28), (0.32, 0.24, 0.22), False, 198612),
        ("MapBG_03_水晶内殿暗紫底.glb", "MapBG_03_clean.png", (0.16, 0.08, 0.24), (0.26, 0.14, 0.36), True,  198613),
    ]
    S = 512
    yy, xx = np.mgrid[0:S, 0:S]
    for glb_name, tex_name, c1, c2, has_inlay, seed in map_specs:
        clear_scene()
        rng = np.random.default_rng(seed)
        hfield = np.zeros((S, S), dtype=np.float32)
        for freq, amp in [(2, 0.42), (4, 0.28), (9, 0.18), (19, 0.12)]:
            ph1, ph2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
            hfield += amp * np.sin(2 * np.pi * freq * xx / S + ph1) * np.cos(2 * np.pi * freq * yy / S + ph2)
        hfield = (hfield - hfield.min()) / (hfield.max() - hfield.min() + 1e-6)

        col1, col2 = np.array(c1, dtype=np.float32), np.array(c2, dtype=np.float32)
        rgb = col1[None, None, :] * hfield[:, :, None] + col2[None, None, :] * (1.0 - hfield[:, :, None])

        if has_inlay:
            # Clean 3D engraved amethyst magic circle inlay (matching MapBG_03 without vignette darkening!)
            dx = (xx - S * 0.5) / (S * 0.5)
            dy = (yy - S * 0.5) / (S * 0.5)
            rr = np.hypot(dx, dy)
            ang = np.arctan2(dy, dx)
            rings = np.exp(-(((rr - 0.65) * 48.0) ** 2)) + np.exp(-(((rr - 0.38) * 48.0) ** 2))
            spokes = np.exp(-((np.abs(np.sin(ang * 8.0)) * rr * 28.0) ** 2)) * (rr > 0.38) * (rr < 0.65)
            inlay = np.clip(rings + spokes, 0.0, 1.0)
            rgb = rgb * (1.0 - 0.55 * inlay[:, :, None]) + np.array([0.48, 0.30, 0.68])[None, None, :] * (0.55 * inlay[:, :, None])
            hfield = hfield - 0.25 * inlay

        gy, gx = np.gradient(hfield)
        nx, ny = -gx * 2.5, gy * 2.5
        nz = np.ones_like(nx)
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        norm_rgba = np.ones((S, S, 4), dtype=np.float32)
        norm_rgba[:, :, 0] = nx / nlen * 0.5 + 0.5
        norm_rgba[:, :, 1] = ny / nlen * 0.5 + 0.5
        norm_rgba[:, :, 2] = nz / nlen * 0.5 + 0.5

        alb_rgba = np.ones((S, S, 4), dtype=np.float32)
        alb_rgba[:, :, :3] = np.clip(rgb, 0.0, 1.0)
        alb_path = save_rgba_png(tex_name.replace(".png", "_alb"), alb_rgba)
        nrm_path = save_rgba_png(tex_name.replace(".png", "_nrm"), norm_rgba)

        # Also save clean seamless PNG into Maps folder for direct TextureLoader tiling!
        for root_dir in ([OUT_ROOT, HULUWA3D_OUT_ROOT] if HULUWA3D_OUT_ROOT else [OUT_ROOT]):
            m_dir = os.path.join(root_dir, "Maps")
            os.makedirs(m_dir, exist_ok=True)
            shutil.copy2(alb_path, os.path.join(m_dir, tex_name))
            shutil.copy2(nrm_path, os.path.join(m_dir, tex_name.replace("_clean.png", "_normal.png")))

        t_mat = make_mat("TerrainPBR", albedo_path=alb_path, normal_path=nrm_path, roughness=0.78 if not has_inlay else 0.42, metallic=0.05 if not has_inlay else 0.22, repeat=True)

        # Build 3D Displaced Terrain Tile Mesh (8m x 8m x 0.25m)
        bm_t = bmesh.new()
        N = 32
        grid = []
        for j in range(N):
            row = []
            for i in range(N):
                u, v = i / (N - 1), j / (N - 1)
                px = (u - 0.5) * 8.0
                py = (v - 0.5) * 8.0
                hz = float(hfield[int(v * (S - 1)), int(u * (S - 1))]) * (0.08 if has_inlay else 0.22)
                row.append(bm_t.verts.new((px, py, hz)))
            grid.append(row)
        for j in range(N - 1):
            for i in range(N - 1):
                bm_t.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
        bmesh.ops.solidify(bm_t, geom=bm_t.faces[:], thickness=0.24)
        bmesh.ops.recalc_face_normals(bm_t, faces=bm_t.faces)
        uv_t = bm_t.loops.layers.uv.new("UVMap")
        for f in bm_t.faces:
            f.smooth = True
            for l in f.loops:
                l[uv_t].uv = (l.vert.co.x / 8.0 + 0.5, l.vert.co.y / 8.0 + 0.5)
        tm = bpy.data.meshes.new("TerrainTile")
        bm_t.to_mesh(tm)
        bm_t.free()
        tobj = bpy.data.objects.new("TerrainTile", tm)
        tobj.data.materials.append(t_mat)
        bpy.context.collection.objects.link(tobj)

        export_glb(f"Maps/{glb_name}")


def main():
    argv = sys.argv
    only = set(argv[argv.index("--") + 1 :]) if "--" in argv else None

    # 1. Characters + Stage_04
    manifest_p = os.path.join(CHAR_CACHE, "manifest.json")
    if os.path.exists(manifest_p):
        with open(manifest_p, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            if only and entry["id"] not in only:
                continue
            build_character_from_npz(entry)

    # 2. Props
    if not only or "prop_vine" in only:
        build_prop_vine()
    if not only or "prop_cauldron" in only:
        build_prop_cauldron()
    if not only or "prop_lotus" in only:
        build_prop_lotus()
    if not only or "prop_mirror" in only:
        build_prop_mirror()
    if not only or "prop_sword" in only:
        build_prop_sword()
    if not only or "prop_peach" in only:
        build_prop_peach()

    # 3. Stages (Stage_04 is handled in Characters manifest)
    if not only or "stage_rock" in only:
        build_stage_rock()
    if not only or "stage_pillar" in only:
        build_stage_pillar()
    if not only or "stage_mire" in only:
        build_stage_mire()
    if not only or "stage_doors" in only:
        build_stage_doors()

    # 4. Orbs
    if not only or "orbs" in only:
        build_all_orbs()

    # 5. VFX
    if not only or "vfx" in only:
        build_all_vfx()

    # 6. Maps
    if not only or "maps" in only:
        build_all_maps()


if __name__ == "__main__":
    main()
