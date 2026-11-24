#!/usr/bin/env python3
"""
Production 2D-to-3D Volumetric Character & Relief Preparation Pipeline (v3).
Converts all 18 Huluwa character sprites (+ Stage_04 bound Grandpa) into:
  - Watertight sub-voxel 3D volumetric meshes (MonsterMesh Poisson elliptical inflation
    + enclosed ink-cell 3D cushion domes + natural anatomical posture camber)
  - 2048x1024 side-by-side Front/Back PBR Albedo, Normal, and Metallic-Roughness atlases
    with deep-eroded ink-free rim extrapolation (zero 90° side streaks) and clean
    connected-component anatomical backside synthesis (zero front-feature ghosting on back).
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from skimage import measure

import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.environ.get("HULUWA_3D_CACHE", os.path.join(tempfile.gettempdir(), "huluwa_3d_char_cache"))

CHARACTER_SPECS = [
    # id, rel_png, out_rel_glb, mode, height_m, extra_opts
    ("hero1", "Sprites_Characters/01_Heroes_Allies/Hero_01_大娃_红娃.png", "Characters/01_Heroes_Allies/Hero_01_大娃_红娃.glb", "calabash_hero", 2.15, {}),
    ("hero2", "Sprites_Characters/01_Heroes_Allies/Hero_02_二娃_橙娃.png", "Characters/01_Heroes_Allies/Hero_02_二娃_橙娃.glb", "calabash_hero", 2.08, {}),
    ("hero3", "Sprites_Characters/01_Heroes_Allies/Hero_03_三娃_黄娃.png", "Characters/01_Heroes_Allies/Hero_03_三娃_黄娃.glb", "calabash_hero", 2.12, {}),
    ("hero4", "Sprites_Characters/01_Heroes_Allies/Hero_04_四娃_绿娃.png", "Characters/01_Heroes_Allies/Hero_04_四娃_绿娃.glb", "calabash_hero", 2.10, {}),
    ("hero5", "Sprites_Characters/01_Heroes_Allies/Hero_05_五娃_青娃.png", "Characters/01_Heroes_Allies/Hero_05_五娃_青娃.glb", "calabash_hero", 2.10, {}),
    ("hero6", "Sprites_Characters/01_Heroes_Allies/Hero_06_六娃_蓝娃.png", "Characters/01_Heroes_Allies/Hero_06_六娃_蓝娃.glb", "calabash_hero", 2.08, {}),
    ("hero6_stealth", "Sprites_Characters/01_Heroes_Allies/Hero_06b_六娃_隐身态.png", "Characters/01_Heroes_Allies/Hero_06b_六娃_隐身态.glb", "calabash_hero", 2.08, {"stealth": True}),
    ("hero7", "Sprites_Characters/01_Heroes_Allies/Hero_07_七娃_紫娃.png", "Characters/01_Heroes_Allies/Hero_07_七娃_紫娃.glb", "calabash_hero", 2.05, {"has_hand_gourd": True}),
    ("hero8_king", "Sprites_Characters/01_Heroes_Allies/Hero_08_葫芦小金刚.png", "Characters/01_Heroes_Allies/Hero_08_葫芦小金刚.glb", "calabash_hero", 3.60, {"is_king": True}),
    ("grandpa", "Sprites_Characters/01_Heroes_Allies/NPC_01_爷爷.png", "Characters/01_Heroes_Allies/NPC_01_爷爷.glb", "grandpa", 2.80, {}),
    ("pangolin", "Sprites_Characters/01_Heroes_Allies/NPC_02_穿山甲.png", "Characters/01_Heroes_Allies/NPC_02_穿山甲.glb", "pangolin", 1.65, {}),
    ("boss_centipede", "Sprites_Characters/02_Enemies_Bosses/Boss_01_蜈蚣精.png", "Characters/02_Enemies_Bosses/Boss_01_蜈蚣精.glb", "centipede", 3.30, {}),
    ("boss_scorpion", "Sprites_Characters/02_Enemies_Bosses/Boss_02_蝎子精.png", "Characters/02_Enemies_Bosses/Boss_02_蝎子精.glb", "scorpion", 3.85, {}),
    ("boss_snake", "Sprites_Characters/02_Enemies_Bosses/Boss_03_金蛇精.png", "Characters/02_Enemies_Bosses/Boss_03_金蛇精.glb", "snake", 3.55, {}),
    ("enemy_toad", "Sprites_Characters/02_Enemies_Bosses/Enemy_01_蛤蟆精.png", "Characters/02_Enemies_Bosses/Enemy_01_蛤蟆精.glb", "toad", 2.05, {}),
    ("enemy_bat", "Sprites_Characters/02_Enemies_Bosses/Enemy_02_蝙蝠精.png", "Characters/02_Enemies_Bosses/Enemy_02_蝙蝠精.glb", "bat", 2.15, {}),
    ("enemy_wasp", "Sprites_Characters/02_Enemies_Bosses/Enemy_03_黄蜂精.png", "Characters/02_Enemies_Bosses/Enemy_03_黄蜂精.glb", "wasp", 2.10, {}),
    ("enemy_boar", "Sprites_Characters/02_Enemies_Bosses/Enemy_04_野猪精.png", "Characters/02_Enemies_Bosses/Enemy_04_野猪精.glb", "boar", 2.40, {}),
    # Stage_04 (Bound Grandpa on spiderweb rock pillar) also uses hybrid 3D character+rock relief!
    ("stage_web_grandpa", "Sprites_Props_Env/02_Stage_Obstacles/Stage_04_困爷爷蛛网石柱.png", "Stages/Stage_04_困爷爷蛛网石柱.glb", "web_pillar", 4.20, {}),
]


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / max(1e-6, (e1 - e0)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def largest_cc_bbox(bin_mask):
    labeled, n = ndimage.label(bin_mask)
    if n == 0:
        return None
    sizes = ndimage.sum(bin_mask, labeled, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    cc = labeled == best
    ys, xs = np.where(cc)
    return cc, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def prepare_one(spec, tex_size=1024):
    cid, rel_png, out_rel_glb, mode, height_m, opts = spec
    png_path = os.path.join(REPO_ROOT, rel_png)
    out_dir = os.path.join(CACHE_DIR, cid)
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(png_path).convert("RGBA")
    if img.size != (tex_size, tex_size):
        img = img.resize((tex_size, tex_size), Image.LANCZOS)
    rgba = np.array(img, dtype=np.uint8)
    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    alpha_thresh = 0.25 if opts.get("stealth") else 0.45
    mask = alpha > alpha_thresh
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_fill_holes(mask)
    labeled, n_cc = ndimage.label(mask)
    if n_cc > 1:
        sizes = ndimage.sum(mask, labeled, range(1, n_cc + 1))
        mask = np.isin(labeled, [i + 1 for i, s in enumerate(sizes) if s >= 120])

    dist_in = ndimage.distance_transform_edt(mask)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Compute local 85th-percentile interior brightness strictly from interior pixels (dist_in > 9)
    # so anti-aliased outer outline pixels NEVER masquerade as valid border colors!
    deep_mask = mask & (dist_in > 10.5)
    if deep_mask.sum() < 100:
        deep_mask = mask & (dist_in > 4.0)

    luma_deep_only = np.where(deep_mask, luma, 0.0).astype(np.float32)
    local_bright = ndimage.maximum_filter(luma_deep_only, size=29)

    # Dark threshold adapts for stealth Liuwa (lighter grey outlines ~135) vs normal dark ink (<105)
    ink_dark_cutoff = 152.0 if opts.get("stealth") else 108.0
    solid_dark_cutoff = 135.0 if opts.get("stealth") else 92.0

    # Deep clean pigment seeds: strictly > 10.5 px inside silhouette (completely past all outer outline & anti-alias rings!)
    # AND not an internal dark contour line!
    is_internal_ink = luma < ink_dark_cutoff
    is_genuinely_dark_region = deep_mask & (local_bright <= solid_dark_cutoff)
    pigment_seed = (deep_mask & (~is_internal_ink)) | is_genuinely_dark_region
    if pigment_seed.sum() < 80:
        pigment_seed = deep_mask

    _, idx_pig = ndimage.distance_transform_edt(~pigment_seed, return_indices=True)
    inkless_rgb = rgb[idx_pig[0], idx_pig[1]].copy()
    inkless_rgb = cv2.bilateralFilter(inkless_rgb, d=9, sigmaColor=28, sigmaSpace=9)

    # Front Albedo: crisp internal details inside (dist_in > 16 px), smoothly blending to 100% streak-free inkless_rgb
    # across [10..18 px] so even after mesh smoothing, the entire 90° side profile is 100% pure regional pigment!
    smooth_orig = cv2.bilateralFilter(rgb, d=5, sigmaColor=16, sigmaSpace=5)
    rim_blend = smoothstep(10.5, 19.5, dist_in)[:, :, None]
    front_rgb = np.clip(inkless_rgb * (1.0 - rim_blend) + smooth_orig * rim_blend, 0, 255).astype(np.uint8)

    # Sub-part 3D dome cushions from enclosed anatomical ink cells (for front sculptural relief)
    internal_ink = mask & (dist_in > 9.0) & (luma < (140 if opts.get("stealth") else 75)) & (local_bright > solid_dark_cutoff)
    cell_interior = mask & (~internal_ink) & (dist_in > 2.0)
    cell_dist = ndimage.distance_transform_edt(cell_interior)
    sub_dome = np.tanh(cell_dist / 15.0) - ndimage.gaussian_filter(internal_ink.astype(np.float32), sigma=1.5) * 0.38
    sub_dome = ndimage.gaussian_filter(sub_dome, sigma=2.0) * mask

    # Tangent-space normal map (fades out completely by dist_in=14 so side rim has zero normal ripples)
    normal_taper = smoothstep(12.0, 22.0, dist_in)
    fine_relief = (sub_dome * 0.85 + ndimage.gaussian_filter(luma / 255.0, sigma=1.1) * 0.15) * normal_taper
    gy, gx = np.gradient(fine_relief)
    strength = 2.4
    nx = -gx * strength
    ny = gy * strength
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    normal_u8 = np.clip(
        np.stack([nx / norm * 0.5 + 0.5, ny / norm * 0.5 + 0.5, nz / norm * 0.5 + 0.5], axis=-1) * 255.0,
        0,
        255,
    ).astype(np.uint8)

    ys, xs = np.mgrid[0:tex_size, 0:tex_size]
    my, mx = np.where(mask)
    y0, y1 = int(my.min()), int(my.max())
    x0, x1 = int(mx.min()), int(mx.max())
    uy = (ys - y0) / max(1, (y1 - y0))
    ux = (xs - x0) / max(1, (x1 - x0))
    hsv = cv2.cvtColor(inkless_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Front Metallic-Roughness vs Clean Uniform Back Metallic-Roughness
    rough_f = np.full((tex_size, tex_size), 0.56, dtype=np.float32)
    metal_f = np.zeros((tex_size, tex_size), dtype=np.float32)
    rough_f[(uy < 0.22) & mask] = 0.26
    rough_f[(luma < 48) & (uy < 0.42) & mask] = 0.36
    if opts.get("is_king"):
        # Glowing 7-color chest gem & gold collar
        is_gold = mask & (h_ch >= 16) & (h_ch <= 32) & (s_ch > 90) & (uy < 0.52)
        metal_f[is_gold] = 0.78
        rough_f[is_gold] = 0.22
    if mode in ("scorpion", "centipede", "snake", "toad", "boar", "web_pillar"):
        is_steel = mask & (s_ch < 48) & (v_ch > 130) & (uy > 0.22)
        is_gold = mask & (h_ch >= 16) & (h_ch <= 32) & (s_ch > 85) & (v_ch > 140)
        metal_f[is_steel] = 0.76
        rough_f[is_steel] = 0.25
        metal_f[is_gold] = 0.72
        rough_f[is_gold] = 0.28

    mr_front = np.stack(
        [
            np.full((tex_size, tex_size), 255, dtype=np.uint8),
            np.clip(rough_f * 255.0, 0, 255).astype(np.uint8),
            np.clip(metal_f * 255.0, 0, 255).astype(np.uint8),
        ],
        axis=-1,
    )
    mr_back = np.stack(
        [
            np.full((tex_size, tex_size), 255, dtype=np.uint8),
            np.full((tex_size, tex_size), int(0.56 * 255), dtype=np.uint8),
            np.zeros((tex_size, tex_size), dtype=np.uint8),
        ],
        axis=-1,
    )

    # Synthesize Clean Connected-Component Anatomical Backside Albedo
    back_base = inkless_rgb.copy()
    z_fg_boost = np.zeros((tex_size, tex_size), dtype=np.float32)

    if mode == "calabash_hero":
        hair_dark_lim = 132 if opts.get("stealth") else 52
        raw_hair = mask & (luma < hair_dark_lim) & (uy >= 0.11) & (uy <= 0.39)
        cc_info = largest_cc_bbox(raw_hair)
        if cc_info is not None:
            hair_cc, hx0, hx1, hy0, hy1 = cc_info
            hair_col = np.median(inkless_rgb[hair_cc], axis=0).astype(np.uint8)
        else:
            hx0, hx1 = int(x0 + 0.30 * (x1 - x0)), int(x0 + 0.70 * (x1 - x0))
            hy0 = int(y0 + 0.15 * (y1 - y0))
            hair_col = np.array([105, 118, 130] if opts.get("stealth") else [20, 20, 24], dtype=np.uint8)

        # Smooth top crown curve of hair_cc so gourd & side leaves above hair stay untouched
        hair_top_line = np.full(tex_size, hy0 + 12, dtype=np.float32)
        if cc_info is not None:
            cols_with_h = np.where(hair_cc.any(axis=0))[0]
            if len(cols_with_h) > 4:
                tops = [int(np.where(hair_cc[:, c])[0].min()) for c in cols_with_h]
                hair_top_line = np.interp(np.arange(tex_size), cols_with_h, tops)

        # Head back: strictly inside head column span [hx0+4 .. hx1-4], replacing skin/eyes/brows/mouth with smooth hair
        head_back = (
            mask
            & (xs >= hx0 + 4)
            & (xs <= hx1 - 4)
            & (ys >= hair_top_line[None, :] + 3)
            & (uy <= 0.396)
        )
        back_base[head_back] = hair_col

        # Sample vest fabric color cleanly
        is_skin = (h_ch >= 6) & (h_ch <= 24) & (s_ch >= (22 if opts.get("stealth") else 48)) & (s_ch <= 185) & (v_ch >= 140)
        is_green_collar = (h_ch >= 42) & (h_ch <= 85) & (uy < 0.49) & (ux > 0.35) & (ux < 0.65)
        if opts.get("is_king"):
            vest_col = np.array([246, 246, 248], dtype=np.uint8)  # Xiaojingang white vest!
        elif opts.get("stealth"):
            vest_zone = mask & (uy > 0.40) & (uy < 0.55) & (xs >= hx0 - 12) & (xs <= hx1 + 12)
            vest_cand = vest_zone & (h_ch >= 90) & (h_ch <= 130) & (s_ch > 45)
            vest_col = np.median(inkless_rgb[vest_cand], axis=0).astype(np.uint8) if vest_cand.sum() > 20 else np.array([128, 168, 226], dtype=np.uint8)
        else:
            vest_zone = mask & (uy > 0.40) & (uy < 0.55) & (xs >= hx0 - 15) & (xs <= hx1 + 15)
            # Colored hero vests (red/orange/yellow/green/cyan/blue/purple) all have high saturation > 125!
            vest_cand = vest_zone & (~is_green_collar) & (s_ch > 125) & (v_ch > 75)
            if vest_cand.sum() < 30:
                vest_cand = vest_zone & (~is_skin) & (~is_green_collar) & (v_ch > 65)
            vest_col = np.median(inkless_rgb[vest_cand], axis=0).astype(np.uint8) if vest_cand.sum() > 20 else np.array([205, 42, 42], dtype=np.uint8)

        # Replace ONLY bare chest skin, white front trim, collar wreath, & gem inside central torso [hx0..hx1] x [0.396..0.585]
        # so bare arms outside the vest armholes stay 100% natural skin without rectangular sleeve cuts!
        king_gem_zone = ((uy > 0.40) & (uy < 0.56)) if opts.get("is_king", False) else False
        is_chest_inner = (
            is_skin
            | is_green_collar
            | ((s_ch < 35) & (v_ch > 195))  # white scalloped trim
            | king_gem_zone
        )
        torso_back = (
            mask
            & (uy >= 0.396)
            & (uy <= 0.585)
            & (xs >= hx0 + 6)
            & (xs <= hx1 - 6)
            & (is_chest_inner | (ux > 0.40) & (ux < 0.60))
        )
        back_base[torso_back] = vest_col

        # Boost foreground Z for overlapping hand/gourd poses (Erwa salute, Siwa/Wuwa palm, Liuwa chest fist, Qiwa gourd)
        if opts.get("has_hand_gourd"):
            gourd_hand = mask & (ux > 0.64) & (uy > 0.18) & (uy < 0.46)
            z_fg_boost[gourd_hand] = 1.1

    elif mode == "scorpion":
        # Sample Scorpion's actual dark purple cape & shoulder armor color so there is ZERO rectangular color mismatch!
        cape_sample = mask & (uy > 0.60) & (uy < 0.80) & (ux > 0.68) & (v_ch < 110)
        cape_col = np.median(inkless_rgb[cape_sample], axis=0).astype(np.uint8) if cape_sample.sum() > 30 else np.array([58, 38, 78], dtype=np.uint8)
        helm_col = np.array([224, 230, 238], dtype=np.uint8)

        # Back of face -> helmet back
        helm_back = mask & (uy >= 0.25) & (uy <= 0.43) & (ux >= 0.35) & (ux <= 0.54)
        back_base[helm_back] = helm_col

        # Back of torso/legs inside cape silhouette -> smooth flowing dark purple cape (excluding outer bare arms & bottom boots!)
        is_outer_arm = ((ux < 0.29) | (ux > 0.68)) & (uy > 0.34) & (uy < 0.66) & (v_ch > 88)
        in_cape_span = mask & (uy >= 0.36) & (uy <= 0.86) & (~is_outer_arm)
        back_base[in_cape_span] = cape_col

        is_blade = mask & (s_ch < 45) & (v_ch > 135) & (uy > 0.54) & (uy < 0.82)
        z_fg_boost[is_blade] = 1.2

    elif mode == "grandpa":
        back_base[mask & (uy >= 0.10) & (uy <= 0.36) & (ux >= 0.36) & (ux <= 0.65)] = [236, 236, 240]
        # Replace front white beard & inner tunic seam on upper back with brown vest color
        vest_sample = mask & (uy > 0.38) & (uy < 0.55) & (h_ch >= 8) & (h_ch <= 22) & (v_ch < 155)
        v_col = np.median(inkless_rgb[vest_sample], axis=0).astype(np.uint8) if vest_sample.sum() > 30 else np.array([124, 78, 45], dtype=np.uint8)
        back_base[mask & (uy >= 0.35) & (uy <= 0.58) & (ux >= 0.35) & (ux <= 0.65)] = v_col

    elif mode == "snake":
        back_base[mask & (uy >= 0.17) & (uy <= 0.34) & (ux >= 0.42) & (ux <= 0.61)] = [22, 24, 28]
        back_base[mask & (uy >= 0.34) & (uy <= 0.64) & (ux >= 0.43) & (ux <= 0.62)] = [38, 134, 74]
        is_pale_belly = mask & (uy > 0.50) & (v_ch > 205) & (s_ch < 140)
        back_base[is_pale_belly] = [204, 152, 38]

    elif mode == "centipede":
        back_base[mask & (uy >= 0.14) & (uy <= 0.33) & (ux >= 0.38) & (ux <= 0.62)] = [138, 38, 56]
        back_base[mask & (uy >= 0.30) & (uy <= 0.48) & (ux >= 0.35) & (ux <= 0.65)] = [38, 176, 86]
        back_base[mask & (uy >= 0.48) & (uy <= 0.82) & (ux >= 0.40) & (ux <= 0.60)] = [66, 60, 96]

    elif mode == "toad":
        is_belly_or_mouth = mask & (ux < 0.72) & ((s_ch < 48) | (h_ch < 22))
        back_base[is_belly_or_mouth] = [98, 148, 88]

    elif mode == "bat":
        back_base[mask & (uy >= 0.24) & (uy <= 0.66) & (ux >= 0.34) & (ux <= 0.62)] = [78, 56, 72]

    elif mode in ("pangolin", "wasp", "boar"):
        bright_eye = mask & (uy < 0.52) & (v_ch > 190) & (s_ch < 60) & (dist_in > 10)
        keep_body = mask & (~bright_eye)
        if keep_body.sum() > 50:
            _, b_idx = ndimage.distance_transform_edt(~keep_body, return_indices=True)
            back_base[bright_eye] = back_base[b_idx[0][bright_eye], b_idx[1][bright_eye]]
        if mode == "boar":
            # Foreground boost for heavy battle axe across body
            is_axe = mask & (uy > 0.52) & (uy < 0.82) & ((s_ch < 40) | ((uy > 0.60) & (uy < 0.72)))
            z_fg_boost[is_axe & (s_ch < 40)] = 1.1

    elif mode == "web_pillar":
        # Back of Stage_04 rock pillar is solid dark cavern slate!
        slate_sample = mask & (uy < 0.25)
        slate_col = np.median(inkless_rgb[slate_sample], axis=0).astype(np.uint8) if slate_sample.sum() > 30 else np.array([58, 60, 76], dtype=np.uint8)
        back_base[mask] = slate_col
        # Grandpa & iron chains protrude strongly in +Z on the front of the rock pillar!
        is_grandpa_or_chain = mask & (uy > 0.22) & ((s_ch > 42) | (v_ch > 165))
        z_fg_boost[is_grandpa_or_chain] = 1.8

    # Smooth backside transitions & feather into inkless_rgb at outer silhouette rim so 90° side profile is 100% seamless!
    back_blurred = cv2.bilateralFilter(cv2.GaussianBlur(back_base, (13, 13), 0), d=9, sigmaColor=35, sigmaSpace=9)
    back_rim_w = smoothstep(8.0, 20.0, dist_in)[:, :, None]
    back_rgb = np.clip(inkless_rgb * (1.0 - back_rim_w) + back_blurred * back_rim_w, 0, 255).astype(np.uint8)

    # Extrapolate outside mask from clean inkless border so grazing vertices NEVER sample dark edge texels!
    border_clean_mask = mask & (dist_in > 6.0)
    if border_clean_mask.sum() < 50:
        border_clean_mask = mask
    _, near_clean_idx = ndimage.distance_transform_edt(~border_clean_mask, return_indices=True)
    front_padded = front_rgb.copy()
    front_padded[~mask] = inkless_rgb[near_clean_idx[0][~mask], near_clean_idx[1][~mask]]
    back_padded = back_rgb.copy()
    back_padded[~mask] = inkless_rgb[near_clean_idx[0][~mask], near_clean_idx[1][~mask]]

    # Pack 2048x1024 Atlases
    atlas_albedo = np.concatenate([front_padded, back_padded], axis=1)
    norm_flat = np.full_like(normal_u8, [128, 128, 255])
    atlas_normal = np.concatenate([normal_u8, norm_flat], axis=1)
    atlas_mr = np.concatenate([mr_front, mr_back], axis=1)

    albedo_path = os.path.join(out_dir, f"{cid}_albedo.png")
    normal_path = os.path.join(out_dir, f"{cid}_normal.png")
    mr_path = os.path.join(out_dir, f"{cid}_mr.png")
    Image.fromarray(atlas_albedo).save(albedo_path)
    Image.fromarray(atlas_normal).save(normal_path)
    Image.fromarray(atlas_mr).save(mr_path)

    # Solve Volumetric 3D Poisson Inflation on 320x320 Grid
    G = 320
    mask_g = cv2.resize(mask.astype(np.uint8), (G, G), interpolation=cv2.INTER_AREA) > 0.45
    mask_g = ndimage.binary_closing(mask_g, structure=np.ones((3, 3)))
    mask_g = ndimage.binary_fill_holes(mask_g)
    labeled_g, n_cc_g = ndimage.label(mask_g)
    if n_cc_g > 1:
        sizes_g = ndimage.sum(mask_g, labeled_g, range(1, n_cc_g + 1))
        mask_g = np.isin(labeled_g, [i + 1 for i, s in enumerate(sizes_g) if s >= 18])

    dist_g = ndimage.distance_transform_edt(mask_g)
    idx_map = -np.ones((G, G), dtype=np.int32)
    pts_y, pts_x = np.where(mask_g)
    N = len(pts_y)
    idx_map[pts_y, pts_x] = np.arange(N)

    rows, cols, vals = [], [], []
    rhs = np.full(N, 4.0, dtype=np.float64)
    for k in range(N):
        r, c = pts_y[k], pts_x[k]
        rows.append(k)
        cols.append(k)
        vals.append(4.0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < G and 0 <= nc < G and idx_map[nr, nc] >= 0:
                rows.append(k)
                cols.append(idx_map[nr, nc])
                vals.append(-1.0)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    u_sol = spsolve(A, rhs)
    u_grid = np.zeros((G, G), dtype=np.float32)
    u_grid[pts_y, pts_x] = np.maximum(0.0, u_sol)
    r_poisson = np.sqrt(u_grid)

    gy_idx, gx_idx = np.mgrid[0:G, 0:G]
    my_g, mx_g = np.where(mask_g)
    yg0, yg1 = int(my_g.min()), int(my_g.max())
    xg0, xg1 = int(mx_g.min()), int(mx_g.max())
    uy_g = np.clip((gy_idx - yg0) / max(1, (yg1 - yg0)), 0.0, 1.0)
    ux_g = np.clip((gx_idx - xg0) / max(1, (xg1 - xg0)), 0.0, 1.0)
    char_h_px = float(max(1, yg1 - yg0))

    if mode == "bat":
        body_band = np.exp(-((ux_g - 0.48) / 0.18) ** 2)
        cap_px = (0.042 + 0.118 * body_band) * char_h_px
    elif mode == "web_pillar":
        cap_px = np.full((G, G), 0.165 * char_h_px, dtype=np.float32)
    else:
        cap_px = np.where(uy_g < 0.40, 0.192 * char_h_px, 0.145 * char_h_px)
        cap_px = ndimage.gaussian_filter(cap_px.astype(np.float32), sigma=8.0)

    z_vol_px = cap_px * np.tanh(r_poisson / np.maximum(1.0, cap_px))

    sub_dome_g = cv2.resize(sub_dome, (G, G), interpolation=cv2.INTER_LINEAR)
    fg_boost_g = ndimage.gaussian_filter(cv2.resize(z_fg_boost, (G, G), interpolation=cv2.INTER_LINEAR), sigma=2.5)
    boundary_taper = np.tanh(dist_g / 5.0)

    posture_camber = (
        0.022 * np.exp(-((uy_g - 0.28) / 0.14) ** 2)
        + 0.024 * np.exp(-((uy_g - 0.48) / 0.10) ** 2)
        - 0.018 * np.exp(-((uy_g - 0.68) / 0.11) ** 2)
        + 0.024 * np.exp(-((uy_g - 0.96) / 0.05) ** 2)
    ) * char_h_px * boundary_taper

    if mode == "scorpion":
        tail_arch = np.where(uy_g < 0.26, (-0.055 * (ux_g - 0.58) * 3.0), 0.0) * char_h_px * boundary_taper
        posture_camber = posture_camber + tail_arch

    z_center_front = posture_camber + fg_boost_g * (0.048 * char_h_px) * boundary_taper
    z_front_px = z_center_front + z_vol_px + sub_dome_g * (0.024 * char_h_px) * boundary_taper
    # Backside is completely free of front sub_dome_g & fg_boost_g so ZERO front facial/weapon relief ghosts onto the back!
    z_back_px = -posture_camber + z_vol_px * 0.94

    scale = height_m / char_h_px
    max_z_px = max(float(np.abs(z_front_px).max()), float(np.abs(z_back_px).max())) + 5.0
    NZ = 116
    z_lin = np.linspace(-max_z_px, max_z_px, NZ, dtype=np.float32)

    d_in = ndimage.distance_transform_edt(mask_g)
    d_out = ndimage.distance_transform_edt(~mask_g)
    sdf_xy = (d_in - d_out).astype(np.float32)

    Z3 = z_lin[:, None, None]
    field = np.minimum(sdf_xy[None, :, :], np.minimum(z_front_px[None, :, :] - Z3, Z3 + z_back_px[None, :, :]))
    field = ndimage.gaussian_filter(field, sigma=0.68)

    verts_zyx, faces, _, _ = measure.marching_cubes(field, level=0.0)
    iz, iy, ix = verts_zyx[:, 0], verts_zyx[:, 1], verts_zyx[:, 2]
    z_px = -max_z_px + (iz / (NZ - 1)) * (2.0 * max_z_px)
    cx_px = 0.5 * (xg0 + xg1)
    feet_y_px = float(yg1)

    wx = (ix - cx_px) * scale
    wy = (feet_y_px - iy) * scale
    wz = z_px * scale
    v0 = np.stack([wx, wy, wz], axis=-1)

    npz_path = os.path.join(out_dir, f"{cid}_mesh.npz")
    np.savez_compressed(
        npz_path,
        verts=v0.astype(np.float32),
        faces=faces.astype(np.int32),
        G=np.int32(G),
        cx_px=np.float32(cx_px),
        feet_y_px=np.float32(feet_y_px),
        scale=np.float32(scale),
        albedo_path=albedo_path,
        normal_path=normal_path,
        mr_path=mr_path,
        out_rel_glb=out_rel_glb,
        stealth=bool(opts.get("stealth", False)),
    )
    return npz_path


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = []
    for spec in CHARACTER_SPECS:
        if only and spec[0] not in only:
            continue
        npz_p = prepare_one(spec)
        manifest.append({"id": spec[0], "npz": npz_p, "out_rel_glb": spec[2], "stealth": bool(spec[5].get("stealth", False))})
        print(f"[prep_characters] OK: {spec[0]} -> {spec[2]}")
    with open(os.path.join(CACHE_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
