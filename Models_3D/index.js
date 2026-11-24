import * as THREE from 'three';
import { GLTFLoader } from './GLTFLoader.js';

export const MODEL_3D_MANIFEST = {
  // 1. Characters — Heroes & Allies (11)
  hero1:          { path: 'Characters/01_Heroes_Allies/Hero_01_大娃_红娃.glb',      category: 'heroes',  name: '大娃（红娃）',   height: 2.15 },
  hero2:          { path: 'Characters/01_Heroes_Allies/Hero_02_二娃_橙娃.glb',      category: 'heroes',  name: '二娃（橙娃）',   height: 2.08 },
  hero3:          { path: 'Characters/01_Heroes_Allies/Hero_03_三娃_黄娃.glb',      category: 'heroes',  name: '三娃（黄娃）',   height: 2.12 },
  hero4:          { path: 'Characters/01_Heroes_Allies/Hero_04_四娃_绿娃.glb',      category: 'heroes',  name: '四娃（绿娃）',   height: 2.10 },
  hero5:          { path: 'Characters/01_Heroes_Allies/Hero_05_五娃_青娃.glb',      category: 'heroes',  name: '五娃（青娃）',   height: 2.10 },
  hero6:          { path: 'Characters/01_Heroes_Allies/Hero_06_六娃_蓝娃.glb',      category: 'heroes',  name: '六娃（蓝娃）',   height: 2.08 },
  hero6_stealth:  { path: 'Characters/01_Heroes_Allies/Hero_06b_六娃_隐身态.glb',   category: 'heroes',  name: '六娃（隐身态）', height: 2.08 },
  hero7:          { path: 'Characters/01_Heroes_Allies/Hero_07_七娃_紫娃.glb',      category: 'heroes',  name: '七娃（紫娃）',   height: 2.05 },
  hero8_king:     { path: 'Characters/01_Heroes_Allies/Hero_08_葫芦小金刚.glb',     category: 'heroes',  name: '葫芦小金刚',     height: 3.60 },
  grandpa:        { path: 'Characters/01_Heroes_Allies/NPC_01_爷爷.glb',            category: 'heroes',  name: '爷爷',           height: 2.80 },
  pangolin:       { path: 'Characters/01_Heroes_Allies/NPC_02_穿山甲.glb',          category: 'heroes',  name: '穿山甲',         height: 1.65 },

  // 2. Characters — Enemies & Bosses (7)
  boss_centipede: { path: 'Characters/02_Enemies_Bosses/Boss_01_蜈蚣精.glb',        category: 'enemies', name: '蜈蚣精',         height: 3.30 },
  boss_scorpion:  { path: 'Characters/02_Enemies_Bosses/Boss_02_蝎子精.glb',        category: 'enemies', name: '蝎子精',         height: 3.85 },
  boss_snake:     { path: 'Characters/02_Enemies_Bosses/Boss_03_金蛇精.glb',        category: 'enemies', name: '金蛇精',         height: 3.55 },
  enemy_toad:     { path: 'Characters/02_Enemies_Bosses/Enemy_01_蛤蟆精.glb',       category: 'enemies', name: '蛤蟆精',         height: 2.05 },
  enemy_bat:      { path: 'Characters/02_Enemies_Bosses/Enemy_02_蝙蝠精.glb',       category: 'enemies', name: '蝙蝠精',         height: 2.15 },
  enemy_wasp:     { path: 'Characters/02_Enemies_Bosses/Enemy_03_黄蜂精.glb',       category: 'enemies', name: '黄蜂精',         height: 2.10 },
  enemy_boar:     { path: 'Characters/02_Enemies_Bosses/Enemy_04_野猪精.glb',       category: 'enemies', name: '野猪精',         height: 2.40 },

  // 3. Props & Treasures (6)
  prop_vine:      { path: 'Props/Prop_01_七色葫芦仙藤.glb',                         category: 'props',   name: '七色葫芦仙藤',   height: 3.80 },
  prop_cauldron:  { path: 'Props/Prop_02_三足青铜炼丹炉.glb',                       category: 'props',   name: '三足青铜炼丹炉', height: 2.20 },
  prop_lotus:     { path: 'Props/Prop_03_幽谷七色彩莲.glb',                         category: 'props',   name: '幽谷七色彩莲',   height: 1.25 },
  prop_mirror:    { path: 'Props/Prop_04_水晶魔镜.glb',                             category: 'props',   name: '水晶魔镜',       height: 2.65 },
  prop_sword:     { path: 'Props/Prop_05_刚柔阴阳剑.glb',                           category: 'props',   name: '刚柔阴阳剑',     height: 2.35 },
  prop_peach:     { path: 'Props/Prop_06_回血蟠桃.glb',                             category: 'props',   name: '回血蟠桃',       height: 1.50 },

  // 4. Stage Obstacles (6)
  stage_rock:        { path: 'Stages/Stage_01_青绿剪纸奇石.glb',                    category: 'stages',  name: '青绿山石',       height: 2.80 },
  stage_pillar:      { path: 'Stages/Stage_02_妖洞钟乳石柱.glb',                    category: 'stages',  name: '妖洞钟乳石柱',   height: 4.80 },
  stage_mire:        { path: 'Stages/Stage_03_迷魂泥潭陷阱.glb',                    category: 'stages',  name: '迷魂泥潭陷阱',   height: 0.55 },
  stage_web_grandpa: { path: 'Stages/Stage_04_困爷爷蛛网石柱.glb',                  category: 'stages',  name: '困爷爷蛛网石柱', height: 4.20 },
  stage_door_closed: { path: 'Stages/Stage_05_妖洞通关石门_闭.glb',                 category: 'stages',  name: '妖洞通关石门（闭）', height: 4.90 },
  stage_door_open:   { path: 'Stages/Stage_05b_妖洞通关石门_开.glb',                category: 'stages',  name: '妖洞通关石门（开）', height: 4.90 },

  // 5. Spirit Orbs (7)
  orb1:           { path: 'Orbs/Orb_01_赤红灵珠.glb',                               category: 'orbs',    name: '赤红灵珠',       height: 1.00 },
  orb2:           { path: 'Orbs/Orb_02_橘橙灵珠.glb',                               category: 'orbs',    name: '橘橙灵珠',       height: 1.00 },
  orb3:           { path: 'Orbs/Orb_03_明黄灵珠.glb',                               category: 'orbs',    name: '明黄灵珠',       height: 1.00 },
  orb4:           { path: 'Orbs/Orb_04_翠绿灵珠.glb',                               category: 'orbs',    name: '翠绿灵珠',       height: 1.00 },
  orb5:           { path: 'Orbs/Orb_05_青蓝灵珠.glb',                               category: 'orbs',    name: '青蓝灵珠',       height: 1.00 },
  orb6:           { path: 'Orbs/Orb_06_靛蓝灵珠.glb',                               category: 'orbs',    name: '靛蓝灵珠',       height: 1.00 },
  orb7:           { path: 'Orbs/Orb_07_紫红灵珠.glb',                               category: 'orbs',    name: '紫红灵珠',       height: 1.00 },

  // 6. Volumetric 3D VFX (8)
  vfx1:           { path: 'VFX/VFX_01_大娃_裂地云雷震波.glb',                       category: 'vfx',     name: '裂地云雷震波',   height: 4.20 },
  vfx2:           { path: 'VFX/VFX_02_二娃_千里眼暴击神光印.glb',                   category: 'vfx',     name: '千里眼暴击神光印', height: 3.20 },
  vfx3:           { path: 'VFX/VFX_03_三娃_铜头铁臂金钟护盾.glb',                   category: 'vfx',     name: '铜头铁臂金钟护盾', height: 3.60 },
  vfx4:           { path: 'VFX/VFX_04_四娃_剪纸烈火舌喷射.glb',                     category: 'vfx',     name: '烈火喷射',       height: 2.50 },
  vfx5:           { path: 'VFX/VFX_05_五娃_剪纸卷浪水柱.glb',                       category: 'vfx',     name: '卷浪水柱',       height: 2.50 },
  vfx6:           { path: 'VFX/VFX_06_七娃_宝葫芦紫金旋涡.glb',                     category: 'vfx',     name: '宝葫芦紫金旋涡', height: 3.60 },
  vfx7:           { path: 'VFX/VFX_07_金蛇精_金如意冰凌寒风.glb',                   category: 'vfx',     name: '金如意冰凌寒风', height: 3.80 },
  vfx8:           { path: 'VFX/VFX_08_金蛇精_刚柔阴阳剑柔丝网.glb',                 category: 'vfx',     name: '刚柔阴阳剑柔丝网', height: 3.40 },

  // 7. 3D Terrain Maps (3)
  map_mountain:   { path: 'Maps/MapBG_01_葫芦山脚宣纸底.glb',                       category: 'maps',    name: '葫芦山脚地形',   height: 0.45, albedo: 'Maps/MapBG_01_clean.png', normal: 'Maps/MapBG_01_normal.png' },
  map_cave:       { path: 'Maps/MapBG_02_妖洞前殿岩彩底.glb',                       category: 'maps',    name: '妖洞前殿地形',   height: 0.45, albedo: 'Maps/MapBG_02_clean.png', normal: 'Maps/MapBG_02_normal.png' },
  map_palace:     { path: 'Maps/MapBG_03_水晶内殿暗紫底.glb',                       category: 'maps',    name: '水晶内殿地形',   height: 0.32, albedo: 'Maps/MapBG_03_clean.png', normal: 'Maps/MapBG_03_normal.png' },
};

export function cloneModelInstance(templateScene, { cloneGeometry = false } = {}) {
  const cloned = templateScene.clone(true);
  cloned.traverse((child) => {
    if (child.isMesh) {
      if (cloneGeometry && child.geometry) {
        child.geometry = child.geometry.clone();
      }
      if (Array.isArray(child.material)) {
        child.material = child.material.map((m) => m.clone());
      } else if (child.material) {
        child.material = child.material.clone();
      }
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
  return cloned;
}

export class Huluwa3DAssetLibrary {
  constructor(baseUrl = './') {
    this.baseUrl = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    this.loader = new GLTFLoader();
    this.templates = new Map();
  }

  async load(key) {
    if (this.templates.has(key)) return this.templates.get(key);
    const meta = MODEL_3D_MANIFEST[key];
    if (!meta) throw new Error(`Unknown Huluwa 3D asset key: "${key}"`);
    const url = encodeURI(`${this.baseUrl}${meta.path}`);
    const gltf = await this.loader.loadAsync(url);
    this.templates.set(key, gltf.scene);
    return gltf.scene;
  }

  async loadAll(onProgress) {
    const keys = Object.keys(MODEL_3D_MANIFEST);
    let done = 0;
    await Promise.all(
      keys.map(async (k) => {
        await this.load(k);
        done += 1;
        if (onProgress) onProgress(done / keys.length, k, done, keys.length);
      })
    );
    return this;
  }

  instantiate(key, opts = {}) {
    const tpl = this.templates.get(key);
    if (!tpl) return null;
    const instance = cloneModelInstance(tpl, { cloneGeometry: Boolean(opts.cloneGeometry) });
    if (opts.fitHeight || opts.fitSize) {
      const box = new THREE.Box3().setFromObject(instance);
      const size = new THREE.Vector3();
      box.getSize(size);
      const scale = opts.fitHeight
        ? opts.fitHeight / Math.max(1e-4, size.y)
        : opts.fitSize / Math.max(1e-4, Math.max(size.x, size.y, size.z));
      instance.scale.multiplyScalar(scale);
      box.setFromObject(instance);
      const center = new THREE.Vector3();
      box.getCenter(center);
      if (opts.pivotBottom) {
        instance.position.set(-center.x, -box.min.y, -center.z);
      } else if (opts.centerOrigin) {
        instance.position.set(-center.x, -center.y, -center.z);
      }
      const wrapper = new THREE.Group();
      wrapper.name = `Model3D_${key}`;
      wrapper.add(instance);
      return wrapper;
    }
    return instance;
  }
}
