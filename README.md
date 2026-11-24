# 葫芦兄弟 · 2D & 3D 游戏素材库 / Calabash Brothers Assets (2D + 3D)

基于 1986 版《葫芦兄弟》重制的非商业粉丝致敬游戏素材库，同时保留完整的 **2D 原画剪纸贴图** 与 **1:1 细节还原的 Three.js 3D 模型（`.glb`）**，并附带纯享国风器乐 BGM 与技能音效。

Non-commercial fan-made game asset pack inspired by the 1986 classic *Calabash Brothers*, providing both the original **2D PNG sprites** and **3D glTF 2.0 `.glb` models** ready for Three.js, plus BGM and SFX audio.

---

## 目录区分 / Directory Layout

仓库按素材维度清晰划分，原有 2D 贴图与音频完整保留在各自目录，所有 3D 模型与 Three.js 加载模块统一位于 `Models_3D/`：

```text
GameAssets/
├── Sprites_Characters/          # [2D 角色贴图] 18 张透明通道 PNG（七兄弟、小金刚、爷爷、穿山甲、三大妖王、四小妖）
│   ├── 01_Heroes_Allies/
│   └── 02_Enemies_Bosses/
├── Sprites_Props_Env/           # [2D 法宝与机关贴图] 19 张 PNG
│   ├── 01_Props_Treasures/      #   七色葫芦仙藤、炼丹炉、七色彩莲、水晶魔镜、刚柔阴阳剑、蟠桃
│   ├── 02_Stage_Obstacles/      #   青绿奇石、钟乳石柱、迷魂泥潭、困爷爷蛛网石柱、妖洞石门（闭/开）
│   └── 03_Drop_Orbs/            #   七色灵珠（赤橙黄绿青蓝紫）
├── Sprites_VFX/                 # [2D 技能特效贴图] 8 张 PNG（震波、千里眼神光印、金钟罩、烈火、水柱、旋涡、寒风、柔丝网）
├── Sprites_Maps/                # [2D 地面底图] 3 张 PNG（葫芦山脚、妖洞前殿、水晶内殿）
│
├── Models_3D/                   # [3D 模型库 · Three.js 专用] 48 个自包含 PBR .glb 模型 + 无缝法线贴图
│   ├── Characters/              #   [3D 角色] 18 个 .glb（去除剪纸黑边与宣纸噪点，立体膨胀浮雕 + 背面解剖结构补全）
│   │   ├── 01_Heroes_Allies/
│   │   └── 02_Enemies_Bosses/
│   ├── Props/                   #   [3D 法宝道具] 6 个 360° 全三维建模 .glb
│   ├── Stages/                  #   [3D 场景机关] 6 个全三维山石/泥潭/石门 .glb
│   ├── Orbs/                    #   [3D 七色灵珠] 7 个双层通透发光葫芦灵珠 .glb
│   ├── VFX/                     #   [3D 立体技能特效] 8 个三维体积特效网格 .glb
│   ├── Maps/                    #   [3D 地形块与 PBR 贴图] 3 个地形 .glb + 6 张去宣纸化无缝 Albedo/Normal PNG
│   ├── index.js                 #   Three.js 48 项 3D 资产清单（MODEL_3D_MANIFEST）与实例化工具类
│   └── GLTFLoader.js            #   兼容 Three.js ES Module 的零外部依赖 GLB 加载器
│
├── Audio_Pure/                  # [重制音频素材] 23 个无损 WAV
│   ├── BGM/                     #   5 首国风器乐背景音乐（含片头、三幕场景与升调片尾）
│   └── SFX/                     #   18 条 UI、拾取、七兄弟神通与妖精法宝音效
├── Audio_Original_1986/         # [1986 原版原声提取] 9 首原版主题曲与经典纯配乐
│   ├── MP3/                     #   9 首 48kHz 320kbps MP3（含 ID3 曲目信息）
│   └── WAV/                     #   9 首 48kHz 16-bit 无损立体声 WAV
│
└── tools_3d/                    # [3D 生成脚本] Python + Blender 5.3 自动化建模与 PBR 烘焙管线
```

---

## 在 Three.js 中使用 3D 素材 / Usage in Three.js

### 方式一：直接使用标准 `GLTFLoader` 加载任意 `.glb`

所有 `.glb` 文件均内嵌 PBR 贴图（BaseColor / Normal / MetallicRoughness），默认遵循 `+Y` 向上、`+Z` 朝前约定：

```js
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('./Models_3D/Characters/01_Heroes_Allies/Hero_01_大娃_红娃.glb', (gltf) => {
  scene.add(gltf.scene);
});
```

### 方式二：使用内置 `Huluwa3DAssetLibrary` 按键名加载与归一化实例化

```js
import { Huluwa3DAssetLibrary, MODEL_3D_MANIFEST } from './Models_3D/index.js';

const library = new Huluwa3DAssetLibrary('./Models_3D/');
await library.load('boss_scorpion');

// 自动按指定高度缩放并将锚点对齐脚底 (y = 0)
const scorpion = library.instantiate('boss_scorpion', {
  fitHeight: 3.85,
  pivotBottom: true,
});
scene.add(scorpion);
```

---

## 2D 与 3D 素材对照一览 / 2D ↔ 3D Asset Mapping

| 类别 | 数量 | 2D 原画目录 | 3D 模型目录（`.glb`） |
|---|---:|---|---|
| 英雄与友方角色 | 11 | `Sprites_Characters/01_Heroes_Allies/` | `Models_3D/Characters/01_Heroes_Allies/` |
| 妖王与小妖 | 7 | `Sprites_Characters/02_Enemies_Bosses/` | `Models_3D/Characters/02_Enemies_Bosses/` |
| 法宝与仙物 | 6 | `Sprites_Props_Env/01_Props_Treasures/` | `Models_3D/Props/` |
| 关卡场景机关 | 6 | `Sprites_Props_Env/02_Stage_Obstacles/` | `Models_3D/Stages/` |
| 七色掉落灵珠 | 7 | `Sprites_Props_Env/03_Drop_Orbs/` | `Models_3D/Orbs/` |
| 技能与法宝特效 | 8 | `Sprites_VFX/` | `Models_3D/VFX/` |
| 场景地面贴图/地形 | 3 | `Sprites_Maps/` | `Models_3D/Maps/` |
| 重制背景音乐与音效 | 23 | `Audio_Pure/BGM/` · `Audio_Pure/SFX/` | —（2D/3D 通用） |
| 1986 原版主题曲与配乐 | 9 | `Audio_Original_1986/MP3/` · `Audio_Original_1986/WAV/` | —（原片原声提取） |

---

## 版权声明 / Disclaimer

本项目为**非商业**粉丝致敬作品，献给上海美术电影制片厂 1986 年出品的《葫芦兄弟》。原作角色、造型与音乐旋律的相关权利归原权利方所有。

- `Models_3D/index.js`、`Models_3D/GLTFLoader.js` 与 `tools_3d/` 中的代码与构建脚本以 **MIT** 协议开源（见 [LICENSE](LICENSE)）。
- 仓库中的 2D 贴图、3D 模型与音频素材为基于原作的二次创作，**不在 MIT 协议范围内**（见 [NOTICE](NOTICE)），仅供个人学习、研究与非商业致敬使用。如权利方认为不妥，请提交 Issue 联系处理。
