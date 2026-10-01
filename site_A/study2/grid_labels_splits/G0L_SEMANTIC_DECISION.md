# A1 G0-L 作物语义核验与分类映射决策备忘录 (G0L_SEMANTIC_DECISION)

**发布日期**：2026-09-23  
**执行规范**：依据 `E:\research\SAR\A1_observation_budget_20260923\OVERNIGHT_AGENT_HANDOFF_20260923.md` 第 2 节第 3 条编写。  
**目标**：针对荷兰官方 BRP/RVO 农作物编码表中未明示播种季节的谷物代码，确立严谨、可证伪的语义归类基准，消除基于农事经验的主观推断。

---

## 1. 语义核验背景与存疑代码分析

荷兰国家企业局（RVO）官方《Tabel gewassen en teelten》（作物与种植编码表）中，部分谷物明确以荷兰语词头标注了季节物候，而另一部分仅标示物种或泛称。

经 `scripts/audit_g0l_season_codes_20260923.py` 统计核验，当前映射中存在 **7 个未明示播种季节的代码**：

| Gewascode | 荷兰官方名称 (Dutch Name) | 当前 8 类映射 | 官方名称中是否包含季节词头 | 常见农事经验倾向 |
|---|---|---|---|---|
| **233** | Wintertarwe (冬小麦) | WinterCereal | 是 (`Winter-`) | 冬季播种 |
| **235** | Wintergerst (冬大麦) | WinterCereal | 是 (`Winter-`) | 冬季播种 |
| **234** | Zomertarwe (春小麦) | SpringCereal | 是 (`Zomer-`) | 春季播种 |
| **236** | Zomergerst (春大麦) | SpringCereal | 是 (`Zomer-`) | 春季播种 |
| **238** | Haver (燕麦) | SpringCereal | **否** (未注明冬/春) | 荷兰多春播，但存在冬燕麦 |
| **314** | Triticale (小黑麦) | WinterCereal | **否** (未注明冬/春) | 欧洲多秋播冬性小黑麦，但有春小黑麦 |
| **382** | Spelt (斯佩尔特小麦) | WinterCereal | **否** (未注明冬/春) | 多秋播，但有春播品种 |
| **670** | Japanse haver (日本燕麦) | SpringCereal | **否** (绿肥/覆盖作物) | 通常夏末或春播 |
| **2652** | Granen, overig (其他谷物) | SpringCereal | **否** (未分类泛称) | 未知混杂 |
| **6636** | Naakte haver (裸燕麦) | SpringCereal | **否** (未注明冬/春) | 多春播 |
| **7130** | Rogge, korrelgewas (黑麦籽粒) | WinterCereal | **否** (未注冬黑麦) | 荷兰传统黑麦多秋播冬性，但代码无词头 |

---

## 2. 数量与统计敏感性分析 (Twente AOI)

根据 `runs/20260923_G0L_COMMON_GRID/crop_season_code_sensitivity.csv` 统计，该 7 个代码在 Twente AOI 内部的地块数量与纯像元合格地块（$\ge 2$ 像元）分布如下：

| 年度 | 代码 | 官方名称 | 当前类别 | 全部地块 | 合格地块 ($\ge 2$ 像元) | 严格季节归类 |
|---|---|---|---|---|---|---|
| **2024** | 238 | Haver | SpringCereal | 46 | 17 | Other |
| 2024 | 314 | Triticale | WinterCereal | 94 | 53 | Other |
| 2024 | 382 | Spelt | WinterCereal | 5 | 3 | Other |
| 2024 | 670 | Japanse haver | SpringCereal | 4 | 2 | Other |
| 2024 | 2652 | Granen, overig | SpringCereal | 192 | 7 | Other |
| 2024 | 6636 | Naakte haver | SpringCereal | 2 | 1 | Other |
| 2024 | 7130 | Rogge, korrelgewas | WinterCereal | 119 | 26 | Other |
| **2025** | 238 | Haver | SpringCereal | 34 | 8 | Other |
| 2025 | 314 | Triticale | WinterCereal | 94 | 45 | Other |
| 2025 | 382 | Spelt | WinterCereal | 4 | 4 | Other |
| 2025 | 670 | Japanse haver | SpringCereal | 7 | 4 | Other |
| 2025 | 2652 | Granen, overig | SpringCereal | 192 | 16 | Other |
| 2025 | 6636 | Naakte haver | SpringCereal | 3 | 1 | Other |
| 2025 | 7130 | Rogge, korrelgewas | WinterCereal | 141 | 44 | Other |

---

## 3. 规范裁决与实验协议 (Semantic Decision)

为贯彻**“官方来源第一、不凭经验主观猜测”**的审计底线，确立以下双轨协议：

### 决策 A：主实验基准 (Strict Season Protocol, 推荐主线)
1. **WinterCereal (冬谷物)**：严格仅包含代码 **233** (Wintertarwe) 与 **235** (Wintergerst)。
   - 2024 年合格地块数：**185** 块（原 267 块扣除 82 块存疑地块）
   - 2025 年合格地块数：**311** 块（原 404 块扣除 93 块存疑地块）
2. **SpringCereal (春谷物)**：严格仅包含代码 **234** (Zomertarwe) 与 **236** (Zomergerst)。
   - 2024 年合格地块数：**214** 块（原 241 块扣除 27 块存疑地块）
   - 2025 年合格地块数：**282** 块（原 311 块扣除 29 块存疑地块）
3. **Other (其他)**：吸收全部 7 个存疑代码（238, 314, 382, 670, 2652, 6636, 7130）。
4. **科学收益**：主类别定义纯净，物候特征严格一致（冬谷物均经历越冬分蘖期，春谷物均在 3–4 月播种出苗），彻底杜绝由于物种混淆造成的物候曲线双峰或异常抖动。且即使剔除存疑代码，冬春两类在两年中合格样本均在 180 块以上，统计量完全充裕。

### 决策 B：敏感性分析对比分支 (Permissive Sowing Season Protocol)
- 保留先前映射方案作为独立的 Sensitivity Branch。
- 在论文/技术报告消融实验中，评估将小黑麦/黑麦/燕麦并入冬/春谷物后，分类器在时间相干性与后向散射时序上的响应变异度。

---

## 4. 结论与冻结状态

- **冻结决议**：主实验划分正式采纳 **决策 A (Strict Season Protocol)**。
- **派生映射表**：将在 `crop_code_mapping_strict_2024_2025.csv` 中固化该版本，并在空间切块划分中分别输出基准与敏感性对照指标。
