# CoDefenderC3 — 模型与论文参考文档

**CoDefenderC3: Multi-Model Collaborative Malware Concept Drift Defense via Spectral Drift Decomposition and LLM-Assisted Analysis**

本项目实现了论文中描述的 **33个异构子模型**，覆盖 PE 文件 **8大分析维度(A-H)、48项特征、11个视角组**，构成对恶意软件的全方位多粒度感知能力。

所有特征由统一特征提取层 (`feature_extraction/unified.py`) 管理，定义了 15 种 `FeatureType`。模型配置由中心注册表 (`model_registry.py`) 统一维护。

---

## 系统架构总览 (论文 §4.1)

```
CoDefenderC3/
├── feature_extraction/
│   ├── unified.py              ← 统一特征提取层 (15种 FeatureType, §4.2.1)
│   └── extract_feature.py      ← BinaryNinja 底层 48 项特征提取器 (附录 C)
├── model_registry.py           ← 中心注册表 (模型→特征→参数→损失→保存路径)
├── models/                     ← 31个模型目录 (模型名_年份/)
│   └── [33个子模型, 按 Table 4 组织]
├── models_running/             ← 36个一键训练/预测脚本
├── core/                       ← SDD谱漂移分解引擎 + 基线 + LLM
└── run_all.py                  ← 一键全流程
```

四层架构 (论文 Fig.1):
- Layer 1: 异构多模型集成层 — 33模型×11视角组 (§4.2)
- Layer 2: SDD谱漂移分解引擎 — K×K分歧矩阵→谱分解 (§4.3, §5)
- Layer 3: LLM辅助分析层 — ATT&CK映射+策略推荐 (§4.4)
- Layer 4: 自适应推断层 — DSIR+DACP+策略选择 (§6.2-§6.3)

---

## V1: 字节级 (Byte-Level)

攻击者技术: 加壳、加密、字节填充 → T1027.002, T1027.001
输入: `FeatureType.RAW_BYTES`

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 1 | MalConv | `MalConv_2017/` | 1D-CNN (门控卷积) | (B,1) Sigmoid | BCELoss |
| 2 | MalConv2 | `MalConv2_2021/` | 1D-CNN+GCT+LowMemConv | (B,2)+三元组 | CE |
| 3 | ByteTransformer | `ByteTransformer_2023/` | 4层Transformer-Encoder | (B,2) logits | CE |

1. **MalConv** — Raff et al. 2018. "Malware Detection by Eating a Whole EXE." AAAI-WS. https://arxiv.org/abs/1710.09435
2. **MalConv2 (MalConvGCT)** — Raff et al. 2021. "Learning the PE Header, Malware Detection with Minimal Domain Knowledge." AISec. https://arxiv.org/abs/1709.01471
3. **ByteTransformer** — 基于 Vaswani et al. 2017. "Attention Is All You Need." NeurIPS. https://arxiv.org/abs/1706.03762

---

## V2: 字节统计 (ByteStat)

攻击者技术: 字节分布操纵、熵伪装 → T1027.005
输入: `FeatureType.EMBER_BYTEHIST` (512d)

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 4 | SaxeBerlinDNN | `SaxeBerlinDNN_2015/` | MLP(1024,512,256)+BN | (B,1) Sigmoid | BCE |
| 5 | EmberDNN | `EmberDNN_2018/` | MLP(300,300)+BN | (B,1) Sigmoid | BCE |
| 6 | NatarajKNN | `NatarajKNN_2011/` | kNN(k=5) | fit/predict | 纯Python |

4. **SaxeBerlinDNN** — Saxe & Berlin 2015. "Deep Neural Network Based Malware Detection Using Two-Dimensional Binary Visualization." IEEE ISI. https://doi.org/10.1109/MALWARE.2015.7413680
5. **EmberDNN** — Anderson & Roth 2018. "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models." https://arxiv.org/abs/1804.04637
6. **NatarajKNN** — Nataraj et al. 2011. "Malware Images: Visualization and Automatic Classification." VizSec. https://doi.org/10.1145/2016904.2016908. 纯Python kNN(k=5, 欧氏距离, 1/d加权投票)

---

## V3: PE结构 (PE-Struct)

攻击者技术: PE头部伪装、节区操纵 → T1036.005
输入: `FeatureType.EMBER_PE_STRUCT` (327d)

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 7 | DREBIN | `DREBIN_2014/` | Linear SVM (SGD+Platt) | fit/predict | 纯Python |
| 8 | DL4MD | `DL4MD_2016/` | Stacked Denoising AE | (B,1) Sigmoid | BCE |
| 9 | PEMiner | `PEMiner_2009/` | Random Forest (200棵CART) | fit/predict | 纯Python |

7. **DREBIN** — Arp et al. 2014. "DREBIN: Effective and Explainable Detection of Android Malware in Your Pocket." NDSS. https://www.ndss-symposium.org/ndss2014/. 纯Python SVM(SGD铰链损失+L2+Platt Scaling)
8. **DL4MD** — Hardy et al. 2016. "DL4MD: A Deep Learning Framework for Intelligent Malware Detection." DMIN. https://worldcomp-proceedings.com/proc/p2016/DMIN16_Contents.html 含DenoisingAutoencoder+pretrain_layer()
9. **PEMiner** — Shafiq et al. 2009. "PE-Miner: Mining Structural Information to Detect Malicious Executables in Realtime." RAID. https://doi.org/10.1007/978-3-642-04342-0_7 纯Python RF(CART+Bootstrap+基尼)

---

## V4: 导入/导出 (Import)

攻击者技术: 动态API解析、DLL注入 → T1027.007, T1055.001
输入: `FeatureType.EMBER_IMPORT` (1280d) / `FeatureType.EMBER_FULL` (2381d)

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 10 | DrebinImport | `DREBIN_2014/` | Linear SVM (同V3) | fit/predict | 纯Python |
| 11 | ALOHANet | `ALOHANet_2020/` | MLP(512,256,128)+BN+L2 | (B,1) Sigmoid | BCE |
| 12 | EmberGBDT | `EmberGBDT_2018/` | GBDT (Friedman 2001) | fit/predict | 纯Python |

11. **ALOHANet** — Aghakhani et al. 2020. "When Malware is Packin' Heat; Limits of ML Classifiers Based on Static Analysis Features." USENIX Security. https://api.semanticscholar.org/CorpusID:211268965
12. **EmberGBDT** — Anderson & Roth 2018. (同#5). 纯Python GBDT: F₀=log(p/(1-p)), r=y-σ(F), γ=Σr/Σ(p(1-p))

---

## V5: 字符串 (String)

攻击者技术: 字符串加密/混淆 → T1027, T1140
输入: `FeatureType.EMBER_STRING` (104d)

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 13 | RaffFeatureNet | `RaffFeatureNet_2018/` | MLP(256,128) | (B,1) Sigmoid | BCE |
| 14 | DrebinString | `DREBIN_2014/` | Linear SVM (同V3) | fit/predict | 纯Python |

13. **RaffFeatureNet** — Raff et al. 2018. (同#1 MalConv论文, Section 4.1特征基线)

---

## V6: 操作码 (Opcode)

攻击者技术: 控制流平坦化、不透明谓词 → T1027.004
输入: `FeatureType.OPCODE_SEQ` / `FeatureType.FUNC_EMBEDDINGS`

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 15 | OpcodeLSTM | `OpcodeLSTM_2017/` | BiLSTM(64,128,2层) | (B,2) logits | CE |
| 16 | OpcodeTransformer | `OpcodeTransformer_2020/` | 4层Transformer(nhead=4) | (B,2) logits | CE |
| 17 | OpcodeStatNet | `MicrosoftBIG_2016/` | XGBoost (二阶泰勒) | fit/predict | 纯Python |
| 18 | AsmEmbedNet | `Asm2Vec_2019/` | PV-DM+注意力LSTM | (B,2) logits | CE |

1.  **OpcodeLSTM** — McLaughlin et al. 2017. "Deep Android Malware Detection." IEEE TIFS. https://doi.org/10.1145/3029806.3029823
2.  **OpcodeTransformer** — Ren et al. 2020. "End-to-end Malware Detection for Android IoT Devices Using Deep Learning." Ad Hoc Networks, 101, 102098. https://doi.org/10.1016/j.adhoc.2020.102098
3.  **OpcodeStatNet** — Ahmadi et al. 2016. "Novel Feature Extraction, Selection and Fusion for Effective Malware Family Classification." ACM CODASPY. https://doi.org/10.1145/2857705.2857713. 纯Python XGBoost(w=-G/(H+λ))
4.  **AsmEmbedNet** — Ding et al. 2019. "Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search." IEEE S&P. https://doi.org/10.1109/SP.2019.00003

---

## V7: 图拓扑 (Graph)

攻击者技术: 控制流变异 → T1027.004, T1027.009
输入: `FeatureType.GRAPH_STAT` (22d) 或 `FeatureType.PYG_GRAPH`

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 19 | MalGraph | `MalGraph_2022/` | HierarchicalGNN | (B,1) Sigmoid | BCE |
| 20 | CFG-GAT | `CFG_GAT_2019/` | 3层GAT(heads=[8,8,1]) | (B,2) logits | CE |
| 21 | CFG-GCN | `GCN_2017/` | 3层GCN(64→32→16) | (B,2) logits | CE |
| 22 | CFG-DGCNN | `DGCNN_2018/` | GCN→SortPool→1D-CNN | (B,2) logits | CE |
| 23 | CallGraph-GNN | `CallGraphGNN_2019/` | SAGEConv+图注意力聚合 | (B,2) logits | CE |
| 24 | GraphStatNet | `GraphStatNet_2022/` | ResidualMLP(128)×2 | (B,2) logits | CE |

20. **CFG-GAT** — Yan et al. 2019. "Classifying Malware Represented as Control Flow Graphs using Deep Graph CNN." 2019 49th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN), pp. 52–63. https://doi.org/10.1109/DSN.2019.00020
21. **CFG-GCN** — Kipf & Welling 2017. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR. https://arxiv.org/abs/1609.02907
22. **CFG-DGCNN** — Zhang et al. 2018. "An End-to-End Deep Learning Architecture for Graph Classification." AAAI. https://doi.org/10.1609/aaai.v32i1.11782
23. **CallGraph-GNN** — 基于 Li et al. 2019. "Graph Matching Networks for Learning the Similarity of Graph Structured Objects." ICML. https://arxiv.org/abs/1904.12787

---

## V8: 可视化 (Visual)

攻击者技术: 大规模字节变换 → T1027 (联合效应)
输入: `GRAY_IMAGE` / `COLOR_IMAGE` / `MARKOV_IMAGE` / `ENTROPY_IMAGE`

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 25 | IMCFN | `IMCFN_2020/` | VGG16+FineTune(jet) | (B,C) logits | CE |
| 26 | InceptionV3 | `InceptionV3_2020/` | InceptionV3(pretrained) | (B,C) logits | CE |
| 27 | GrayscaleCNN | `GrayscaleCNN_2011/` | Gabor滤波器+MLP | (B,2) logits | CE |
| 28 | ColorCNN | `ColorCNN_2019/` | VGG-style 3-Block | (B,2) logits | CE |
| 29 | MarkovCNN | `MarkovCNN_2018/` | 5×5大核+4层CNN | (B,2) logits | CE |
| 30 | EntropyMapCNN | `EntropyMapCNN_2018/` | 7×7+5×5大核CNN | (B,2) logits | CE |

25. **IMCFN** — Vasan et al. 2020. "IMCFN: Image-based Malware Classification using Fine-tuned CNN." Computer Networks. https://doi.org/10.1016/j.comnet.2020.107138
26. **InceptionV3** — Szegedy et al. 2016. "Rethinking the Inception Architecture for Computer Vision." CVPR. https://arxiv.org/abs/1512.00567
29. **MarkovCNN** — Ni et al. 2018. "Malware Identification Using Visualization Images and Deep Learning." Computers & Security. https://doi.org/10.1016/j.cose.2018.04.005
30. **EntropyMapCNN** — Conti et al. 2018. "Visual Reverse Engineering of Binary and Data Files." VizSec. https://doi.org/10.1007/978-3-540-85933-8_1

---

## V9: 哈希 (Hash)

攻击者技术: 段哈希多样化 → T1036
输入: `FeatureType.EMBER_BYTEHIST` (512d)

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 31 | HashEmbedNet | `HashEmbedNet_2019/` | 4种Embedding+MLP | (B,2) logits | CE |

31. **HashEmbedNet** — Pagani et al. 2019. "Towards Interpretable and Robust ML-based Malware Detection." CCS Workshop. https://doi.org/10.1145/3338466

---

## V10: 元数据 (Metadata)

输入: `FeatureType.EMBER_METADATA` (158d)

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 32 | MetadataNet | `MetadataNet_2009/` | MLP(128,64) | (B,1) Sigmoid | BCE |
| 33 | ResourceNet | `MicrosoftBIG_2016/` | Random Forest(100棵) | fit/predict | 纯Python |

---

## V11: 全特征集成 (Full)

输入: `FeatureType.EMBER_FULL` (2381d)

| # | 子模型 | 目录 | 架构 | 输出 | 损失 |
|---|--------|------|------|------|------|
| 34 | Ember (NDF) | `Ember_2018/` | Neural Decision Forest | (B,1) Sigmoid | BCE |
| 35 | EmberGBDT | `EmberGBDT_2018/` | GBDT (同V4, 全特征) | fit/predict | 纯Python |

34. **Ember NDF** — Anderson & Roth 2018 + Kontschieder et al. 2015. "Deep Neural Decision Forests." ICCV.

---

## 实现类型: 26个nn.Module + 5个纯Python (不使用sklearn)

## 论文引用的7种基线 (§7.8)

| 基线 | 会议 | 年份 | 技术路线 |
|:---|:---|:---:|:---|
| Transcend | USENIX Security | 2017 | 共形评估器p值 |
| CADE | USENIX Security | 2021 | 对比自编码器 |
| HCC | USENIX Security | 2023 | 层次化对比学习 |
| DroidEvolver | IEEE EuroS&P | 2019 | 伪标签自更新 |
| DREAM | ACM CCS | 2025 | 概念增强AE |
| MADCAT | ICML Workshop | 2025 | 掩码AE测试时训练 |
| Ens.Disagree | — | — | 朴素分歧阈值 |

## SDD四个定理 (§5): 机制可辨识性(99%) + 检测保证(Type-I≤α) + 适应质量(ρ=0.960) + 共形覆盖(+6.0%)
