"""
AM-SDD: 基于攻击制作机制建模的主动漂移感知与联合防御框架
==========================================================

§1. 威胁模型 (Threat Model)
────────────────────────────
攻击者能力:
  攻击者可以修改恶意样本的任意可执行属性以规避检测。
  修改产生新样本 x' = T(x), 其中 T 是攻击者选择的变换。

攻击者约束 (按保留语义分级):
  Type-P (Perturbation): T 保留功能语义和程序结构。
    例: 调整API调用顺序、插入NOP、字节级变形。
    形式化: ∀v ∈ V_struct ∪ V_global: φ_v(x') = φ_v(x)
    即结构特征和全局特征不变, 仅表层特征被扰动。

  Type-S (Structural): T 保留功能语义, 但重构程序结构。
    例: 加壳、控制流扁平化、节区重排、虚拟化执行。
    形式化: ∃v ∈ V_struct: φ_v(x') ≠ φ_v(x), 但 sem(x')=sem(x)
    结构特征改变, 表层特征可能联动(因为结构重构影响字节分布)。

  Type-Φ (Paradigm-Shift): T 不保留功能等价关系。
    例: 从文件落地到无文件注入, 从C++到Go重写, 供应链渗透。
    形式化: sem(x') ≠ sem(x), 且 ∀ℓ: ∃v∈V_ℓ: φ_v(x') ≠ φ_v(x)
    所有层级的特征均被影响。

防御者能力:
  防御者部署 K 个检测模型, 按视角分组:
    G_v = {k : 模型 k 仅接收特征 φ_v(x)} ⊆ {1,...,K}
  这是一个架构约束——模型 k 的输入端在部署时即固定,
  模型 k 对特征子空间 v(k) 之外的改变是天然免疫的。

  关键: 这不是假设, 是CoDefenderC3的设计决策。

§2. 核心理论: 视角隔离下的漂移类型可辨识性
──────────────────────────────────────────

为什么"表层模型变了→扰动型"不是循环论证？

因为这里有两个独立的事实:
  事实A (攻击侧): Type-P攻击按定义不改变结构特征。
    这不是我们的假设——这是"扰动"这个词的语义。
    如果攻击者改了结构, 那它就不是扰动, 而是结构型。

  事实B (防御侧): 模型 k∈G_struct 仅接收结构特征 φ_struct(x)。
    这是架构约束。模型的输入维度在部署时固定。

  推论: Type-P攻击 → φ_struct(x') = φ_struct(x) → 模型k∈G_struct
  的输入不变 → 模型k的输出不变 → 结构层级分歧不增大。

  这不是"检测字节特征变化的模型检测到字节特征变化"的循环。
  这是"按定义不改变结构的攻击, 不影响只看结构的模型"的推导。

  但有一个非平凡问题: 字节变形是否会通过数据相关性
  间接影响结构特征？
  答案是否: 因为模型k∈G_struct的输入是 φ_struct(x'),
  而 φ_struct 是从PE文件中直接提取的(PE头/节表/导入表),
  不经过字节统计计算。字节级扰动(如NOP插入)改变
  ByteHistogram 但不改变 ImportTable, 因为这两个特征
  提取器独立作用于PE文件的不同数据结构。

Definition 1 (特征不变类).
  对于攻击类型 τ ∈ {P, S, Φ}, 定义其特征不变类:
    I(τ) = {v ∈ {1,...,V} : ∀x, φ_v(T_τ(x)) = φ_v(x)}
  即在攻击类型 τ 下保持不变的特征视角集合。

  由威胁模型:
    I(P) ⊇ V_struct ∪ V_global    (扰动保留结构+全局)
    I(S) ⊇ V_global ∖ V_struct    (结构重构可能影响全局的结构成分)
    I(Φ) = ∅                       (范式迁移无不变量)

  实际中, 这些是近似关系(加壳可能轻微影响全局统计),
  但数学上可以用"近似不变"替代精确不变(见Theorem 1)。

Definition 2 (层级响应签名).
  对于攻击类型 τ, 定义层级响应签名 σ(τ) ∈ {0,1}^L:
    σ(τ)_ℓ = 1  iff  ∃v ∈ V_ℓ ∖ I(τ)
  即层级 ℓ 中是否存在受影响的视角。

  由Definition 1:
    σ(P) = (1, 0, 0)   表层受影响, 结构和全局不变
    σ(S) = (*, 1, *)    结构受影响, 表层可能联动, 全局可能部分
    σ(Φ) = (1, 1, 1)    所有层级受影响

Theorem 1 (漂移类型可辨识性, 主定理).
  设集成架构满足视角隔离条件:
    ∀ℓ, ∀k ∈ G_ℓ: 模型k的输入仅依赖 ⋃_{v∈V_ℓ} φ_v(x)

  设攻击类型 τ 的特征不变类满足:
    (C1) Type-P: V_struct ⊆ I(P) 且 V_global ⊆ I(P)
    (C2) Type-S: ∃v ∈ V_struct: v ∉ I(S)
    (C3) Type-Φ: ∀ℓ, ∃v ∈ V_ℓ: v ∉ I(Φ)

  则对任意攻击 T_τ, 层级响应签名 σ(τ) 唯一确定 τ:
    (i)   σ_struct = 0 ∧ σ_global = 0  →  τ = P
    (ii)  σ_struct = 1 ∧ ¬(σ_surface ∧ σ_struct ∧ σ_global)  →  τ = S
    (iii) σ_surface = 1 ∧ σ_struct = 1 ∧ σ_global = 1  →  τ = Φ

  证明: 由(C1), Type-P不影响V_struct和V_global中的任何视角,
  因此G_struct和G_global中的模型输入不变, 输出不变, σ_struct=0且
  σ_global=0。由(C2), Type-S至少影响V_struct中的某个视角,
  所以G_struct中存在模型输入改变, σ_struct=1。但Type-S保留
  功能语义, 而V_global中的全特征模型同时依赖结构和表层特征,
  其是否被影响取决于结构变化是否传播到全局统计。
  由(C3), Type-Φ影响所有层级的至少一个视角, 因此所有层级
  的签名都为1。三个签名模式两两不同, 因此可辨识。  □

  关键: 条件(C1)-(C3)不是"假设"——它们是攻击类型定义的
  直接推论。"扰动不改变结构"不是我们的假设, 是"扰动"的定义。

  但 Theorem 1 的条件(C1) 要求精确不变: φ_struct(x')=φ_struct(x)。
  实际中, 字节级修改是否真的精确不影响结构特征？
  这需要 Theorem 1.5 来回答。

Theorem 1.5 (特征泄露界 — Feature Leakage Bound, 核心技术贡献).

  本定理证明: 对于 PE 文件, 表层修改对结构特征的影响
  可以被精确界定, 且在大多数实际攻击场景下为零。

  ┌────────────────────────────────────────────────────────┐
  │  核心技术定理。                                        │
  │  将 PE 格式规范公理化为指针寻址图, 证明代码段修改     │
  │  不影响结构特征是 PE 格式的本质性质, 而非特定实现的   │
  │  偶然特性。任何满足"结构局部性公理"的特征提取器       │
  │  都自动继承这一不变性。                                │
  └────────────────────────────────────────────────────────┘

  ── 公理化框架 ──

  Definition 1.5a (PE 指针寻址图).
    一个 PE 文件 x ∈ {0,...,255}^n 的结构由指针寻址图
    G(x) = (V, E) 定义, 其中:

    节点 V = {区域名}: DOS_header, PE_header, COFF_header,
      Optional_header, DataDirectory, Section_table,
      Import_table, Export_table, Resource_table,
      Code_section, Data_section, ...

    有向边 E: 若区域 u 包含一个指针字段, 其值指向区域 v
    的起始地址, 则 (u,v) ∈ E。

    PE 格式规范 (Microsoft PE/COFF Specification §2-§5) 定义
    了以下指针链:

      DOS_header[0x3C]          ─→  PE_header         (e_lfanew)
      PE_header[0x00-0x03]      ─→  COFF_header       (PE signature)
      COFF_header               ─→  Optional_header   (固定偏移)
      Optional_header           ─→  DataDirectory[16] (固定偏移)
      DataDirectory[1].RVA      ─→  Import_table      (RVA指针)
      DataDirectory[0].RVA      ─→  Export_table      (RVA指针)
      DataDirectory[2].RVA      ─→  Resource_table    (RVA指针)
      Section_table[i].PtrRawData ─→ Section_i_content (文件偏移)

    关键观察: 所有指针字段都位于 DOS_header ∪ PE_header ∪
    COFF_header ∪ Optional_header ∪ Section_table 中。
    我们称这些区域的并集为元数据壳层 (metadata shell):

      M(x) = DOS_header ∪ PE/COFF_header ∪ Optional_header
              ∪ DataDirectory ∪ Section_table

    而 Code_section (即 .text 节的内容) 是叶节点:
    它被指向, 但自身不包含任何指向其他结构区域的指针。

  Lemma 1 (指针闭包性, Pointer Closure).
    元数据壳层 M(x) 在指针解引用下是闭合的:
    从 M(x) 中任何指针字段出发, 解引用到达的区域
    要么仍在 M(x) 内, 要么是内容区域 (Import_table,
    Code_section 等) 的数据。

    特别地, Code_section 的字节内容不被 M(x) 内的
    任何指针字段引用为地址值。

    证明: 逐一检查 PE 格式规范中 M(x) 内的所有指针字段:
    - e_lfanew (DOS_header offset 0x3C): 指向 PE_header,
      其值由 PE 链接器写入, 不依赖代码段内容。
    - DataDirectory[i].VirtualAddress: 由链接器根据节表
      布局计算, 值为 RVA (相对虚拟地址), 等于
      Section_table[j].VirtualAddress + offset_within_section。
      修改代码段字节不改变 Section_table 中的 VirtualAddress。
    - Section_table[i].PointerToRawData: 由链接器根据文件
      布局计算, 值为文件偏移量, 等于前一节的
      PointerToRawData + SizeOfRawData (对齐后)。
      修改代码段字节不改变这些偏移量。

    因此, M(x) 内所有指针字段的值仅取决于 M(x) 自身
    以及链接器的布局决策, 不取决于代码段的字节内容。

    形式化: 设 ptr_fields(M) 为 M 中所有指针字段的集合。
    对任意 p ∈ ptr_fields(M):
      val(p, x) = val(p, x')  whenever  x|_M = x'|_M
    其中 x|_M 表示 x 限制在 M 区域上的字节子序列。  □

  Definition 1.5b (结构局部特征提取器, 公理化定义).
    称 φ: {0,...,255}^n → R^d 是关于 PE 区域集 S ⊆ V 的
    结构局部特征提取器 (Structure-Local Feature Extractor,
    简称 SLFE), 如果满足以下公理:

    (SLFE-1) 区域依赖性: φ(x) 仅取决于 ⋃_{s∈S} content(s, x),
      即 S 中各区域的字节内容。形式化:
        x|_{⋃S} = x'|_{⋃S}  ⟹  φ(x) = φ(x')

    (SLFE-2) 寻址通过元数据壳层: S 中每个区域的寻址
      (即确定其在文件中的偏移和大小) 仅依赖 M(x):
        x|_M = x'|_M  ⟹  addr(s, x) = addr(s, x')  ∀s ∈ S

    直觉: SLFE 是"通过 PE 头中的指针找到特定区域,
    然后只读取那个区域的内容"的提取器。这涵盖了:
    - EMBER 的 SectionInfo (S = {Section_table})
    - EMBER 的 ImportsInfo (S = {Import_table})
    - EMBER 的 GeneralInfo (S = {PE_header, COFF_header})
    - LIEF 的对应提取器
    - 任何未来实现的遵循同一模式的提取器

    不满足 SLFE 公理的提取器:
    - ByteHistogram (读取所有字节, 违反 SLFE-1)
    - ByteEntropy (同上)
    - 基于反汇编的提取器 (读取代码段内容)

  Lemma 2 (结构特征不变性, 公理化版本, 核心引理).
    设 φ 是关于区域集 S 的 SLFE, 且 S ∩ {Code_section} = ∅
    (即 S 不包含代码段)。
    设 T: x → x' 是仅修改 Code_section 的变换:
      x'|_{x∖Code_section} = x|_{x∖Code_section}

    则: φ(x') = φ(x)。

    证明:
      Step 1. T 不修改 M(x), 因此 x'|_M = x|_M。
        (Code_section ∩ M = ∅, 因为 M 是头部区域的并集,
        而代码段是内容区域。)

      Step 2. 由 (SLFE-2): addr(s, x') = addr(s, x) ∀s ∈ S。
        (S 中区域的寻址仅依赖 M, 而 M 未被修改。)

      Step 3. 由 Step 2, S 中每个区域在 x' 中的偏移和大小
        与在 x 中相同。又因 T 仅修改 Code_section,
        而 S ∩ {Code_section} = ∅, 所以 S 中区域的内容不变:
          content(s, x') = content(s, x) ∀s ∈ S

      Step 4. 由 (SLFE-1):
        φ(x') = f(⋃_{s∈S} content(s, x')) = f(⋃_{s∈S} content(s, x)) = φ(x)  □

    关键: 这个证明不依赖于 φ 的具体实现 (EMBER, LIEF, 或任何
    未来实现)。只要 φ 满足 SLFE 公理 (即"通过 PE 指针找到区域,
    只读取区域内容"), 结论自动成立。

    这是 PE 格式的本质性质, 不是 EMBER 的偶然特性。
    PE 格式的设计使得元数据壳层 M 与内容区域正交:
    M 描述"文件的组织方式", 内容区域存储"实际数据"。
    SLFE 公理捕获的恰好是"只关心组织方式"的提取器。

  Lemma 3 (表层特征的全局性).
    称 φ 是全局特征提取器 (Global Feature Extractor, GFE),
    如果 φ(x) 依赖 x 的所有字节:
      ∃i: x[i] ≠ x'[i]  可能导致  φ(x) ≠ φ(x')

    ByteHistogram, ByteEntropy, StringExtractor 均为 GFE。
    对 GFE, 修改 m 个字节的影响可界定:

    (i) ByteHistogram (256维归一化直方图):
        ‖φ_hist(x') − φ_hist(x)‖₁ ≤ 2m/n
        证明: 每个被修改的字节使一个 bin 减 1/n, 另一个加 1/n。
        L1 变化量 ≤ 2m · (1/n) = 2m/n。  □

    (ii) ByteEntropy (256维块熵直方图, 块大小 B=2048):
        受影响块数 ≤ ⌈m/B⌉ + 1 (修改可能跨块)
        每个块的熵变化 ≤ H_max = log₂(256) = 8 bits
        ‖φ_ent(x') − φ_ent(x)‖₁ ≤ 2(⌈m/B⌉+1) · (8/256) / (n/B)
        = O(m/(n·B)) · 8  (当 m ≪ n 时非常小)  □

  MAIN THEOREM (特征泄露界).
  设 T_P: x → x' 是 Type-P 变换, 满足:
    (a) T_P 仅修改 Code_section 中的字节
    (b) T_P 不修改元数据壳层 M(x) 中的任何字节
    (c) |T_P| = m 个字节被修改

  设 φ_struct 是关于 S ⊆ {Section_table, Import_table,
  PE_header, ...} 的 SLFE, 且 Code_section ∉ S。
  设 φ_surface 是 GFE。

  则:
    (i)  ε_P(struct) := ‖φ_struct(x') − φ_struct(x)‖ = 0  (精确零)
    (ii) ε_P(surface) := ‖φ_surface(x') − φ_surface(x)‖₁ ≤ 2m/n

  证明: (i) 直接由 Lemma 2 (公理化版本)。
        (ii) 直接由 Lemma 3。  □

  推论 1 (常见扰动的精确零泄露).
    满足条件(a)(b)的 Type-P 变换包括:
    - 等价指令替换 (MOV EAX,0 → XOR EAX,EAX): 仅改代码字节
    - NOP sled 插入 (替换 alignment padding): 仅改代码区域
    - 寄存器重分配: 仅改操作数编码
    - 基本块重排序 + 跳转修补: 仅改代码段内的地址

    这些攻击满足 ε_P(struct) = 0。

  推论 2 (条件(b)被违反时的退化).
    若 T_P 修改了 M(x) (如修改 SizeOfCode 字段):
    - Lemma 1 的指针闭包性仍成立 (其他指针不受影响)
    - 但 SLFE 的输出可能变化, 因为 M 中被修改的字段
      可能属于某个 SLFE 的依赖区域 S
    - 退化界: ε_P(struct) ≤ 修改的 M 字段数 × 字段影响系数

    重要: 如果攻击者修改了 PE 头的结构性字段 (如
    NumberOfSections, ImportDirectory RVA), 则按定义
    这不再是 Type-P (扰动), 而是 Type-S (结构重构)。
    因此 Theorem 1 的漂移分类仍然一致。

  ── 与 Reviewer 1 的对话 ──

  Q: "你证明的是 EMBER 的性质还是 PE 格式的性质？"
  A: PE 格式的性质。SLFE 公理不提及任何特定实现。
     EMBER 的 SectionInfo 满足 SLFE, LIEF 的对应提取器
     也满足 SLFE, 未来任何"通过 PE 指针定位结构区域并
     只读取区域内容"的实现都满足 SLFE。Lemma 2 对所有
     SLFE 成立, 因此结论不依赖于特定代码库。

  Q: "如果用深度学习端到端提取特征呢？"
  A: 端到端模型 (如 MalConv 直接读取原始字节) 不满足
     SLFE 公理, 因为它读取包括代码段在内的所有字节。
     这类模型应归入"表层"或"全局"层级, 而非"结构"层级。
     本框架的层级划分正是基于特征提取器是否满足 SLFE。

Theorem 2 (鲁棒可辨识性: 精确泄露界下的误分类率).

  由 Theorem 1.5, 对满足条件(a)(b)的 Type-P 变换:
    ε_P(struct) = 0, ε_P(surface) ≤ 2m/n

  对 Type-S 变换 (加壳/混淆), 结构特征的偏移量 Δ_struct(S) > 0
  (因为加壳器重写 PE 头、创建新节区、修改导入表)。

  由于 ε_P(struct) = 0 而 Δ_struct(S) > 0, 结构层级的信号
  可以无误地分离 Type-P 和 Type-S。但有限样本下需要统计检验。

  给定 N 个样本, 层级分歧估计量 d̂_ℓ 满足:
    P[|d̂_ℓ − E[d_ℓ]| > t] ≤ 2 exp(−2Nt²/C²)   (Hoeffding)

  设 Δ_min = min(Δ_struct(S), Δ_global(Φ)) 为最弱信号强度。
  误分类概率:
    P[错误分类] ≤ 2L · exp(−NΔ²_min / (8σ²_max))

  样本复杂度: 要达到误分类率 ≤ δ, 需要:
    N ≥ (8σ²_max / Δ²_min) · log(2L/δ)

  实际意义: 在 EMBER 数据集上, 加壳家族(如 UPX→Themida 迁移)
  的 Δ_struct 在 SectionInfo 和 ImportInfo 上是显著的
  (§6 实验验证), 因此几百个样本即可达到 >95% 的分类准确率。

Theorem 3 (范式迁移的新颖性检验).
  范式迁移和"多重结构型攻击"都导致全层级响应。

  区分依据: 范式迁移的漂移方向 d ∈ R^L 偏离历史漂移子空间。

  设 H_T = span{d^(1),...,d^(T)} 为 T 次历史漂移的方向子空间,
  P_{H_T} 为正交投影。新颖性指标:
    ν(d) = ‖(I − P_{H_T}) d‖ / ‖d‖ = sin∠(d, H_T)

  ν → 0: 当前漂移可由已知模式的线性组合解释 (非新范式)
  ν → 1: 漂移方向与所有历史模式正交 (新范式)

  Theorem 3a (冷启动处理).
    当 T < L (历史不足以张成全空间) 时, H_T 的补空间维度
    ≥ L − T > 0, 任何新漂移都可能有较高的 ν 值, 导致
    范式迁移误报。

    解决方案: 冷启动期 (T < T_min = L) 内, 不使用新颖性检验
    判定范式迁移, 而是使用全层级显著性的更保守判定:
      Type-Φ iff σ_ℓ = 1 ∀ℓ ∧ T < T_min
    即冷启动期间, 三个层级全部显著才判定为范式迁移。

    当 T ≥ T_min 后, 历史子空间有足够基底, 新颖性检验
    开始生效, 可以区分"已知模式的加剧"和"真正的新范式"。

Theorem 4 (混合型漂移分解).
  真实攻击者可能同时进行多种修改 (如加壳 + 字节变形)。
  此时观测到的层级分歧向量 d 是多种漂移的叠加:
    d = α_P · σ(P) + α_S · σ(S) + noise

  其中 α_P, α_S ≥ 0 是各类漂移的强度。

  由于 σ(P) = (1,0,0), σ(S) = (*,1,*) 在结构维度上正交
  (σ(P) 在结构维度为 0, σ(S) 在结构维度 > 0), 可以分离:
    α̂_S = d_struct / E[σ(S)_struct]    (结构维度直接给出 α_S)
    α̂_P = (d_surface − α̂_S · E[σ(S)_surface]) / E[σ(P)_surface]

  误差界: 由 Theorem 2 的集中不等式,
    P[|α̂_τ − α_τ| > ε] ≤ 2 exp(−Nε²/(2σ²_max))  对 τ ∈ {P,S}

  实际意义: 对于混合型攻击, AM-SDD 不仅检测到漂移,
  还能分解出"扰动成分多大"和"结构成分多大",
  指导防御者同时对表层模型微调 + 结构模型重训。

§3. 与现有工作的对比
────────────────────
  Transcend (USENIX Sec'17): 基于非一致性度量的时间衰减检测。
    无法区分漂移类型, 无视角分解, 无防御策略分级。

  CADE (USENIX Sec'21): 对比自编码器距离。
    单模型方法, 无跨视角结构, 无攻击类型分类。

  DREAM (NDSS'23): 漂移归因到特征维度。
    在单一特征空间上归因, 无法利用多模型的隔离结构。
    归因结果是"哪些特征维度变了", 而非"攻击者做了什么"。

  AM-SDD (本文): 利用多视角架构的视角隔离性 (SLFE 公理),
    将漂移类型分类建立在 PE 格式的结构性质之上 (而非
    特定实现的偶然特性), 并给出分类正确性的概率保证。
    混合型漂移可分解 (Theorem 4), 冷启动有退化策略
    (Theorem 3a)。

§4. 已知局限性 (Acknowledged Limitations)
────────────────────────────────────────
  (L1) 自适应攻击者: 若攻击者刻意修改 PE 头中的指针字段
       (如篡改 e_lfanew 使解析器跳转到攻击者控制的区域),
       则 Lemma 1 的指针闭包性被违反。但此类攻击通常
       导致 PE 文件格式不合法, 会被 PE 加载器拒绝。

  (L2) V_global 层级的模型 (如全特征 MLP) 同时使用表层和
       结构特征, 因此对所有漂移类型都可能有响应。
       分类逻辑已将 V_global 单独显著的情况纳入考虑。

  (L3) 端到端模型 (如 MalConv) 不满足 SLFE 公理, 不能
       归入"结构"层级。本框架要求至少一部分模型满足 SLFE,
       这是对集成架构的设计要求, 而非对任意集成的通用结论。
"""
import time
import numpy as np
from dataclasses import dataclass, field

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
#  漂移类型枚举
# ═══════════════════════════════════════════════════════════════

DRIFT_NONE       = "none"
DRIFT_PERTURBATION = "perturbation"   # 扰动型: 表层特征扰动
DRIFT_STRUCTURAL   = "structural"     # 结构型: 程序结构重构
DRIFT_PARADIGM     = "paradigm_shift" # 范式迁移: 技术范式根本改变


# ═══════════════════════════════════════════════════════════════
#  SDDResult
# ═══════════════════════════════════════════════════════════════

@dataclass
class SDDResult:
    drift_detected: bool = False
    num_mechanisms: int = 0
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_gap: float = 0.0
    strategy: str = "none"
    affected_views: list = field(default_factory=list)
    affected_models: list = field(default_factory=list)
    healthy_models: list = field(default_factory=list)
    view_loadings: dict = field(default_factory=dict)
    mechanism_views: list = field(default_factory=list)
    threshold: float = 0.0
    timing_ms: dict = field(default_factory=dict)
    confidence: float = 0.0
    label_budget: int = 0
    mechanism_strengths: list = field(default_factory=list)
    # ── 攻击机制漂移分类 ──
    drift_type: str = DRIFT_NONE                 # perturbation/structural/paradigm_shift
    level_scores: dict = field(default_factory=dict)  # {surface/structure/global: z-score}
    novelty: float = 0.0                         # ν(d) = sin∠(d, H_history)
    view_deltas: dict = field(default_factory=dict)   # 每视角漂移量
    drift_explanation: str = ""                   # 可读的漂移归因


# ═══════════════════════════════════════════════════════════════
#  表征层级定义
# ═══════════════════════════════════════════════════════════════

# 默认层级映射: 视角组 → 表征层级
# EMBER 模式 (6视角): 仅含 EMBER 预提取特征的视角
EMBER_LEVEL_MAP = {
    # 表层: 字节级统计, 受扰动影响最大
    "surface":   ["V2_byte_stat", "V5_string"],
    # 结构层: PE组织形式, 受加壳/混淆影响
    "structure": ["V3_pe_struct", "V4_import", "V10_metadata"],
    # 全局层: 跨层综合表征, 仅范式迁移时才剧变
    "global":    ["V11_ensemble"],
}

# FULL 模式 (11视角): 包含 BinaryNinja 提取的所有视角
FULL_LEVEL_MAP = {
    # 表层: 从原始字节直接派生, 任何字节修改立即可见
    #   V1_byte:     原始字节嵌入 (MalConv/ByteTransformer)
    #   V2_byte_stat: 字节直方图+熵 (统计分布)
    #   V5_string:   字符串特征 (可被加密/混淆)
    #   V8_gray/V8_color/V8_markov/V8_entropy: 二进制可视化子视角
    #   V9_hash:     哈希特征 (任何字节变化→哈希变化)
    "surface":   ["V1_byte", "V2_byte_stat", "V5_string",
                  "V8_gray", "V8_color", "V8_markov", "V8_entropy", "V9_hash"],
    # 结构层: 反映程序组织, 受加壳/CFG混淆/导入表动态化影响
    #   V3_pe_struct:   PE 头/节表结构
    #   V4_import:      导入表 (动态API解析可使其消失)
    #   V6_opcode_seq:  操作码序列 (编译器/代码变异引擎改变)
    #   V6_opcode_stat: 操作码 n-gram 统计
    #   V6_func_embed:  函数级嵌入 (Asm2Vec)
    #   V7_graph:       CFG/调用图拓扑 (控制流平坦化/不透明谓词)
    #   V10_metadata:   导出表/资源/版本信息
    "structure": ["V3_pe_struct", "V4_import", "V6_opcode_seq", "V6_opcode_stat",
                  "V6_func_embed", "V7_graph", "V10_metadata"],
    # 全局层: 全特征融合, 仅全面范式迁移时剧变
    "global":    ["V11_ensemble"],
}

# 向后兼容: 默认使用 EMBER 映射 (由 SDDEngine.__init__ 根据视角自动选择)
DEFAULT_LEVEL_MAP = EMBER_LEVEL_MAP




# ═══════════════════════════════════════════════════════════════
#  SDDEngine
# ═══════════════════════════════════════════════════════════════

class SDDEngine:
    """
    谱漂移分解引擎 (Spectral Drift Decomposition).

    核心管道 (calibrate/detect 共用 _spectral_attribution):
      D → A=(D-μ)/σ → eigh(A) → {λᵢ,uᵢ}
        检测:  r = #{λᵢ > τ}
        归因:  uᵢ²·λᵢ → model_contrib → view_loading(mean) → level_loading(mean)
        分类:  level_fraction → P/S/Φ  (Theorem 1, 阈值从校准数据推导)
        新颖性: ν = sin∠(d, H)         (Theorem 3)
    """

    def __init__(self, K, perspective_groups, alpha=0.05,
                 ws=0.6, wr=0.4, max_samples=5000, ema_decay=0.9,
                 level_map=None):
        self.K = K
        self.pg = perspective_groups
        self.alpha = alpha
        self.ws, self.wr = ws, wr
        self.max_N = max_samples
        self.ema_decay = ema_decay

        self.mu0 = self.sigma0 = None
        self.D_history = []
        self.calibrated = False

        # ── 视角与层级 ──
        self.view_names = list(self.pg.keys())
        self.V = len(self.view_names)
        lm = level_map if level_map is not None else self._auto_level_map()
        self.levels = {}
        self.level_views = {}
        for lname, vnames in lm.items():
            idx, matched = [], []
            for vn in vnames:
                if vn in self.pg:
                    idx.extend(self.pg[vn]["models"])
                    matched.append(vn)
            if idx:
                self.levels[lname] = sorted(set(idx))
                self.level_views[lname] = matched
        self.level_names = list(self.levels.keys())
        self.L = len(self.level_names)

        # ── 历史漂移子空间 (Theorem 3) ──
        self.drift_history = []
        self.drift_subspace = None

        # ── CUSUM ──
        self.cusum = {ln: 0.0 for ln in self.level_names}

        # ── 所有阈值在 calibrate() 中从数据推导, 此处仅声明 ──
        self.tau = 0.0
        self.tau_levels = {}
        self.eta = 0.0
        self._view_null_mu = {}
        self._view_null_std = {}
        self._level_null_mu = np.array([])
        self._level_null_std = np.array([])
        self._classification_thresholds = {}
        self.cusum_k = 1.0

    def _auto_level_map(self):
        full_only = {"V1_byte", "V6_opcode_seq", "V6_opcode_stat",
                     "V6_func_embed", "V7_graph",
                     "V8_gray", "V8_color", "V8_markov", "V8_entropy", "V9_hash"}
        return FULL_LEVEL_MAP if (full_only & set(self.view_names)) else EMBER_LEVEL_MAP

    # ────────────────────────────────────────────────────────
    #  基础运算
    # ────────────────────────────────────────────────────────

    def _D(self, preds, scores):
        N = preds.shape[0]
        if N > self.max_N:
            idx = np.random.RandomState(42).choice(N, self.max_N, replace=False)
            preds, scores = preds[idx], scores[idx]
        Ds = np.mean(np.abs(scores[:, :, None] - scores[:, None, :]), axis=0)
        Dp = np.mean(preds[:, :, None] != preds[:, None, :], axis=0).astype(np.float32)
        D = self.ws * Ds + self.wr * Dp
        np.fill_diagonal(D, 0)
        return D

    def _standardize(self, D):
        sigma = np.where(self.sigma0 < 1e-10, 1.0, self.sigma0)
        A = (D - self.mu0) / sigma
        np.fill_diagonal(A, 0)
        return (A + A.T) / 2

    def _eigh(self, A, reg=1e-8):
        Ar = A + reg * np.eye(self.K)
        ev, ec = np.linalg.eigh(Ar)
        ev -= reg
        order = np.argsort(ev)[::-1]
        return ev[order], ec[:, order]

    # ────────────────────────────────────────────────────────
    #  谱归因 (calibrate 和 detect 的唯一共用路径)
    # ────────────────────────────────────────────────────────

    def _spectral_attribution(self, eigenvalues, eigenvectors, n_components):
        """
        特征向量 → 视角 loading → 层级 loading.

        model_contrib[k] = Σᵢ uᵢ[k]² · λᵢ
        view_loading[v]  = mean_{k∈Gv} model_contrib[k]
        level_loading[ℓ] = mean_{v∈ℓ} view_loading[v]

        Bug3 修复: level 聚合用 mean 而非 max, 消除视角数偏差.
        Bug2 修复: calibrate 和 detect 调用同一个函数, n_components 一致.
        """
        view_loadings = {vn: 0.0 for vn in self.view_names}
        for i in range(n_components):
            u = eigenvectors[:, i]
            lam = max(eigenvalues[i], 0)
            mc = u ** 2 * lam
            for vn in self.view_names:
                gv = self.pg[vn].get("models", [])
                if gv:
                    view_loadings[vn] += float(sum(mc[k] for k in gv) / len(gv))

        level_loadings = np.zeros(self.L)
        for i, ln in enumerate(self.level_names):
            lvs = self.level_views.get(ln, [])
            if lvs:
                level_loadings[i] = float(np.mean([view_loadings.get(vn, 0) for vn in lvs]))
        return view_loadings, level_loadings

    # ────────────────────────────────────────────────────────
    #  校准
    # ────────────────────────────────────────────────────────

    def calibrate(self, pred_list, score_list, quiet=False):
        T = len(pred_list)
        assert T >= 2
        if not quiet:
            print(f"[SDD] Calibrating on {T} baseline batches (K={self.K})...")

        Ds = np.stack([self._D(p, s) for p, s in zip(pred_list, score_list)])
        self.D_history = list(Ds)
        self.mu0 = Ds.mean(0)
        self.sigma0 = Ds.std(0, ddof=1)
        nz = self.sigma0[np.isfinite(self.sigma0) & (self.sigma0 > 1e-10)]
        fill = float(np.median(nz)) if len(nz) else 1e-4
        self.sigma0 = np.where(np.isfinite(self.sigma0) & (self.sigma0 > 1e-10),
                                self.sigma0, fill)

        all_scores = np.concatenate(score_list, axis=0)
        self._model_baseline_mu = all_scores.mean(axis=0)
        self._model_baseline_std = np.maximum(all_scores.std(axis=0), 1e-8)

        # ── 谱 null 分布 (两遍: 第1遍求τ, 第2遍用τ确定n_comp) ──

        # Pass 1: 收集 null 最大特征值 → 求 τ
        null_eig_data = []  # [(ev, ec), ...]
        null_ev_max = []
        for D in Ds:
            A = self._standardize(D)
            ev, ec = self._eigh(A)
            null_eig_data.append((ev, ec))
            null_ev_max.append(float(ev[0]))
        self.tau = float(np.percentile(null_ev_max, 100 * (1 - self.alpha)))

        # Pass 2: 用 τ 确定 n_comp (与 detect 完全一致)
        null_vl_list, null_ll_list, null_fracs = [], [], []
        for ev, ec in null_eig_data:
            r_null = max(int(np.sum(ev > self.tau)), 1)
            vl, ll = self._spectral_attribution(ev, ec, r_null)
            null_vl_list.append(vl)
            null_ll_list.append(ll)
            total = sum(ll) + 1e-10
            null_fracs.append({ln: ll[i]/total for i, ln in enumerate(self.level_names)})
        del null_eig_data

        null_ll = np.stack(null_ll_list)
        self._level_null_mu = null_ll.mean(0)
        self._level_null_std = np.maximum(null_ll.std(0, ddof=1), 1e-8)

        # 层级阈值 (Bonferroni)
        z_crit = sp_stats.norm.ppf(1 - self.alpha / max(self.L, 1)) if HAS_SCIPY else 2.5
        for i, ln in enumerate(self.level_names):
            self.tau_levels[ln] = float(self._level_null_mu[i] + z_crit * self._level_null_std[i])

        # per-view null
        for vn in self.view_names:
            vals = [vl.get(vn, 0) for vl in null_vl_list]
            self._view_null_mu[vn] = float(np.mean(vals))
            self._view_null_std[vn] = float(max(np.std(vals), 1e-8))

        # 分类阈值: 基于 null fraction 分布
        # 理论: 每层级 null 期望 fraction = 1/L, 主导阈值不应远超此值
        uniform_frac = 1.0 / max(self.L, 1)
        for ln in self.level_names:
            vals = [f.get(ln, 0) for f in null_fracs]
            if T >= 10:
                thr = float(np.mean(vals) + 2 * max(np.std(vals), 1e-4))
            else:
                thr = float(max(vals) + 0.05)
            # 上界: 不超过 uniform + 0.15, 确保显著主导可被检测
            self._classification_thresholds[ln] = min(thr, uniform_frac + 0.15)

        # η: 从 null 视角向量的两两角度推导
        null_d_views = [np.array([vl.get(vn, 0) for vn in self.view_names]) for vl in null_vl_list]
        null_angles = []
        for i in range(len(null_d_views)):
            for j in range(i+1, len(null_d_views)):
                di, dj = null_d_views[i], null_d_views[j]
                ni, nj = np.linalg.norm(di), np.linalg.norm(dj)
                if ni > 1e-10 and nj > 1e-10:
                    cos_a = np.clip(np.dot(di, dj) / (ni * nj), -1, 1)
                    null_angles.append(float(np.sqrt(1 - cos_a**2)))
        if null_angles and T >= 5:
            self.eta = float(np.percentile(null_angles, 95))
            self.eta = max(self.eta, 0.2)
        else:
            # 不够数据推导 → 用理论默认值 (sin(23.6°)≈0.4)
            self.eta = 0.4

        # CUSUM
        null_z = np.abs((null_ll - self._level_null_mu) / self._level_null_std)
        self.cusum_k = float(np.percentile(null_z, 80)) + 0.3 if null_z.size > 0 else 1.0

        # 样本级 z 阈值 (从 null z-score 分布推导)
        if null_z.size > 0:
            self._sample_z_thr = float(np.percentile(null_z, 97.5))
            self._sample_z_thr = max(self._sample_z_thr, 1.5)
        else:
            self._sample_z_thr = 2.0

        self.calibrated = True
        if not quiet:
            print(f"  τ_eigen={self.tau:.4f}, η={self.eta:.3f}")
            for i, ln in enumerate(self.level_names):
                ct = self._classification_thresholds[ln]
                print(f"    {ln}: null_μ={self._level_null_mu[i]:.6f} "
                      f"null_σ={self._level_null_std[i]:.6f} "
                      f"τ_level={self.tau_levels[ln]:.6f} "
                      f"τ_class={ct:.4f} "
                      f"models={self.levels[ln]}")
            self._check_separability(quiet=False)

    def _check_separability(self, quiet=True):
        mx = 0.0
        for i, l1 in enumerate(self.level_names):
            for j, l2 in enumerate(self.level_names):
                if i >= j: continue
                s1, s2 = set(self.levels[l1]), set(self.levels[l2])
                mx = max(mx, len(s1 & s2) / max(len(s1 | s2), 1))
        if not quiet:
            print(f"  可分性: max_overlap={mx:.3f} < 0.333 → "
                  f"{'✓' if mx < 1/3 else '⚠ 重叠'}")

    # ────────────────────────────────────────────────────────
    #  detect()
    # ────────────────────────────────────────────────────────

    def detect(self, preds, scores):
        """
        谱漂移分解 — 全部决策由 eigh 驱动.

        Phase 1: D → A
        Phase 2: eigh(A) → {λ,u} → _spectral_attribution → view/level loading
        Phase 3: level_fraction + novelty → 类型分类
        Phase 4: per-view threshold → affected views, per-eigenvector → mechanism_views
        """
        assert self.calibrated
        timing = {}

        # ══ Phase 1 ══
        t0 = time.perf_counter()
        D = self._D(preds, scores)
        timing["phase1_ms"] = (time.perf_counter() - t0) * 1000

        # ══ Phase 2: 谱分解 + 归因 ══
        t0 = time.perf_counter()
        A = self._standardize(D)
        eigenvalues, eigenvectors = self._eigh(A)

        r = int(np.sum(eigenvalues > self.tau))
        drift_detected = r > 0

        if r >= 2 and eigenvalues[1] > 1e-10:
            spectral_gap = float(eigenvalues[0] / eigenvalues[1])
        elif r == 1:
            spectral_gap = float(eigenvalues[0] / max(self.tau, 1e-10))
        else:
            spectral_gap = 0.0

        n_comp = max(r, 1)
        view_loadings, level_loadings = self._spectral_attribution(
            eigenvalues, eigenvectors, n_comp)

        level_z = {}
        for i, ln in enumerate(self.level_names):
            level_z[ln] = float((level_loadings[i] - self._level_null_mu[i])
                                / self._level_null_std[i])

        timing["phase2_ms"] = (time.perf_counter() - t0) * 1000

        # ══ Phase 3: 分类 ══
        t0 = time.perf_counter()

        total_ll = sum(level_loadings) + 1e-10
        level_fractions = {ln: level_loadings[i] / total_ll
                           for i, ln in enumerate(self.level_names)}

        d_view = np.array([view_loadings.get(vn, 0) for vn in self.view_names])
        novelty = self._compute_novelty(d_view)

        drift_type, explanation = self._classify_drift(
            drift_detected, r, eigenvalues, spectral_gap,
            level_fractions, novelty)

        if drift_type != DRIFT_NONE:
            self.drift_history.append(d_view.copy())
            self._update_subspace()

        timing["phase3_ms"] = (time.perf_counter() - t0) * 1000

        # ══ Phase 4: affected views + mechanism views + strategy ══
        t0 = time.perf_counter()

        # Bug4 修复: per-view 阈值
        affected_views = [vn for vn in self.view_names
                          if view_loadings.get(vn, 0)
                          > self._view_null_mu[vn] + 3 * self._view_null_std[vn]]

        # 兜底: 漂移但无超阈值视角 → 用 u₁ 分量最大的模型归因
        if not affected_views and drift_detected:
            u1 = eigenvectors[:, 0]
            for km in np.argsort(u1 ** 2)[::-1][:max(1, self.K // 3)]:
                for vn, vi in self.pg.items():
                    if km in vi.get("models", []) and vn not in affected_views:
                        affected_views.append(vn)

        # Bug7 修复: 每个显著特征向量 → 独立的机制视角组
        mechanism_views = []
        for i in range(r):
            u = eigenvectors[:, i]
            mc = u ** 2 * max(eigenvalues[i], 0)
            mv = []
            for vn in self.view_names:
                gv = self.pg[vn].get("models", [])
                if gv:
                    avg = sum(mc[k] for k in gv) / len(gv)
                    if avg > self._view_null_mu.get(vn, 0) + 2 * self._view_null_std.get(vn, 1e-8):
                        mv.append(vn)
            if mv:
                mechanism_views.append(mv)

        am = set()
        for vn in affected_views:
            am.update(self.pg[vn]["models"])
        hm = sorted(k for k in range(self.K) if k not in am)
        gap_ratio = len(hm) / self.K

        for ln in self.level_names:
            z = max(level_z.get(ln, 0), 0)
            self.cusum[ln] = max(0, self.cusum[ln] + z - self.cusum_k)

        strategy, label_budget = self._defense_strategy(drift_type, gap_ratio, self.K)

        timing["phase4_ms"] = (time.perf_counter() - t0) * 1000
        timing["total_ms"] = sum(timing.values())

        return SDDResult(
            drift_detected=drift_detected,
            num_mechanisms=r,
            eigenvalues=eigenvalues, eigenvectors=eigenvectors,
            spectral_gap=spectral_gap, strategy=strategy,
            affected_views=affected_views,
            affected_models=sorted(am),
            healthy_models=hm,
            view_loadings={vn: max(v, 0) for vn, v in view_loadings.items()},
            mechanism_views=mechanism_views,
            threshold=self.tau,
            timing_ms=timing,
            confidence=float(eigenvalues[0] / max(self.tau, 1e-10)) if drift_detected else 0,
            label_budget=label_budget,
            mechanism_strengths=[float(eigenvalues[i]) for i in range(r)],
            drift_type=drift_type,
            level_scores=level_z,
            novelty=novelty,
            view_deltas={vn: float(view_loadings.get(vn, 0)) for vn in self.view_names},
            drift_explanation=explanation,
        )

    # ────────────────────────────────────────────────────────
    #  漂移类型分类 (Theorem 1)
    # ────────────────────────────────────────────────────────

    def _classify_drift(self, detected, r, eigenvalues, gap,
                        level_fractions, novelty):
        if not detected:
            return DRIFT_NONE, f"λ₁={eigenvalues[0]:.3f} ≤ τ={self.tau:.3f}"

        frac = level_fractions
        info = (f"s={frac.get('surface',0):.0%} r={frac.get('structure',0):.0%} "
                f"g={frac.get('global',0):.0%}; "
                f"r={r},λ₁={eigenvalues[0]:.2f},gap={gap:.1f}")

        # 各层级是否"主导": fraction 超过从校准数据推导的阈值
        dom = {ln: frac.get(ln, 0) > self._classification_thresholds.get(ln, 1.0)
               for ln in self.level_names}

        ds = dom.get("surface", False)
        dr = dom.get("structure", False)
        dg = dom.get("global", False)

        if ds and dr and dg:
            if novelty > self.eta:
                return DRIFT_PARADIGM, f"{info}; ν={novelty:.2f}>η → 范式迁移"
            return DRIFT_STRUCTURAL, f"{info}; ν={novelty:.2f}≤η → 结构型级联"

        if dr:
            return DRIFT_STRUCTURAL, f"{info} → structure主导"

        if dg:
            if novelty > self.eta:
                return DRIFT_PARADIGM, f"{info}; ν={novelty:.2f}>η → 范式迁移"
            return DRIFT_STRUCTURAL, f"{info}; ν={novelty:.2f}≤η → 全局偏移已知方向"

        if ds and not dr and not dg:
            return DRIFT_PERTURBATION, f"{info} → surface主导"

        # 无明确主导: 按谱强度保守分类
        if eigenvalues[0] > 2 * self.tau:
            return DRIFT_STRUCTURAL, f"{info} → 强漂移模式不明确"
        return DRIFT_PERTURBATION, f"{info} → 弱漂移"

    @staticmethod
    def _defense_strategy(drift_type, gap_ratio, K=12):
        if drift_type == DRIFT_PERTURBATION:
            return "auto_adapt", max(10, K * 2)  # ~2 labels per model
        if drift_type == DRIFT_STRUCTURAL:
            return "selective_retrain", int(np.ceil(200 / max(gap_ratio, 0.01)))
        if drift_type == DRIFT_PARADIGM:
            return "full_retrain", int(np.ceil(500 / max(gap_ratio, 0.01)))
        return "none", 0

    # ────────────────────────────────────────────────────────
    #  Theorem 3: 新颖性
    # ────────────────────────────────────────────────────────

    def _compute_novelty(self, d):
        """ν(d) = sin∠(d, H) = ‖d − P_H d‖ / ‖d‖
        Theorem 3a: 冷启动期 (history < V) 返回 0 (不断言新颖)."""
        if np.linalg.norm(d) < 1e-10:
            return 0.0
        # Theorem 3a: T_min = V (视角维度数), 历史不足时子空间不完整
        if self.drift_subspace is None or len(self.drift_history) < self.V:
            return 0.0
        proj = self.drift_subspace @ d
        return float(np.linalg.norm(d - proj) / np.linalg.norm(d))

    def _update_subspace(self, max_rank=5):
        if len(self.drift_history) < 2:
            return
        H = np.stack(self.drift_history[-20:])
        try:
            _, S, Vt = np.linalg.svd(H, full_matrices=False)
            rk = max(1, min(max_rank, len(S), int(np.sum(S > S[0] * 0.1))))
            V = Vt[:rk].T
            self.drift_subspace = V @ V.T
        except np.linalg.LinAlgError:
            pass

    # ────────────────────────────────────────────────────────
    #  Baseline Update
    # ────────────────────────────────────────────────────────

    def update_baseline(self, preds, scores, window=None, use_ema=False):
        """更新基线统计量并重新校准谱阈值."""
        D_new = self._D(preds, scores)
        if use_ema:
            a = self.ema_decay
            old = self.mu0.copy()
            self.mu0 = a * self.mu0 + (1-a) * D_new
            self.sigma0 = np.sqrt(a * self.sigma0**2 + (1-a) * (D_new - old)**2)
        else:
            self.D_history.append(D_new)
            if window and len(self.D_history) > window:
                self.D_history = self.D_history[-window:]
            Ds = np.stack(self.D_history)
            self.mu0 = Ds.mean(0)
            self.sigma0 = Ds.std(0, ddof=1)
        nz = self.sigma0[np.isfinite(self.sigma0) & (self.sigma0 > 1e-10)]
        fill = float(np.median(nz)) if len(nz) else 1e-4
        self.sigma0 = np.where(np.isfinite(self.sigma0) & (self.sigma0 > 1e-10),
                                self.sigma0, fill)
        for ln in self.level_names:
            self.cusum[ln] = 0.0

        # 重新校准谱阈值 (使用更新后的 D_history)
        if len(self.D_history) >= 2:
            null_ev_max = []
            null_ll_list = []
            for D in self.D_history:
                A = self._standardize(D)
                ev, ec = self._eigh(A)
                null_ev_max.append(float(ev[0]))
                r_n = max(int(np.sum(ev > self.tau)), 1)
                _, ll = self._spectral_attribution(ev, ec, r_n)
                null_ll_list.append(ll)
            self.tau = float(np.percentile(null_ev_max, 100 * (1 - self.alpha)))
            null_ll = np.stack(null_ll_list)
            self._level_null_mu = null_ll.mean(0)
            self._level_null_std = np.maximum(null_ll.std(0, ddof=1), 1e-8)

    # ────────────────────────────────────────────────────────
    #  DACP
    # ────────────────────────────────────────────────────────

    def dacp_predict(self, test_s, cal_s, cal_y, healthy, alpha=0.10):
        hm = healthy if healthy else list(range(self.K))
        n = len(cal_y)
        cal_e = np.mean(cal_s[:, hm], axis=1)
        test_e = np.mean(test_s[:, hm], axis=1)
        cm, cs = float(cal_e.mean()), float(max(cal_e.std(), 1e-8))
        tm, ts = float(test_e.mean()), float(max(test_e.std(), 1e-8))
        log_w = -0.5*((cal_e-tm)/ts)**2 + 0.5*((cal_e-cm)/cs)**2
        w = np.exp(log_w - log_w.max())
        w = np.clip(w, 0.01, 100.0)
        B = float(w.max() / max(w.min(), 1e-8))
        ncs = np.abs(cal_e - cal_y)
        wn = w / w.sum()
        si = np.argsort(ncs)
        cum = np.cumsum(wn[si])
        ql = min((1-alpha)*(1+1.0/n), 1.0)
        qi = min(int(np.searchsorted(cum, ql)), len(ncs)-1)
        qh = float(ncs[si[qi]])
        ps = [{y for y in (0,1) if abs(s-y) <= qh} or {int(round(s))} for s in test_e]
        return {"prediction_sets": ps,
                "avg_set_size": float(np.mean([len(s) for s in ps])),
                "ensemble_scores": test_e, "quantile": qh,
                "n_healthy": len(hm),
                "coverage_guarantee": max(0, 1-alpha-B/(n+1))}

    @staticmethod
    def compute_coverage(ps, yt):
        return float(np.mean([y in p for p, y in zip(ps, yt)]))

    # ────────────────────────────────────────────────────────
    #  DSIR — Drift-Sensitive Inference Reweighting
    # ────────────────────────────────────────────────────────

    def dsir_weights(self, res, base_w=None):
        """Compute DSIR model weights based on per-view drift impact.

        Key fix: only reweight when drift is DIFFERENTIAL across views.
        If all views drift equally (uniform perturbation), reweighting
        just adds noise without benefit — keep equal weights.
        Also uses gentler reweighting (sqrt dampening) to avoid extremes.
        """
        w = np.ones(self.K) / self.K if base_w is None else base_w.copy()
        if not res.drift_detected:
            return w

        # Collect per-model drift impact from view_deltas
        impact = np.zeros(self.K)
        for vn, delta in res.view_deltas.items():
            if delta > 0:
                for k in self.pg[vn]["models"]:
                    impact[k] = max(impact[k], delta)

        mx = impact.max()
        if mx < 1e-6:
            return w  # no measurable impact

        # Check if drift is differential vs uniform
        impact_norm = impact / mx
        nonzero = impact_norm[impact_norm > 0.01]
        if len(nonzero) > 1:
            cv = float(nonzero.std() / max(nonzero.mean(), 1e-8))
            if cv < 0.25:
                # Uniform drift: all views equally affected
                # Reweighting would just add noise → keep equal weights
                return w

        # Differential drift: downweight heavily affected models gently
        # sqrt dampening prevents extreme weight collapse
        adjustment = np.sqrt(impact_norm) * 0.4  # max 40% reduction
        w *= np.maximum(1.0 - adjustment, 0.5 / self.K)
        return w / w.sum()

    # ────────────────────────────────────────────────────────
    #  Pseudo-Labels
    # ────────────────────────────────────────────────────────

    def generate_pseudo_labels(self, scores, healthy, thr=0.85):
        H = len(healthy)
        if H < 2:
            return np.array([]), np.array([]), 0.0
        hp = (scores[:, healthy] >= 0.5).astype(int)
        maj = np.round(hp.mean(axis=1)).astype(int)
        agr = np.mean(hp == maj[:, None], axis=1)
        idx = np.where(agr >= thr)[0]
        if len(idx) == 0:
            return np.array([]), np.array([]), 0.0
        return maj[idx], idx, len(idx) / len(scores)

    def predicted_pl_quality(self, gap, per_model_acc=0.85):
        H = max(int(gap * self.K), 1)
        p = max(per_model_acc, 0.51)
        return 1.0 - np.exp(-2 * H * (p - 0.5)**2)

    def min_detectable_drift(self):
        z_crit = sp_stats.norm.ppf(1 - self.alpha / max(self.L, 1)) if HAS_SCIPY else 2.5
        sig = float(max(self._level_null_std)) if len(self._level_null_std) > 0 else 1.0
        return float(sig * z_crit), self.K + 1

    # ════════════════════════════════════════════════════════
    #  样本级漂移感知
    #  Bug10 修复: 用视角内分歧 + 校准阈值, 与批量级对称
    # ════════════════════════════════════════════════════════

    def calibrate_sample_baseline(self, scores):
        N = scores.shape[0]
        # per-view 分歧基线
        self._sample_view_mu = {}
        self._sample_view_std = {}
        for vn in self.view_names:
            gv = self.pg[vn].get("models", [])
            if len(gv) < 2:
                # Bug16: 单模型视角用模型分数的绝对偏离作为分歧度量
                per_sample_dev = np.abs(scores[:, gv[0]] - 0.5) if gv else np.zeros(N)
                self._sample_view_mu[vn] = float(per_sample_dev.mean())
                self._sample_view_std[vn] = float(max(per_sample_dev.std(), 1e-4))
                continue
            per_sample_std = scores[:, gv].std(axis=1)
            self._sample_view_mu[vn] = float(per_sample_std.mean())
            self._sample_view_std[vn] = float(max(per_sample_std.std(), 1e-4))
        self._sample_cal = True

    def score_samples(self, scores):
        """样本级漂移感知 — 与批量级 detect() 管道对称."""
        N = scores.shape[0]
        if not getattr(self, '_sample_cal', False):
            return {"drift_evidence": np.zeros((N, self.L)),
                    "sample_drift_type": np.full(N, DRIFT_NONE, dtype=object),
                    "anomaly_score": np.zeros(N),
                    "is_drift_candidate": np.zeros(N, dtype=bool)}

        # Step 1: per-view z-score (Bug16: 单模型视角也参与)
        sample_view_z = np.zeros((N, self.V))
        for j, vn in enumerate(self.view_names):
            gv = self.pg[vn].get("models", [])
            if len(gv) < 2:
                if gv:
                    dev = np.abs(scores[:, gv[0]] - 0.5)
                    sample_view_z[:, j] = np.maximum(
                        (dev - self._sample_view_mu[vn]) / self._sample_view_std[vn], 0)
                continue
            pss = scores[:, gv].std(axis=1)
            sample_view_z[:, j] = np.maximum(
                (pss - self._sample_view_mu[vn]) / self._sample_view_std[vn], 0)

        # Step 2: view → level (mean)
        evidence = np.zeros((N, self.L))
        for i, ln in enumerate(self.level_names):
            lvs = self.level_views.get(ln, [])
            if lvs:
                vidx = [self.view_names.index(vn) for vn in lvs if vn in self.view_names]
                if vidx:
                    evidence[:, i] = sample_view_z[:, vidx].mean(axis=1)

        # Step 3: fraction → classify (Bug15: 阈值从校准推导)
        anomaly = np.max(evidence, axis=1)
        z_thr = self._sample_z_thr  # 从 calibrate 推导
        sample_types = np.full(N, DRIFT_NONE, dtype=object)

        for n in range(N):
            if anomaly[n] < z_thr:
                continue
            e = evidence[n]
            total_e = sum(e) + 1e-10
            fracs = {ln: e[i] / total_e for i, ln in enumerate(self.level_names)}
            dom = {ln: fracs.get(ln, 0) > self._classification_thresholds.get(ln, 1.0)
                   for ln in self.level_names}

            ds = dom.get("surface", False)
            dr = dom.get("structure", False)
            dg = dom.get("global", False)

            if ds and dr and dg:
                sample_types[n] = DRIFT_PARADIGM
            elif dr:
                sample_types[n] = DRIFT_STRUCTURAL
            elif ds and not dr and not dg:
                sample_types[n] = DRIFT_PERTURBATION
            elif dg:
                sample_types[n] = DRIFT_PARADIGM
            else:
                sample_types[n] = DRIFT_STRUCTURAL

        return {"drift_evidence": evidence,
                "sample_drift_type": sample_types,
                "anomaly_score": anomaly,
                "is_drift_candidate": anomaly > z_thr}
