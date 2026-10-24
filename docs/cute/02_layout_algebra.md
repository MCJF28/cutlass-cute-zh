# CuTe 布局代数运算法则 (Layout Algebra)

> [!NOTE] 原文出处
> 本文翻译自 NVIDIA CUTLASS 官方核心规范文档 [`media/docs/cpp/cute/02_layout_algebra.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)。

CuTe 构建了一套严密的**“布局代数（Algebra of Layouts）”**数学体系，用于支撑布局之间的各种高阶组合与代数变换。这套代数包含以下核心操作：

* **函数复合（Functional Composition）**：将一个布局作为坐标变换代入另一个布局。
* **布局乘法（Layout Product）**：根据一个目标布局的模式规则，对另一个源布局进行平铺与复写。
* **布局除法（Layout Divide）**：根据特定子块规则，将一个大布局精准拆解为分块（Tile）与余量（Rest）。

在 GPU 体系结构中，**将数据布局（Data Layout）精准分发映射到线程网格布局（Thread Layout）**的核心机制正是依赖布局除法；而构建深层嵌套的分块数据结构则依赖布局乘法。所有这些高级特性的底层基石，全部统一于**函数复合（Composition）**。

---

## 1. 合并化简 (Coalesce)

前文指出：**Layout 在数学上就是从整数映射到整数的函数（Functions from Integers to Integers）**。

`coalesce` 操作是对这一函数系统的**“最简化运算（Simplify）”**。如果我们只关注输入的物理坐标，那么在不改变该函数整体映射行为的前提下，我们可以任意精简它的形状、减少它的模式数量（Rank）。
在整个化简过程中，**唯一严格保持不变的只有布局的容量大小 `size`**。

在单元测试中定义的后验条件如下：
```cpp
// @post size(result) == size(layout)           // 元素容量守恒
// @post depth(result) <= 1                     // 嵌套层级被完全展平
// @post 对于所有 0 <= i < size(layout), result(i) == layout(i) // 函数映射值严格一致
Layout coalesce(Layout const& layout);
```

### 示例解析
```cpp
auto layout = Layout<Shape <_2, Shape <_1, _6>>,
                     Stride<_1, Stride<_6, _2>>>{};
auto result = coalesce(layout);    // 化简结果为: _12:_1
```
原本带有两层嵌套、包含 3 个维度的复杂布局，经过 `coalesce` 后被瞬间化简为仅有 1 个维度、连续步长为 1 的平坦标量向量 `_12:_1`！这在硬件动态计算时能直接省去数次多余的坐标求模与乘加运算。

### 化简推导规则 (4 种核心判定)
对于任意相邻的两个一维模式 $s_0:d_0$ 与 $s_1:d_1$（记化简操作为 $\text{++}$）：

1. **退化维度消除（左）**：`_1:d0 ++ s1:d1 => s1:d1`。凡是静态大小为 1 的模式，无论其步长多大，其计算结果永远为 0，直接消除。
2. **退化维度消除（右）**：`s0:d0 ++ _1:d1 => s0:d0`。同理直接消除。
3. **物理连续维度合并**：
   $$s_0:d_0 \text{ ++ } s_1:(s_0 \cdot d_0) \implies (s_0 \cdot s_1):d_0$$
   若第二维的步长恰好等于第一维的“形状 $\times$ 步长”，说明这两维在内存中完全连续，直接合并为一个宏维度！
4. **无法合并**：`s0:d0 ++ s1:d1 => (s0, s1):(d0, d1)`。保留原状。

### 依模式化简 (By-mode Coalesce)
如果我们需要在化简的同时**保留特定的高维张量外层形状**（例如希望一个 2D 矩阵在化简内部碎片后依然维持 2D），可以使用带目标外形的重载：

```cpp
auto a = Layout<Shape <_2, Shape <_1, _6>>,
                Stride<_1, Stride<_6, _2>>>{};
auto result = coalesce(a, Step<_1, _1>{});   // 化简为: (_2, _6):(_1, _2)
```

---

## 2. 函数复合 (Composition)

**函数复合（Functional Composition）是 CuTe 整个宇宙运转的核心引擎。**

若 $A$ 与 $B$ 为两个布局，则它们的复合运算记作：

$$R := A \circ B \implies R(c) := A(B(c))$$

### 数值推导实例
设 $A = (6, 2):(8, 2)$，$B = (4, 3):(3, 1)$：
* $R(0) = A(B(0)) = A(B(0,0)) = A(0) = A(0,0) = 0$
* $R(1) = A(B(1)) = A(B(1,0)) = A(3) = A(3,0) = 24$
* $R(2) = A(B(2)) = A(B(2,0)) = A(6) = A(0,1) = 2$
* $\dots$
* $R(11) = A(B(11)) = A(B(3,2)) = A(11) = A(5,1) = 42$

**极其惊人的数学结论**：上述逐点计算得到的映射关系 $R(c) = k$，**其本身恰好可以严丝合缝地表达为另一个全新的 CuTe Layout！**

$$R = ((2, 2), 3):((24, 2), 8)$$

且满足：

$$\text{compatible}(B, R)$$

这意味着：**输入布局 $B$ 的每一个合法坐标，都可以无缝作为复合结果布局 $R$ 的合法输入坐标！** 因为从数学上讲，$B$ 的定义域直接决定了复合函数 $R$ 的定义域。

```cpp
// 核心后验断言
// @post compatible(layout_b, result)
// @post 对于所有 0 <= i < size(layout_b), result(i) == layout_a(layout_b(i))
Layout composition(LayoutA const& layout_a, LayoutB const& layout_b);
```

---

### 2.1 复合运算的代数推导原理

1. **左分配律**：若 $B = (B_0, B_1, \dots)$ 且 $B$ 为单射，则：
   $$A \circ B = A \circ (B_0, B_1, \dots) = (A \circ B_0, A \circ B_1, \dots)$$
2. 当 $A = a:b$ 且 $B = s:d$ 均为一维标量时：
   $$R = (a:b) \circ (s:d) = s:(b \cdot d)$$
3. 当 $A$ 为多维模式而 $B = s:d$ 时，求解分为两步：
   * **消除步长（Div out stride $d$）**：通过从左至右用 $d$ 依次整除 $A$ 的各维度形状，得到包含间隔抽取效果的新步长。此步需满足**步长整除条件（Stride Divisibility Condition）**。
   * **截取形状（Mod out shape $s$）**：从左至右取模，截取前 $s$ 个元素，确保输出形状与 $B$ 相容。此步需满足**形状整除条件（Shape Divisibility Condition）**。

---

### 2.2 依模式复合 (By-mode Composition 与 Tiler)

当 `composition` 的第二个参数不是单一布局，而是一个由多个子布局组成的元组时，该参数被称为 **`Tiler`（分块器）**。

```cpp
// 原始布局 A: (12, (4, 8)):(59, (13, 1))
auto a = make_layout(make_shape (12, make_shape ( 4, 8)),
                     make_stride(59, make_stride(13, 1)));

// 定义分块器 Tiler: 沿着第 0 维按 3:4 截取，沿着第 1 维按 8:2 截取
auto tiler = make_tile(Layout<_3, _4>{}, Layout<_8, _2>{});

// 得到截取后的局部子布局
auto result = composition(a, tiler); // (_3, (2, 4)):(236, (26, 1))
```

> [!TIP] Tiler 的本质
> `Tiler` 可以是一个单独的 `Layout`、一个包含多个 `Layout` 的元组、或者一个纯粹的 `Shape`（纯 `Shape` 会被默认隐式解释为步长全为 1 的基础平铺器）。

---

## 3. 补全运算 (Complement)

我们可以把复合运算 $A \circ B$ 视作使用子布局 $B$ 在父布局 $A$ 中“圈定并选择”了一部分特定元素。
**那么，剩下的那些没有被圈定的元素该如何排布？**

为了实现工业级的算子平铺，我们不仅需要选出当前的局部 Tile，还必须精确描述**所有其余未被选择的 Tile 的排布规律（The Rest / Leftovers）**。

`complement` 操作的目标，正是寻找这样一个与原布局在值域上**完全互补、互不重叠**的补全布局：

```cpp
// 计算 layout_a 相对于总空间大小 cotarget 的补全布局
Layout complement(LayoutA const& layout_a, Shape const& cotarget);
```

### 关键数学性质：
1. **有界性**：补全布局 $R$ 的大小受限于 `size(cotarget)`；
2. **正交无交集（Disjoint Codomain）**：$A$ 与补全布局 $R$ 的物理索引值域完全不相交（零碰撞）；
3. **空间填补**：$R$ 描述了原布局 $A$ 在目标空间内**周期性平移复制的规律**。

### 经典算例：
* `complement(4:1, 24)` $\implies$ **`6:4`**（原布局跨度为 4，连续重复 6 次填满 24）。
* `complement(6:4, 24)` $\implies$ **`4:1`**（原布局跨步为 4 产生大量空洞，正好由步长为 1 的 4 个元素填补）。
* `complement((4, 6):(1, 4), 24)` $\implies$ **`1:0`**（原布局已占满 24 个元素，补全为空）。
* `complement(4:2, 24)` $\implies$ **`(2, 3):(1, 8)`**。

---

## 4. 布局划分与平铺 (Logical Division / Tiling)

至此，我们拥有了布局代数的三大神器：**拼接（Concatenation）**、**复合（Composition）** 与 **补全（Complement）**。现在我们可以正式定义**布局除法（Division）**！

### 4.1 逻辑划分定义

逻辑划分 `logical_divide(A, B)` 在语义上将布局 $A$ 切分为两个模式：
* **第 0 模式**：所有被 $B$ 命中的数据元素集合（即当前局部分块 Tile）；
* **第 1 模式**：所有未被 $B$ 命中的剩余元素（即遍历所有分块的网格排布 Rest）。

其代数定义极其优雅唯美：

$$A \oslash B := A \circ (B, B^*)$$

在 CuTe 中仅需一行源码即可表达：

```cpp
template <class LShape, class LStride, class TShape, class TStride>
auto logical_divide(Layout<LShape, LStride> const& layout,
                    Layout<TShape, TStride> const& tiler)
{
  // 先求补全 B*，再将 (B, B*) 整体复合回原始 layout
  return composition(layout, make_layout(tiler, complement(tiler, size(layout))));
}
```

---

### 4.2 四种划分形态：Zipped、Tiled、Flat

为了方便在 CUDA 内核中对分块进行切片操作，CuTe 提供了四种重排格式：

| 划分函数 | 输出模式排布结构 | 适用工程场景 |
| :--- | :--- | :--- |
| **`logical_divide`** | `((TileM, RestM), (TileN, RestN), ...)` | 保留原始各模式的语义不变 |
| **`zipped_divide`** | `((TileM, TileN), (RestM, RestN, ...))` | **最常用！** 将整个 Tile 聚为 mode-0，遍历网格聚为 mode-1 |
| **`tiled_divide`** | `((TileM, TileN), RestM, RestN, ...)` | 保持多维外部网格索引独立 |
| **`flat_divide`** | `(TileM, TileN, RestM, RestN, ...)` | 完全展平所有层级 |

#### 在代码中使用 `zipped_divide` 轻松索引分块：
```cpp
auto zd = zipped_divide(layout_a, tiler);

// 提取第 3 个分块的物理基地址偏移：
auto offset_tile_3 = zd(0, 3);

// 提取第 (1, 2) 号网格分块的物理基地址偏移：
auto offset_tile_1_2 = zd(0, make_coord(1, 2));

// 任意分块内部的固有数据布局恒等于：
// layout<0>(zd) == composition(layout_a, tiler);
```

---

## 5. 布局乘法与广播复写 (Logical Product)

若布局除法是“切分子块”，那么**布局乘法（Logical Product）**则是“用子块平铺构建宏观张量”。

数学定义为：

$$A \otimes B := (A, A^* \circ B)$$

实现形式为：
```cpp
template <class LShape, class LStride, class TShape, class TStride>
auto logical_product(Layout<LShape, LStride> const& layout,
                     Layout<TShape, TStride> const& tiler)
{
  return make_layout(layout, composition(complement(layout, size(layout) * cosize(tiler)), tiler));
}
```

### 块状乘积与交错循环乘积 (Blocked & Raked Products)
* **`blocked_product(A, B)`（块状平铺）**：
  * 将小分块 $A$ 作为一个不可分割的实体，按照 $B$ 的宏观网格排列成连续的大矩阵块。
* **`raked_product(A, B)`（交错循环分布 / Cyclic Distribution）**：
  * 将 $A$ 的内部元素与 $B$ 的周期进行**交错交织排布**。这在 GPU 线程分配中极具价值（例如：让相邻的 32 个线程以跨步轮询方式读取全局内存，以达成完美的一维内存合并访存合并 Coalescing！）。
