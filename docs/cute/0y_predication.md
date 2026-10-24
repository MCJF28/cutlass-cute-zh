# CuTe 谓词判定与非对齐边缘裁剪 (Predication)

> [!NOTE] 原文出处
> 本文翻译自 NVIDIA CUTLASS 官方核心规范文档 [`media/docs/cpp/cute/0y_predication.md`](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0y_predication.md)。

在基础教程中，我们通常假定矩阵的规模能够被分块大小整除（例如 $1024 \times 1024$ 的矩阵被整除为 $128 \times 128$ 的块）。然而在真实世界的工业级大模型算子中，输入的序列长度（Sequence Length）、Batch Size 或特征维度往往是**任意动态数值**（例如将一个 $41 \times 55$ 的矩阵切分为 $4 \times 8$ 的分块，行剩余 1，列剩余 7）。

**当矩阵边缘无法整除时，如何处理这些边界残余元素并防止非法显存越界崩溃？**

---

## 1. 为什么“向上取整 + 谓词判定”是最佳策略？

在 CuTe 的布局代数中，`logical_divide` 面对无法整除的尺寸时**天然采用“向上取整（Round Up）”**策略。
例如：若数据布局为 $1000:1$，平铺块为 $128:1$，则逻辑划分结果为 `(128, 8):(1, 128)`。这相当于将数据向上扩充为了一个 $128 \times 8 = 1024$ 的虚拟张量。

面对多出来的 24 个无效位置，CUDA 传统的错误做法是试图在 Host 端将分块拆解为“7 个 128 块 + 1 个 104 块”，这会导致复杂的循环分支与剧烈的 **Warp 分歧（Warp Divergence）**。

CuTe 的核心哲学是：**保持所有 CTA 的循环结构与分块完全同构统一，在指令发射层级通过布尔谓词（Predication Mask）裁剪无效越界访问！**

---

## 2. 核心架构：恒等坐标张量 (Coordinate Identity Tensor)

CuTe 解决高维非对齐边界的终极武器，是**“影子平铺机制（Shadow Tiling）”**：

1. **构造恒等坐标张量**：使用 `make_identity_tensor` 创建一个与原始大矩阵形状相同的虚拟坐标张量（该张量的值就是其逻辑坐标本身 $(m, n) \to (m, n)$）；
2. **影子同步平铺**：对该坐标张量执行与真实数据张量**完全一模一样**的 CTA Tiling 与 Thread Partitioning；
3. **坐标有界性判定**：通过比较影子坐标张量与原始矩阵的边界，自动推导出该线程负责的各个寄存器元素的合法性；
4. **指令掩码阻断**：在 `copy_if` 或尾声段循环中直接应用该布尔谓词！

---

## 3. 实战案例一：GEMM 尾声段 (Epilogue) 边缘裁剪

在将计算结果累加写回全局内存 $C$ 矩阵时进行谓词判定：

```cpp
// 1. 构造与原始 mC 矩阵形状完全同构的恒等坐标张量: (m, n) -> (m, n)
Tensor cC = make_identity_tensor(shape(mC));

// 2. 施加与真实数据完全镜像的 CTA 平铺与线程分块:
Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N) -> (m, n)
Tensor tCcC   = thr_mma.partition_C(cta_cC);                             // (MMA, MMA_M, MMA_N) -> (m, n)

// 3. 在写回主循环中通过 elem_less 判定全局坐标是否落在合法边界内:
for (int i = 0; i < size(tCgC); ++i) {
  if (elem_less(tCcC(i), shape(mC))) { // 若当前元素未越界，执行写入
    tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  }
}
```

#### 机制解析：
坐标张量 `tCcC` 与实际持有的寄存器片段 `tCrC` 以及全局显存视图 `tCgC` **在元组层级上完全全等同构（Congruent）**！但 `tCcC(i)` 记录的是该元素在全局矩阵 $mC$ 中的真实绝对坐标。将其与 `shape(mC)` 比较即可实现零歧义的越界判定。

---

## 4. 实战案例二：主循环访存 `copy_if` 谓词张量生成

在 GEMM 主循环从全局显存拉取 $A$ 和 $B$ 矩阵时，若在内层 $K$ 循环中反复做边界判断会严重拉低 IPC。
由于 $M$ 和 $N$ 维度的越界情况在整个 $K$ 轴求和过程中是固定不变的，我们可以**在序言段（Prologue）提前计算好谓词，并在 $K$ 轴广播复用**！

```cpp
// 1. 创建坐标恒等张量并同步分区:
Tensor cA = make_identity_tensor(shape(mA));
Tensor tAcA = local_partition(local_tile(cA, cta_tiler, cta_coord, Step<_1, X, _1>{}), tA, thread_idx);

// 2. 创建仅需存储 M 维判断、步长为 0 的广播谓词张量:
Tensor tApA = make_tensor<bool>(make_shape (size<0>(tAcA), size<1>(tAcA)),
                                make_stride(     Int<1>{},      Int<0>{})); // Stride-0 实现沿着 K 轴广播复用

// 3. 在序言段预先计算 M 维合法性:
CUTE_UNROLL
for (int m = 0; m < size<0>(tApA); ++m) {
  tApA(m, 0) = elem_less(get<0>(tAcA(m, 0, 0)), shape<0>(mA));
}

// 4. 在主循环中直接使用 copy_if:
for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
  copy_if(tApA, tAgA(_, _, k_tile), tAsA); // 越界线程自动屏蔽，不触发非法访存
}
```

---

## 5. 方案核心优势总结

1. **绝对解耦**：完全不依赖张量的底层物理存储步长，纯粹依赖抽象的逻辑几何边界；
2. **极度泛化**：可无缝穿透任意深度的 CTA 分块、TiledMMA 与 TiledCopy 变换；
3. **保持硬件最高能效**：规避了动态可变循环边界（Variable Loop Bounds），将控制流转化为 GPU 汇编原生的谓词指令执行（Predicated Execution），最大程度保持了线程束（Warp）的执行一致性与高吞吐负载均衡。
