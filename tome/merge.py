# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple

import torch


def do_nothing(x, mode=None):
    """
    一个空操作函数，返回原输入。
    由于 ToMe 会返回 `merge` 和 `unmerge` 两个函数，当我们不需要进行合并时（如 r<=0），返回此占位函数。
    """
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).
    【核心实现】将输入 token 集合划分为两半，找出二分图上的最优匹配（基于特征空间的余弦相似度），
    并返回一个“合并(前向)”函数及一个“解开(逆向)”函数。

    Input size is [batch, tokens, channels]. this is the embeddings of the tokens.
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: 图像全局的分类 token
     - distill_token: 蒸馏训练模型可能有的蒸馏 token
     启用了这两个配置时，这些 token 将不会参与被缩减。
    """
    # 记录需要保护的全局特征 token 数量，防止它们被错误合并
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    # t 是序列总的 token 数量 (包含 protected 特殊 token)
    t = metric.shape[1]
    # ToMe 算法基于二分图匹配（类似 1 对 1 选择），也就是将可用 token 拆分成两组进行匹配。
    # 每一对匹配就会消耗一边的一个 token，所以最多只能合并/去除可用 token 数量的一半（向下取整）。
    r = min(r, (t - protected) // 2)

    # 如果需要消去的数量 <= 0，就没必要计算了。直接返回啥都不做的两个占位函数。
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # 【关键处：计算匹配对】
        # 将 metric (例如 ViT 里面的 Attention Keys) 沿着特征维度归一化。
        # 这样之后计算内积即为两个 token 间的余弦相似度。cz the cos(theta)=a*b/|a|*|b|
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # 按照奇偶步长把所有 Token 分段拆成两个互不相交的集合：a 组 (偶数索引) 和 b 组 (奇数索引)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        
        # 对 a 组中的每个 token 和 b 组中的每个 token 做矩阵点积。
        # scores 矩阵形状: [batch, len(a), len(b)]，这就是它们两两间的余弦相似度打分。
        scores = a @ b.transpose(-1, -2)

        # 保护特殊 token，使它们的相似度被置为负无穷，从而不可能被挑选为匹配项 (因为后续使用 max 找最大相似度)
        if class_token:
            scores[..., 0, :] = -math.inf # 假如 class_token 存在，它被分在了 a 组的第 0 位 (索引为 0)
        if distill_token:
            scores[..., :, 0] = -math.inf # 假如 distill_token 存在，它刚好占满第一、二位，可能在 b 组第 0 位

        # 取出 a 组中每个 token 对应 b 组里打分最高的那个 b_token。
        # node_max: 这个最高的分数 ([batch, len(a)])
        # node_idx: 被匹配最高分的 b 组 token 索引号 
        node_max, node_idx = scores.max(dim=-1)
        
        # 按照相似度 max 值从大到小对 a 组的 node 排序（我们只要找出前 r 个最高置信度的连线）。
        # argsort 返回降序排列后的 a 组自身原序列索引。
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        # 从中划分界限，由于排序是降序：
        # 前 r 个最应该被“拿去合并掉的” a 组 token。这部分作为“合并源”。
        src_idx = edge_idx[..., :r, :]  
        # 剩下那些不主动合并的 a 组孤立 token，它们将被孤零零地直接原样保留下来。
        unm_idx = edge_idx[..., r:, :]  
        
        # dst_idx: 这个步骤中，既然 src_idx 要被消灭并合并去 b 组，那么它们具体要和 b组 的哪一个去同化呢？
        # 利用 node_idx（我们在刚刚找到的最大匹配对应编号），用 src_idx 进行 gather 将其对应取出。
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            # 虽然由于 class_token 被置为了 -inf 我们肯定没合并它，所以它流落到了 unm_idx 区间。
            # 为了能在拼凑回整体 token 序列时它依然处于开头，这里手工让 unm_idx 做一次升序保证其索引 0 回归首位。
            unm_idx = unm_idx.sort(dim=1)[0]

    # 定义具体的 merge 闭包（Closure），因为之后 transformer 处理前向计算时就是调它。
    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # x 为输入进来的需要被压缩/融合的实际特征 (比如 keys / values / attention后隐层)
        # 用同样的奇偶手法切出 a(src), b(dst) 空间
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        
        # unm: a 组当中无需合并的孤立 token（按照 unm_idx 索引拉出来，它们依然自己过日子）
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        # src: a 组当中这些要求做出牺牲（去融合到 dst 里）的代表队伍
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # scatter_reduce 魔法：将在 b 空间中等待着它们的对应的目标位置 (dst_idx)，
        # 追加源 token 特征 (src)。如果 mode="mean"，那就是对特征取均值。如果 mode="sum" 就是累加。
        # 执行完这句，部分 dst_token 变成了 (原本的自己 + 新吸纳过来的 src) 的融合体，数量还是 len(b)。
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        # 把 a空间的幸存者 (unm) 和 带着融合体、甚至什么都没带的幸存者 (dst) 拼接到一起。
        # 注意此处的拼凑意味着整体 tokens 数量比来的时候减少了 `r` 个。
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    # 从已经被压扁的时空，重新铺张开来（如 MAE 解码器时如果需要恢复序列像素尺寸以进行重建，由于只在 encoder 做 tome，此时可以把特征还原出去）
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        # 解构出两段：未合并者特征、带上所有残余及被合并痕迹的目标特征。
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        # 所谓 unmerge 并不是真通过一个时空机器把最初没合并的那个状态一模一样变回来（特征不可逆转回两份独立），
        # 而是直接从已经融合以后的 dst 里面，再 "抄" 一份出来，原样填充回曾经被删除的 a_token 的那些位置。
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        # 构建一个满配原本尺寸 (metric.shape[1]) 的全零大空盘
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        # 把 b组(dst) 放飞到所有奇数位
        out[..., 1::2, :] = dst
        # 把 a组 未涉世事的人安排回他原来的偶数位
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        # 把 a组 那帮去 b 组历练过了而且抄录回来特征的 token 也塞回原来的偶数位
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    # 一共抛出两个锤子：把序列砸扁减耗时的 merge，和强行复原出全员但包含连带共享特征的 unmerge。
    return merge, unmerge



def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    ToMe 的一种参照对照片段变异版：按照固定步长 k 取出作为一组，剩余作另一组。
    不再是 50%, 50% 均分。
    
    If n is the number of tokens, resulting number of tokens will be n // z.
    Input size is [batch, tokens, channels].
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        """将 x 分离成待匹配的两边"""
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        # a 组 (每个 k 块切分中，前 k-1 个元素)；b 组 (该大块里面的最后一个即第 k 个元素)
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        # 找出 a 组每个元素对应在 b 组的最高相似度目标的序号
        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        """K 倍匹配合并逻辑，同理把 a 组的所有元素强行去寻找匹配融合进稀少的 b 组内"""
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        """K 倍匹配的解绑还原，切片复制粘回"""
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    这是论文里的另个消融（Ablation）强随机基准：并不依靠奇偶拆分组，
    而是从总列表中【彻底随机】挑选出 r 个倒霉蛋(A组)送上合并祭坛！剩余当作继承者(B组)。
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        # 取 B * N 维度上的白噪声 rand 计算序数 argsort，得出无规则打乱数组作为抽取参考
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        # 切分为随机拿掉的那 r 个 以及 保卫家园的剩余 N-r 个
        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        # 求余弦相似度算出要融合到哪个对应点上最合适
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        # 同理，随机版的所有 A组都被扔过去跟合并者(也就是剩下的B组)吸取均值，缩短输出为仅剩的 dst
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        # 参考当时保存下来的乱序 a_idx, b_idx 索引将他们重新安放在本属于他们原先所在的相对空间。
        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    【加权平均合并】这是 ToMe 的重点辅助函数。
    由于 token 合并后会变得类似一个聚拢滚雪球。如果两个 token 合并了，它的影响力在后续再和别的发生合并时理应更大。
    为了不让最后求平均被稀释失真，我们不能只做朴素的 `mode="mean"`。我们要按这个复合 Token 里面包含的原始像素数量（size）进行加权求均值。
    """
    if size is None:
        # 起初在第一层时，如果没有赋 size，每一个特征点(token) 自身就是 1 个基本单元体。
        size = torch.ones_like(x[..., 0, None])

    # 先用这个 token 已累积的质量 (size) 去放大此处的原特征值
    # 在进行 merge 时，我们必须传递 mode="sum"! 让两个被放大的特征做求和累积，
    # 这样新的目标结点的 “未整除特征总质量” 就是两者加权之结合。
    x = merge(x * size, mode="sum")
    
    # 相应地，我们记录在这个目标节点里，最终堆叠聚集了总共多少个基础小 token 单元。
    size = merge(size, mode="sum")

    # 根据刚刚加和膨胀过的数据，再除以最新总 size ，这便获得了严密无误的“当前融合组的整体加权特征平均数”。
    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    用于跟踪源的可视化溯源函数模块。用于记录和关联：追踪分析原始层级的某个小网格像素特征，它是如何一步一步最终被合并汇入进某一个极深层宏观 token 的。
    """
    if source is None:
        # 初始没数据时根据前向张量开一份记录盘。
        n, t, _ = x.shape
        # 用对角线为1的单位矩阵 eye 表示：最开始我只属于我自己
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    # 通过合并网络追踪合并走势。注意此处的合并策略并不是求和或者平摊均分，
    # 而是用 "amax"（取绝对最大值），这相当于是建立并维护包含从属关系的邻接矩阵（即保留最大溯源连线），形成 Boolean OR 级的传承合并！
    source = merge(source, mode="amax")
    return source

