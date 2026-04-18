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
    tome_info: dict = None,
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

    if tome_info is None: tome_info = {}

    with torch.no_grad():
        dist_func = tome_info.get("distance_func", "cosine")
        part_style = tome_info.get("partition_style", "alternating")

        if dist_func == 'cosine':
            metric_norm = metric / metric.norm(dim=-1, keepdim=True)
        else:
            metric_norm = metric

        if part_style == 'sequential':
            mid = math.ceil(t / 2)
            a, b = metric_norm[:, :mid, :], metric_norm[:, mid:, :]
        elif part_style == 'random':
            B, N, C = metric_norm.shape
            rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)
            a_idx, b_idx = rand_idx[:, :N//2, :], rand_idx[:, N//2:, :]
            a = metric_norm.gather(dim=1, index=a_idx.expand(B, N//2, C))
            b = metric_norm.gather(dim=1, index=b_idx.expand(B, N - N//2, C))
        else: # alternating
            a, b = metric_norm[..., ::2, :], metric_norm[..., 1::2, :]
        
        if dist_func == 'eucl':
            scores = -torch.cdist(a, b)
        elif dist_func == 'softmax':
            scores = (a @ b.transpose(-1, -2)).softmax(dim=-1)
        else: # cosine or dot
            scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        src_idx = edge_idx[..., :r, :]  
        unm_idx = edge_idx[..., r:, :]  
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        part_style = tome_info.get("partition_style", "alternating")
        combine_method = tome_info.get("combine_method", "weighted avg")
        
        if part_style == 'sequential':
            mid = math.ceil(t / 2)
            src, dst = x[:, :mid, :], x[:, mid:, :]
        elif part_style == 'random':
            C = x.shape[-1]
            src = x.gather(dim=1, index=a_idx.expand(-1, -1, C))
            dst = x.gather(dim=1, index=b_idx.expand(-1, -1, C))
        else: # alternating
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        
        # Determine combine method
        # 注意：当 combine_method == 'weighted avg' 时，不能覆盖 mode！
        # 因为 merge_wavg 会传入 mode="sum" 来实现加权平均，如果被覆盖成 "mean" 就全废了
        act_mode = mode
        if combine_method == 'max pool':
            act_mode = 'amax'
        elif combine_method == 'avg pool':
            act_mode = 'mean'
        # weighted avg: 保持外部传入的 mode 不变（merge_wavg 传 "sum"，默认传 "mean"）
            
        if combine_method == 'keep one':
            pass
        else:
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=act_mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        part_style = tome_info.get("partition_style", "alternating")
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        if part_style == 'sequential':
            mid = math.ceil(t / 2)
            out[:, mid:, :] = dst
            out.scatter_(dim=-2, index=unm_idx.expand(n, unm_len, c), src=unm)
            out.scatter_(dim=-2, index=src_idx.expand(n, r, c), src=src)
        elif part_style == 'random':
            out.scatter_(dim=-2, index=b_idx.expand(n, n - n//2, c), src=dst)
            src_orig = torch.zeros(n, n//2, c, device=x.device, dtype=x.dtype)
            src_orig.scatter_(dim=-2, index=unm_idx.expand(n, unm_len, c), src=unm)
            src_orig.scatter_(dim=-2, index=src_idx.expand(n, r, c), src=src)
            out.scatter_(dim=-2, index=a_idx.expand(n, n//2, c), src=src_orig)
        else:
            out[..., 1::2, :] = dst
            out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
            out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

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
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, tome_info: dict = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tome_info is None: tome_info = {}
    
    if tome_info.get("combine_method", "weighted avg") == 'weighted avg':
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = merge(x * size, mode="sum")
        size = merge(size, mode="sum")
        x = x / size
        return x, size
    else:
        x = merge(x)
        return x, None


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

