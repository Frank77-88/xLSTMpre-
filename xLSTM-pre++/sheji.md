# xLSTM-pre++ 最终设计文档 v1

## 1. 设计目标

本项目面向**停车场车辆轨迹预测**，目标是在保留现有主干优势的前提下，形成一套更统一、更可解释、且更适合地图约束场景的最终架构。

本版设计坚持以下原则：

- 保留 `LSTM / xLSTM` 作为**动态体时序编码器**
- 保留**地图软信号 / 硬信号**
- 将原先仅建模车车关系的 `GAT` 升级为**局部异构图融合模块**
- 将多模态预测改为**行为条件控制生成**
- 保留**运动学展开**
- 删除旧的、重复的、解释性较弱的分支设计，保持项目干净

### 当前正式协议

- 继续保留 `dt = 0.04` 的 dense 采样
- 统一采用 **4s observation + 4s prediction**
- 因此当前正式配置统一为：
  - `obs_len = 100`
  - `pred_len = 100`
- 该协议用于项目内部五个正式版本的主实验与消融实验

---

## 2. 全局范围约束：统一使用 ego 20m 局部范围

为降低计算量并统一特征来源，最终版采用**以 ego 为中心、半径 20m 的局部感知范围**。

### 2.1 统一规则

- 所有参与 encoder 的动态体特征，只保留 **ego 20m 内** 的邻居 / 障碍车
- 所有参与 encoder 的地图特征，只保留 **ego 20m 内** 的 `soft map` 与 `hard map`
- 所有参与异构图融合的节点，只来自 **ego 20m 内**
- 行为层输入的场景特征，也只来自 **ego 20m 内**
- decoder 端逐步刷新的局部地图，也继续使用 **ego 当前预测位置附近 20m 内** 的局部地图子集

### 2.2 具体含义

这里的“20m 内”建议统一定义为：

- 以 ego 当前时刻位置为中心
- 使用局部坐标系或全局坐标系中的二维欧氏距离
- 阈值为 `20.0m`

### 2.3 这样做的好处

- 显著减少 `GAT` 节点数和边数，降低计算量与显存占用
- 统一各模块看到的场景范围，减少不一致
- 更符合停车场局部决策场景：远处信息通常对当前短期预测贡献有限

---

## 3. 最终总体思路

最终模型采用四段式结构：

1. **动态体时序编码**
2. **局部异构场景图融合**
3. **行为条件控制生成**
4. **运动学约束轨迹展开**

一句话概括：

> 用 xLSTM 编码每个动态体在 20m 局部范围内的历史运动，用异构 GAT 融合 ego-obstacle-hard-soft 的局部场景关系，再基于融合后的 ego 场景表征生成行为条件控制，最终通过运动学层展开成多模态未来轨迹。

---

## 4. 最终结构流程图

```text
[ Ego observed trajectory ]
            │
            ▼
   [ Temporal Encoder ]
   (LSTM / xLSTM / xLSTM-V3)
            │
            ▼
        [ Ego token ]
            │
            │
            ├──────────────────────────────────────────────┐
            │                                              │
            │                                              │
[ Obstacles within ego 20m ]                    [ Local map subset within ego 20m ]
            │                                              │
            ▼                                              ▼
   [ Temporal Encoder ]                     [ Soft map token encoder / Hard map token encoder ]
            │                                              │
            ▼                                              │
    [ Obstacle tokens ]                                    │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
                 [ Local Heterogeneous Scene Graph ]
      (nodes: ego / obstacle / hard_map / soft_map, all within ego 20m)
                                   │
                                   ▼
                   [ Relation-aware Hetero-GAT Fusion ]
                                   │
                                   ▼
                     [ Fused ego scene representation ]
                                   │
                                   ▼
                       [ Behavior / Strategy Layer ]
                                   │
                                   ▼
                  [ Coarse control generation per mode ]
                                   │
                                   ▼
                   [ Residual control sequence decoder ]
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
      [ hard local signal refresh ]   [ local topology refresh within 20m ]
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                        [ Kinematic rollout layer ]
                                   │
                                   ▼
                  [ Multimodal future trajectories + scores ]
```

---

## 5. 核心模块设计

## 5.1 动态体时序编码
**Dynamic Temporal Encoding**

### 目标

对每个动态体各自的历史轨迹进行时间建模，提取稳定的运动语义。

### 输入

- ego 历史轨迹
- `ego 20m 内` 的邻车 / 障碍车历史轨迹

### 输出

- `ego token`
- `obstacle tokens`

### 设计原则

- 时序编码仍由 `LSTM / xLSTM` 完成
- 这一步**不用 GAT**
- 因为这里回答的是：**“这个体过去怎么运动”**

### 说明

- `xLSTM` 负责**时间建模（temporal modeling）**
- `GAT` 负责**关系建模（relational modeling）**
- 两者职责不能混淆

---

## 5.2 局部异构场景图融合
**Local Heterogeneous Scene Graph Fusion**

### 目标

统一建模以下局部关系：

- ego 与 obstacle
- ego 与 hard map
- ego 与 soft map
- 可选：obstacle 与 map

### 节点类型
**Node Types**

- `ego`
- `obstacle`
- `hard_map`
- `soft_map`

### 边类型
**Edge Types**

- `ego ↔ obstacle`
- `ego ↔ hard_map`
- `ego ↔ soft_map`

可选边：

- `obstacle ↔ hard_map`
- `obstacle ↔ soft_map`

### 范围约束

局部异构图中的所有节点和边，都只在 **ego 20m 内** 构建。

### 为什么仍然是 GAT

这一步本质上仍是**基于 GAT 的图融合**，但不是普通同构 GAT，而是：

- `heterogeneous GAT`
- `relation-aware GAT`
- `typed GAT`

因此论文表述可以写成：

- **基于 GAT 的局部异构场景图融合模块**
- **GAT-based local heterogeneous scene graph fusion**

### 设计要点

- 不做全图 attention
- 不做全场景全连接图
- 采用 **ego-centric local graph**
- 优先保留 `ego` 的一跳重要关系

---

## 5.3 地图建模
**Soft / Hard Map Modeling**

## 5.3.1 Soft Map

### 含义

软地图表示可行驶趋势、弱引导信息，例如：

- waypoint
- guide path
- 局部通行带
- 局部拓扑引导线

### 作用

- 提供“更可能往哪走”的弱偏好
- 帮助行为层形成更合理的分叉模式

### 处理方式

- 先做局部裁剪，只保留 **ego 20m 内**
- 保留为 `soft map tokens`
- 进入异构图，而不是一开始就池化成单个向量

## 5.3.2 Hard Map

### 含义

硬地图表示强约束信息，例如：

- 障碍边界
- 硬边界
- 不可穿越区域
- 墙体 / 障碍物轮廓

### 作用

- 在 encoder 端参与异构图融合，帮助理解局部场景
- 在 decoder 端继续作为逐步约束信号刷新

### 处理方式

- encoder 端：作为 `hard_map tokens` 进入异构图
- decoder 端：继续保留 `hard_signal + topology refresh`
- 所有 hard map token 也只来自 **ego 20m 内**

---

## 5.4 行为条件层
**Behavior-conditioned Mode Layer**

### 目标

不再直接把模态理解为“纯轨迹分支”，而是理解为**高层行为 / 策略条件**。

### 行为层负责回答

- 接下来是继续前进还是倒车
- 是否即将出车位
- 若出车位，是否存在左 / 右转分叉
- 当前局部场景下有哪些合理控制策略

### 关键原则

行为层输入必须来自：

- `ego token`
- `ego 20m 内` 的 obstacle / hard / soft 图融合结果

也就是说，行为层只看**局部场景表征**，不看远处无关特征。

### 输出

- 多个行为条件查询 `behavior-conditioned mode queries`
- 对应每个 mode 的 logits / probabilities

---

## 5.5 控制生成层
**Control-space Generation**

### 目标

在每个行为条件下，先生成粗控制，再做细化，而不是直接从行为跳到整条轨迹。

### 两阶段生成

1. **粗控制关键点**
   - 生成少量 control keyframes
2. **残差控制序列**
   - 用 decoder 对每步控制做细化修正

### 为什么在控制空间建模

停车场轨迹是否合理，更依赖：

- 转向过程
- 速度变化
- 倒车 / 前进切换
- 出车位动作

这些都更适合在控制空间表达，而不是只在终点空间表达。

---

## 5.6 运动学展开层
**Kinematic Rollout Layer**

### 目标

将控制序列展开成未来轨迹，同时显式保证基础动力学合理性。

### 保留原因

- 停车场低速场景非常适合运动学约束
- 比纯位置回归更稳定
- 对倒车、进 / 出车位动作更自然

### 与地图信号的关系

在 rollout 过程中继续使用：

- `hard_signal`
- `topology_refresh`

并且 decoder 刷新的局部地图也统一限制在**当前预测位置附近 20m 内**。

这与 encoder 端异构图不是重复，而是两个层次：

- encoder：理解局部场景
- decoder：生成时逐步受局部约束

---

## 6. 最终五个模型版本

为了便于消融实验，项目保留以下五个正式版本。

## 6.1 `LSTM++`

### 定义

- 时序编码器：`LSTM`
- 地图：有 soft / hard map
- 图融合：无异构 GAT
- 行为条件：无

### 用途

- 最基础 baseline

## 6.2 `xLSTM++`

### 定义

- 时序编码器：`xLSTM V3`
- 地图：有 soft / hard map
- 图融合：无异构 GAT
- 行为条件：无

### 用途

- 验证时序主干从 LSTM 升级到 xLSTM 的效果

## 6.3 `GAT-xLSTM++`

### 定义

- 时序编码器：`xLSTM`
- 图融合：局部异构 GAT
- 地图：soft / hard map 进入异构图
- 行为条件：无

### 用途

- 验证异构场景图融合的作用

## 6.4 `GAT-xLSTM-K++`

### 定义

- 时序编码器：`xLSTM`
- 图融合：局部异构 GAT
- 地图：soft / hard map 进入异构图
- 行为条件：无显式行为层
- 解码：直接多模态控制生成

### 用途

- 验证加入控制空间多模态后、但不做显式行为层的性能

## 6.5 `GAT-xLSTM-K-I++`

### 定义

- 时序编码器：`xLSTM`
- 图融合：局部异构 GAT
- 地图：soft / hard map 进入异构图
- 行为条件：有
- 解码：行为条件控制生成 + 运动学展开

### 用途

- 最终完整模型
- 论文主模型候选

---

## 7. Loss 设计（v1）

v1 建议保持简洁，不做过多辅助项。

## 7.1 主损失

- `L_reg`
- `L_cls`
- `L_inertia`
- `L_ctrl_smooth`

## 7.2 各损失含义

### `L_reg`

轨迹回归损失，用于约束预测轨迹与 GT 轨迹的时序位置误差。

### `L_cls`

模态 / 行为分数损失，用于让最接近 GT 的 mode 获得更高概率。

### `L_inertia`

惯性平滑损失，约束运动变化不要过于抖动，防止产生不合理控制跳变。

### `L_ctrl_smooth`

控制平滑损失，作用在控制序列上，鼓励相邻时刻控制变化平滑。

## 7.3 v1 明确不加入

为了保持项目干净，v1 不加入以下内容：

- legality rerank
- 全图 attention
- 旧 intent-goal candidate 分支
- div loss
- branch × submode 旧设计
- turn_into_obstacle penalty
- `L_region_aux`
- 复杂多阶段 rerank

---

## 8. 需要删除 / 废弃的旧思路

## 8.1 废弃：仅车车 `Social GAT` 主路径

原因：

- 只建模车车，不适合停车场地图主导场景

## 8.2 废弃：encoder 端地图先池化再主融合

原因：

- 会损失 token-level 结构信息
- 与异构图设计冲突

## 8.3 废弃：旧 intent-goal candidate 路线

原因：

- 与当前行为条件控制生成方案重复
- 解释链条过长

## 8.4 废弃：branch × submode 设计

原因：

- 会在明显直行场景产生错误分叉
- 对停车位内部误导较大

---

## 9. 最终需要更改什么

本节给出最终实现时的**直接修改清单**。

## 9.1 `xLSTM-pre++/xlstm_prepp/models/social_gat.py`

### 原实现问题

- 原实现是 `SocialInteractionGAT`
- 只建模 `ego + neighbors`
- 没有节点类型 / 边类型区分

### 最终要改成

- 新的 `HeteroSceneGAT`
- 支持：
  - `ego`
  - `obstacle`
  - `hard_map`
  - `soft_map`
- 支持：
  - `node type embeddings`
  - `edge type aware attention`

### 同时新增

- 只构建 **ego 20m 内** 的节点与边

---

## 9.2 `xLSTM-pre++/xlstm_prepp/models/topology_lite_encoder.py`

### 当前问题

- 现在主要输出 pooled `topology_context`
- 更像“局部地图向量编码器”

### 最终要改成

- `soft map token encoder`
- `hard map token encoder`
- 输出 token-level 表征，而不是只输出池化后的单向量

### 同时新增

- 地图 token 统一只保留 **ego 20m 内**

---

## 9.3 `xLSTM-pre++/xlstm_prepp/models/topology_lite.py`

### 当前问题

- 当前主路径更接近：
  - `xLSTM -> social GAT -> pooled topology concat -> behavior layer`

### 最终要改成

- `xLSTM(dynamic tokens)`
- `map token encoder`
- `hetero scene graph`
- `hetero-GAT fusion`
- `fused ego scene token`
- `behavior-conditioned mode builder`

### 关键变更

- 删除 encoder 端主路径里的“地图先池化再直接拼接”核心逻辑
- 改为 token-level graph fusion
- 所有参与 scene fusion 的特征统一限制在 **ego 20m 内**

---

## 9.4 `xLSTM-pre++/xlstm_prepp/models/topology_lite_decoder.py`

### 当前建议

这部分**保留**：

- `hard_signal`
- `topology_refresh_steps`
- decoder 端局部 topology refresh

### 需要补充的约束

- decoder 刷新的局部地图，也统一限制在**当前预测位置附近 20m 内**

---

## 9.5 `xLSTM-pre++/xlstm_prepp/models/xlstm_encoder.py`

### 当前建议

- 保留其作为**动态体时序编码器**
- 不在图融合步骤里重复使用 xLSTM

### 后续可选增强

如果要更接近论文原版，可继续增强：

- `causal Conv4`
- `block-diagonal multi-head projections`
- `GroupNorm / head-wise norm`
- `up/down projection`
- `learnable skip`

但这些属于**xLSTM 论文对齐增强**，不是这次结构重构的主线。

---

## 9.6 配置文件需要新增 / 统一的关键项

建议在最终 yaml 中统一出现以下配置概念：

- `local_scene_radius: 20.0`
- `use_hetero_scene_gat: true`
- `hetero_num_heads`
- `hetero_hidden_dim`
- `hetero_use_node_types: true`
- `hetero_use_edge_types: true`
- `max_obstacles_in_range`
- `max_soft_map_tokens_in_range`
- `max_hard_map_tokens_in_range`
- `decoder_local_map_radius: 20.0`
- `enable_behavior_conditioning: true`

其中最重要的统一约束是：

- `local_scene_radius: 20.0`
- `decoder_local_map_radius: 20.0`

---

## 10. 论文表述建议

### 中文

可写成：

> 本文提出一种基于 xLSTM 时序编码与局部异构图注意力融合的停车场轨迹预测框架。模型首先对自车及 ego 20m 局部范围内的周围动态体进行时序编码，再构建包含自车、障碍车、硬地图与软地图 token 的局部异构场景图，通过关系感知 GAT 进行场景融合，随后在行为条件下生成控制序列，并结合运动学层展开为多模态未来轨迹。

### 英文

可写成：

> We propose a parking-lot trajectory prediction framework based on xLSTM temporal encoding and local heterogeneous graph attention fusion. The model first encodes the motion histories of the ego vehicle and surrounding dynamic agents within an ego-centric 20 m local range, then builds a local heterogeneous scene graph containing ego, obstacle, hard-map, and soft-map tokens. A relation-aware GAT is used for scene fusion, followed by behavior-conditioned control generation and kinematic rollout for multimodal trajectory prediction.

---

## 11. 最终一句话总结

**v1 最终版 = xLSTM 时序编码 + ego 20m 局部异构 GAT 场景融合 + 行为条件控制生成 + 运动学展开 + decoder 端 20m 局部地图约束。**
