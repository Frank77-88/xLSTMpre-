# xLSTM-pre++

`xLSTM-pre++` 现在整理为 **5 个正式版本**，统一使用同一套数据协议、训练脚本、评估脚本和可视化脚本，便于做消融实验。

## 正式版本

- `LSTM++`
- `xLSTM++`
- `GAT-xLSTM++`
- `GAT-xLSTM-K++`
- `GAT-xLSTM-K-I++`

它们对应的配置文件分别是：

- `configs/LSTM++.yaml`
- `configs/xLSTM++.yaml`
- `configs/GAT-xLSTM++.yaml`
- `configs/GAT-xLSTM-K++.yaml`
- `configs/GAT-xLSTM-K-I++.yaml`

## 实验配置

除上面 **5 个正式版本** 外，项目额外提供一个**独立实验配置**：

- `configs/GAT-xLSTM-K-F++.yaml`

它不计入正式五版本，也不会替换当前默认训练入口。这个实验配置专门用于验证：

- `winner_ade_weight = 0.3` 的更强终点导向 winner 选择
- `decoder LSTMCell -> xLSTM block` 的解码器替换
- 当前 loss 已回退为 **soft cls only**
- 当前不再使用单独 `endpoint/FDE loss`

也就是说，当前正式五版本保持不变，`K-F++` 只是新增的隔离实验分支。

## 统一实验协议

5 个版本统一采用：

- `obs_len: 100`
- `pred_len: 100`
- `dt: 0.04`
- `dense 4s observation + 4s prediction`
- `train: DJI_0001 ~ DJI_0016`
- `val: DJI_0021, DJI_0022`
- `test: DJI_0028, DJI_0029`
- `filter_reverse: false`
- `window_stride: 10`
- `num_epochs: 100`
- `scheduler: CosineAnnealing`
- `eta_min: 0.00005`

其中：

- 5 个正式版本统一使用：
  - `lambda_cls: 0.5`
  - `tau_start: 2.0`
  - `tau_end: 0.3`
  - `tau_anneal_epochs: 100`
- `LSTM++ / xLSTM++ / GAT-xLSTM++ / GAT-xLSTM-K++` 默认 `min_future_displacement: 0.0`
- `GAT-xLSTM-K-I++` 默认 `min_future_displacement: 0.2`
- 当前 `GAT-xLSTM-K-I++` 不再使用旧的 soft winner / hard / branch 训练项

## 当前结构说明

### 1. `LSTM++`

- `LSTM + displacement decoder`
- 不使用 GAT
- 不使用 kinematic decoder
- 使用 **地图软信号**

### 2. `xLSTM++`

- `xLSTM_v3 + displacement decoder`
- 不使用 GAT
- 使用 **地图软信号**

### 3. `GAT-xLSTM++`

- `hetero GAT + xLSTM_v3 + displacement decoder`
- 使用 **地图软信号**

### 4. `GAT-xLSTM-K++`

- `hetero GAT + xLSTM_v3 + kinematic decoder`
- 使用 **TopologyLite 局部地图软信号**
- **不使用意图识别**
- 作为 `K-I++` 的直接对照基线

### 5. `GAT-xLSTM-K-I++`

- `hetero GAT + xLSTM_v3 + kinematic decoder`
- 使用 **TopologyLite 局部地图软信号**
- 启用 **behavior condition -> control -> trajectory** 两阶段预测

它是当前主推版本。

---

## 局部场景范围

5 个正式版本现在统一采用 **ego-centric 20m 局部范围**：

- 所有参与 scene fusion 的邻居特征只保留 **ego 20m 内**
- 所有参与 encoder 的地图特征只保留 **ego 20m 内**
- decoder 逐步刷新的局部地图也统一使用 **20m 范围**

对应配置项：

- `data.local_scene_radius: 20.0`
- `map.decoder_local_map_radius: 20.0`

---

## 地图软信号

5 个正式版本现在都启用同一套 **局部地图软信号**：

- 不做全图 attention
- 不依赖 waypoint 箭头方向
- 只编码当前位置附近的：
  - `waypoint segments`
  - `hard segments`
  - `parking slot polygons`

非 GAT 版本的地图编码结构统一为：

- `MLP + masked mean pooling + masked max pooling`

GAT 版本则使用：

- `soft map tokens`
- `hard map tokens`
- `ego / obstacle / hard_map / soft_map` 的局部异构图融合

这套地图信号现在以 **20m 局部 token / context** 的方式注入模型，而不是做硬规则约束。

当前项目中已经删除旧的：

- `hard projection / legality projection`
- `hetero` 独立实验支线

---

## `GAT-xLSTM-K-I++` 设计

当前 `K-I++` 已切换为 **behavior-conditioned control generation**：

- **先用 xLSTM 编码动态体历史**
- **再用局部异构 GAT 融合 ego-obstacle-hard-soft**
- **再生成高层行为条件**
- **再在控制空间中生成 coarse control**
- **随后通过 TopologyLite residual decoder 做控制细化**
- **最终通过车辆运动学展开为 K 条轨迹**
- **再由 mode score head 对多行为轨迹进行概率排序**

当前不再使用：

- `intent classification`
- `goal candidate generation`
- `goal scoring`
- `top_m`
- `weighted K-means`

旧的 `intent-goal` 训练支线代码已经从主项目实现中移除，当前 `K-I++` 只保留行为条件版本。

### 行为条件设计

当前固定采用：

- `K = num_modes` 个 learnable behavior queries
- 每个 behavior query 生成一个高层 `behavior condition`
- 每个 condition 输出 `coarse_control_points` 个控制关键帧
- 控制关键帧插值为完整 `(a_x, psi_dot)` 控制序列
- 再由 decoder 预测残差控制 `Δu`

最终控制为：

- `u_final = u_coarse + Δu`

### 训练 loss

`GAT-xLSTM-K-I++` 使用 4 个核心 loss：

- `L_reg`
  - 轨迹回归损失
  - 当前实现为：**逐点轨迹误差 + 终点误差加权**
- `L_cls`
  - 多行为模态概率监督
- `L_inertia`
  - 轻量运动连续性约束
- `L_ctrl_smooth`
  - 控制序列平滑约束

当前已经删除或停用：

- `hard loss`
- `branch × submode`
- `branch supervision`
- `diversity loss`
- `intent loss`
- `goal candidate / goal ranking loss`

### 非意图版 loss

其余 4 个正式版本现在统一收敛到更干净的训练目标：

- `L_reg`
  - 多模态轨迹回归损失
- `L_cls`
  - 模态概率监督
- `L_inertia`
  - 仅在 kinematic 版本中启用的轻量连续性约束

---

## 训练

激活环境：

- `source ../venv/bin/activate`

默认训练入口现在指向：

- `configs/GAT-xLSTM-K-I++.yaml`

也就是直接运行：

- `python train.py`

等价于：

- `python train.py --config configs/GAT-xLSTM-K-I++.yaml`

训练新增实验版本：

- `python train.py --config configs/GAT-xLSTM-K-F++.yaml`

训练所有正式版本：

- `python train.py --config configs/LSTM++.yaml`
- `python train.py --config configs/xLSTM++.yaml`
- `python train.py --config configs/GAT-xLSTM++.yaml`
- `python train.py --config configs/GAT-xLSTM-K++.yaml`
- `python train.py --config configs/GAT-xLSTM-K-I++.yaml`

---

## 评估

验证集示例：

- `python evaluate.py --config configs/GAT-xLSTM-K-I++.yaml --checkpoint checkpoints/GAT-xLSTM-K-I++/best.pth --split val`
- `python evaluate.py --config configs/GAT-xLSTM-K++.yaml --checkpoint checkpoints/GAT-xLSTM-K++/best.pth --split val`
- `python evaluate.py --config configs/GAT-xLSTM++.yaml --checkpoint checkpoints/GAT-xLSTM++/best.pth --split val`
- `python evaluate.py --config configs/xLSTM++.yaml --checkpoint checkpoints/xLSTM++/best.pth --split val`
- `python evaluate.py --config configs/LSTM++.yaml --checkpoint checkpoints/LSTM++/best.pth --split val`

实验版本评估示例：

- `python evaluate.py --config configs/GAT-xLSTM-K-F++.yaml --checkpoint checkpoints/GAT-xLSTM-K-F++/best.pth --split val`

说明：

- `evaluate.py` 已经适配 `configs/GAT-xLSTM-K-F++.yaml`
- 评估脚本会自动读取该配置中的 `pred_len` 和 `dt`
- 因此当前 `dense 4s + 4s` 设定下，会自动输出 `1s ~ 4s` 的 `minADE / minFDE`
- 不需要再手动改 horizon 参数

输出指标包括：

- `ADE@1`
- `FDE@1`
- `minADE@K`
- `minFDE@K`
- `MR@2m`
- `Intrusion@1`
- `minIntrusion@K`
- `minADE@1s`
- `minADE@2s`
- `minADE@3s`
- `minADE@4s`
- `minFDE@1s`
- `minFDE@2s`
- `minFDE@3s`
- `minFDE@4s`

当前评估时域会自动跟随配置中的完整预测 horizon；在当前 dense 4+4 协议下，不再输出 `5s` 指标。

---

## 可视化

单样本可视化：

- `python visualize_predictions.py --config configs/GAT-xLSTM-K-I++.yaml --checkpoint checkpoints/GAT-xLSTM-K-I++/best.pth --sample_idx 0 --top_k 3`

批量导出：

- `python visualize_predictions.py --config configs/GAT-xLSTM-K-I++.yaml --checkpoint checkpoints/GAT-xLSTM-K-I++/best.pth --sample_idx 120 --sample_idx_end 150 --top_k 3`

### `F++` 完整命令块

`visualize_predictions.py` 已适配 `GAT-xLSTM-K-F++`，`visualize_sliding.py` 也已适配 `GAT-xLSTM-K-F++`。其中滑动窗口动画使用的是“保留历次预测、当前帧继续滚动”的版本。

```bash
source ../venv/bin/activate

python visualize_predictions.py \
  --config configs/GAT-xLSTM-K-F++.yaml \
  --checkpoint checkpoints/GAT-xLSTM-K-F++/best.pth \
  --split val \
  --sample_idx 0 \
  --top_k 3

python visualize_predictions.py \
  --config configs/GAT-xLSTM-K-F++.yaml \
  --checkpoint checkpoints/GAT-xLSTM-K-F++/best.pth \
  --split val \
  --sample_idx 120 \
  --sample_idx_end 150 \
  --top_k 3

python visualize_sliding.py \
  --config configs/GAT-xLSTM-K-F++.yaml \
  --checkpoint checkpoints/GAT-xLSTM-K-F++/best.pth \
  --split val \
  --traj_idx 0 \
  --top_k 3 \
  --save_gif \
  --output visualizations/GAT-xLSTM-K-F++/sliding_prediction_accumulated.gif

python visualize_sliding.py \
  --config configs/GAT-xLSTM-K-F++.yaml \
  --checkpoint checkpoints/GAT-xLSTM-K-F++/best.pth \
  --split val \
  --traj_idx 0 \
  --top_k 3 \
  --prediction_history 30 \
  --save_gif \
  --output visualizations/GAT-xLSTM-K-F++/sliding_prediction_recent30.gif
```

说明：

- `visualize_predictions.py` 已适配 `GAT-xLSTM-K-F++`
- `visualize_sliding.py` 也已适配 `GAT-xLSTM-K-F++`
- `visualize_sliding.py` 现在支持“每次滑窗预测都保留、当前帧继续向前滚动”的播放方式
- 默认 `--prediction_history 0` 表示保留全部历史预测

当前 `K-I++` 行为条件版本的图例不再依赖显式 intent 名称，
而是直接显示概率最高的多条预测轨迹。

---

## 训练曲线

根据训练历史绘图：

- `python plot_training_curves.py --history logs/GAT-xLSTM-K-I++/training_history.json`

当前曲线图会分别画：

默认会在 `training_history_plots/` 目录下导出多张独立图片。
现在每种 loss 都会单独保存成一张图，并且每张图只包含对应的 `train / val` 两条曲线：

- `total_loss.png`
- `reg_loss.png`
- `cls_loss.png`
- `inertia_loss.png`
- `ctrl_smooth_loss.png`

如果训练历史里包含新增分量，绘图脚本还会自动额外导出：

- `end_loss.png`
- `cls_soft_loss.png`
- `cls_hard_loss.png`

其他指标也会分开导出：

- `ade_at_1.png`
- `fde_at_1.png`
- `minade_at_k.png`
- `minfde_at_k.png`
- `safety_metrics.png`

---

## 配置建议

如果你现在主要想做对比实验，建议先跑这 3 组：

- `GAT-xLSTM++`
- `GAT-xLSTM-K++`
- `GAT-xLSTM-K-I++`

这样可以分别对比：

- GAT 主干 + 地图软信号
- GAT 主干 + kinematic decoder + 地图软信号
- GAT 主干 + kinematic decoder + 地图软信号 + behavior-conditioned control generation

---

## 备注

- 当前项目保留倒车样本：`filter_reverse: false`
- 5 个正式版本都已经统一使用地图软信号
- GAT 版本现在统一使用 **局部异构 GAT**
- 所有 scene-level 特征统一限制在 **ego 20m 内**
- 当前正式协议已经切换到 **dense 4s + 4s**
- 当前行为条件版本不依赖 lane arrow / route id
- 当前设计目标是让项目在**换地图后仍可继续使用**
- 数据集 cache key 已升级；旧的 `3s + 5s` 缓存不会再被当前配置复用
- 完整设计文档见 `sheji.md`
