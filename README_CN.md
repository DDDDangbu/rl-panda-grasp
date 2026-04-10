# RL-Panda-Grasp

**基于自适应课程学习的机械臂强化学习抓取**

[English](README.md) | 中文文档

![Demo](results/videos/demo_d0.1.gif)

## 项目亮点

- **SAC + HER (后见经验回放)**: 解决稀疏奖励下的机器人操作难题
- **自适应课程学习**: 根据训练表现自动调整任务难度 (reaching → pushing → lifting → pick-and-place)
- **完整消融实验**: SAC vs TD3、有/无课程学习、不同 HER 策略、奖励函数对比
- **PyBullet + panda-gym** 仿真平台，自定义扩展环境
- 专业代码结构：YAML 配置、单元测试、TensorBoard 日志、可复现性

## 核心方法

### 1. 为什么用稀疏奖励 + HER？

在机器人抓取任务中，**稠密奖励**(如负距离)容易导致策略陷入局部最优——机器人可能学会把物体推向目标，而不是拿起来。**稀疏奖励**（到了就是0，没到就是-1）定义清晰但信号稀少。

**后见经验回放 (HER)** 解决了这个矛盾：即使机器人没有把物体放到目标位置，HER 会"假装"实际到达的位置就是目标，从而把失败经验变成成功经验。这大幅提升了学习效率。

### 2. 为什么用课程学习？

直接训练完整的 pick-and-place 任务仍然很难——机器人需要同时学会：
1. **接近** 物体
2. **抓住** 物体
3. **搬运** 到目标位置
4. **释放** 物体

课程学习将这个过程分解为由易到难的阶段：

| 阶段 | 难度 | 描述 |
|------|------|------|
| Reaching | 0.0-0.2 | 目标就在物体旁边，只需接近 |
| Pushing | 0.2-0.5 | 目标在桌面上较远处 |
| Lifting | 0.5-0.8 | 目标在桌面上方，需要抓起物体 |
| Full P&P | 0.8-1.0 | 完整 pick-and-place，全范围随机化 |

**自适应调度**: 连续 3 次评估成功率 > 60% → 难度 +0.1；成功率 < 10% → 难度 -0.05（防止崩溃）。

### 3. SAC vs TD3

| 特点 | SAC | TD3 |
|------|-----|-----|
| 探索策略 | 熵正则化（自动调整） | 高斯噪声 |
| 适合场景 | 多模态策略、复杂探索 | 稳定、低方差 |
| 预期表现 | 更好（抓取动作多样性高） | 次优但更稳定 |

## 快速开始

### 环境安装

```bash
git clone https://github.com/<your-username>/rl-panda-grasp.git
cd rl-panda-grasp

# 创建虚拟环境（推荐 Python 3.11+）
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# macOS: pybullet 编译需要抑制警告
CFLAGS="-w" pip install pybullet

# 安装项目和剩余依赖
pip install -e ".[dev]"
```

**常见问题排查**:
- PyBullet 编译错误 (macOS): 先执行 `CFLAGS="-w" pip install pybullet`，再安装其余依赖
- CUDA 版本不匹配: 用 `pip install torch --index-url https://download.pytorch.org/whl/cu118` 安装对应版本
- 国内镜像加速: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"`

### 训练

```bash
# SAC + HER + 课程学习（推荐）
python -m training.train --config configs/sac_her.yaml

# TD3 + HER + 课程学习
python -m training.train --config configs/td3_her.yaml

# 自定义参数
python -m training.train --config configs/sac_her.yaml --seed 123 --total_timesteps 100000

# 不使用课程学习（对比用）
python -m training.train --config configs/sac_her.yaml --no_curriculum
```

### 消融实验

```bash
# 运行全部 5 个消融实验
python -m training.ablation

# 快速测试（减少训练步数）
python -m training.ablation --timesteps 100000
```

### 评估

```bash
# 评估模型
python -m evaluation.evaluate --model results/models/sac_her_curriculum/best_model.zip

# 不同难度下的表现
python -m evaluation.evaluate --model results/models/sac_her_curriculum/best_model.zip --sweep
```

### 录制 Demo

```bash
python -m evaluation.record_video --model results/models/sac_her_curriculum/best_model.zip
```

### 查看训练曲线

```bash
tensorboard --logdir results/logs
# 浏览器打开 http://localhost:6006
```

### 生成对比图

```bash
python -m evaluation.plot_results --log_dir results/logs --output results/plots
```

## 项目结构

```
rl-panda-grasp/
├── configs/                # YAML 训练配置
│   ├── sac_her.yaml        # SAC + HER 超参数
│   └── td3_her.yaml        # TD3 + HER 超参数
├── envs/                   # 自定义 Gymnasium 环境
│   ├── curriculum_task.py  # ★ 核心：难度自适应的 pick-and-place 任务
│   ├── curriculum_env.py   # 支持课程学习的 Panda 环境
│   ├── wrappers.py         # Gymnasium 包装器
│   └── env_factory.py      # 向量化环境工厂
├── agents/                 # RL 智能体构建
│   ├── builder.py          # SAC/TD3 + HER 构建器
│   └── callbacks.py        # ★ 课程学习回调 + 指标跟踪
├── training/               # 训练脚本
│   ├── train.py            # 主训练入口
│   └── ablation.py         # 消融实验批量运行
├── evaluation/             # 评估 & 可视化
│   ├── evaluate.py         # 模型评估
│   ├── plot_results.py     # 训练曲线 & 对比图
│   └── record_video.py     # Demo 视频录制
├── utils/                  # 工具函数
├── tests/                  # 单元测试
├── scripts/                # Shell 脚本
└── results/                # 输出结果 (gitignored)
    ├── models/             # 模型检查点
    ├── logs/               # TensorBoard 日志
    ├── videos/             # Demo 视频
    └── plots/              # 训练曲线图
```

## 消融实验设计

| 编号 | 算法 | HER | 课程学习 | 目的 |
|------|------|-----|---------|------|
| E1 (基线) | SAC | ✓ future | ✓ | 完整方案 |
| E2 | TD3 | ✓ future | ✓ | 算法对比 |
| E3 | SAC | ✓ future | ✗ | 验证课程学习效果 |
| E4 | SAC | ✗ (稠密奖励) | ✗ | 验证 HER 效果 |
| E5 | SAC | ✓ final | ✓ | HER 策略对比 |

## 关键超参数

| 参数 | 值 | 说明 |
|------|---|------|
| 折扣因子 γ | 0.95 | 短时域任务（50步）用更低的 γ |
| 学习率 | 0.001 | 操作任务的标准配置 |
| 网络结构 | [256,256,256] | 三层 MLP，足够表达复杂策略 |
| HER 采样数 | 4 (future) | 每个真实转换产生 4 个虚拟成功经验 |
| 回放缓冲区 | 1M | 离线算法需要大缓冲区 |
| 总训练步数 | 500K | 有课程学习时可较快收敛 |

## 参考文献

- Andrychowicz et al., *"Hindsight Experience Replay"*, NeurIPS 2017
- Haarnoja et al., *"Soft Actor-Critic"*, ICML 2018
- Fujimoto et al., *"TD3"*, ICML 2018
- Gallouédec et al., *"panda-gym"*, 2021
- Bengio et al., *"Curriculum Learning"*, ICML 2009

## License

MIT
