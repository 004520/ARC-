# 人脸识别与考勤系统

基于 **ArcFace** 深度学习算法的人脸识别系统，集成 **GUI 可视化界面**、**实时摄像头识别**、**人脸库管理** 和 **考勤打卡统计** 功能，使用 PyTorch + Tkinter + MySQL 构建。

---

## 1. 主要功能

### 1.1 GUI 人脸识别

打开摄像头，点击"识别"按钮即可检测画面中的所有人脸，并与数据库中的注册人脸进行匹配。

- 支持**多人同时识别**，每张人脸标注姓名和置信度
- 识别结果实时叠加在摄像头画面上
- 未检测到人脸时显示半透明提示文字
- **陌生人警告**：未识别的人脸以**红色框**标注并显示"陌生人"提示，已知人脸以**绿色框**标注并显示姓名和相似度
  - 🔵 **蓝色框**：检测到人脸，尚未点击识别（待验证状态）
  - 🟢 **绿色框**：识别成功，显示姓名 + 相似度
  - 🔴 **红色框**：陌生人警告，未匹配到脸库中的任何人
- 状态栏实时反馈识别结果，陌生人时显示"⚠ 警告：陌生人！"
- **陌生人自动截图存证**：检测到陌生人时，自动将当前摄像头画面保存到 `strangers/` 目录，文件名带时间戳，便于事后追溯
- **陌生人声光告警**：检测到陌生人时，触发蜂鸣提示音，状态栏红色闪烁 3 次

### 1.2 人脸录入

提供两种录入方式：

- **摄像头拍照录入**：面对摄像头，系统自动检测并裁剪最大人脸，一键注册
- **上传照片录入**：从本地选择照片，系统自动检测人脸后注册

录入时自动保存头像照片到 `faces/` 目录，在脸库页面可查看。

- **人脸录入质量检测**：录入前自动检查人脸质量，包括：
  - 人脸过小/过大检测
  - 人脸位置偏移检测（是否在画面中央）
  - 光线亮度检测
  - 质量不合格时给出中文提示，合格时显示"人脸质量：优秀"

### 1.3 人脸库管理

以卡片网格形式展示所有已注册用户：

- 照片、姓名、注册时间
- 一键删除用户（同时清除特征向量和识别日志）

### 1.4 实时统计面板

主页摄像头下方展示实时统计信息：

- 脸库人数、今日识别次数、今日陌生人次数、最后识别记录、累计识别次数
- 每 30 秒自动刷新

### 1.5 考勤打卡

- **考勤模式开关**：点击开启后，每次识别成功自动记录打卡日志
- **考勤记录页**：按日期范围筛选，表格展示姓名、日期、首次打卡时间、识别次数、打卡状态
- **迟到判定**：首次打卡时间 > 9:00 标记为迟到（红色），正常为绿色
- **导出报表**：一键导出为 Excel（.xlsx，带样式表头）或 CSV 文件

### 1.5 实时摄像头识别（命令行）

```bash
python realtime.py                           # 使用 Haar 检测器
python realtime.py --detector mediapipe      # 使用 MediaPipe（推荐）
python realtime.py --detector mtcnn          # 使用 MTCNN
```

- 实时画面叠加姓名和相似度
- 按 `N` 切换识别/仅检测模式，按 `R` 注册当前画面中的人脸
- 支持四种检测器：Haar / DNN SSD / MediaPipe / MTCNN

### 1.6 命令行管理

```bash
python register_face.py                # 交互式管理控制台
python register_face.py --list         # 查看已注册用户
python register_face.py --delete <id>  # 删除用户
```

提供交互菜单：摄像头录入、图片录入、批量录入、查看/删除用户、日志查看、统计信息。

---

## 2. 快速开始

### 2.1 环境要求

| 组件 | 版本/说明 |
|------|----------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0（支持 CUDA） |
| MySQL | 8.0+（本地运行） |
| GPU | NVIDIA 显卡（可选，CPU 也可推理） |
| 摄像头 | 内置/USB 摄像头 |

### 2.2 安装依赖

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow pymysql tqdm numpy openpyxl
```

### 2.3 初始化数据库

编辑 `database.py` 和 `init_db.py`，修改 MySQL 连接信息（host、user、password），然后运行：

```bash
python init_db.py
```

脚本会自动创建 `face_system` 数据库和 `users`、`face_features`、`recognition_logs` 三张表。

### 2.4 启动应用

```bash
python gui.py
```

程序加载模型和摄像头后即可使用。

### 2.5 训练自己的模型（可选）

如果希望在自己的数据集上训练模型：

```bash
python train.py
```

训练配置（`train.py` 内可调）：

- 默认数据集路径：`D:/face_dataset/Cleaned_CASIA_FaceV5`
- 默认 150 轮，带 Warmup 预热、混合精度训练、早停机制
- 最佳模型自动保存到 `checkpoints/best_face_model.pth`

---

## 3. 技术原理

### 3.1 人脸识别的基本思路

计算机看人脸，本质是把一张照片转换成一个**特征向量**（一串数字）。如果同一个人的两张照片，它们的特征向量足够"像"（余弦相似度高），计算机就认为这是同一个人。

普通的分类网络（比如 Softmax）只能学会区分不同的人，但它不保证同一类人的特征向量**很紧凑**。打个比方：Softmax 只是把每个人划到一个小格子，但格子里的人可能站得很散；ArcFace 则是把同一个格子里的人**往中心推**，同时把不同格子**推得更远**。

### 3.2 ArcFace 损失函数

ArcFace 全称 **Additive Angular Margin Loss**（加法角度间隔损失），2019 年由邓健康等人提出。

#### 从 Softmax 说起

普通 Softmax 分类的损失函数：

$$L_{\text{softmax}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{W_{y_i}^T x_i + b_{y_i}}}{\sum_{j=1}^{C} e^{W_j^T x_i + b_j}}$$

这一步只是让每个样本被正确分类，对特征向量没有额外的约束。

#### 归一化 + 角度表示

当特征向量 $x$ 和权重向量 $W$ 都做了 L2 归一化之后，内积就等于它们夹角的余弦值：

$$W^T x = \|W\| \cdot \|x\| \cdot \cos\theta = \cos\theta$$

此时 Softmax 输出变为：

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s \cdot \cos\theta_{y_i}}}{\sum_{j=1}^{C} e^{s \cdot \cos\theta_j}}$$

其中 $s$ 是缩放因子（本系统默认 $s=30$），控制概率分布的"陡峭程度"。

#### 引入角度间隔 $m$

ArcFace 的**核心创新**：在正确类别的角度 $\theta_y$ 上**额外加一个间隔 $m$**，让模型必须把特征向量推得更靠近类中心才能正确分类。用大白话说就是——增加了"及格线"的难度：

$$L_{\text{arc}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$

#### 关键实现细节

编程时不能用 `cos(θ + m)` 直接算，需要用三角恒等式展开：

$$\cos(\theta + m) = \cos\theta \cdot \cos m - \sin\theta \cdot \sin m$$

其中 $\sin\theta = \sqrt{1 - \cos^2\theta}$（因为归一化后向量在单位球面上）。

对应到代码中（`ArcFace/model.py` ArcMarginProduct.forward）：

```python
cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # cosθ
sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))      # sinθ
phi = cosine * self.cos_m - sine * self.sin_m                     # cos(θ+m)
```

然后对正确类别用带 margin 的 `phi` 替换原始 `cosine`，乘以缩放因子 $s$ 后送入交叉熵。

#### 超参数说明

| 参数 | 含义 | 本项目取值 |
|------|------|-----------|
| $s$ | 缩放因子，控制概率分布陡峭度 | 30.0 |
| $m$ | 角度间隔（弧度），越大类间分离越强 | 0.50 |

> **为什么不用论文的 $s=64$？** 论文基于百万级数据集（MS-Celeb-1M），本项目使用千级数据集（CASIA 500人），过大的 $s$ 会导致初始损失爆炸、梯度稀疏。将 $s$ 降到 30 后，初始损失从 ~37 降至 ~21，训练稳定。

### 3.3 模型架构

#### 整体结构

模型基于 **IR-ResNet**（Improved Residual Network），这是 ArcFace 论文推荐的骨干网络。一张 112×112 的人脸照片经过以下流程变成 512 维特征向量：

```
输入 (3×112×112)
  │
  ▼  Conv2d 3→64, BN + PReLU
  │
  ▼  Layer1: 3 个 IRBlock, 64 通道 → 28×28
  │
  ▼  Layer2: 4 个 IRBlock, 128 通道 → 14×14
  │
  ▼  Layer3: 14 个 IRBlock, 256 通道 → 7×7
  │
  ▼  Layer4: 3 个 IRBlock, 512 通道 → 4×4
  │
  ▼  BN → 1×1 Conv → BN → PReLU → GAP → Dropout → L2 Norm
  │
  输出: 512 维归一化特征向量
```

#### 关键设计：参数压缩

传统的做法是在最后一层用 Flatten + 全连接层（FC），会产生巨大的参数量：

> Flatten(4×4×512) → FC(8192, 512) = **419 万参数** ❌

本项目改用 **1×1 卷积 + 全局平均池化（GAP）**：

> 1×1 Conv(512, 512) → GAP → **26 万参数** ✅ （减少 94%）

这有效抑制了过拟合，尤其适合中小规模数据集。

#### 双重注意力机制

每个 IRBlock 内部集成了两种注意力：

- **SE Block（通道注意力）**：自动学习哪些通道（特征维度）更重要。比如某个通道负责眼睛特征，SE 模块会放大它的权重。
- **Spatial Attention（空间注意力）**：自动学习图片中哪些区域（眼睛、鼻子、嘴巴）更重要，抑制背景干扰。

流程：输入 → BN → Conv → PReLU → Conv → **SE → SA** → 与输入残差相加 → 输出。

#### 总参数量

约 **2.8M** 可训练参数，在 RTX 4060 Laptop（8GB）上训练时显存占用约 3GB。

### 3.4 训练策略

| 配置项 | 值 | 说明 |
|--------|---|------|
| 优化器 | SGD, momentum=0.9 | 标准人脸识别优化器 |
| 初始学习率 | 0.01 | 配合 batch_size=128 线性缩放 |
| Warmup | 前 5 轮线性增长 | 防止初始梯度爆炸 |
| StepLR | 每 40 轮 ×0.1 | 分段衰减细化收敛 |
| 混合精度 | AMP (FP16) | 节省显存，加速训练 |
| 梯度裁剪 | max_norm=3.0 | 防止梯度爆炸 |
| Dropout | p=0.4 | 防止过拟合 |
| 早停 | patience=40 | 验证损失不改善则自动停止 |

---

## 4. 项目结构

```
Facial_recognition_system/
├── ArcFace/                     # 模型定义
│   ├── __init__.py
│   └── model.py                 # FaceEncoder + ArcMarginProduct
├── checkpoints/                 # 训练好的模型权重
│   ├── best_face_model.pth      # 最佳验证模型
│   └── latest_checkpoint.pth    # 断点续训用
├── data/                        # 数据加载
│   ├── __init__.py
│   └── dataset.py               # 数据集划分 + 数据增强
├── faces/                       # 用户头像照片存储
├── strangers/                   # 陌生人自动截图存证
├── snapshots/                   # 识别日志缩略图（预留）
├── database.py                  # MySQL 数据库操作
├── gui.py                       # Tkinter 图形界面（主入口）
├── inference.py                 # 推理引擎（特征提取+识别）
├── init_db.py                   # 数据库初始化
├── monitor_training.py          # 训练进度监控
├── realtime.py                  # 实时摄像头识别
├── register_face.py             # 命令行注册管理
├── test_dataset.py              # 批量测试脚本
├── test_simple.py               # 简单验证脚本
├── train.py                     # 模型训练脚本
└── README.md                    # 本文档
```

---

## 5. 附录

### 核心依赖

```
torch>=2.0, torchvision, opencv-python, pillow
pymysql, numpy, tqdm, openpyxl
```

### 数据集

默认使用 **CASIA-FaceV5**（500 人，每人 5 张原始 + 12 种变体，共约 3.2 万张），数据集路径在 `train.py` 中配置。

### 参考

- Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019
