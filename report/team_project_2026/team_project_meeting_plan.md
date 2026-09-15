# 49274 Space Robotics — Team Project 分工会议文件

> 项目：Exploring a Martian Cave  
> 团队人数：3 人  
> 会议日期：2026-09-15  
> 目的：了解作业、比较两种分工模式，并在会议中确定最终方案  
> 原始说明：`/home/tianya/UTS/Space Robotics/team_project_2026.pdf`

### 已确认的团队决定

- 采用模式一：A 负责 Perception 1、2；B 负责 Perception 3、Planning 2；C 负责 Planning 1、3。
- 先完成基础任务并跑通完整流程，再选择 Advanced 和分配负责人。
- 以下 Advanced 选项和阶段日期均为候选建议，尚未锁定。

---

## 1. 作业基本信息

### 项目目标

开发一个 ROS 2 火星车系统，在没有人工键盘或鼠标控制的情况下：

1. 自主探索最初未知的火星洞穴；
2. 使用 SLAM 建立地图；
3. 检测并分类洞穴中的 artefact；
4. 估计 artefact 在地图中的位置；
5. 自动接近目标进行近距离检查；
6. 在探索和检查之间自动切换；
7. 根据团队能力完成 Advanced task。

评分重点包括：

- Performance：准确率、速度、覆盖率和效率；
- Novelty：是否有合理的新方法或改进；
- Robustness：多次运行是否稳定，失败后能否恢复；
- Understanding：成员是否理解并能解释系统。

### 截止日期

| 项目 | 时间 | 占课程总成绩 |
|---|---|---:|
| Live demonstration | 2026-10-19 星期一 09:30 | 25% |
| Report、video、code | 2026-10-23 星期五 23:59 | 15% |

### 最终交付物

- 三名成员参加现场演示；
- 一份团队报告，通常约 10–20 页；
- 约 5 分钟的演示视频，无需旁白；
- 视频上传至 YouTube、OneDrive 或 Google Drive，并保证阅卷人可以访问；
- 提交团队创建或修改的源代码；
- 不向 Canvas 上传大型模型文件；
- 报告必须包含个人贡献表和 Generative-AI disclosure。

本项目已经使用 OpenAI Codex 阅读作业要求、检查模板代码并协助制定计划，最终报告需要如实披露。

---

## 2. 任务和评分结构

### Perception：35%

| 任务 | 主要要求 |
|---|---|
| Perception 1 | 采集图像数据集；包括不同角度、距离以及无目标的负样本 |
| Perception 2 | 检测多种 artefact；输出类别、置信度和 bounding box；在 RViz 显示 |
| Perception 3 | 使用像素方向、深度图和 TF 估计地图坐标；融合重复观测；发布 RViz marker |

### Planning：35%

| 任务 | 主要要求 |
|---|---|
| Planning 1 | 自主探索未知洞穴；建议使用 frontier-based exploration |
| Planning 2 | 检测到选定的 2–3 类 artefact 后，计算安全的近距离观察位置并导航过去 |
| Planning 3 | 探索与检查自动切换；避免重复检查；目标丢失时重试；失败后恢复探索 |

### Advanced：30%

Advanced 有六个选项：

1. Robust perception；
2. Cave geometry analysis；
3. Communication network deployment；
4. Online roadmap construction；
5. Science tour planning；
6. Adaptive scientific sampling。

不要求全部完成。两个高质量 Advanced task 可以获得该部分满分。

结合现有 Assignment 2 代码，比较适合本组的候选是：

- Advanced 2：Cave geometry analysis；
- Advanced 4：Online roadmap construction。

两者都能复用 OccupancyGrid、distance transform、图结构和 RViz marker。复用旧代码前需要修复已发现的地图维度、碰撞检查和不可达目标处理问题。

---

## 3. 当前代码状态

主文件：`src/cave_explorer/cave_explorer/cave_explorer.py`

### 已经具备

- Gazebo 火星洞穴环境；
- SLAM Toolbox、Nav2 和 RViz；
- RGB、depth、LiDAR、IMU 和 odometry topic bridge；
- RGB 图像订阅；
- stop sign Haar Cascade 检测示例；
- Nav2 goal 发送；
- 固定位置、返回原点和随机预设目标；
- artefact RViz marker 框架。

### 尚未完成

- 没有图像采集工具和正式数据集；
- 没有真实 artefact 检测模型；
- 没有使用 depth image 和 TF 定位目标；
- `localise_artifact()` 仍把 marker 放在机器人位置；
- 没有目标聚类和去重；
- 没有 frontier exploration；
- 没有 close-range inspection；
- 没有 behaviour state machine；
- 没有 Advanced task；
- 没有自动测试。

### Artefact 类型

仿真世界中共有 15 个 artefact，五类、每类三个：

- Green alien；
- Green crystals；
- White sphere；
- Ice formation；
- Blue mushrooms。

Stop sign 只是模板检测示例，不是正式目标。程序不能读取 SDF 中的真实 artefact 坐标进行导航；这些坐标只能用于开发阶段评估定位误差。

---

## 4. 分工模式一：按照系统模块和数据流分工

### 核心思想

不完全按照评分标题切割，而是按照系统的数据流和耦合关系分工：

```text
相机图像
   ↓
目标检测（成员 A）
   ↓
深度定位、目标管理、接近目标（成员 B）
   ↓
探索与行为切换（成员 C）
   ↓
Nav2
```

### 具体分配

| 成员 | 主要负责 | 对应评分任务 | 可选 Advanced |
|---|---|---|---|
| 成员 A | 数据集、目标检测、bounding box、检测指标 | Perception 1、2 | Advanced 1 |
| 成员 B | 深度定位、TF、目标聚类、RViz marker、approach pose | Perception 3、Planning 2 | Advanced 2 |
| 成员 C | Frontier exploration、Nav2 lifecycle、行为状态机、恢复逻辑 | Planning 1、3 | Advanced 4 |

团队只需从三个 Advanced 候选中选择两个高质量任务，不需要三项全部完成。

### 各成员输出

#### 成员 A：Detection 模块

- 图像采集和数据集统计；
- 模型训练与验证；
- ROS 图像推理；
- 输出类别、置信度和 bounding box；
- 保存检测准确率、失败案例和 RViz 截图。

#### 成员 B：Localisation and inspection 模块

- 读取检测结果和 depth image；
- 将像素位置转换到相机坐标；
- 使用 TF 转换到 `map` frame；
- 聚合同一目标的重复观测；
- 计算 standoff distance 和面向目标的 approach pose；
- 管理目标的 detected/approaching/visited 状态。

#### 成员 C：Autonomy 模块

- 从地图中提取 frontier；
- 计算 frontier information gain 和路径成本；
- 发送、取消和监控 Nav2 goals；
- 管理探索、接近、检查和恢复状态；
- 处理 timeout、goal rejection 和 navigation failure；
- 记录地图覆盖率、时间和行驶距离。

### 优点

- 三个人可以较早并行开发；
- Planning 2 与 Perception 3 都涉及目标几何，由同一人负责更容易衔接；
- 模块边界接近实际数据流，最终集成风险较低；
- 每个人都会接触系统不同部分，更容易理解整体工作方式；
- 适合拆成多个 Python 文件，Git 冲突较少。

### 缺点

- 个人贡献不完全对应 Perception/Planning/Advanced 三个评分标题；
- 报告贡献表需要更详细地写清每个人参与了哪些任务；
- 成员 B 同时跨越 Perception 和 Planning，需要与 A、C 都保持沟通；
- Advanced 工作量需要另外协调，避免某人任务明显过多。

### 主要风险

- 如果 A 修改 detection 输出格式，B 的定位模块可能需要同步修改；
- 如果 B 的目标状态定义不稳定，C 的行为切换会受到影响；
- 如果没有提前规定接口，仍然会在集成时产生大量改动。

### 适合选择模式一的情况

- 三个人编程能力比较接近；
- 希望尽早并行开发；
- 团队愿意频繁沟通和 review；
- 最优先目标是完成一个稳定的端到端系统；
- 不介意个人贡献横跨多个评分任务。

---

## 5. 分工模式二：按照评分大类分工

### 核心思想

每个人拥有一个完整评分方向：

```text
成员 A：全部 Perception（35%）
成员 B：全部 Planning（35%）
成员 C：Advanced（30%）+ Integration
```

### 具体分配

| 成员 | 主要负责 | 对应评分任务 | 额外职责 |
|---|---|---|---|
| 成员 A | 完整感知系统 | Perception 1–3 | 向 Planning 提供标准 artefact 接口 |
| 成员 B | 完整规划系统 | Planning 1–3 | 使用 fake detection 提前测试状态机 |
| 成员 C | 两个 Advanced task | Advanced 2、4 | 系统集成、测试、GitHub 和完整演示 |

### 各成员输出

#### 成员 A：Perception owner

- 数据采集、整理和统计；
- 模型训练和测试；
- bounding box 与 RViz detection image；
- depth + TF artefact localisation；
- 重复观测聚类和 artefact marker；
- 检测与定位结果评估。

#### 成员 B：Planning owner

- Frontier exploration；
- Frontier 评分和目标选择；
- Nav2 goal lifecycle；
- Close-range approach goal；
- Exploration/inspection state machine；
- timeout、retry、visited 和 recovery；
- 探索与检查性能评估。

#### 成员 C：Advanced and integration owner

- Advanced 2：distance transform、最宽区域和瓶颈；
- Advanced 4：online roadmap nodes、edges 和碰撞验证；
- 集成 A 和 B 的模块；
- 建立测试脚本、运行日志和结果表；
- 管理 `main` 分支和完整演示版本；
- 组织多次端到端测试。

### 优点

- 与 35% / 35% / 30% 的评分结构直接对应；
- 每个人责任范围非常清楚；
- 报告章节和个人贡献表容易分配；
- 适合成员技能方向明确的团队，例如视觉、规划和算法各有一人；
- 各评分大类有明确 owner，出现问题时容易确定负责人。

### 缺点

- 存在明显的依赖顺序：Perception 3 → Planning 2/3 → 部分 Advanced；
- A 的 Perception 工作量可能很大，包括数据、训练、ROS 和几何定位；
- C 的 Advanced 工作可能依赖实时地图，前期容易等待；
- 如果三个方向开发到最后才合并，集成风险很高；
- 成员可能只理解自己的评分部分，不利于现场交叉提问。

### 主要风险

- 如果 Perception localisation 延期，Planning 的目标接近和状态机无法使用真实数据测试；
- 如果 Planning exploration 延期，Advanced roadmap 难以进行完整在线验证；
- Advanced 做得很复杂但基础六项不稳定，会影响整体表现；
- C 必须真正承担集成和测试，否则三人的实际工作量可能不均衡。

### 适合选择模式二的情况

- 三个人技能方向非常明确；
- 希望每个人对一个评分大类负责到底；
- 团队重视贡献记录简单清楚；
- A 能独立完成视觉训练和坐标定位；
- B 能独立完成探索和状态机；
- C 愿意同时负责 Advanced、集成和系统测试。

---

## 6. 两种模式直接比较

| 比较项 | 模式一：系统模块/数据流 | 模式二：评分大类 |
|---|---|---|
| 分工依据 | 功能耦合和数据流 | Rubric 的 35/35/30 |
| 责任清晰度 | 中等，需要详细记录 | 很高 |
| 前期并行性 | 较高 | 中等，必须使用 fake data 才能提高 |
| 最终集成风险 | 较低 | 较高 |
| Git 冲突 | 拆文件后较少 | 若不拆文件则较多 |
| 工作量平衡 | 通常较均衡 | 取决于 C 是否承担集成测试 |
| 报告分配 | 一个任务可能有多人贡献 | 章节可直接按成员分配 |
| 对接口设计要求 | 高 | 同样需要接口，但容易被忽略 |
| 单人延期影响 | 相邻模块受影响 | 可能阻塞整个后续大类 |
| 整体系统理解 | 通常较好 | 容易形成知识孤岛 |
| 最适合 | 重视并行和稳定集成 | 重视专业方向和清晰 ownership |

### 无论选哪种模式都必须做到

- 第一周确定 detection 和 tracked artefact 的数据格式；
- Planning 使用 fake detection 开发，不能等待最终视觉模型；
- Advanced 使用保存地图或实时 OccupancyGrid 提前开发；
- 每个 Pull Request 至少由另一名成员 review；
- 每周至少进行一次完整集成；
- 每个人都要理解其他模块的基本原理；
- 所有成员共同参加完整系统测试和彩排。

---

## 7. 建议模块接口

建议将模板拆成以下文件，避免三个人同时修改 `cave_explorer.py`：

```text
cave_explorer/
├── cave_explorer.py          # ROS 主节点和模块连接
├── perception.py             # 图像检测
├── artifact_tracker.py       # 深度定位、聚类和 visited 状态
├── frontier_explorer.py      # Frontier 提取和目标评分
├── behaviour_manager.py      # 行为状态机
├── geometry_analyser.py      # Advanced 2
└── online_roadmap.py         # Advanced 4
```

Detection 输出建议：

```python
ArtifactDetection(
    class_name,
    confidence,
    bbox,
    depth,
    map_x,
    map_y,
    timestamp,
)
```

Tracked artefact 建议：

```python
TrackedArtifact(
    artifact_id,
    class_name,
    map_x,
    map_y,
    observation_count,
    last_seen,
    inspection_status,
)
```

Behaviour states 建议：

```text
WAITING_FOR_SYSTEM
EXPLORING
APPROACHING_ARTEFACT
INSPECTING
RECOVERY
FINISHED
```

---

## 8. GitHub 协作规则

建议建立 private GitHub repository，只邀请团队成员。

- `main` 必须保持可构建，尽量保持可运行；
- 不直接在 `main` 上开发；
- 每个小功能使用独立分支；
- 通过 Pull Request 合并；
- 每个 PR 至少一人 review；
- 每个人使用自己的 GitHub 账号和 commit identity；
- 不要直接 `git add .`，先配置 `.gitignore`。

建议分支：

```text
feature/image-collector
feature/artifact-detector
feature/artifact-localisation
feature/frontier-exploration
feature/inspection-state-machine
feature/cave-geometry
feature/online-roadmap
fix/nav2-goal-lifecycle
```

建议忽略：

```gitignore
build/
install/
log/
__pycache__/
*.pyc
.vscode/
dataset/
models/*.pt
models/*.onnx
*.bag
*.mp4
```

大型数据集、视频和模型使用 OneDrive 或 Google Drive 共享；GitHub 保存版本号、训练配置和下载说明。

---

## 9. 共同里程碑

### 9 月 15–20 日：基础架构

- 确定分工模式、人员和 Advanced tasks；
- 建立 GitHub；
- 确保三个人都能启动模板系统；
- 定义模块接口；
- 完成数据采集工具；
- 建立 fake detection；
- 开始地图和 distance transform 处理。

### 9 月 21–27 日：独立功能第一版

- 第一版 artefact detector；
- 第一版 artefact localisation；
- 第一版 frontier exploration；
- 第一版 cave geometry analysis；
- 各模块能独立显示 RViz 结果。

### 9 月 28 日–10 月 4 日：第一次端到端集成

- Detection 接入 localisation；
- Localisation 接入 approach goal；
- Detection event 接入 state machine；
- Frontier exploration 接入 Nav2；
- 开始 online roadmap；
- 完成第一次完整运行。

### 10 月 5–11 日：Advanced 与稳定性

- 完成两个 Advanced tasks；
- 加入 timeout、retry 和 navigation recovery；
- 至少进行三次完整测试；
- 记录检测、定位、覆盖率、用时和失败案例；
- 开始整理报告和视频证据。

### 10 月 12–18 日：冻结与彩排

- 停止新增高风险功能；
- 只修复影响演示的缺陷；
- 完成多次现场演示彩排；
- 准备备用视频；
- 确认每个人都能解释完整系统；
- 保存最终参数、模型版本和 commit hash。

---

## 10. Meeting 中如何选择

每个人先独立选择，再讨论原因。

| 成员 | 更倾向模式一 | 更倾向模式二 | 主要理由 |
|---|:---:|:---:|---|
| 成员 1：________ | ☐ | ☐ | |
| 成员 2：________ | ☐ | ☐ | |
| 成员 3：________ | ☐ | ☐ | |

讨论问题：

- [ ] 三个人目前分别最熟悉视觉、规划还是地图算法？
- [ ] 是否有人愿意完整负责 Perception 1–3？
- [ ] 是否有人愿意完整负责 Planning 1–3？
- [ ] 谁愿意承担最终集成和测试责任？
- [ ] 我们更重视清晰 ownership，还是更重视并行和低集成风险？
- [ ] 如何保证实际工作量公平？
- [ ] 某成员延期时，谁负责接手关键任务？

### 最终选择

- [x] **模式一：按照系统模块和数据流分工**
- [ ] **模式二：按照评分大类分工**

选择理由：

> 团队决定按照系统模块和数据流分工，以提高并行开发效率、减少后期集成风险，并让高度耦合的功能由同一名成员连续负责。三条主要职责线分别为：目标检测、目标定位与近距离检查、自主探索与行为切换。

---

## 11. 最终分工记录

| 角色/模块 | 成员 | 主要任务 | 第一阶段截止时间 |
|---|---|---|---|
| Detection / Perception | | 数据集、目标检测、bounding box、检测评估 | |
| Localisation / Tracking | | Depth、TF、目标聚类、RViz marker、visited 状态 | |
| Close-range inspection | 与 Localisation 负责人相同 | 安全观察位置、面向目标的 approach pose，配合自主节点执行 | |
| Exploration / Planning | | Frontier、目标评分、Nav2 goal lifecycle | |
| Behaviour switching | | 探索/接近/检查/恢复状态机，与 Exploration 负责人相同 | |
| Advanced task 1 | | | |
| Advanced task 2 | | | |
| Integration / Testing | 全体成员 | 模块接口、端到端测试、演示彩排 | 持续进行 |
| GitHub 管理 | | | |
| Report 主编 | | | |
| Video 整理 | | | |

### 其他会议决定

| 决策项 | 最终决定 |
|---|---|
| 视觉模型 | |
| Advanced tasks | 基础任务完成后选择 |
| 数据集共享位置 | |
| GitHub repository owner | |
| 每周 meeting 时间 | |
| 第一次完整集成日期 | |
| 演示负责人和顺序 | |

---

## 12. 建议会议流程（约 35 分钟）

1. **5 分钟**：确认作业目标、评分和截止日期；
2. **5 分钟**：介绍当前代码完成情况；
3. **8 分钟**：说明模式一；
4. **8 分钟**：说明模式二；
5. **5 分钟**：三个人表态、比较技能和工作量；
6. **4 分钟**：确定模式、负责人和第一周任务。
