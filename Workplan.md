# 49274SpaceRobotics-Team7-Project — Work Plan

## 1. 已确认的方向

- 采用模式一：按照系统模块与数据流分工。
- 先完成 Perception 1–3、Planning 1–3，并跑通完整自主流程；随后共同选择 Advanced，重新评估工作量。
- 成员姓名暂用 A/B/C，待团队填写。
- 以下文件结构、接口和验收标准是开发建议；当前仓库仍为课程模板，已创建模块占位文件及 test/ 目录，模块逻辑尚未实现或接入主节点。

目标流程：**自主探索 → 检测 artefact → 地图定位 → 安全接近并检查 → 记录已检查 → 恢复探索。**

### 作业依据

依据课程 `team_project_2026.pdf`：第 8 页为评分与任务安排，第 9–12 页为基础任务，第 13–16 页为 Advanced，第 17–18 页为提交要求。

| 项目 | 要求 |
|---|---|
| Perception | 项目评分的 35%，包括数据集、检测、定位及显示 |
| Planning | 项目评分的 35%，包括探索、近距离检查、行为切换 |
| Advanced | 项目评分的 30%；两个高质量任务可获得此部分满分，不要求完成全部六项 |
| Live demo | 2026-10-19 09:30 开始，具体小组时段以 Canvas 为准；占课程总成绩 25% |
| 报告、视频、代码 | 2026-10-23 23:59 截止；占课程总成绩 15% |

正式任务运行应完全自主；数据采集阶段允许遥控。检测框、目标位置和探索决策需要有 RViz 展示。报告需包含个人贡献、方法与结果、可访问的视频链接和生成式 AI 使用披露。

## 2. 三人分工

| 角色 | 姓名 | 负责的基础任务 | 核心交付 |
|---|---|---|---|
| A：视觉检测 | 待填写 | Perception 1、2 | 数据采集工具、数据集、检测器、检测框图像、检测评估 |
| B：定位与近距离检查 | 待填写 | Perception 3、Planning 2 | 深度定位、目标关联与融合、目标 markers、安全观察位姿、检查条件 |
| C：探索与行为管理 | 待填写 | Planning 1、3 | Frontier 探索、Nav2 管理、行为状态机、重试与恢复、检查记录 |
| 全体 | A+B+C | 集成与验证 | 接口评审、完整场景测试、视频证据、各自任务报告和演示 |

### A：图像采集与检测

1. 从机器人相机选择性保存图像，包含多角度、不同距离和无目标负样本；统计每类数量。
2. 数据集尽量覆盖 green alien、green crystals、white sphere、ice formation、blue mushrooms；优先让容易识别的类别可靠运行。
3. 按采集运行或观察视角划分训练、验证和测试数据，避免相邻近似帧跨集合。
4. 实现检测接口，输出类别、检测框及置信度；提供模型版本、配置、依赖和权重获取说明。
5. 发布带检测框的 `/detections_image`，保留输入图像的采集时间和 frame 信息。
6. 记录每类 precision/recall、误检漏检例子和推理耗时；与 B/C 共同选定 2–3 个检查类别。

**职责边界：** A 输出二维图像检测，不估计 `map` 坐标，不操作 Nav2。Stop sign 可用于早期接口测试，最终需扩展到实际 artefact。

### B：定位、关联与检查目标

1. 确认 RGB/depth 的对应关系、时间同步、深度单位、相机内参和 optical frame。
2. 在检测框内选取有效深度，剔除 NaN、Inf、零值及明显背景；估计目标相机坐标并转换到 `map`。
3. 根据类别与空间距离关联多次观测，为同一目标提供稳定 `artifact_id`，融合位置并显示 RViz markers。
4. 为目标生成安全的观察位姿：考虑机器人尺寸、障碍净空、standoff distance 和相机朝向。
5. 与 C 联合检查目标可达性；接近过程中提供最新目标观测和必要的目标更新建议。
6. 定义“近距离检查成功”的可配置条件，如目标近期可见、距离与朝向符合要求，并保持一段观察时间。
7. 记录定位误差、重复目标率及近距离检查结果。

**职责边界：** B 返回观察位姿与检查评估结果；C 负责实际发送、取消目标和切换行为。Planning 2 由 B 主责，B/C 联合完成运行验收。

### C：探索、导航与状态机

1. 保存完整 OccupancyGrid，区分未知、自由和占据区域，提取 frontier（自由空间与未知区域的边界）。
2. 先实现简单目标选择，再加入信息增益、路径成本、重复访问和失败目标惩罚；显示候选及选中目标。
3. 统一管理 Nav2 action：异步发送、接受/拒绝、反馈、结果、取消、超时；只根据真实结果判定成功。
4. 检测到符合条件的新目标时暂停探索，调用 B 的观察位姿接口并执行接近与检查。
5. 维护已检查目标、当前目标和尝试次数；目标丢失时重试一次，仍失败则放弃本次检查并恢复探索。
6. 保存覆盖面积随时间的变化、实际行驶距离、导航失败次数和检查成功率。
7. 主责 `cave_explorer.py` 的模块连接；A/B review 各自模块的接入。

**职责边界：** 只有 C 的导航模块持有 `NavigateToPose` action client。A/B 不另外发送目标或发布 `/cmd_vel`，防止行为互相抢占。

## 3. 建议文件结构与维护人

初期采用一个 ROS 主节点加普通 Python 模块，通过函数调用传递数据，避免立即增加自定义 ROS 消息包。

```text
src/cave_explorer/
├── cave_explorer/
│   ├── cave_explorer.py       # C：ROS I/O、模块初始化与调度；A/B review
│   ├── interfaces.py          # 共享数据类型；A/B/C 共同评审
│   ├── dataset_collector.py   # A：选择性保存图像与元数据
│   ├── perception.py          # A：检测与图像标注
│   ├── artifact_tracker.py    # B：定位、数据关联与位置融合
│   ├── inspection_planner.py  # B：观察位姿和检查条件
│   ├── frontier_explorer.py   # C：frontier 与探索目标
│   ├── navigation_client.py   # C：Nav2 action 生命周期
│   └── behaviour_manager.py   # C：状态机与检查记录
├── config/                   # 各自负责相关参数；共享参数一起 review
├── launch/                   # C 集成；新增依赖由引入者补齐
└── test/                     # A/B/C 各自提供模块测试
```

现有 `urdf/`、`worlds/`、资源与打包文件继续保留。新增算法先放各自模块，尽量避免三个人同时编辑主文件。`interfaces.py` 修改需通知所有使用方，并随 PR 更新本文。

## 4. 建议接口契约

### 4.1 共同约定

- Python 内部类型建议用 `dataclass`；数组、数据对象以只读快照交给其他模块。
- 距离统一为米，角度为弧度；地图位置采用 `map` frame，导航位姿采用 `PoseStamped`。
- `stamp_ns` 为原始观测的 ROS 仿真时间（整数纳秒）；超时使用同一 ROS 时钟。仿真重置/时间回退时清理旧缓存、目标与 action 记录。
- 检测框采用原始 RGB 分辨率的 `(x_min, y_min, x_max, y_max)`，右下边界不包含在框内；A 负责撤销模型输入的缩放和 padding。
- `class_name` 使用固定标签表。建议：`green_alien`、`green_crystals`、`white_sphere`、`ice_formation`、`blue_mushrooms`。
- `confidence` 是检测器分数，不代表定位精度；具体检测阈值、聚类半径和检查参数放配置文件。
- 未准备好、无检测、定位失败、目标不可达应有不同状态；不得用 `(0, 0)` 代表失败，因为原点是有效坐标。

### 4.2 A → B：二维检测

建议数据：

```python
Detection2D:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

DetectionBatch:
    stamp_ns: int
    source_frame_id: str
    image_width: int
    image_height: int
    detections: list[Detection2D]
```

建议调用：`Detector.detect(rgb_image, stamp_ns, source_frame_id) -> DetectionBatch`。

- 无检测返回空列表；模型加载或推理失败需单独上报。
- 输出即使在推理完成后才到达，也保留采集时间；B 必须使用对应时刻的 depth/TF。
- A 不向此接口填入 `depth` 或 `map_x/map_y`，这些字段属于 B 的计算结果。
- 原图用于定位与采集；绘制标注使用副本，避免修改下游输入。

### 4.3 B → C：目标地图快照

建议数据：

```python
ArtifactTrack:
    artifact_id: int
    class_name: str
    position_map: tuple[float, float, float]
    last_seen_ns: int
    observation_count: int
    confirmed: bool
```

建议调用：

```text
ArtifactTracker.update(detections, matched_depth, camera_model, transform_at_stamp)
    → list[ArtifactTrack]
ArtifactTracker.snapshot() → list[ArtifactTrack]
```

- B 负责 ID 和观测位置；同一目标重复观测应保持 ID，不同位置的同类目标不能仅按类别合并。
- 无有效深度或 TF 时，跳过该次定位并报告原因；历史目标仍可保留，但不能更新其 `last_seen_ns`。
- `confirmed` 表示满足配置的观测确认规则。C 只选择已确认、属于检查类别且未成功检查的目标。
- `last_seen_ns` 用于判断当前是否丢失目标；“历史上见过”不等于“当前仍在画面中”。
- 最小版本在一次仿真运行内保持 ID 稳定；如后续允许合并已发布 ID，必须向 C 提供旧 ID 到新 ID 的映射。

### 4.4 B → C：观察目标与检查评估

```text
InspectionPlanner.make_goal(track, robot_pose, map_snapshot)
    → {status: READY | NOT_READY | NO_SAFE_GOAL,
       goal: PoseStamped | None, reason: str}

InspectionPlanner.evaluate(track, robot_pose, now_ns)
    → {status: PENDING | SATISFIED | LOST, reason: str}
```

- `goal` 表示机器人应到达的观察位姿，不是 artefact 本体坐标。
- B 检查局部安全距离和朝向；C 通过规划结果确认路径是否可达。几何安全不保证 Nav2 一定能到达。
- 接近时继续使用实时检测，避免频繁追逐位置噪声；重规划频率由 C 控制。
- `SATISFIED` 应反映近期观测与观察距离/朝向等条件。仅收到 Nav2 成功结果不足以认定检查成功。

### 4.5 C 内部：探索与导航

```text
FrontierExplorer.select_goal(map_snapshot, robot_pose, failed_goals)
    → {status: READY | NOT_READY | NO_REACHABLE_FRONTIER,
       goal: PoseStamped | None, reason: str}

NavigationClient.send(goal) → request_id
NavigationClient.cancel(request_id) → 异步取消请求
NavigationClient.poll(request_id)
    → PENDING | ACTIVE | SUCCEEDED | REJECTED | ABORTED | CANCELED | ERROR
```

- 没有地图与暂时找不到可达 frontier 是两种情况；后者不能直接宣称整个洞穴已探索完成。
- 每个回调携带 request ID，避免旧目标的结果覆盖新目标状态。
- 取消请求不等于取消已完成；等待原目标结束或确认取消后再下发新的检查目标。
- B 的 `evaluate` 与 C 的导航反馈共同驱动状态机。

### 4.6 检查状态的唯一维护方

**C 维护检查记录；B 维护物体观测。** C 使用 `artifact_id` 记录：

```text
UNVISITED → IN_PROGRESS → VISITED
                       → DEFERRED（失败后暂缓，恢复探索）
```

每条记录包含本次尝试次数、状态及失败原因。C 将检查状态的只读快照传给 B 的 marker 发布函数，用颜色区分已检查、待检查和暂缓目标。

`VISITED` 仅在检查成功时设置。重试上限为首次尝试后再重试一次；失败目标进入暂缓集合，防止下一帧检测立即触发无限重试。再次启用的冷却/新观测条件需由团队确定。

## 5. ROS 接入与现有模板注意点

| 接口 | 当前情况 | 接入责任 |
|---|---|---|
| `/camera/image` | 已桥接 RGB Image | A 使用；C 主节点接入 |
| `/camera/depth/image` | 已桥接 depth Image，主节点尚未用于定位 | B 使用；C 主节点接入 |
| CameraInfo | 当前桥接配置未配置 | B 确认来源，必要时添加桥接或显式相机模型配置 |
| TF | 模板查询 `map → base_link` | B 增加观测时刻的相机坐标变换 |
| `/map` | 主节点目前只保存边界 | C 保存完整地图并提供给 B |
| `/navigate_to_pose` | 已有 action client | C 接管并完善结果、取消、失败处理 |
| `/detections_image` | 已用于 RViz | A 保留输出及采集 header |
| `/marker_array_artifacts` | 已有 marker 框架 | B 发布位置，结合 C 提供的检查状态 |

模板相机配置含 `gz_frame_id=camera_link`，URDF 同时定义了 optical frames。B 必须验证图像投影坐标与 TF 的对应关系，不能直接把 optical 坐标当作 `camera_link` 坐标。RGB 与 depth 的尺寸、配准和时间偏差也需先确认；不满足时应转换或拒绝该次定位。

主节点应保留能按采集时间匹配的 depth 缓存，并设置最大同步误差。推理不能长时间阻塞导航回调；若使用工作线程，采用有界队列和线程安全快照，训练过程放在节点之外。

## 6. 集成顺序与验收

这些是内部开发门槛，不是课程规定的额外评分项。

| 阶段 | A | B | C | 联合验收 |
|---|---|---|---|---|
| 1：接口与环境 | 提供样例图和假检测 | 提供假目标和观察位姿 | 接入样例数据，修复 Nav2 状态处理 | 三人从同一 commit 构建启动；接口字段和单位一致 |
| 2：模块独立运行 | 第一版真实检测 | RGB/depth/TF 定位、聚类和观察位姿 | Frontier 和导航；用假目标测状态机 | 检测、定位、探索各自有可观察输出 |
| 3：端到端 | 持续实时检测 | 真实目标与检查评估 | 探索/检查切换 | 自主完成一次完整流程，无手动下发目标 |
| 4：基础任务验收 | 测漏检与误检 | 测坏深度、重复目标、定位与检查 | 测拒绝、超时、丢失、重试和恢复 | 选择 2–3 类检查目标，重复运行并记录失败案例 |
| 5：Advanced 决策 | 全员评估 | 全员评估 | 全员评估 | 基础流程可重复运行后，根据剩余时间与兴趣选题和分工 |

建议 10 月 4 日前完成基础集成，10 月 5–11 日用于选定的 Advanced，10 月 12–18 日用于稳定性测试和彩排；若基础进度滞后，及时调整 Advanced 范围。这些日期是建议，待团队确认。

### 基础功能验收清单

- [ ] 数据集包含多视角与负样本，有每类数量和测试划分记录。
- [ ] RViz 显示真实 artefact 检测框，而不只运行 stop sign 示例。
- [ ] 目标 marker 表示估计物体位置，重复观测不会无限生成新目标。
- [ ] 从未知地图自主探索，展示 frontier/选中目标，记录覆盖面积与时间。
- [ ] 对选定 2–3 类生成可达的安全观察位姿，接近时保持目标可见。
- [ ] 成功检查后恢复探索，已检查目标不会被重复触发。
- [ ] 目标丢失能重试一次；再次失败、目标拒绝或超时后能继续探索。
- [ ] 用相同条件重复运行，保留成功与失败数据、参数和 commit hash。

## 7. 协作与交付

- 按小功能开分支与 PR；各模块负责人提交自己的实现，至少另一名成员 review。
- C 协调主节点集成；A/B 分别验证自己的数据进入主节点后仍满足契约。
- 接口变更先更新 `interfaces.py` 设计和本文，随同调用方一起修改；每周至少一次联合运行。
- 引入者负责补充依赖声明、配置和 README，使其他成员可从干净 checkout 复现。
- A/B/C 分别整理自己对应任务的截图、方法、数据与贡献记录；全体准备现场讲解。
- 数据集、权重、录制、报告及内部会议材料按现有忽略规则另行共享；本文件作为仓库中的开发协作规范。

### 尚待确认

- A/B/C 姓名、每周集成时间。
- 检测模型、标签表、数据与权重共享方式。
- 相机模型来源、RGB/depth 同步容差、depth 单位和 TF 校验结果。
- 检查类别、安全距离、丢失超时、确认阈值、失败目标再次启用规则。
- 基础验收后选择的 Advanced 及其负责人；当前不指定任何一项。
