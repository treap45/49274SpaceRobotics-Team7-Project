# Exploring a Martian Cave — 49274 Team Project 2026

三人 ROS 2 团队项目：在 Gazebo 洞穴中自主探索、检测与定位 artefact，并完成近距离检查。

## 团队决定

采用模式一，先完成六个基础任务，再根据效果、时间和工作量选择 Advanced。

| 角色 | 基础任务 | 主要输出 |
|---|---|---|
| A：目标检测 | Perception 1、2 | 数据集、检测模型、检测框与类别 |
| B：定位与检查 | Perception 3、Planning 2 | 目标地图坐标、聚类、近距离观察目标 |
| C：自主行为 | Planning 1、3 | Frontier 探索、Nav2 目标管理、状态机与恢复 |

姓名和 Advanced 分工待填写；集成、测试和演示由全体成员共同参与。

详细要求及会议记录：[团队会议文件](report/team_project_2026/team_project_meeting_plan.md)。

## 仓库范围

- `src/cave_explorer/`：课程模板源码、launch、配置、机器人和洞穴资源。
- `report/team_project_2026/`：团队计划与文档。
- `.gitignore`：忽略构建输出、个人作业、训练数据、模型权重和录制文件。

保留课程提供的 Gazebo mesh 和纹理以便组员复现环境；这些资源与训练模型权重不同。当前源码是模板起点，尚未完成基础任务。最终 Canvas 提交需按课程要求整理相关源码。

## 构建与运行

环境：ROS 2 Humble、Gazebo Fortress；依赖安装步骤参见课程 PDF 的 Getting Started。首次构建前，各组员需安装课程列出的依赖。

在仓库根目录执行：

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select cave_explorer
source install/setup.bash
```

分别打开三个终端，各自在仓库根目录执行前两条 source 命令，再执行对应 launch：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

# 终端 1：Gazebo + RViz
ros2 launch cave_explorer cave_explorer_startup.launch.py

# 终端 2：SLAM + Nav2
ros2 launch cave_explorer cave_explorer_navigation.launch.py

# 终端 3：自主节点
ros2 launch cave_explorer cave_explorer_autonomy.launch.py
```

## Git 协作

默认分支为 `main`。后续在 GitHub 创建 private repository 并邀请组员，再配置远程地址。当前本地初始化不会自动建立 GitHub 仓库。

连接远程后，每个小功能创建分支：

```bash
git switch main
git pull --ff-only
git switch -c feature/image-collector
# 修改代码并验证，然后只暂存本功能涉及的文件
git add src/cave_explorer/cave_explorer/perception.py
git commit -m "Add image collection module"
git push -u origin feature/image-collector
```

上面的 `perception.py` 是未来模块示例，当前尚未创建。通过 Pull Request 合并，由另一位成员 review。每人配置自己的 Git 身份；每周做一次集成测试。数据集和权重另行共享，并记录版本及获取方式。

## 截止时间（课程 PDF）

- Live demo：2026-10-19 09:30，占课程总成绩 25%；具体分组时段以 Canvas 为准。
- 报告、视频和代码：2026-10-23 23:59，占课程总成绩 15%。
- 报告应包含个人贡献和生成式 AI 使用披露。
