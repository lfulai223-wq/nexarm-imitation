# nexarm-imitation

在 NexArm 主从臂上做模仿学习：用遥操作采集"抓药盒放进收纳盒"的示教数据，训练 ACT、微调 π0.5，
在真机上用同样的条件评测两者的成功率和推理延迟。

接在项目一 [nexarm-teleop-latency](https://github.com/lfulai223-wq/nexarm-teleop-latency) 后面：
遥操作链路、串口驱动和延迟测量方法都来自那里。

**状态：准备阶段**（布置桌面、试录前检查）。计划和进度见 [docs/roadmap.md](docs/roadmap.md)。

## 环境

- 录制和训练用商家改版的 LeRobot（`nexarm_follower` / `nexarm_leader`），在它的虚拟环境里运行，
  并装上项目一的包：`pip install -e ~/nexarm-teleop-latency`。
- `scripts/make_print_sheets.py` 需要 matplotlib（项目一的环境里有）。

## 文件

| 文件 | 用途 |
|---|---|
| `configs/record_pilot.yaml` | 试录 10 条的 LeRobot 录制配置，开头写了用法、按键和收尾方法 |
| `configs/teleop_practice.yaml` | 录之前练手：只遥操作不录数据，控制频率和录制时一样 |
| `scripts/check_setup.py` | 开工检查：设备、相机帧率和亮度、ArUco 码有没有偏、从臂读舵机是否正常、主从姿势差 |
| `scripts/make_print_sheets.py` | 生成 ArUco 码和标定棋盘格的 A4 打印文件 |
