# SeiyuuMatch

当前版本：`1.0.0`

> 上传一张照片，看看你长得最像哪位邦多利女声优。

SeiyuuMatch 是一个面向邦多利企划粉丝的趣味识别小站。你可以上传自己的照片，选择想参与匹配的乐队范围，然后获得最相似的女声优结果、相似度和 Top 5 候选。

[![Version](https://img.shields.io/badge/version-1.0.0-ff6b9d)](./CHANGELOG.md)
[![Status](https://img.shields.io/badge/status-online-c44dff)](#)
[![Dataset](https://img.shields.io/badge/dataset-48%20seiyuu-6c5ce7)](#)
[![Privacy](https://img.shields.io/badge/privacy-upload%20notice-2d3436)](#隐私说明)

## 可以玩什么

| 功能 | 体验 |
| --- | --- |
| 上传照片识别 | 自动检测照片里的人脸，给出最像的声优 |
| 多人照片 | 一张图里有多个人时，会分别给出结果 |
| Top 5 相似度 | 不只看第一名，还能展开候选排行 |
| 乐队范围筛选 | 只测 `MyGO!!!!!`、`Ave Mujica`、`sumimi`，或者全团一起测 |
| 二挡阈值 | 识别不到时，可以降低阈值再试一次 |
| 声优头像展示 | 结果卡片会展示匹配声优头像 |
| 数据集贡献 | 可以上传公开、清晰的候选照片，帮助补数据 |
| 反馈意见 | 页面内直接提交反馈，方便后续修正 |

## 当前内容

| 项目 | 状态 |
| --- | --- |
| 声优条目 | 48 |
| 默认推荐范围 | `mygo`、`avemujica`、`sumimi` |
| 头像展示 | 已独立到 `avatar/` |
| 乐队图标 | 已独立到 `icon/` |
| 数据集上传 | 进入 `faces_upload/`，需要人工审核 |
| 固定域名部署 | 支持 Cloudflare Tunnel |

## 页面入口

正式站点建议通过 Cloudflare 域名访问：

```text
https://seiyuumatch.org
```

本地开发访问：

```text
http://localhost:3724
```

## 使用提醒

- 请上传清晰、正脸或半侧脸照片，遮挡太多会影响结果。
- 识别结果只是娱乐向相似度，不代表真实身份、关系或评价。
- 页面会先提示隐私说明，确认后才能使用。
- 数据集贡献入口适合上传公开照片，不要上传敏感照片或没有权利处理的图片。

## 本地运行

想在本地跑起来，可以按下面操作：

```bash
# 1. 创建环境
conda create -n seiyumatch python=3.10
conda activate seiyumatch
pip install opencv-python numpy torch pytorch-lightning pillow requests

# 2. 注册人脸特征
./start_register.zsh

# 3. 启动服务
./start_server.zsh
```

## 环境检测与性能测试

服务器配好环境后，可以先运行：

```bash
python3 bench_env.py
```

它会输出 Python、Torch、CPU、内存、模型加载时间，并从 `tests/` 或 `faces/` 自动选几张图跑识别耗时测试。

常用参数：

```bash
# 指定一张图片测试
python3 bench_env.py --image tests/羊宫妃那.png

# 测试更多轮，观察平均耗时和 P95
python3 bench_env.py --samples 5 --rounds 5

# 临时指定 PyTorch CPU 线程数
python3 bench_env.py --torch-threads 4

# 测试低阈值识别
python3 bench_env.py --relaxed
```

如果单次识别平均耗时超过 5 秒，可以优先尝试降低 `server.py` 里的 `MAX_IMAGE_DIM`，例如从 `1280` 调到 `960`。

## Cloudflare Tunnel

### 固定域名

在 Cloudflare Zero Trust 中创建 Tunnel 后，将 Public Hostname 指向：

```text
Type: HTTP
URL: 127.0.0.1:3724
```

本机保持以下服务运行：

```bash
./start_server.zsh
cloudflared tunnel run --token <Cloudflare 给你的 token>
```

如果已经用 `cloudflared service install ...` 安装成系统服务，通常只需要启动 Python 服务。

### 临时地址

开发测试时也可以用 Quick Tunnel，不需要域名：

```bash
brew install cloudflared
./start_server.zsh
./start_tunnel.zsh
```

终端会打印一个 `https://*.trycloudflare.com` 临时地址。

## 服务器部署

推荐用一台 Linux 云服务器长期运行 Python 服务，再用 Cloudflare Tunnel 绑定固定域名。这样不需要开放服务器公网端口，也不需要自己配置 HTTPS 证书。

### 1. 准备服务器

建议配置：

- Ubuntu 22.04/24.04
- 2 核 CPU / 4GB 内存起步
- 磁盘按 `faces/`、`faces_upload/` 和模型大小预留，建议 20GB 以上

安装基础依赖：

```bash
sudo apt update
sudo apt install -y git python3.10 python3.10-venv python3-pip curl
```

拉取项目：

```bash
git clone https://github.com/satoshinji2992/SeiyuuMatch.git
cd SeiyuuMatch
```

创建虚拟环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python-headless numpy torch pytorch-lightning pillow requests
```

确认以下文件或目录已经放到服务器：

```text
AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
features.npz
faces/
```

### 2. 本地启动测试

```bash
source .venv/bin/activate
python3 -u server.py --host 127.0.0.1 --port 3724
```

另开一个终端检查：

```bash
curl http://127.0.0.1:3724/health
```

### 3. 用 systemd 常驻服务

创建服务文件：

```bash
sudo nano /etc/systemd/system/seiyuumatch.service
```

写入：

```ini
[Unit]
Description=SeiyuuMatch recognition server
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/ubuntu/SeiyuuMatch
ExecStart=/home/ubuntu/SeiyuuMatch/.venv/bin/python3 -u server.py --host 127.0.0.1 --port 3724
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

如果项目不在 `/home/ubuntu/SeiyuuMatch`，把 `WorkingDirectory` 和 `ExecStart` 改成实际路径。

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now seiyuumatch
sudo systemctl status seiyuumatch
```

查看日志：

```bash
journalctl -u seiyuumatch -f
```

### 4. 绑定 Cloudflare 固定域名

在 Cloudflare Zero Trust 创建 Tunnel，Public Hostname 建议这样填：

```text
Hostname: seiyuumatch.org 或 www.seiyuumatch.org
Type: HTTP
URL: http://127.0.0.1:3724
```

然后在服务器执行 Cloudflare 给出的安装命令，例如：

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
sudo cloudflared service install <Cloudflare 给你的 token>
```

确认 Tunnel 服务运行：

```bash
sudo systemctl status cloudflared
```

### 5. 更新代码

以后更新服务器代码：

```bash
cd /home/ubuntu/SeiyuuMatch
git pull
sudo systemctl restart seiyuumatch
```

如果修改了正式人脸库，需要重新生成特征文件：

```bash
source .venv/bin/activate
python3 register.py
sudo systemctl restart seiyuumatch
```

## API

### GET `/`

返回 Web 页面。

### GET `/health`

健康检查：

```json
{
  "ok": true,
  "people": 47
}
```

### GET `/face_groups`

返回 `faces/` 中的团、声优列表和照片数量，用于前端选择识别范围与数据集上传。

### POST `/`

上传图片并识别。支持查询参数：

- `bands=mygo,avemujica`：限制识别候选团。
- `mode=relaxed`：使用低阈值检测人脸。

示例：

```bash
curl --noproxy localhost \
  -X POST 'http://localhost:3724/?bands=mygo,avemujica' \
  --data-binary @photo.jpg
```

响应包含最相似声优、bbox、当前相似度与 Top 5：

```json
{
  "faces": ["羊宮妃那"],
  "details": [
    {
      "name": "羊宮妃那",
      "band": "mygo",
      "similarity": 0.7812,
      "top5": [
        {"name": "羊宮妃那", "band": "mygo", "similarity": 0.7812}
      ],
      "bbox": [0.14, 0.21, 0.44, 0.78]
    }
  ],
  "mode": "default",
  "bands": ["avemujica", "mygo"]
}
```

### POST `/upload_faces`

数据集贡献上传接口。前端会保存到：

```text
faces_upload/<团>/<声优>/
```

这些照片不会自动进入正式识别库，需要人工审核后移动到 `faces/`，再重新注册。

## 人脸注册

正式数据放在：

```text
faces/<团>/<声优>/
```

例如：

```text
faces/
├── mygo/
│   ├── 羊宮妃那/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── 立石凛/
└── avemujica/
    └── 渡瀬結月/
```

前端识别结果展示的头像单独放在：

```text
avatar/<声优>/1.jpg
```

服务端收到 `/avatar/<声优>` 请求时，会优先读取 `avatar/` 目录；如果没有找到头像，会临时回退到 `faces/` 中对应声优目录下的 `1.jpg`。

添加或修改正式照片后：

```bash
./start_register.zsh
./start_server.zsh
```

`features.npz` 不是热更新，服务启动时只加载一次。

## 隐私说明

识别功能会把用户选择的图片上传到服务器进行处理，并保存一份压缩后的历史记录。数据集贡献入口会把照片保存到 `faces_upload/` 待审核目录。请不要上传敏感照片、他人隐私照片，或没有权利处理的图片。
