# SeiyuFinder

基于 MTCNN 人脸检测对齐 + AdaFace 人脸识别的端到端管线。

## 目录结构

```
├── faces/                  # 注册人脸库（每人一个文件夹）
│   ├── 青木阳菜/           # 文件夹名即人名
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   └── ...
│   └── ...
├── AdaFace/                # AdaFace 模型仓库
│   └── pretrained/
│       └── adaface_ir50_ms1mv2.ckpt
├── register.py             # 人脸注册脚本
├── server.py               # HTTP 识别服务
├── features.npz            # 注册生成的特征文件
├── start_register.zsh      # 注册启动脚本
├── start_server.zsh        # 服务器启动脚本
├── run.py                  # 单图离线识别
├── run_camera.py           # 摄像头实时识别 + HTTP 推流
├── index.html              # 前端页面（拖拽上传 + 结果展示）
└── crawl_faces.py          # 爬取声优人脸图片
```

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy torch pytorch-lightning

# 2. 注册人脸
python3 register.py

# 3. 启动服务器
python3 server.py
```

或使用启动脚本：
```bash
./start_register.zsh
./start_server.zsh
```

## API 协议

### 端口

默认 `3724`，可通过 `--port` 参数修改。

### GET / — Web 识别页

浏览器访问 `http://localhost:3724` 即可打开前端页面，支持拖拽上传或点击选择图片，识别结果以卡片形式展示，包含匹配声优的头像和姓名。

### POST / — 人脸识别

发送图片，返回图中所有人脸的识别结果。

**请求：**

```
POST http://localhost:3724
Content-Type: application/octet-stream
Body: <图片二进制数据（支持 jpg/png）>
```

**成功响应（200）：**

```json
{
  "faces": ["青木阳菜", "立石凛"]
}
```

- `faces`: 数组，每张脸对应最相似的注册人名

**失败响应（500）：**

```json
{
  "error": "错误信息"
}
```

### 调用示例

```bash
# curl
curl --noproxy localhost -X POST http://localhost:3724 --data-binary @photo.jpg

# Python
import requests
with open("photo.jpg", "rb") as f:
    resp = requests.post("http://localhost:3724", data=f.read())
    print(resp.json())
```

## 人脸注册

将照片放入 `faces/<人名>/` 文件夹，每人可放任意数量照片（推荐 3 张以上），支持 jpg/png 格式。

```bash
faces/
├── 青木阳菜/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.png
├── 立石凛/
│   └── 1.jpg
```

添加或修改照片后需重新注册：

```bash
python3 register.py
```

注册流程：MTCNN 检测并对齐人脸 → AdaFace 提取 512 维特征 → 同人多张取均值 → 保存到 `features.npz`。

## 预置特征

仓库中已包含 MyGO!!!!! 五位声优的注册特征（`features.npz`）：

| 角色 | 声优 |
|------|------|
| 高松灯 | 羊宫妃那 |
| 千早爱音 | 立石凛 |
| 要乐奈 | 青木阳菜 |
| 长崎爽世 | 小日向美香 |
| 椎名立希 | 林鼓子 |

启动服务后可直接使用，无需重新注册。
