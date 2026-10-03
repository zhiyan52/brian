# 队友上手指南（Windows）

本指南面向新加入的队友，目标是在 **20 分钟内**把项目跑起来，包括：环境搭建、数据与模型放置、启动网页、以及后续的 Git 协作流程。

- 仓库地址：<https://github.com/zhiyan52/brian>
- 操作系统：Windows 10/11（macOS/Linux 命令见文末附注）
- 硬件：无独立显卡也能跑网页（CPU 推理，约 30~60 秒/例）；有 NVIDIA 显卡（显存 ≥6GB）可训练，单例推理约 1~2 秒

---

## 0. 你需要拿到的两样东西

| 资源 | 获取方式 | 说明 |
|---|---|---|
| 代码 | `git clone` 仓库（见第 1 步） | 全部源码、文档、网页 |
| `ibsr18-team-resources.zip`（约 71MB） | 向组长索取（微信/网盘/U盘） | 内含 18 例 MRI 数据 + 训练好的模型，**不在 Git 仓库中** |

> 数据集为 IBSR-18 医学影像，按其使用条款仅限本项目学习/研究使用，不要二次上传到公开网络。

---

## 1. 安装基础软件

### 1.1 Git

- 下载：<https://git-scm.com/download/win>，安装时一路默认即可。
- 安装后在开始菜单打开 **PowerShell**，验证：

```powershell
git --version
```

### 1.2 Python 3.11（必须是 3.11.x，不要用 3.12+）

- 下载：<https://www.python.org/downloads/release/python-3119/> → 页面底部选 **Windows installer (64-bit)**
- 安装第一步**务必勾选 `Add python.exe to PATH`**，再点 Install Now。
- 验证（重新打开一个 PowerShell）：

```powershell
python --version   # 应显示 Python 3.11.9
```

---

## 2. 克隆代码

```powershell
# 选择一个不含中文和空格的目录，例如 D:\projects
cd D:\
git clone https://github.com/zhiyan52/brian.git
cd brian
```

> 如果组长已把你加为 Collaborator（仓库 Settings → Collaborators），你才拥有推送权限；只读/下载不需要邀请。

---

## 3. 创建虚拟环境并安装依赖

**以下所有命令都在项目根目录 `brian\` 下执行。**

```powershell
# 3.1 创建并激活虚拟环境（激活成功后命令行前会出现 (.venv)）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3.2 升级 pip
python -m pip install --upgrade pip
```

如果激活时报"禁止运行脚本"，先执行一次（仅当前用户，安全）：

```powershell
Set-ExecutionPolicy -Scope CurrentUser -RemoteSigned
```

然后重新执行激活命令。

### 3.3 安装 PyTorch

- **有 NVIDIA 显卡**（推荐，训练和网页推理都快）：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

- **没有 NVIDIA 显卡**（CPU 版，只能跑网页/推理，训练会非常慢）：

```powershell
pip install torch torchvision
```

> 注意：安装 torch 时**不要加** `-i https://pypi.tuna.tsinghua.edu.cn` 之类的国内镜像参数，否则会被装成 CPU 版。其余包可以用清华源加速（见第 3.4）。

验证 GPU 是否可用（有显卡时应输出 `True` 和显卡名）：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 3.4 安装项目本体与全部依赖

```powershell
# 含网页（FastAPI）+ 开发测试工具（pytest/ruff）
pip install -e ".[web,dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> SimpleITK、fastapi、uvicorn、monai、nibabel 等都会随这条命令自动装好，无需单独安装。

---

## 4. 放置数据和模型

解压组长发的 `ibsr18-team-resources.zip`，把里面的内容放到对应位置：

```text
brian\
├── data\
│   └── raw\
│       ├── IBSR_01\
│       │   ├── IBSR_01.nii.gz
│       │   └── IBSR_01_seg.nii.gz
│       ├── IBSR_02\ ...
│       └── IBSR_18\        （共 18 个受试者文件夹）
└── outputs\
    └── checkpoints\
        └── best_model.pt   （约 55MB）
```

即：压缩包里的 `raw\` 内容放进 **`data\raw\`**，`best_model.pt` 放进 **`outputs\checkpoints\`**（`outputs\checkpoints` 文件夹没有就手动新建）。

### 4.1 校验数据（可选但推荐）

```powershell
python scripts\prepare_data.py --data-dir data/raw --splits-dir data/splits
```

看到末尾 `PASSED` 即数据放置正确。

### 4.2 跑单元测试，确认环境完整

```powershell
python -m pytest -q
```

应显示 **14 passed**。

---

## 5. 启动网页

```powershell
python web\server.py
```

看到 `Uvicorn running on http://0.0.0.0:8000` 后，浏览器打开：

- 本机访问：<http://127.0.0.1:8000>
- 同一 WiFi/局域网的手机或其他电脑：`http://你的内网IP:8000`（内网 IP 用 `ipconfig` 查看，形如 192.168.x.x；首次访问若被 Windows 防火墙拦截，点"允许访问"）

上传 `data\raw\IBSR_02\IBSR_02.nii.gz` 即可体验完整流程：自动分割 → 三视图 + 3D 可视化 → 组织体积与脑萎缩定量报告。

停止服务：在运行服务的窗口按 `Ctrl + C`。

---

## 6. 训练 / 评估 / 测试集推理（按需）

```powershell
# 重新训练（有 GPU 约 30 分钟，400 epochs；CPU 不建议）
python scripts\train.py --config configs\train.yaml

# 用验证集评估已训练模型，输出各类 Dice
python scripts\evaluate.py

# 对 3 个无标签测试受试者推理 + 完整性检查
python scripts\predict_test.py
python scripts\check_test_predictions.py
```

当前模型在验证集的参考指标：平均前景 Dice ≈ **0.917**（CSF 0.893 / GM 0.932 / WM 0.925）。

---

## 7. Git 协作流程（重要）

### 7.1 一次性配置（只做一次）

```powershell
git config user.name "你的名字"
git config user.email "你的GitHub邮箱"
```

### 7.2 标准开发循环

```powershell
# (1) 开工前先拉最新代码
git checkout main
git pull

# (2) 基于 main 建自己的功能分支
git checkout -b feature/你的功能名      # 例如 feature/add-n4-experiment

# (3) 开发、提交（小步多次提交，信息写清楚）
git add 改动的文件
git commit -m "feat: 简述这次改了什么"

# (4) 推送自己的分支
git push -u origin feature/你的功能名
```

然后在 GitHub 仓库页面点 **Compare & pull request**，填写说明并创建 PR，@队友 review，通过后点 **Merge** 合并进 main。

合并后同步本地：

```powershell
git checkout main
git pull
```

### 7.3 分支与提交命名约定

- 分支：`feature/xxx`（新功能）、`fix/xxx`（修 bug）、`docs/xxx`（文档）
- 提交信息前缀：`feat:` 新功能、`fix:` 修复、`docs:` 文档、`chore:` 杂项、`refactor:` 重构

### 7.4 协作纪律

1. **不要直接往 main 推代码**，一律走分支 + PR。
2. 开工前 `git pull`，减少冲突。
3. 不要提交数据（`data/raw/`）、模型（`*.pt`）、虚拟环境（`.venv/`）和临时日志——`.gitignore` 已自动排除，请勿手动 `git add -f`。
4. 任务分工和问题讨论用 GitHub 的 **Issues**；代码评审意见写在 PR 评论里。
5. 提交前可本地跑 `python -m pytest -q` 和 `ruff check .`，避免把坏代码合进去。

---

## 8. 常见问题排查

| 现象 | 原因与解决 |
|---|---|
| `python` 命令找不到或版本是 3.12 | 安装时没勾选 Add to PATH，或装错版本。重装 Python 3.11 并勾选 PATH |
| 激活 venv 报"禁止运行脚本" | 执行 `Set-ExecutionPolicy -Scope CurrentUser -RemoteSigned` 后重试 |
| `import torch` 后 `cuda.is_available()` 为 `False`，但你有 N 卡 | 装成了 CPU 版 torch。执行 `pip uninstall -y torch torchvision` 后，重新用第 3.3 的 cu126 命令安装（**不要加镜像 -i 参数**） |
| 启动网页报 `No module named 'simpleitk'/'fastapi'` | 第 3.4 没做或没装 web 扩展。执行 `pip install -e ".[web,dev]"` |
| 网页启动报找不到 `best_model.pt` | 模型没放对位置，应为 `outputs\checkpoints\best_model.pt` |
| `prepare_data.py` 报形状/标签 FAILED | 数据没放全或目录层级不对，对照第 4 节检查 18 个 `IBSR_xx` 文件夹 |
| `git push` 报 403 / 无权限 | 你还没被加为 Collaborator，或当前登录的 GitHub 账号不对，联系组长邀请 |
| `git pull` 后一堆冲突 | 先不要慌，在群里说明；冲突文件用编辑器处理后 `git add` + `git commit`，不要随意 `reset --hard` |
| 端口 8000 被占用 | 关掉占用进程，或修改 `web\server.py` 末尾的 `port=8000` 为其他端口 |
| pip 下载很慢 | 第 3.4 的普通包可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；**torch 除外** |

---

## 9. macOS / Linux 附注

流程相同，仅两处命令不同：

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # 激活
pip install -e ".[web,dev]"        # 其余一致
```

---

## 10. 项目速览

- 代码核心库：`src/ibsr_unet/`（模型 / 数据 / 训练 / 推理 / 评估）
- 运行入口：`scripts/`（训练、评估、推理、QC、可视化）
- 网页：`web/server.py`（FastAPI 后端）+ `web/static/index.html`（前端）
- 配置：`configs/train.yaml`
- 完整实验与架构文档：`docs/architecture.md`、`docs/experiments.md`
- 临床需求与竞品分析：`docs/clinical_requirements.md`、`docs/competitive_analysis.md`

遇到本指南没覆盖的问题，在仓库 **Issues** 里提问并附上完整报错截图。
