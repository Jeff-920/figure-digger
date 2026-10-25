# DIG

**Data from Images and Graphs**：从 PDF 论文的折线图、散点图中提取数据，输出逐系列 CSV、原图叠加核对图和报告。支持网页与命令行、按图号或内容选图、多子图、双 Y 轴及批量处理。

视觉模型负责识别图意、坐标轴和系列，代码完成曲线追踪、坐标换算与导出；可读取的 PDF 矢量路径会用于直接提取。

## 安装

需要 Python 3.10+。进入项目目录后，先激活自己的 Python 环境；已有 Conda 或虚拟环境可直接使用。新建环境的 Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux 的激活命令为 `source .venv/bin/activate`。依赖安装与运行应使用同一解释器。

完整安装包含算法与网页：

```bash
python -m pip install -r requirements.txt
```

也可以分开安装：

```bash
python -m pip install -r requirements-core.txt -r requirements-web.txt
```

## 网页使用

```bash
python app.py
```

打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)，使用期间保持终端运行：

1. 在“开始提取”上传一篇或多篇 PDF；可一次多选，也可分多次从不同目录添加。已知 Figure 编号时可直接在“已知 Figure 编号”填数字（优先按图号定位），否则填写目标描述；提交后会自动记住描述，下次新建预填，也可留空处理全部数据图。
2. 所有任务都使用视觉模型，需要填写 API Key；勾选“记住”后会在提交任务时保存到本机。
3. 按需设置页码、DPI、批量并行数或上传输出模板，然后开始处理。
4. 任务在后台运行，可离开新建页；在“任务结果”查看状态与日志。任务 ID 可展开，按文献查看状态与结果；文献右侧的眼睛按钮只打开该文献的核对视图。
5. 完成后先按文献核对状态，再点图例确认当前曲线，点选起点并确定采样间隔；重叠曲线会优先响应当前曲线，点图例可随时切换。对采样后的点可用左侧编辑栏移动/擦除/补充，或新建未识别曲线。鼠标附近会显示 3× 放大镜辅助定位。右侧可编辑轴标签和单位换算（× 倍率 + 偏移），确认绘制成图；保存会按换算值覆盖对应 CSV，并同步更新面板轴范围与标签。

端口被占用时可运行 `python app.py --port 8502`，也可设置环境变量 `DIG_PORT`。网页默认只监听 `127.0.0.1`，仍是本地单机服务，不提供账户或公网访问保护。

网页密钥以明文保存在本机 `deepseek_key.txt`，可在“密钥设置”中清除；不会进入任务配置、命令行参数、任务日志或结果压缩包。上传文件、日志和任务结果保存在 `.ui_workspace/`，兼容已有任务目录和旧版 `ui-run.json`。

详细步骤、状态含义、坐标确认及常见问题见 [网页指南](docs/web-ui.md)。

## 命令行使用

视觉模型的密钥可通过网页保存，也可写入根目录 `deepseek_key.txt`，或设置环境变量 `DEEPSEEK_API_KEY`。

```bash
# 单篇，按描述选图；使用图号时可将需求改为 "Figure 5"
python run.py all paper.pdf --out out --vlm --want "目标物理量随时间变化的曲线"

# 批量处理一个目录，每次并行两篇
python run.py batch papers --out out --vlm --jobs 2

# 仅分析，先检查坐标与配置，再提取
python run.py analyze paper.pdf --out out --vlm

# 查看环境状态和完整参数
python run.py doctor
python run.py --help
python run.py all --help
```

常用选项：`--pages 3,5,7` 限定 PDF 实际页序，`--template format.dat` 按文本模板导出，`--verbose` 输出详细日志，`--keep-intermediates` 保留诊断图片。位图任务的坐标轴读数需要 `--vlm`；矢量图可直接从 PDF 路径和文字读取。按内容或工况查找及模板导出也需要视觉模型。

## 结果与目录

每篇论文的输出包括：

| 文件或目录 | 内容 |
| --- | --- |
| `csv/` | 逐系列数据 |
| `verify/` | 提取结果重投影到原图的核对图 |
| `report.md` | 处理情况、质量检查及失败原因 |
| `paper_config.json` | 坐标、系列与提取配置 |
| `templates/` | 使用模板时生成的文件 |

批量任务另有 `batch_summary.md` 和 `batch_summary.json`。网页任务在 `.ui_workspace/<任务ID>/output/` 下保存对应结果。

源码分为 `app.py` / `web/` / `templates/` / `static/`（网页）、`ui_utils.py`（CLI 与结果公共逻辑）、`run.py`（命令行）、`dig/`（算法）及 `tests/`（自动化回归测试）。`requirements-core.txt`、`requirements-web.txt` 分别管理算法和网页依赖。Bootstrap 5.3.8 与 Font Awesome Free 6.7.2 随项目提供，页面不依赖 CDN。

## 使用提示

- 自动结果需要结合核对图复查，重点检查坐标数量级、系列完整性和曲线贴合情况；低清晰度、严重重叠和特殊图形可能提取不完整。
- 已知目标页或图号时尽量指定范围，减少无关处理；视觉模型耗时还受网络和接口响应影响。
- 模型响应缓存位于 `.vlm_cache/`，调用明细位于 `.vlm_log.jsonl`。本地密钥、论文、缓存及运行结果均已加入 Git 忽略规则。

项目采用 [MIT License](LICENSE)。
