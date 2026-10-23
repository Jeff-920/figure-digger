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

完整安装包含算法、网页和 OCR。只使用视觉模型时，可改为轻量安装：

```bash
python -m pip install -r requirements-core.txt -r requirements-web.txt
```

后续需要本地 OCR 或 OCR 交叉核对时，补装 `python -m pip install -r requirements-ocr.txt`。

## 网页使用

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

打开 [http://127.0.0.1:8501](http://127.0.0.1:8501)，使用期间保持终端运行：

1. 上传一篇或多篇 PDF，填写图号或目标描述，也可留空处理全部数据图。
2. 选择视觉模型或本地 OCR。视觉模型需填写 API Key；网页默认记住密钥，下次无需重复输入。
3. 按需设置页码、DPI、批量并行数或上传输出模板，然后开始处理。
4. 先检查“图像核对”，再下载 CSV、模板或全部结果；历史记录位于“任务结果”。

网页密钥以明文保存在本机 `deepseek_key.txt`，可在页面清除；不会进入任务配置、任务日志或结果压缩包。上传文件和任务结果保存在 `.ui_workspace/`。

详细步骤、坐标确认及常见问题见 [网页指南](docs/web-ui.md)。

## 命令行使用

视觉模型的密钥可通过网页保存，也可写入根目录 `deepseek_key.txt`，或设置环境变量 `DEEPSEEK_API_KEY`。

```bash
# 单篇，按描述选图；使用图号时可将需求改为 "Figure 5"
python run.py all paper.pdf --out out --vlm --no-ocr --want "目标物理量随时间变化的曲线"

# 批量处理一个目录，每次并行两篇
python run.py batch papers --out out --vlm --no-ocr --jobs 2

# 仅分析，先检查坐标与配置，再提取
python run.py analyze paper.pdf --out out --vlm --no-ocr

# 查看环境状态和完整参数
python run.py doctor
python run.py --help
python run.py all --help
```

常用选项：`--pages 3,5,7` 限定 PDF 实际页序，`--ocr` 开启 OCR 交叉核对，`--template format.dat` 按文本模板导出，`--verbose` 输出详细日志，`--keep-intermediates` 保留诊断图片。按内容或工况查找及模板导出需要视觉模型；本地 OCR 仅接受纯图号约束。

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

源码分为 `app.py` / `ui_utils.py`（网页）、`run.py`（命令行）、`dig/`（算法）及 `tests/`（自动化回归测试）。`requirements-core.txt`、`requirements-web.txt`、`requirements-ocr.txt` 分别管理算法、网页和可选 OCR 依赖。

## 使用提示

- 自动结果需要结合核对图复查，重点检查坐标数量级、系列完整性和曲线贴合情况；低清晰度、严重重叠和特殊图形可能提取不完整。
- 已知目标页或图号时尽量指定范围，减少无关处理；视觉模型耗时还受网络和接口响应影响。
- 模型响应缓存位于 `.vlm_cache/`，调用明细位于 `.vlm_log.jsonl`。本地密钥、论文、缓存及运行结果均已加入 Git 忽略规则。

项目采用 [MIT License](LICENSE)。
