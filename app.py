"""DIG Studio web entry point.

Run ``python app.py`` for the local Flask application.  The extraction
algorithms remain available through ``python run.py ...`` unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from web import create_app

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / ".ui_workspace"


def _default_port() -> int:
    raw = os.environ.get("DIG_PORT", "8501").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8501
    return value if 1 <= value <= 65535 else 8501


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DIG Studio 本地网页")
    parser.add_argument("--host", default=os.environ.get("DIG_HOST", "127.0.0.1"),
                        help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=_default_port(),
                        help="监听端口，默认 8501，也可设置 DIG_PORT")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")

    app = create_app(ROOT, WORKSPACE)
    url = f"http://{args.host}:{args.port}"
    print(f"DIG Studio 正在启动：{url}")
    print("任务在后台运行；上传文件、日志和结果保存在 .ui_workspace/")
    print("按 Ctrl+C 停止服务。")
    try:
        from waitress import serve
    except ImportError:
        print("未安装 waitress，使用 Flask 内置服务器；"
              "可运行 python -m pip install -r requirements-web.txt 启用 Waitress。")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    else:
        serve(app, host=args.host, port=args.port, threads=6)


if __name__ == "__main__":
    main()
