"""Flask application factory for the DIG web interface."""

from __future__ import annotations

import mimetypes
import os
import secrets
from pathlib import Path

from flask import Flask

from .routes import bp
from .tasks import RunStore, TaskManager, format_clock, format_duration


def create_app(root: Path | str | None = None,
               workspace: Path | str | None = None) -> Flask:
    project_root = (Path(root) if root is not None
                    else Path(__file__).resolve().parent.parent).resolve()
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    app.config.update(
        SECRET_KEY=os.environ.get("DIG_SECRET_KEY") or secrets.token_hex(32),
        DIG_PROJECT_ROOT=str(project_root),
        DIG_WORKSPACE=str(Path(workspace).resolve() if workspace is not None
                          else (project_root / ".ui_workspace").resolve()),
    )
    app.json.ensure_ascii = False
    app.jinja_env.filters["duration"] = format_duration
    app.jinja_env.filters["clock"] = format_clock
    mimetypes.add_type("font/woff2", ".woff2")

    store = RunStore(project_root, Path(app.config["DIG_WORKSPACE"]))
    manager = TaskManager(store)
    app.extensions["dig_store"] = store
    app.extensions["dig_tasks"] = manager
    app.register_blueprint(bp)
    manager.recover_interrupted()
    return app
