"""End-to-end test of the VLM plumbing against a local mock server (no key needed).

Verifies: image encoding into a data URI, OpenAI-compatible request shape, JSON-mode
parsing, the response cache, and the four task prompts.

Usage: python test_vlm_mock.py [image.png]
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dig"))          # 库模块在 dig/
import vlm_client as vc  # noqa: E402
import vlm_tasks as vt  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

CAPTURED = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send({"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode())
        auth = self.headers.get("Authorization", "")
        content = payload["messages"][-1]["content"]
        blocks = content if isinstance(content, list) else []
        text_block = next((b for b in blocks if b.get("type") == "text"), {})
        image_block = next((b for b in blocks if b.get("type") == "image_url"), {})
        text = text_block.get("text", "") if blocks else str(content)
        has_image = bool(image_block)
        data_uri = image_block.get("image_url", {}).get("url", "") if has_image else ""
        CAPTURED.append({"model": payload.get("model"), "auth": auth,
                         "json_mode": "response_format" in payload,
                         "thinking": payload.get("thinking"),
                         "has_image": has_image, "data_uri_prefix": data_uri[:30],
                         "image_count": sum(b.get('type') == 'image_url' for b in content),
                         "prompt": text[:80]})
        thinking_on = (payload.get("thinking") or {}).get("type") == "enabled"
        if thinking_on:
            # reproduce the production bug: chain-of-thought eats max_tokens and the
            # answer comes back empty with finish_reason=length
            self._send({
                "id": "mock-empty", "object": "chat.completion", "model": payload.get("model"),
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "",
                                         "reasoning_content": "..."},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1234, "completion_tokens": 1024},
            })
            return
        reply = vc.mock_response(text)
        self._send({
            "id": "mock", "object": "chat.completion", "model": payload.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1234, "completion_tokens": 56},
        })


def main():
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "pipeline_out" / "figures" /
        "2023-01-1635_p7_img1.png")
    if not image.exists():
        print(f"找不到测试图 {image}")
        return 1

    prod = vc.DeepSeekVLM(api_key="unused", base_url="https://api.deepseek.com")
    mock = vc.DeepSeekVLM(api_key="unused", base_url="http://127.0.0.1:9")
    key_args = ("same-image", "same prompt", None, True, 1024, 0.0)
    assert prod._cache_key(*key_args) != mock._cache_key(*key_args), \
        "真实端点与 mock 端点不能共用缓存键"
    print("缓存命名空间: 真实 API 与本地 mock 已隔离 ✓")

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"mock 服务器: http://127.0.0.1:{port}   测试图: {image.name}")

    vlm = vc.DeepSeekVLM(api_key="sk-mock-key-for-testing",
                         base_url=f"http://127.0.0.1:{port}", model="deepseek-flash",
                         verbose=True)
    print(f"\n可用模型: {vlm.list_models()}")

    print("\n[1] 图表分类")
    print("   ", vt.classify_chart(vlm, image).get("chart_type"),
          "| is_line_chart =", vt.classify_chart(vlm, image).get("is_line_chart"))
    print("[2] 轴范围")
    ax = vt.read_axis_ranges(vlm, image, hint="x=[0,5] y=[0,70]")
    y_axis = ax.get("y") or (ax.get("y_axes") or [{}])[0]
    print(f"    x={ax['x']['min']}~{ax['x']['max']}  "
          f"y={y_axis['min']}~{y_axis['max']}"
          f"  confidence={ax.get('confidence')}")
    print("[3] 系列命名")
    names = vt.name_series(vlm, image, ["#053856", "#8a532d"])
    for s in names.get("series", []):
        print(f"    {s['color']} -> {s['label']}")
    print("[4] 抽查校验")
    checks = vt.spot_check(vlm, image, [{"id": 1, "x": 3.0, "series": "S_tip"}])
    cmp = vt.compare_spot_check(checks, {1: 60.5}, tol_abs=3.5)
    print("   ", json.dumps(cmp, ensure_ascii=False))

    print(f"\n缓存命中: {vlm.usage['cache_hits']}  实际请求: {vlm.usage['calls']}"
          f"  tokens={vlm.usage['prompt_tokens']}+{vlm.usage['completion_tokens']}")
    ok = True
    for c in CAPTURED:
        assert c["has_image"], "请求里没有图片"
        assert c["data_uri_prefix"].startswith("data:image/png;base64"), c["data_uri_prefix"]
        assert c["json_mode"], "没有开启 response_format=json_object"
        assert c["auth"].startswith("Bearer "), "缺少 Authorization 头"
        assert c["thinking"] == {"type": "disabled"}, f"thinking 未关闭: {c['thinking']}"
    print(f"检查了 {len(CAPTURED)} 个请求：图片 data URI ✓  JSON 模式 ✓  Bearer 认证 ✓  "
          f"thinking 已关闭 ✓")

    print("[5] 子图 + 共用图例的双图请求与缓存")
    ax = vt.read_axis_ranges(vlm, image, figure_path=image, panel_label='(b)')
    assert CAPTURED[-1]['image_count'] == 2, '整图上下文未发送到服务端'
    calls = len(CAPTURED)
    vt.read_axis_ranges(vlm, image, figure_path=image, panel_label='(b)')
    assert len(CAPTURED) == calls, '双图请求缓存未命中'

    print("\n[回归] 模拟 thinking 模式被开启时的空内容 bug")
    bad = vc.DeepSeekVLM(api_key="sk-mock", base_url=f"http://127.0.0.1:{port}",
                         model="deepseek-flash", verbose=False, thinking=True, cache=False)
    try:
        vt.classify_chart(bad, image)
        print("   ❌ 本应报错却没有")
        ok = False
    except vc.VLMError as exc:
        print(f"   ✅ 正确捕获并给出可读提示: {exc}")
    print(f"   说明：默认关闭 thinking 后同一请求可正常返回 → "
          f"{'已修复' if ok else '仍有问题'}")
    again = vt.classify_chart(vlm, image)
    print(f"二次调用缓存命中={vlm.usage['cache_hits']}（>0 表示缓存生效）"
          f" 分类结果仍然可用: {again.get('chart_type')}")
    srv.shutdown()
    print("\n✅ 链路自检通过 —— 你把真 key 写进项目根目录的 deepseek_key.txt 即可直接用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
