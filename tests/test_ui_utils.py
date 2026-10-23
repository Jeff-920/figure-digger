import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ui_utils import (RunOptions, build_command, collect_artifacts, make_archive,
                      safe_filename, valid_pages, is_figure_number_request,
                      save_api_key, clear_saved_api_key)


class UiUtilsTests(unittest.TestCase):
    def test_saved_key_is_reusable_and_can_be_replaced_and_cleared(self):
        import run as pipeline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(pipeline.vlmc, "ROOT", root), \
                    patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
                save_api_key(root, " test-first-key ")
                self.assertEqual(pipeline.vlmc.load_api_key(), "test-first-key")
                save_api_key(root, "test-second-key")
                self.assertEqual(pipeline.vlmc.load_api_key(), "test-second-key")
                self.assertEqual([p.name for p in root.iterdir()], ["deepseek_key.txt"])
                clear_saved_api_key(root)
                self.assertIsNone(pipeline.vlmc.load_api_key())
                clear_saved_api_key(root)

    def test_invalid_key_does_not_overwrite_saved_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_api_key(root, "test-existing-key")
            for value in ("", " \n ", "first\nsecond"):
                with self.assertRaises(ValueError):
                    save_api_key(root, value)
            self.assertEqual((root / "deepseek_key.txt").read_text().strip(), "test-existing-key")

    def test_safe_filename_removes_paths_and_shell_punctuation(self):
        self.assertEqual(safe_filename("../../paper; rm.pdf"), "paper_rm.pdf")
        self.assertEqual(safe_filename("图 1.pdf"), "图_1.pdf")

    def test_pages_validation(self):
        self.assertTrue(valid_pages(""))
        self.assertTrue(valid_pages("3, 5,7"))
        self.assertFalse(valid_pages("2-4"))
        self.assertFalse(valid_pages("1;whoami"))
        self.assertFalse(valid_pages("0"))
        self.assertFalse(valid_pages("1,0"))

    def test_ocr_rejects_semantic_conditions_even_with_figure_number(self):
        for want in ("Figure 5 中 800 bar 的曲线", "Figure 5(a)", "速度随时间"):
            with self.subTest(want=want), self.assertRaises(ValueError):
                build_command(Path("run.py"), [Path("a.pdf")], Path("out"), RunOptions(want=want))
        for want in ("Figure 5", "图5", "圖5", "Fig. 5 and Figure 6"):
            self.assertTrue(is_figure_number_request(want))

    def test_disabled_vlm_controls_do_not_leak_to_ocr(self):
        cmd = build_command(Path("run.py"), [Path("a.pdf")], Path("out"),
                            RunOptions(use_vlm=False, want_deep=True, find_series=False))
        self.assertIn("--ocr", cmd)
        self.assertNotIn("--want-deep", cmd)
        self.assertNotIn("--no-find-series", cmd)

    def test_vlm_defaults_and_force_match_cli(self):
        cmd = build_command(Path("run.py"), [Path("a.pdf")], Path("out"),
                            RunOptions(use_vlm=True, force=False, want="800 bar 速度随时间"))
        self.assertIn("--no-ocr", cmd)
        self.assertNotIn("--force", cmd)
        self.assertEqual(cmd[cmd.index("--want") + 1], "800 bar 速度随时间")

    def test_template_is_not_silently_ignored(self):
        for vlm, analyze in ((False, False), (True, True)):
            with self.subTest(vlm=vlm, analyze=analyze), self.assertRaises(ValueError):
                build_command(Path("run.py"), [Path("a.pdf")], Path("out"),
                              RunOptions(use_vlm=vlm, template=Path("format.dat")),
                              analyze_only=analyze)

    def test_single_pdf_command_covers_ui_options(self):
        opts = RunOptions(dpi=400, pages="3, 5", force=True, use_vlm=True,
                          use_ocr=True, want="Figure 5", want_deep=True,
                          keep_intermediates=True, template=Path("format.dat"))
        command = build_command(Path("run.py"), [Path("paper.pdf")], Path("out"), opts)
        self.assertIn("all", command)
        self.assertIn("--force", command)
        self.assertIn("--vlm", command)
        self.assertIn("--ocr", command)
        self.assertEqual(command[command.index("--pages") + 1], "3,5")
        self.assertEqual(command[command.index("--want") + 1], "Figure 5")
        self.assertIn("--template", command)

    def test_multiple_pdfs_use_batch(self):
        inputs = [Path("papers/a.pdf"), Path("papers/b.pdf")]
        command = build_command(Path("run.py"), inputs, Path("out"), RunOptions(jobs=4))
        self.assertIn("batch", command)
        self.assertEqual(command[command.index("--jobs") + 1], "4")

    def test_commands_reach_real_cli_with_expected_values(self):
        import run as pipeline
        for count, analyze_only, target in ((1, False, "analyze"), (1, True, "analyze"),
                                             (2, False, "batch")):
            with self.subTest(count=count, analyze=analyze_only):
                command = build_command(Path("run.py"),
                    [Path(f"papers/{i}.pdf") for i in range(count)], Path("out"),
                    RunOptions(use_vlm=True, use_ocr=True, want="Figure 5 的速度曲线",
                               dpi=400, pages="3, 5", want_deep=True, find_series=False,
                               jobs=3, model="test-model", base_url="http://localhost:1234"),
                    analyze_only=analyze_only)
                with patch.object(sys, "argv", command[1:]), \
                     patch.object(pipeline.vlmc, "DeepSeekVLM") as client, \
                     patch.object(pipeline, "analyze", return_value="config.json") as analyze, \
                     patch.object(pipeline, "batch") as batch, \
                     patch.object(pipeline, "extract", return_value={}), \
                     patch.object(pipeline, "print_vlm_usage"):
                    pipeline.main()
                kwargs = (batch if target == "batch" else analyze).call_args.kwargs
                self.assertEqual(kwargs["want"], "Figure 5 的速度曲线")
                self.assertEqual(kwargs["pages"], "3,5")
                self.assertTrue(kwargs["use_ocr"])
                self.assertTrue(kwargs["want_deep"])
                self.assertFalse(kwargs["find_series"])
                self.assertEqual(client.call_args.kwargs["model"], "test-model")
                self.assertEqual(client.call_args.kwargs["base_url"], "http://localhost:1234")

    def test_collect_and_archive_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "csv").mkdir()
            (root / "verify").mkdir()
            (root / "csv" / "curve.csv").write_text("x,y\n1,2", encoding="utf-8")
            (root / "verify" / "curve.png").write_bytes(b"png")
            (root / "report.md").write_text("# Report", encoding="utf-8")
            artifacts = collect_artifacts(root)
            self.assertEqual(len(artifacts["csv"]), 1)
            self.assertEqual(len(artifacts["images"]), 1)
            with zipfile.ZipFile(BytesIO(make_archive(root))) as archive:
                self.assertIn("csv/curve.csv", [name.replace("\\", "/") for name in archive.namelist()])

    def test_image_review_excludes_retrieval_candidates_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            # Ancestor folder names must not turn arbitrary images into overlays.
            root = Path(temp) / "verify" / "output"
            names = [
                "paper/verify/curve_verify.png",
                "paper/select/originals/rejected.png",
                "paper/select/contact_sheet.jpg",
                "paper/figures/figure.png",
                "paper/panels/panel.png",
                "paper/review/critic.png",
                "paper/gaps/crop.png",
            ]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            self.assertEqual(collect_artifacts(root)["images"],
                             [root / "paper/verify/curve_verify.png"])


if __name__ == "__main__":
    unittest.main()
