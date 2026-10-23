"""Retrieval and extraction contracts, using synthetic data with known answers.

Run with python -m unittest discover -s tests -p test_retrieval_invariants.py.
These tests require neither user PDFs nor an API key.
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dig")]

import batch_extract as be
import axis_ranges as ar
import extract_lines as el
import figure_index as fi
import imgio as iio
import run as app
import series_seed as ss
import split_panels as sp
import triage_pdf as tp
import vector_extract as vx
import vlm_tasks as vt
import vlm_client as vc
import verify_overlay as vo


def inspection(match=True, axes=True, conditions=True):
    return {"is_data_chart": True, "panels": [{
        "x_title": "Time", "y_titles": ["Penetration"],
        "matches_request": match, "axes_match": axes,
        "conditions_match": conditions, "evidence": ["Time [ms]", "Penetration [mm]"]}]}


class FakeVLM:
    def __init__(self, replies, selected=(), unresolved=()):
        self.replies = replies
        self.selected, self.unresolved = list(selected), list(unresolved)
        self.calls = []
        self.usage = {}

    def ask_json(self, image, prompt, **kwargs):
        self.calls.append((image, prompt))
        if prompt.startswith("逐图语义检索"):
            n = int(re.search(r"候选记录：\s*(\d+)\.", prompt)[1])
            data = self.replies[n]
            if isinstance(data, Exception):
                raise data
        else:
            data = {"selected": self.selected, "unresolved": self.unresolved,
                    "confidence": .99, "reason": "合成证据汇总"}
        return data, json.dumps(data)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.image = Path(self.tmp.name) / "原图.png"
        iio.imwrite(self.image, np.full((600, 900, 3), 255, np.uint8))

    def cards(self, n):
        return [{"id": f"f{i}", "page": 1, "label": f"Figure {i}",
                 "caption": "通用标题", "thumb": self.image,
                 "bbox": [0, 0, 100, 100]} for i in range(1, n + 1)]

    def test_late_target_and_caption_tail_are_not_lost(self):
        cards = self.cards(25)
        cards[19]["caption"] = "general " * 60 + "800 bar velocity versus time"
        replies = {i: inspection(False, False) for i in range(1, 26)}
        replies[20] = inspection()
        vlm = FakeVLM(replies, [20])
        with patch.object(vt, "shortlist_figures", side_effect=AssertionError("unsafe shortlist")):
            result = fi.select("速度随时间", cards, vlm)
        self.assertEqual(result["ids"], ["f20"])
        self.assertEqual(result["screened_n"], 25)
        self.assertTrue(all(image == self.image for image, _ in vlm.calls[:-1]))
        self.assertIn("800 bar velocity versus time", vlm.calls[19][1])

    def test_body_receives_full_inventory_and_can_resolve_conditions(self):
        replies = {i: inspection(False, False) for i in range(1, 19)}
        replies[18] = inspection(None, True, None)
        vlm = FakeVLM(replies, [18])
        result = fi.select("800 bar 贯穿距随时间", self.cards(18), vlm,
                           body_text=lambda: "Figure 18 presents penetration at 800 bar.")
        self.assertEqual(result["ids"], ["f18"])
        self.assertTrue(result["body_used"])
        self.assertIn("18. Figure 18", vlm.calls[-1][1])
        self.assertIn("1. Figure 1", vlm.calls[-1][1])

    def test_body_cannot_override_visible_axis_conflict(self):
        vlm = FakeVLM({1: inspection(False, False)}, [1])
        result = fi.select("时间与贯穿距", self.cards(1), vlm,
                           body_text=lambda: "Figure 1 discusses spray penetration.")
        self.assertEqual(result["ids"], [])

    def test_unknown_and_failed_calls_are_not_absence(self):
        for reply in (inspection(None, None, None), RuntimeError("timeout")):
            with self.subTest(reply=reply):
                result = fi.select("贯穿距", self.cards(1), FakeVLM({1: reply}))
                self.assertIsNone(result["missing"])
                self.assertEqual(result["unresolved_ids"], ["f1"])

    def test_number_with_content_does_not_bypass_semantics(self):
        self.assertEqual(fi.local_match("Figure 1", self.cards(2))[0], ["f1"])
        self.assertEqual(fi.local_match("Figure 1 中的贯穿距", self.cards(2))[0], [])
        self.assertEqual(fi._pick({"selected": [True, 1.5, "2"]}, self.cards(2)), ["f2"])

    def test_planned_axis_still_checks_individual_panel(self):
        axis = {"x": {"min": 0, "max": 3, "title": "Time"},
                "y_axes": [{"min": 0, "max": 3, "title": "Mass flow rate"}]}
        answer = {"matches_request": False, "axes_match": False,
                  "conditions_match": True, "note": "纵轴为质量流率"}
        with patch.object(vt, "check_panel_request", return_value=answer) as reader:
            checked, _ = app._ensure_panel_request(object(), self.image, "贯穿距随时间", axis, {})
        reader.assert_called_once()
        self.assertEqual(checked["x"], axis["x"])
        entry = {"axis": {}}
        app._record_axis_check(entry, checked, "贯穿距随时间", lambda _: None)
        self.assertTrue(entry["skip"])


class PixelInvariantTests(unittest.TestCase):
    def test_small_coloured_markers_outside_legend_are_not_erased_as_text(self):
        img = np.full((380,640,3),255,np.uint8)
        frame = (30,20,610,350)
        legend = (270,200,100,70)
        cv2.circle(img,(245,245),4,(0,0,255),-1)
        cv2.circle(img,(305,230),4,(0,0,255),-1)
        mask = el.color_mask(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),frame,(0,0,255),
                             exclude_boxes=[legend])
        self.assertGreater(mask[245,245],0)
        self.assertEqual(mask[230,305],0)

    def test_slowly_rising_black_curve_is_not_a_long_horizontal_grid(self):
        mask = np.zeros((340,780),np.uint8)
        frame = (30,20,750,310)
        points = [(float(x),280.-100.*((x-50)/680.)**.5) for x in range(50,731)]
        cv2.polylines(mask,[np.array(points,np.int32)],False,255,2)
        self.assertLess(ss.flat_run_frac(points),.1)
        trace,kind,_info = ss.trace_in_mask(mask,frame,[points[0],points[340],points[-1]],
                                           allow_loop=False,reject_rules=True)
        self.assertEqual(kind,'line')
        self.assertGreater(len(trace),600)

    def test_partly_covered_filled_squares_require_their_own_corner(self):
        mask = np.zeros((320,740),np.uint8)
        other = np.zeros_like(mask)
        expected = [(80+i*42,260-i*10) for i in range(13)]
        for x,y in expected:
            cv2.rectangle(mask,(x-6,y-6),(x+6,y+6),255,-1)
            other[y:y+8,x-7:x+8] = mask[y:y+8,x-7:x+8]
            mask[y:y+8,x-7:x+8] = 0
        # A different series' filled square crossed by a diagonal same-colour
        # model stroke cannot fabricate another data marker.
        cv2.rectangle(other,(654,54),(666,66),255,-1)
        cv2.line(mask,(645,45),(675,75),255,2)
        cv2.rectangle(other,(694,84),(706,96),255,-1)  # Completely hidden, no own ink.
        centres = ss.shape_marker_centers(mask,'square',(30,20,720,300),hollow=False,
                                          threshold=.8,occlusion_mask=other)
        self.assertEqual(len(centres),len(expected))
        self.assertLess(max(min(np.linalg.norm(np.array(p)-q) for q in centres)
                            for p in expected),2)

    def test_model_hint_cannot_mix_grid_top_with_decorative_bottom(self):
        gray = np.full((640,840),255,np.uint8)
        cv2.rectangle(gray,(80,40),(780,590),225,2)
        cv2.rectangle(gray,(150,100),(750,490),110,1)
        for x in range(250,750,100):
            cv2.line(gray,(x,100),(x,490),230,1)
        for y in range(175,490,75):
            cv2.line(gray,(150,y),(750,y),230,1)
        refined = sp.refine_plot_box(gray,(150,170,750,575))
        self.assertLess(max(abs(a-b) for a,b in zip(refined,(150,100,750,490))),3)

    def test_chart_grid_does_not_turn_true_axes_into_multiple_panels(self):
        gray = np.full((500,740),255,np.uint8)
        cv2.rectangle(gray,(100,80),(680,430),110,1)
        for x in range(200,680,100):
            cv2.line(gray,(x,80),(x,430),230,1)
        for y in range(150,430,70):
            cv2.line(gray,(100,y),(680,y),230,1)
        refined = sp.refine_plot_box(gray,(100,65,680,455))
        self.assertLess(max(abs(a-b) for a,b in zip(refined,(100,80,680,430))),3)

    def test_gray_axes_are_refined_from_coarse_model_box(self):
        gray = np.full((500,700),255,np.uint8)
        cv2.rectangle(gray,(80,130),(620,420),200,2)
        refined = sp.refine_plot_box(gray,(75,65,625,420))
        self.assertLess(max(abs(a-b) for a,b in zip(refined,(80,130,620,420))),3)

    def test_stacked_frames_are_not_calibrated_as_one_large_frame(self):
        gray = np.full((680,850),255,np.uint8)
        cv2.rectangle(gray,(80,65),(760,450),0,2)
        cv2.rectangle(gray,(80,480),(760,630),0,2)
        refined = sp.refine_plot_box(gray,(80,65,760,530))
        self.assertLess(max(abs(a-b) for a,b in zip(refined,(80,65,760,450))),3)

    def test_shape_fit_recovers_open_hollow_glyphs(self):
        mask = np.zeros((340,720),np.uint8)
        expected = [(80+i*40,260-i*9) for i in range(14)]
        for x,y in expected:
            cv2.circle(mask,(x,y),6,255,2)
            mask[y+2:y+9,x-7:x+8] = 0  # Another colour covers part of the outline.
        centres = ss.shape_marker_centers(mask,'circle',(30,20,700,310),hollow=True)
        self.assertEqual(len(centres),len(expected))
        self.assertLess(max(min(np.linalg.norm(np.array(p)-q) for q in centres)
                            for p in expected),2)

    def test_occluded_glyph_fit_requires_its_own_visible_outline(self):
        mask = np.zeros((300,620),np.uint8)
        other = np.zeros_like(mask)
        expected = [(80+i*40,240-i*10) for i in range(12)]
        for x,y in expected:
            cv2.circle(mask,(x,y),6,255,2)
            other[y:y+9,x-8:x+9] = mask[y:y+9,x-8:x+9]
            mask[y:y+9,x-8:x+9] = 0
        # A wholly hidden marker has no identifying evidence and must not appear.
        cv2.circle(other,(570,70),6,255,2)
        centres = ss.shape_marker_centers(mask,'circle',(30,20,600,280),
                                          hollow=True,occlusion_mask=other)
        self.assertEqual(len(centres),12)
        self.assertLess(max(min(np.linalg.norm(np.array(p)-q) for q in centres)
                            for p in expected),2)

    def test_isolated_palette_colour_corrects_approximate_model_hue(self):
        colour, source = ss.resolve_color(np.full((30,30,3),255,np.uint8), [],
                                         '#00ff00',['#64d413','#0000ff'])
        self.assertEqual(colour, (19,212,100))
        self.assertEqual(source,'palette')

    def test_black_identity_is_not_replaced_by_a_coloured_shared_legend(self):
        colour, source = ss.resolve_color(np.full((30,30,3),255,np.uint8), [],
                                         '#000000', ['#ff0000', '#b746ff'])
        self.assertIsNone(colour)
        self.assertEqual(source, 'dark')

    def test_repeated_semantic_colours_do_not_tighten_hue_tolerance(self):
        self.assertEqual(ss.hue_tolerances([(255,0,0),(255,0,0),(0,0,255)]),[14,14,14])

    def test_colour_calibration_uses_current_plot_pixels(self):
        img = np.full((320,600,3),255,np.uint8)
        cv2.line(img,(45,280),(550,80),(255,0,0),3)
        cv2.line(img,(45,280),(550,50),(19,212,100),3)
        palette = ss.pixel_palette(cv2.cvtColor(img,cv2.COLOR_BGR2HSV),(30,20,570,300))
        blue, _ = ss.resolve_color(img, [], '#0000ff', palette)
        green, _ = ss.resolve_color(img, [], '#00b050', palette)
        self.assertEqual(blue,(255,0,0))
        self.assertLessEqual(max(abs(a-b) for a,b in zip(green,(19,212,100))),4)

    def test_page_separator_is_not_a_coordinate_axis(self):
        gray = np.full((650,900),255,np.uint8)
        cv2.line(gray,(0,35),(899,35),170,1)
        cv2.rectangle(gray,(100,100),(790,560),0,2)
        refined = sp.refine_plot_box(gray,(100,35,790,560))
        self.assertLess(max(abs(a-b) for a,b in zip(refined,(100,100,790,560))),3)

    def test_black_mask_excludes_dark_saturated_curves(self):
        img = np.full((300,600,3),255,np.uint8)
        cv2.line(img,(40,240),(550,50),(0,0,0),3)
        cv2.line(img,(40,240),(550,180),(0,0,255),3)
        mask = el.dark_mask(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
                            (20,20,580,280),source_img=img)
        self.assertEqual(mask[193,440],0)
        self.assertGreater(mask[91,440],0)

    def test_global_pixel_path_keeps_same_colour_dashes_above_markers(self):
        mask = np.zeros((360,740),np.uint8)
        expected = lambda x: 320-.4*(x-50)
        for x in range(50,700,24):
            cv2.line(mask,(x,int(expected(x))),(min(699,x+13),int(expected(min(699,x+13)))),255,2)
        lower = [(x,int(320-.10*(x-50))) for x in range(50,700,32)]
        cv2.polylines(mask,[np.array(lower)],False,255,1)
        for point in lower:
            cv2.circle(mask,point,5,255,2)
        trace = ss.soft_guided_trace(mask,[(55,318),(380,188),(690,64)],(20,20,720,340))
        self.assertGreater(max(x for x,y in trace)-min(x for x,y in trace),600)
        self.assertLess(np.median([abs(y-expected(x)) for x,y in trace]),2)
        self.assertTrue(all(mask[int(round(y)),int(round(x))]>0 for x,y in trace))
        self.assertTrue(any(b[0]-a[0]>2 for a,b in zip(trace,trace[1:])))

    def test_coarse_anchors_do_not_cut_off_concave_marker_tracks(self):
        frame = (20,20,720,420)
        points = [(float(x),400.-340.*np.sqrt((x-40)/640.)) for x in range(40,681,40)]
        anchors = [(40.,400.),(360.,210.),(680.,60.)]
        track = ss.best_marker_track(points,frame,min_points=3,anchors=anchors)
        self.assertEqual(track,points)

    def test_marker_identity_beats_denser_same_colour_model_candidates(self):
        frame = (20,20,720,420)
        points = [(float(x),380.-.12*x) for x in range(80,681,60)]
        model = [(float(x),380.-.42*x) for x in range(80,681,20)]
        anchors = [points[0],points[5],points[-1]]
        track = ss.best_marker_track(points+model,frame,min_points=3,anchors=anchors)
        self.assertLess(np.median([abs(y-(380.-.12*x)) for x,y in track]),2)
        self.assertGreater(track[-1][0]-track[0][0],500)

    def test_series_reader_receives_panel_and_shared_legend(self):
        vlm = FakeVLM({})
        with patch.object(vlm, 'ask_json', return_value=({'series': []}, '{}')) as ask:
            vt.find_series(vlm,'panel.png','legend',figure_path='figure.png',panel_label='(b)',frame=(50,40,600,400))
        self.assertEqual(ask.call_args.args[0], ['panel.png','figure.png'])
        self.assertIn('第一张图',ask.call_args.args[1])
        self.assertIn('left=50, top=40, right=600, bottom=400',ask.call_args.args[1])

    def test_verification_points_do_not_draw_unmeasured_connecting_segments(self):
        canvas = np.full((500,800,3),255,np.uint8)
        series = [('data',(0,0,255),[1.,2.,8.],[1.,3.,7.])]
        with patch.object(vo.cv2,'polylines') as lines:
            vo.draw_plot(canvas,(80,40,700,400),series,(0,10),(0,10),'x','y',styles={'data':'points'})
            lines.assert_not_called()
            vo.draw_plot(canvas,(80,40,700,400),series,(0,10),(0,10),'x','y',styles={'data':'line'})
            lines.assert_called_once()

    def test_critic_receives_original_before_measurement_overlay(self):
        vlm = FakeVLM({})
        with patch.object(vlm,'ask_json',return_value=({'ok':True},'{}')) as ask:
            vt.revise_traces(vlm,'overlay.png','series','measured',original_path='original.png')
        self.assertEqual(ask.call_args.args[0],['original.png','overlay.png'])

    def test_filled_marker_measurements_survive_same_colour_model_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((400,650,3),255,np.uint8)
            frame = (50,40,610,340)
            points = [(70+i*35,320-i*20) for i in range(14)]
            cv2.polylines(img,[np.array(points)],False,(255,0,0),2)
            for point in points:
                cv2.rectangle(img,(point[0]-6,point[1]-6),(point[0]+6,point[1]+6),(255,0,0),-1)
            image = Path(tmp)/'plot.png'
            iio.imwrite(image,img)
            spec = [{'label':'data','role':'data','color':'#0000ff','draw':'markers_only',
                     'marker_shape':'square','marker_fill':'filled','closed':False,
                     'anchors':[[.04,.07],[.5,.5],[.85,.94]]},
                    {'label':'model','role':'model','color':'#0000ff','draw':'line_only',
                     'closed':False,'anchors':[[.04,.07],[.5,.5],[.85,.94]]}]
            csv = Path(tmp)/'csv'
            csv.mkdir()
            result = app._extract_seeded({'id':'synthetic','frame':frame},image,
                                         [0,10,0,100],csv,spec)
            entry = next(s for s in result['series'] if s['series_name']=='data')
            self.assertEqual(entry['points'],14)

    def test_same_colour_series_corridor_does_not_add_pixels(self):
        mask = np.zeros((300,500),np.uint8)
        cv2.line(mask,(40,250),(460,30),255,2)
        cv2.line(mask,(40,250),(460,210),255,2)
        frame = (20,10,480,280)
        anchors = [(40,250),(250,140),(460,30)]
        filtered = ss.mask_near_anchors(mask,anchors,frame)
        self.assertTrue(np.all(filtered[mask==0]==0))
        self.assertEqual(filtered[210,460],0)
        self.assertGreater(filtered[30,460],0)

    def test_coherent_spines_exclude_a_legend_row(self):
        gray = np.full((500, 700), 255, np.uint8)
        cv2.rectangle(gray, (80, 130), (620, 420), 0, 2)
        cv2.putText(gray, 'Experiment A --- Model', (130, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, 0, 2)
        refined = sp.refine_plot_box(gray, (80, 65, 620, 420))
        self.assertLess(max(abs(a-b) for a,b in zip(refined, (80,130,620,420))), 3)

    def test_duplicate_left_axes_cannot_silently_replace_calibration(self):
        result = app._norm_y_axes({'y_axes': [
            {'side': 'left', 'min': 0, 'max': 80},
            {'side': 'left', 'min': 0, 'max': 30}]})
        self.assertEqual(result, {})

    def test_axis_schema_retries_valid_json_with_wrong_structure(self):
        vlm = FakeVLM({})
        replies = [({'type': 'json_object'}, '{}'),
                   ({'x': {'min': 0, 'max': 3}, 'y_axes': [{'side': 'left', 'min': 0, 'max': 80}]}, '{}')]
        with patch.object(vlm, 'ask_json', side_effect=replies) as ask:
            result = vt.read_axis_ranges(vlm, 'unused.png')
        self.assertEqual(result['y_axes'][0]['max'], 80)
        self.assertEqual(ask.call_count, 2)

    def test_truncated_json_retries_with_larger_budget(self):
        client = vc.DeepSeekVLM(api_key='unused', cache=False, log=False)
        with patch.object(client, 'ask', side_effect=['{"panels":[', '{"panels": []}']) as ask:
            result, _raw = client.ask_json('unused.png', 'prompt', max_tokens=100)
        self.assertEqual(result, {'panels': []})
        self.assertEqual([c.kwargs['max_tokens'] for c in ask.call_args_list], [100, 200])

    def test_pdf_strips_join_transitively_and_keep_separate_figures(self):
        def image(b):
            return {"bbox": b, "display_pt": [b[2] - b[0], b[3] - b[1]]}
        groups = tp.image_groups([image([10, 10, 200, 45]), image([10, 45, 200, 80]),
                                  image([10, 80, 200, 140]), image([10, 150, 200, 240])])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["bbox"], [10, 10, 200, 140])
        self.assertEqual(groups[0]["indices"], [1, 2, 3])

    def test_calibration_frame_is_authoritative(self):
        gray = np.full((500, 700), 255, np.uint8)
        cv2.rectangle(gray, (10, 10), (690, 490), 0, 2)
        hint = (90, 70, 620, 400)
        self.assertEqual(be._frame_of(gray, hint), hint)
        self.assertEqual(el.to_data([(90, 400), (620, 70)], hint, 0, 3, 0, 80),
                         [(0., 0.), (3., 80.)])

    def test_subplot_crop_keeps_context_and_transform(self):
        frames = [(100, 100, 500, 350), (620, 100, 1020, 350)]
        x0, y0, x1, y1 = sp.panel_crop_bounds(frames[1], frames, (500, 1100))
        self.assertGreaterEqual(x0, 500)
        self.assertLessEqual(x0, 620 - 60)
        self.assertLess(y0, 100)
        self.assertGreater(y1, 350)
        relative = (620 - x0, 100 - y0, 1020 - x0, 350 - y0)
        self.assertEqual(el.to_data([(relative[0], relative[3])], relative, 0, 3, 0, 80),
                         [(0., 0.)])

    def test_gutter_preserves_bottom_axis_title_without_neighbouring_plot(self):
        frames = [(120,80,690,430),(120,575,690,880)]
        crop = sp.panel_crop_bounds(frames[0],frames,(1000,850))
        self.assertGreater(crop[3],535)  # tick row + x title occupy the lower gutter
        self.assertLess(crop[3],575)
        lower = sp.panel_crop_bounds(frames[1],frames,(1000,850))
        self.assertGreater(lower[1],535)

    def test_multiplot_vector_region_is_not_one_frame(self):
        rects = [{"x0": x, "top": y, "x1": x + 180, "bottom": y + 140}
                 for y in (20, 220) for x in (20, 260)]
        page = SimpleNamespace(rects=rects, lines=[])
        result = vx.extract_vector_figure(page, [0, 0, 460, 390])
        self.assertEqual(len(result["axes_boxes"]), 4)
        self.assertNotIn("x_calibration", result)

    def test_hollow_markers_are_measurements(self):
        mask = np.zeros((300, 600), np.uint8)
        expected = [(50 + i * 35, 240 - i * 12) for i in range(12)]
        for x, y in expected:
            cv2.circle(mask, (x, y), 7, 255, 2)
        cv2.polylines(mask, [np.array(expected)], False, 255, 1)
        centres = ss.marker_centers(mask)
        self.assertEqual(len(centres), len(expected))
        self.assertLess(max(np.linalg.norm(np.array(a) - b) for a, b in zip(centres, expected)), 2)

    def test_blank_tail_cannot_be_filled_from_model_or_neighbour(self):
        frame = (10, 10, 590, 280)
        mask = np.zeros((300, 600), np.uint8)
        trace = [(float(x), 250. - .3 * x) for x in range(30, 250)]
        for x, y in trace:
            mask[int(round(y)), int(x)] = 255
        neighbour = {x: 250. - .3 * x for x in range(30, 570)}
        filled, _ = ss.complete_trace(trace, mask, frame, [("another series", neighbour)])
        self.assertLessEqual(max(x for x, _ in filled), 249)
        repaired, _ = ss.retrace_from_anchors(trace, [(400, 130), (540, 88)], mask, frame)
        self.assertLessEqual(max(x for x, _ in repaired), 249)

    def test_wrong_semantic_legend_box_does_not_erase_curve(self):
        img = np.full((350, 600, 3), 255, np.uint8)
        cv2.line(img, (50, 280), (560, 80), (0, 0, 255), 3)
        panel = {"extraction_plan": {"exclude_regions": [
            {"role": "legend", "box": [.7, .55, .95, .8]}]}}
        self.assertEqual(app._planner_exclude_boxes(panel, (30, 20, 580, 300), img=img), [])

    def test_marker_csv_does_not_contain_interpolated_or_model_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((400, 650, 3), 255, np.uint8)
            frame = (50, 40, 610, 340)
            points = [(70 + i * 30, 320 - i * 18) for i in range(11)]
            cv2.polylines(img, [np.array(points)], False, (255, 0, 0), 1)
            for point in points:
                cv2.circle(img, point, 7, (255, 0, 0), -1)
            image = Path(tmp) / "plot.png"
            iio.imwrite(image, img)
            spec = [{"label": "data", "role": "data", "color": "#0000ff",
                     "draw": "markers_connected", "closed": False,
                     "anchors": [[.1, .1], [.4, .4], [.9, .7]]}]
            csv = Path(tmp) / "csv"
            csv.mkdir()
            result = app._extract_seeded({"id": "synthetic", "frame": frame}, image,
                                         [0, 10, 0, 100], csv, spec)
            self.assertEqual(len(result["series"]), 1)
            output = result["series"][0]
            self.assertEqual(output["points"], 11)
            self.assertLess(output["x_range"][1], 6)


class VectorPanelTests(unittest.TestCase):
    def test_pdf_gray_black_is_not_a_missing_colour(self):
        self.assertEqual(vx._color_key({'stroke':True,'stroking_color':0.}),(0.,0.,0.))
        self.assertEqual(vx._color_key({'stroke':True,'stroking_color':(.2,)}),(.2,.2,.2))

    def page(self):
        color = (1.,0.,0.)
        points = [(10.+15.*i,100.-5.*i) for i in range(10)]
        model = {'pts':points,'x0':10.,'x1':145.,'top':55.,'bottom':100.,
                 'stroke':True,'fill':False,'stroking_color':color,'path':[]}
        markers = [{'x0':x-1.5,'x1':x+1.5,'top':y-1.5,'bottom':y+1.5,
                    'stroke':True,'fill':True,'stroking_color':color,
                    'non_stroking_color':color}
                   for x,y in [(15.+18*i,110.-3*i) for i in range(8)]]
        markers.append({'x0':95.,'x1':98.,'top':12.,'bottom':15.,
                        'stroke':True,'fill':True,'stroking_color':color})
        return SimpleNamespace(curves=[model],lines=[],rects=markers)

    def specs(self):
        return [{'label':'experiment','color':'#ff0000','draw':'markers_only',
                 'marker_shape':'square','anchors':[[.075,.083],[.39,.17],[.705,.258]]},
                {'label':'model','color':'#ff0000','draw':'line_only',
                 'anchors':[[.05,.167],[.35,.333],[.725,.458]]}]

    def test_vector_markers_survive_same_colour_model_and_ignore_legend(self):
        rows = vx.extract_series(self.page(),[0,0,200,120],
                                 {'slope':1.,'intercept':0.},{'slope':-1.,'intercept':120.},
                                 series_spec=self.specs(),exclude_boxes=[[90,0,110,25]])
        self.assertEqual([(r['series_name'],r['n_points']) for r in rows],
                         [('experiment',8),('model',10)])
        self.assertEqual(rows[0]['source'],'markers:rect')
        self.assertEqual(rows[0]['data'][0],(15.,10.))
        self.assertEqual(rows[0]['data'][-1],(141.,31.))

    def test_panel_vector_coordinates_are_mapped_through_calibrated_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((320,500,3),255,np.uint8)
            image = Path(tmp)/'panel.png'
            iio.imwrite(image,img)
            panel = {'id':'synthetic','frame':[40,30,440,270],
                     'vector_panel_axes':[0,0,200,120]}
            result = app._extract_vector_panel(panel,image,[0,20,0,120],Path(tmp),
                                                self.specs(),self.page())
            self.assertEqual(result['method'],'vector-panel')
            self.assertEqual(result['series'][0]['points'],8)
            xs,ys = vo.read_csv(Path(tmp)/result['series'][0]['file'])
            self.assertAlmostEqual(xs[0],1.5)
            self.assertAlmostEqual(ys[0],10.)
            self.assertEqual(result['frame'],[40,30,440,270])

    def test_axis_snap_cannot_replace_panel_with_legend(self):
        lines = [{'x0':0.,'x1':200.,'top':0.,'bottom':0.},
                 {'x0':0.,'x1':200.,'top':120.,'bottom':120.},
                 {'x0':0.,'x1':0.,'top':0.,'bottom':120.},
                 {'x0':200.,'x1':200.,'top':0.,'bottom':120.},
                 {'x0':2.,'x1':100.,'top':2.,'bottom':2.}]
        page = SimpleNamespace(lines=lines)
        self.assertEqual(vx.snap_axes(page,[1.,1.,199.,119.]),[0.,0.,200.,120.])

    def test_pdf_diamond_geometry_is_independent_of_start_vertex(self):
        page = self.page()
        diamonds = []
        for i in range(8):
            x,y = 15.+18*i,110.-3*i
            vertices = [(x-2,y),(x,y+2),(x+2,y),(x,y-2),(x-2,y)]
            diamonds.append({'pts':vertices,'path':[('m',vertices[0])]+[('l',p) for p in vertices[1:]],
                             'x0':x-2,'x1':x+2,'top':y-2,'bottom':y+2,
                             'stroke':True,'fill':False,'stroking_color':(1.,0.,0.)})
        page.curves += diamonds
        page.rects = []
        spec = self.specs()
        spec[0]['marker_shape']='diamond'
        rows = vx.extract_series(page,[0,0,200,120],{'slope':1.,'intercept':0.},
                                 {'slope':-1.,'intercept':120.},series_spec=spec)
        self.assertEqual(rows[0]['source'],'markers:diamond')
        self.assertEqual(rows[0]['n_points'],8)


class TickAndLegendTests(unittest.TestCase):
    def test_logarithmic_axis_cannot_enter_linear_calibration(self):
        self.assertIsNone(app._norm_axis({'min':.001,'max':100,'scale':'log'}))
        self.assertEqual(app._norm_axis({'min':0,'max':100,'scale':'linear'}),[0,100])

    def test_nonfinite_axis_values_are_not_valid_ranges(self):
        for values in ([0,float('inf')],[-float('inf'),100],[0,float('nan')]):
            self.assertIsNone(app._norm_axis(values))

    def test_dashed_lines_preserve_visible_strokes_without_filling_blank_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((450,760,3),255,np.uint8)
            frame = [50,30,710,420]
            for x in range(70,690,24):
                for shift,color in ((0,(0,0,255)),(-35,(255,0,0))):
                    y = 385-.4*(x-70)+shift
                    cv2.line(img,(x,round(y)),(x+12,round(y-4.8)),color,2)
            image = Path(tmp)/'dashes.png'
            iio.imwrite(image,img)
            spec = [{'label':label,'color':color,'role':'data','draw':'line_only','closed':False,
                     'anchors':[[.03,.1],[.5,.45],[.95,.75]]}
                    for label,color in (('warm','#ff0000'),('cool','#0000ff'))]
            result = app._extract_seeded({'id':'synthetic','frame':frame},image,[0,2,0,100],Path(tmp),spec)
            self.assertEqual(len(result['series']),2)
            for row in result['series']:
                self.assertEqual(row['visible_points']+row['inferred_points'],row['points'])
                self.assertEqual(row['inferred_points'],0)
                self.assertGreater(row['visible_points'],20)
                self.assertFalse(any('推断点' in w for w in row['warnings']))
                # Check against the image itself: gaps between dashes are not
                # observations, even when the series is semantically continuous.
                xs,ys = vo.read_csv(Path(tmp)/row['file'])
                self.assertEqual(len(xs),row['points'])
                ink = ((img[:,:,2]>200)&(img[:,:,0]<40) if row['series_name']=='warm'
                       else (img[:,:,0]>200)&(img[:,:,2]<40))
                for x,y in zip(xs,ys):
                    px = round(frame[0]+x/2*(frame[2]-frame[0]))
                    py = round(frame[3]-y/100*(frame[3]-frame[1]))
                    self.assertTrue(ink[max(0,py-3):py+4,px].any(),(row['series_name'],x,y))

    def test_grid_and_error_bands_cannot_become_large_exclusion_boxes(self):
        img = np.full((450,700,3),255,np.uint8)
        frame = (60,30,660,390)
        cv2.rectangle(img,frame[:2],frame[2:],(100,100,100),1)
        for x in range(160,660,100):
            cv2.line(img,(x,30),(x,390),(230,230,230),1)
        for y in range(90,390,60):
            cv2.line(img,(60,y),(660,y),(230,230,230),1)
        cv2.rectangle(img,(160,180),(560,240),(225,225,245),-1)
        cv2.line(img,(100,370),(620,110),(0,0,255),3)
        cv2.rectangle(img,(475,310),(645,375),(160,160,160),1)
        for y,label in ((330,'Warm'),(355,'Cool')):
            cv2.line(img,(490,y),(515,y),(0,0,255),2)
            cv2.putText(img,label,(525,y+5),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        boxes = be.inner_boxes(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),frame,img=img)
        self.assertTrue(any(x<=500<x+w and y<=345<y+h for x,y,w,h in boxes))
        self.assertFalse(any(x<=300<x+w and y<=210<y+h for x,y,w,h in boxes))
        self.assertFalse(any(x<=300<x+w and y<=270<y+h for x,y,w,h in boxes))

    def test_zoom_reference_edges_cannot_mask_real_data(self):
        img = np.full((520,800,3),255,np.uint8)
        frame = (40,30,740,470)
        cv2.rectangle(img,frame[:2],frame[2:],(0,0,0),2)
        cv2.rectangle(img,(35,325),(95,475),(0,195,255),2)
        cv2.line(img,(95,425),(300,360),(0,195,255),2)
        cv2.rectangle(img,(300,300),(530,430),(0,195,255),2)
        cv2.rectangle(img,(330,315),(515,405),(0,0,0),2)
        cv2.line(img,(330,400),(510,330),(255,0,0),3)
        for y in (340,370,400):
            cv2.putText(img,'20',(304,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
        cv2.polylines(img,[np.array([(40,470),(90,380),(170,315),(250,280),(700,80)])],
                      False,(255,0,0),4)
        boxes = be.inner_boxes(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),frame,img=img)
        self.assertTrue(any(x<=350<x+w and y<=350<y+h for x,y,w,h in boxes))
        self.assertFalse(any(x<=85<x+w and y<=380<y+h for x,y,w,h in boxes))
        self.assertFalse(any(x<=180<x+w and y<=350<y+h for x,y,w,h in boxes))

    def test_dense_marker_centre_noise_does_not_remove_measured_samples(self):
        frame = (20,20,720,420)
        points = [(float(x),380.-.23*x+(.9 if i%2 else -.9))
                  for i,x in enumerate(range(60,681,20))]
        track = ss.best_marker_track(points,frame,min_points=3,anchors=[points[0],points[15],points[-1]])
        self.assertEqual(track,points)

    def test_filled_marker_core_is_not_replaced_by_same_colour_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = np.full((460,760,3),255,np.uint8)
            frame = [50,40,710,420]
            navy = (125,8,0)
            cv2.line(img,(70,360),(690,150),navy,7)
            points = [(100+45*i,390-13*i) for i in range(13)]
            for x,y in points:
                cv2.rectangle(img,(x-6,y-6),(x+6,y+6),(190,115,0),-1)
                cv2.rectangle(img,(x-6,y-6),(x+6,y+6),navy,2)
            image = Path(tmp)/'plot.png'
            iio.imwrite(image,img)
            spec = [{'label':'experiment','role':'data','color':'#002b80','marker_color':'#002b80',
                     'draw':'markers_only','marker_shape':'square','marker_fill':'filled',
                     'anchors':[[.08,.08],[.5,.26],[.9,.5]]},
                    {'label':'model','role':'model','color':'#002b80','draw':'line_only',
                     'anchors':[[.03,.15],[.5,.45],[.97,.7]]}]
            result = app._extract_seeded({'id':'synthetic','frame':frame},image,[0,10,0,100],Path(tmp),spec)
            experiment = next(s for s in result['series'] if s['series_name']=='experiment')
            self.assertEqual(experiment['points'],len(points))
            xs,ys = vo.read_csv(Path(tmp)/experiment['file'])
            expected = el.to_data(points,frame,0,10,0,100)
            self.assertLess(max(abs(a-b[1]) for a,b in zip(ys,expected)),.6)

    def test_highest_label_below_frame_does_not_compress_values(self):
        img = np.full((450,600,3),255,np.uint8)
        frame = [80,40,560,400]
        cv2.rectangle(img,(80,40),(560,400),(180,180,180),1)
        for y in (400,250,100):
            cv2.line(img,(80,y),(91,y),(80,80,80),2)
        reading = {'x':None,'y_axes':[{'side':'left','scale':'linear','min':0,'max':100,
                    'ticks':[{'value':0,'position':0},{'value':50,'position':.42},
                             {'value':100,'position':.83}]}]}
        result = ar.calibrate_vlm_ticks(img,frame,reading)['y_axes'][0]
        self.assertEqual(result['max'],120.)
        self.assertTrue(result['tick_calibration']['accepted'])

    def test_visual_review_cannot_replace_tick_fitted_range(self):
        panel = {'vlm_axis':{'y_axes':[{'side':'left','tick_calibration':
                  {'accepted':True,'frame_range':[0,120]}}]}}
        fix,rejected = app._validated_axis_fix(panel,{'y':[0,80]})
        self.assertEqual(fix,{})
        self.assertEqual(rejected,['y'])

    def test_shared_legend_preserves_physical_style_without_exp_convention(self):
        specs = [{'label':'-7° EXP','color':'#ff0000','draw':'markers_connected'},
                 {'label':'-7° ML','color':'#ff0000','draw':'line_only'}]
        contracts = [{'panel_index':2,'shared':True,'entries':[
            {'label':'-7°EXP','color':'#ff0000','draw':'line_only'},
            {'label':'-7°ML','color':'#ff0000','draw':'markers_only','marker_shape':'circle',
             'marker_fill':'hollow'}]}]
        result = app._bind_legend_styles(specs,contracts,3)
        self.assertEqual([s['draw'] for s in result],['line_only','markers_only'])
        self.assertEqual(result[1]['marker_fill'],'hollow')
        wrong = [{'label':'another case','color':'#ff0000','draw':'markers_connected'}]
        self.assertEqual(app._bind_legend_styles(wrong,contracts,3)[0]['draw'],'markers_connected')
        opposite = [{'label':'7° EXP','color':'#ff0000','draw':'markers_only'}]
        self.assertEqual(app._bind_legend_styles(opposite,contracts,3)[0]['draw'],'markers_only')

    def test_spot_check_preserves_physical_identity_and_shared_legend_context(self):
        vlm = FakeVLM({})
        with patch.object(vlm,'ask_json',return_value=({'checks':[]},'{}')) as ask:
            vt.spot_check(vlm,'panel.png',[{'id':1,'x':.5,'series':'data',
                          'color':'#000000','draw':'markers_only'}],figure_path='figure.png')
            self.assertEqual(ask.call_args.args[0],['panel.png','figure.png'])
            self.assertIn('#000000',ask.call_args.args[1])
            self.assertIn('markers_only',ask.call_args.args[1])

    def test_tall_gray_legend_against_axis_is_detected(self):
        img = np.full((400,700,3),255,np.uint8)
        frame = [60,30,660,360]
        cv2.rectangle(img,(60,30),(660,360),(180,180,180),1)
        cv2.rectangle(img,(460,70),(660,260),(220,220,220),1)
        for y in (95,140,185,230):
            cv2.line(img,(480,y),(505,y),(0,0,255),2)
            cv2.putText(img,'EXP',(520,y+5),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        boxes = be.legend_boxes(img,frame)
        self.assertTrue(any(abs(x-460)<=3 and abs(y-70)<=3 and h>=185 for x,y,w,h in boxes))


if __name__ == "__main__":
    unittest.main()
