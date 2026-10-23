"""Image-grounded regression cases; no model/API or paper-specific constants."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'dig')]
import run as R
import series_seed as ss
import extract_lines as el


class TraceEvidenceTests(unittest.TestCase):
    def test_compressed_black_is_not_a_saturated_coloured_stroke(self):
        image=np.full((160,240,3),255,np.uint8)
        cv2.circle(image,(60,80),3,(6,6,3),-1)
        cv2.line(image,(110,30),(200,120),(5,5,120),3)
        mask=el.dark_mask(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),
                          (10,10,230,150),source_img=image)
        self.assertGreater(mask[80,60],0)
        self.assertEqual(mask[70,150],0)

    def test_small_filled_markers_survive_at_different_resolutions(self):
        for radius in (2,3,6):
            mask=np.zeros((200,320),np.uint8)
            truth=[(40+40*i,110+10*(i%2)) for i in range(6)]
            for p in truth:cv2.circle(mask,p,radius,255,-1)
            pts=ss.shape_marker_centers(mask,'circle',(10,10,310,190),hollow=False,threshold=.8)
            for x,y in truth:
                self.assertTrue(any(np.hypot(x-a,y-b)<=2 for a,b in pts),(radius,x,y,pts))

    def test_vertical_scatter_branch_preserves_measured_points(self):
        track=[(x,80.) for x in range(20,101,10)]
        branch=[(100.,y) for y in range(90,161,10)]
        remote=[(190.,y) for y in range(80,161,10)]
        found=ss.extend_vertical_markers(track,track+branch+remote,(0,0,240,200))
        self.assertTrue(all(p in found for p in branch))
        self.assertTrue(all(p not in found for p in remote))

    def test_same_colour_connected_branches_have_independent_seeds(self):
        mask=np.zeros((220,340),np.uint8)
        cv2.line(mask,(20,190),(310,30),255,2)
        cv2.line(mask,(20,190),(310,140),255,2)
        c=ss.candidates_in_mask(mask,(10,10,320,200),[],allow_loop=False)
        paths=[v['trace'] for v in c if v['span']>.7]
        self.assertTrue(any(abs(t[-1][1]-30)<8 for t in paths))
        self.assertTrue(any(abs(t[-1][1]-140)<8 for t in paths))

    def test_review_uses_same_scaled_axis_as_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.csv';p.write_text('x,y\n0,0\n0.001,1\n',encoding='utf-8')
            with patch.object(R.vo,'compose_overlay') as compose:
                R.make_verify_image('unused.png',(0,0,100,100),
                    {'x':[0,1],'y':[0,1],'multiplier':{'x':.001}},['#ff0000'],[p],'out.png')
            self.assertEqual(compose.call_args.args[2],(0.,.001))

    def test_same_colour_model_line_does_not_follow_experimental_dots(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);image=np.full((300,440,3),255,np.uint8);frame=(20,20,420,280)
            xs=list(range(40,401))
            for x in xs:cv2.circle(image,(x,int(250-.40*x)),1,(0,0,220),-1)
            for x in range(60,400,20):cv2.circle(image,(x,int(180-.10*x)),5,(0,0,220),-1)
            path=folder/'panel.png';cv2.imwrite(str(path),image);csv=folder/'csv';csv.mkdir()
            def norm(x,y):return [(x-20)/400,(280-y)/260]
            spec=[{'label':'points','color':'#dc0000','marker_color':'#dc0000',
                   'dark':False,'role':'data','draw':'markers_only','marker_shape':'circle',
                   'marker_fill':'filled','closed':False,
                   'anchors':[norm(x,180-.10*x) for x in (60,200,380)]},
                  {'label':'model','color':'#dc0000','line_color':'#dc0000',
                   'dark':False,'role':'model','draw':'line_only','closed':False,
                   'anchors':[norm(x,250-.40*x) for x in (40,200,400)]}]
            result=R._extract_seeded({'frame':frame,'id':'p'},path,[0,400,0,260],csv,spec)
            model=next(s for s in result['series'] if s['series_name']=='model')
            x,y=R.vo.read_csv(csv/model['file'])
            px=np.array(x)+20;py=280-np.array(y)
            self.assertLess(float(np.quantile(abs(py-(250-.40*px)),.95)),3.)
            self.assertEqual(len(result['series']),2)
            points=next(s for s in result['series'] if s['series_name']=='points')
            x,y=R.vo.read_csv(csv/points['file'])
            px=np.array(x)+20;py=280-np.array(y)
            self.assertGreaterEqual(len(x),12)
            self.assertLess(float(np.quantile(abs(py-(180-.10*px)),.95)),3.)


if __name__=='__main__':unittest.main()
