"""Four VLM jobs for the extraction pipeline.

  1. classify_chart     - is this a line chart? how many panels? dual Y axis?
  2. read_axis_ranges   - tick ranges for x and y (replaces / cross-checks OCR)
  3. name_series        - map detected colours to legend labels
  4. spot_check         - re-read a few y values independently, to audit our tracing

All prompts demand json (required by DeepSeek's JSON Output mode) and include an
example of the expected shape.
"""

CLASSIFY_SYSTEM = "你是科研论文图表的分析助手，只输出 json。"

CLASSIFY_PROMPT = """你是图表提取 Agent。先理解整张论文插图，再给像素代码一份可执行计划。
只输出 json，结构如下：
{"is_line_chart": true, "chart_type": "line", "panel_count": 2,
 "panel_layout": {"rows": 1, "cols": 2}, "dual_y_axis": false,
 "has_legend": true, "reason": "两个独立坐标面板",
 "shared_axis": {
   "x": {"min": 0, "max": 1.8, "scale": "linear", "title": "Time",
         "unit": "ms", "multiplier": 1},
   "y_axes": [{"side": "left", "min": 0, "max": 90, "scale": "linear",
               "title": "Penetration", "unit": "mm", "multiplier": 1}]},
 "panels": [
   {"label": "150 MPa", "plot_box": [0.07, 0.04, 0.49, 0.45],
    "axis": null,
    "exclude_regions": [{"role": "legend", "box": [0.55, 0.08, 0.97, 0.35]}],
    "series": [
      {"label": "Exp 150 MPa", "role": "data", "y_axis": "left", "draw": "markers_only",
       "line_color": null, "marker_color": "#2448ff", "marker_shape": "circle",
       "dark": false, "anchors": [[0.05, 0.05], [0.30, 0.48], [0.55, 0.92]]},
      {"label": "Sim 150 MPa", "role": "model", "y_axis": "left", "draw": "line_only",
       "line_color": "#2448ff", "marker_color": null, "marker_shape": "none",
       "dark": false, "anchors": [[0.05, 0.04], [0.30, 0.44], [0.55, 0.90]]}
    ]}
 ]}
字段要求：
- chart_type 只能是 line|scatter|bar|pie|photo|schematic|table|mixed 之一
- is_line_chart 表示"是否是可以提取数据点的折线图/散点图"
- panel_count 是子图数量（(a)(b)(c) 这种分开计数）
- panel_layout 是视觉排列的行列数；单图填 rows=1, cols=1。不要把同一子图里的
  多条曲线误算成多个 panel
- panels 必须与 panel_count 一样多，按从上到下、每行从左到右排列
- plot_box=[x0,y0,x1,y1] 是该面板**数据绘图区**在整张图中的归一化位置；原点在左上，
  x 向右、y 向下。边界必须对应坐标轴最小/最大位置，不含刻度文字和轴标题
- 所有面板坐标相同时填写一次 shared_axis，各 panel 的 axis 填 null；只有某面板不同才在
  panel.axis 覆盖。不要为每格重复相同刻度，避免输出冗长；看不清的字段填 null，不要猜
- exclude_regions 的 box 使用当前面板绘图区归一化坐标，原点在左下；图例框必须整框排除
- 图例框必须以实际文字/边框为依据，不能圈住数据线或误差带；box 的 y=0在绘图区底部，y=1在顶部。
- series 是该面板需要独立输出的数据/模型系列。实验点和模拟线是两条系列，不能因为同色
  合成一条；同一系列的 marker+连接线才是一条
- y_axis 填 left 或 right；只有一个纵轴时统一填 left
- anchors 使用当前绘图区归一化笛卡尔坐标（左下 0,0，右上 1,1），每条曲线给 3~8 个
  沿走势分布的粗锚点。锚点只指导代码找线，不直接作为数值输出
- anchors 的 x/y 必须都在0~1，不能把轴上的时间/距离数值直接填入。每条series再写
  expected_x_extent=[开始位置,结束位置]（同样归一化）；终点取真实笔画/最后一个marker，禁止延伸到空白。
- 完全重合的独立系列在overlaps列出对方label，并用occluded_ranges列出隐藏的归一化x区间；
  不能仅因曲线靠近就宣称重合，实验marker与模拟实线须分开。
- draw 只能是 markers_only|markers_connected|line_only；role 只能是 data|model
- line_color/marker_color 用实际十六进制颜色；黑色或深灰线必须 dark=true
- dual_y_axis 表示是否有左右两条 Y 轴
- reason 用一句话说明判断依据
注意：实验装置照片、示意图、表格都不是折线图（is_line_chart=false）。"""

PANEL_SYSTEM = "你是科研论文多面板图的版面定位助手，只输出 json。"

PANEL_PROMPT = """上一轮图表计划没有给出有效面板框。这张论文插图已经确认含有
{expected} 个独立坐标面板。请重新逐个定位每个面板的**数据绘图区矩形**，只输出 json：
{{"rows": 3, "cols": 2,
  "panels": [
    {{"label": "150 MPa", "plot_box": [0.08, 0.02, 0.51, 0.29]}},
    {{"label": "120 MPa", "plot_box": [0.56, 0.02, 0.97, 0.29]}}
  ]}}

严格要求：
- panels 必须正好有 {expected} 项，按从上到下、每行从左到右排列
- plot_box=[x0,y0,x1,y1]，坐标相对**整张输入图**归一化到 0~1；原点在左上，
  x 向右、y 向下
- plot_box 是坐标轴包围的数据区域：左/右边界对应 x 轴最小/最大位置，
  上/下边界对应 y 轴最大/最小位置
- 不要把刻度数字、轴标题、图例文字或相邻子图包含进 plot_box
- label 优先填写该面板图例/标题里的工况（如 150 MPa）；没有就填 (a)/(b) 或 null
- 多条曲线、散点和拟合线属于同一个坐标框，不能因此重复报面板
"""

AXIS_SYSTEM = "你是科研论文图表的读数助手，只输出 json。"

AXIS_PROMPT = """读出这张图表坐标轴的刻度范围和刻度值，只输出 json。
示例 json 输出：
{{"x": {{"min": 0, "max": 5, "labels": ["0", "1", "2", "3", "4", "5"], "scale": "linear",
        "title": "Time after start of injection", "unit": "ms"}},
  "y_axes": [{{"side": "left", "min": 0, "max": 70, "labels": ["0", "10", "20", "30", "40", "50", "60", "70"],
              "scale": "linear", "multiplier": 1e-4, "unit": "mm", "title": "Spray tip penetration",
              "multiplier_note": "轴标题左上角写着 ×10^-4"}}],
  "matches_request": true, "match_note": "横轴是时间(ms)、纵轴是贯穿距(mm)，与需求一致",
  "confidence": 0.9, "note": ""}}
字段要求：
- min/max 是坐标框两端对应的数值（不是数据的最小最大值，是坐标轴的范围）
- labels 按顺序列出你能看清的刻度数字文本
- 每个轴增加 ticks=[{{"value": 刻度数值, "position": 0到1位置}}, ...]，读至少3个明确刻度。
  position 相对当前数据坐标框：x从左向右0到1；y从下向上0到1。位置对应刻度短线/数字中心，
  不要默认等分，也不要把最小最大标签强行放到边界。没有标数字的端点不要编造刻度。
  特别注意最高标签可能低于上边框（如100刻度在5/6高度，上边框对应120），min/max应推算边界。
- scale 只能是 linear 或 log
- **multiplier**：这张图有没有数量级/单位换算标注（轴旁边的 "×10^-4"、"×10^3"、
  "1e-3"、"(×10⁻⁴)" 这类，常在轴标题或图的左上/右上角）？有就填成**乘数**
  （例如 ×10^-4 填 0.0001），没有填 1。这条决定最终数据的数值大小，请务必看清
  ——**注意区分是乘在 x 轴还是 y 轴**，分别填到对应的轴里
- unit：轴标题里的单位文字（没有填 null）
- **title**：轴标题文字（例如 "Time after start of injection"、"Spray tip penetration"），
  **不带单位**；看不清填 null
- **y_axes 是数组**：只有一个纵轴时放一个元素；若图中有左右两条纵轴，请放两个元素，
  分别用 "side": "left" / "right" 标明，并各自给出自己的 min/max/labels
- **matches_request**：只有补充信息里给了"用户要的是什么图"时才判断——这张图的横轴/
  纵轴是否**就是**用户要的？是填 true，不是填 false，看不清填 null。
  要严格：用户要"贯穿距随时间变化"，则横轴必须是时间、纵轴必须是贯穿距；
  横轴是曲轴转角、或纵轴是喷雾锥角/动量通量的都算 false
- match_note：一句话说明依据（横轴是…、纵轴是…，所以符合/不符合）
{hint}
只依据图像内容填写；不确定时把 confidence 调低，不要编造数字。"""

NAME_SYSTEM = "你是科研论文图例解析助手，只输出 json。"

NAME_PROMPT = """这张图的图例把每个数据系列标了颜色。我按顺序给出颜色，请给出对应的图例文字，只输出 json。
示例 json 输出：
{{"series": [{{"color": "#336699", "label": "Series A", "style": "solid",
             "y_axis": "left", "is_data": true, "role": "data",
             "reason": "图例里是这个颜色的实测曲线"}}],
 "dark_series": {{"has_dark_line": true, "kind": "model", "desc": "黑色虚线是模型预测"}}}}
字段要求：
- color 原样返回我给你的颜色
- label 是图例里对应的文字；如果这个颜色在图例里找不到，label 填 null
- style 是 solid|dashed|dashdot|marker|band 之一；无法判断填 null
- **y_axis**：这条曲线读的是哪一条纵轴，填 "left" 或 "right"；只有一个纵轴时填 "left"
- **is_data**：这个颜色的线是不是"要提取的实验数据曲线"。
- **role** 只能是 data|model|tangent|fit|annotation|legend|inset|other 之一：
  * data   = 实测/仿真得到的数据曲线（要提取）
  * model  = 模型预测曲线（通常也要，但请在 reason 里说明）
  * tangent/fit = 作者画的切线、拟合线、辅助斜线（形如 "S ∝ t^0.5"、"S ∝ t" 那种），**不是数据**
  * annotation = 标注文字、箭头、"EOI"、"11 MPa" 这类注释
  * inset = 左上/右上那种放大子图里的线
- reason 用一句话说明判断依据（例如"这条线旁边写着 S ∝ t^0.5，是作者的斜率分析线"）
- **dark_series**：图里有没有黑色/深灰色的线？它是数据、模型还是坐标轴辅助线？
  有就填 has_dark_line=true 并说明；没有填 false
颜色列表（按顺序）：{colors}"""

SPOT_SYSTEM = "你是科研论文图表的读数助手，只输出 json。"

SPOT_PROMPT = """这是一张折线图。请**只根据图像本身**（图上印着的刻度数字）回答下面的问题，只输出 json。
示例 json 输出：
{{"checks": [{{"id": 1, "read_x": 3.0, "read_y": 42.0, "note": ""}}]}}
要读的位置：
{queries}
要求：
- 位置用**视觉描述**给出（如"横轴最左端"、"横轴正中间"），请你自己在图上去定位
- read_y：该处曲线在**纵轴**上的数值，按图上刻度读数
- read_x：你定位到的**横轴**数值（按图上刻度读数），不要猜测题目中的数字
- 不要把任何外部信息当作已知条件；这一题没有任何提示答案
- 查询附带的颜色和画法用于确定要读哪条曲线；同一工况可能同时有实验标记和模型线。
  必须核对指定颜色/画法，不得仅凭名称去读另一条线；身份不符或无法确认时填 null。
- 看不清就填 null 并在 note 里说明"""

SELECT_SYSTEM = "你是科研论文的图检索助手，只输出 json。"

PHYSICAL_MATCH_POLICY = """
物理量匹配规则：比较测量对象、方向、位置/时间定义与工况，不能按轴标题逐字匹配。
同一对象的前缘距离、最远到达距离、相应液相/气相长度可能是同一物理量的不同表述；
结合图注、正文和图中对象确认含义。不能仅因 Length 与 Penetration 等措辞不同就否定。
相同单位也不代表同一个量：轴向前缘距离、横向宽度、液滴直径仍需按对象和方向区分。
只有能指出测量对象、方向或定义不同的可见证据时，才把语义近似的标题判为明确矛盾；
无法确认时填 null，并说明需要的上下文。不要把“术语不同”写成物理矛盾。
"""

PANEL_REQUEST_HINT = """
用户只需要：{want}
仍然定位全部面板，但在每个 panels 元素里增加 request_check：
{{"matches_request": true, "axes_match": true, "conditions_match": true,
  "note": "本面板的横纵轴、图例和工况如何符合需求", "evidence": ["图上的原文"]}}
- 对每个面板独立判断，不能把整图图注或相邻面板的结论套给本面板。
- 先核对本面板的横纵轴物理量，再核对用户指定的燃料/压力/温度等条件；没有指定的条件填 true。
- 中文需求与英文标签按物理含义匹配；单位换算等价，但时间和曲轴转角等不同物理量不能互换。
- 需要的横轴和纵轴必须在同一个面板内，不能跨子图组合；双纵轴只要其中一条对应即可。
- 清楚读到不同物理量或不同工况时填 false；确实读不清或缺少条件证据时填 null 并说明缺什么。
- 有轴范围或 shared_axis 并不代表符合需求；request_check 必须逐面板填写。
"""

INSPECT_FIGURE_PROMPT = """逐图语义检索：请读这张原始论文插图，而不是凭图注关键词猜测。
用户需求：{want}
候选记录：
{index}
只输出 json：
{{"is_data_chart": true, "panels": [
  {{"label": "(a)", "x_title": "Time ASI", "x_unit": "ms",
    "y_titles": ["Spray penetration"], "y_units": ["mm"],
    "legend": ["Experiment 800 bar"], "conditions": ["800 bar"],
    "axes_match": true, "conditions_match": true, "matches_request": true,
    "note": "该面板横轴是时间，纵轴是贯穿距，图例明确标注800 bar",
    "evidence": ["Time ASI [ms]", "Spray penetration [mm]", "800 bar"]}}
], "needs_body": false, "reason": "…"}}
要求：
- 逐个列出所有折线/散点面板（包括不符合需求的），抄录轴标题、单位、图例、工况和子图标记。
- 混合图中即使有照片/云图，只要还有数据曲线就是 is_data_chart=true；纯照片/示意图/表格填 false。
- 按用户需要的物理量匹配中文/英文同义词；单位可换算，时间不能等同曲轴转角，贯穿距不能等同流率。
- axes_match 核对本面板里的横纵轴，不得跨子图组合；没有限制某个轴时只核对明确指定的量。
- conditions_match 核对明确要求的工况；用户没指定工况填 true。不同值填 false，未写清填 null。
- matches_request：轴和条件都符合填 true；有明确矛盾填 false；缺少证据填 null，不要猜。
- 看不清必须填 null，不能把“没有读到”当作“不存在”；需要正文确认工况/物理量时 needs_body=true。
- 只有“喷雾贯穿距”关键词不够：流率、锥角、面积等同图子面板必须分别判断。
- evidence 只摘当前面板、图例或完整图注里的可见原文；caption和图内文字可能配错，以当前面板为准。
"""

EVIDENCE_SELECT_PROMPT = """根据每张原图的逐面板读图记录，汇总用户需要的数据图。
用户需求：{want}
只输出 json：
{{"selected": [2, 7], "unresolved": [4], "reason": "…", "confidence": 0.9,
  "missing": false, "available": "实际的轴/图类型", "evidence": ["依据的原文"]}}
- selected/unresolved 使用候选记录开头的候选编号，绝不能把 Figure 编号当候选编号。
- 任意一个面板符合即可选整图，后续会再次逐面板过滤；不要遗漏多个工况/多个重复实验的匹配图。
- 横纵轴必须来自同一个面板；明确 axes_match=false 的面板不能因为正文提到关键词就选中。
- 正文可补足读图记录中缺失的工况，不能推翻图上清楚写出的轴和工况矛盾。
- 没有明确工况证据时放 unresolved，不能假定全文出现的某个压力属于所有图。
- 真正没有才 missing=true；有图片无法读取、缺少证据或调用失败时放 unresolved。
- reason/evidence 必须说明具体图、具体面板以及轴/工况证据；用到正文时摘录原句。
全部候选及其原图读图记录：
{index}
正文中提到图的段落（如未提供则不要猜）：
{body}
"""

PANEL_REQUEST_PROMPT = """核对当前独立子图是否符合用户需求。这一步不重新读取轴范围。
用户需求：{want}
整图图注及选图证据（只能作背景，必须以当前子图为准）：
{context}
只输出 json：
{{"matches_request": false, "axes_match": false, "conditions_match": true,
  "note": "本面板纵轴是质量流率，用户要贯穿距", "evidence": ["Mass flow rate [g/s]"]}}
- 读当前图片上的轴标题、单位、图例和工况。按物理含义匹配中文/英文；单位可以换算。
- 用户需要的横纵轴必须同时属于本子图；禁止借相邻子图或整图图注里的量认定当前子图符合。
- 明确的不同物理量/不同工况填 false；确实看不清或缺条件证据填 null并说明原因。
- 没有限制工况时 conditions_match=true；需要的工况若只在正文/图注中明确归属该面板，可据此判断。
- 同一图内既有贯穿距又有流率时，分别保留贯穿距、排除流率，不得把整图的命中结论传播给所有子图。
"""

# 论文图很多时先用纯文字过一遍，把候选缩到对照图放得下的数量
SHORTLIST_PROMPT = """用户要从一篇论文里找出他需要的数据图。下面是这篇论文的图清单。
用户需求：{want}
只输出 json：
{{"shortlist": [2, 5, 7], "reason": "…", "missing": false, "available": "…"}}
字段要求：
- shortlist：可能是目标图的编号（整数数组，按可能性从高到低，最多 12 个）
- 可能沾边的都要留下：这一步只看图注文字，宁可多留，不要漏
- 一个都不沾边就填空数组、missing=true，并在 available 里一句话说明这篇里实际有哪些数据图
图清单：
{index}"""

SELECT_PROMPT = """用户要从这篇论文里找出他需要的数据图。我给了两份材料：
（1）这篇论文的图清单；（2）一张**对照图**：把所有候选图的缩略图按编号排在一起，每张缩略图左上角的红字就是它的编号。
用户需求：{want}
只输出 json：
{{"selected": [2, 7], "reason": "…", "confidence": 0.8, "missing": false, "available": "…"}}
字段要求：
- selected：符合需求的编号（整数数组，按符合程度从高到低）
- confidence：你对自己选择的把握（0~1）。图注没写清、只能靠缩略图猜时给低分
- 判断依据可以是图注文字，也可以是你从缩略图里看到的坐标轴标题、图例、工况标注
- 只选**能取数据**的图（折线图、散点图）；照片、装置照片、示意图、二维云图、表格一律不选
- 用户提到了条件（某种燃料、压力、工况等），条件不符的**不要**选
- **这篇论文里可能根本没有**用户要的图：这时 selected 填空数组、missing=true，并在 available 里
  一句话说明这篇里实际有哪些数据图（列出图号+画的是什么）
- 不要为了给出答案而勉强选一张最像的——选错比说"没有"更糟
图清单：
{index}"""

# 第三步：正文里"Figure N shows …"这类句子，常常是唯一写清"哪张图对应哪个工况"的地方
BODY_SELECT_PROMPT = """用户要从这篇论文里找出他需要的数据图。我给了两份材料：
（1）这篇论文的图清单（编号 + 图注 + 图内文字）；
（2）论文正文里**所有提到图的段落**（只摘这些，不是全文）。
用户需求：{want}
只输出 json：
{{"selected": [2, 7], "reason": "…", "confidence": 0.8, "missing": false, "available": "…"}}
字段要求：
- selected：符合需求的编号（整数数组，按符合程度从高到低）
- 正文里常写 "Figure 5 presents the penetration at Pi = 800 bar" 这类句子：**优先用正文明确定位
  "哪个条件属于哪张图"**，图注和缩略图可能都没写
- reason 里**引用你依据的正文原句片段**（方便人工复核），并说明是哪张图
- 只选**能取数据**的图（折线图、散点图）；正文提到的照片、示意图、云图、表格都不算
- 条件不符的不要选；**真的没有**就填空数组、missing=true，并在 available 里说明这篇实际有哪些数据图
- 不要为了给出答案而勉强选：选错比说"没有"更糟
图清单：
{index}
正文中提到图的段落：
{body}"""


def classify_chart(vlm, image_path, want=None, caption=None):
    # A multi-panel executable plan is larger than the old one-line classifier.
    # Give it enough room and retry transient/empty replies; truncated JSON must not
    # reach the pipeline as a half-valid plan.
    prompt = CLASSIFY_PROMPT
    if caption:
        prompt += "\n完整图注（可能包含各子图的含义）：" + caption
    if want:
        prompt += PANEL_REQUEST_HINT.format(want=want)+PHYSICAL_MATCH_POLICY
    data, raw = vlm.ask_json(image_path, prompt, system=CLASSIFY_SYSTEM,
                             max_tokens=4096, retries=2)
    data["_raw"] = raw
    return data


def locate_panels(vlm, image_path, expected):
    """Ask for plot rectangles only when pixel geometry disagrees with panel count."""
    data, raw = vlm.ask_json(
        image_path, PANEL_PROMPT.format(expected=int(expected)),
        system=PANEL_SYSTEM, retries=2)
    data["_raw"] = raw
    return data


def read_axis_ranges(vlm, image_path, hint=None, figure_path=None, panel_label=None, frame=None):
    h = f"补充信息：{hint}" if hint else \
        "补充信息：没有（不用判断 matches_request，填 null）"
    prompt = AXIS_PROMPT.format(hint=h)
    if hint:
        prompt += PHYSICAL_MATCH_POLICY
    if frame:
        prompt += f"\n第一张图的精确绘图区像素框为 left,top,right,bottom={list(frame)}；请用此框计算 ticks.position。"
    if figure_path:
        image_path = [image_path, figure_path]
        prompt += ("\n第一张图是需要标定的当前子图 " + str(panel_label or "") +
                   "，第二张图是完整原图，只用来查共用的刻度、轴标题和单位。"
                   "只返回第一张图这一坐标框的轴。第一张图没有横轴数字时，在第二张图读明确共用的横轴，"
                   "不得猜测；不允许把相邻子图的纵轴、插图小坐标轴、图例内数值当成当前子图的轴。")
    for attempt in range(2):
        data, raw = vlm.ask_json(image_path, prompt, system=AXIS_SYSTEM, max_tokens=2048)
        if isinstance(data, dict) and isinstance(data.get("x"), dict) \
                and isinstance(data.get("y_axes"), list):
            sides = [a.get("side", "left") for a in data["y_axes"] if isinstance(a, dict)]
            if len(sides) == len(set(sides)):
                data["_raw"] = raw
                return data
        prompt += "\n上次输出不符合读轴结构。必须返回当前子图的 x 对象和 y_axes 数组；每侧最多一个轴，不能输出 response_format/type 对象。"
    raise ValueError("子图轴读数缺少有效 x/y_axes，或同一侧混入了多个子图的轴")


def name_series(vlm, image_path, colors):
    data, raw = vlm.ask_json(image_path, NAME_PROMPT.format(colors=", ".join(colors)),
                             system=NAME_SYSTEM)
    data["_raw"] = raw
    return data


def read_legend_contract(vlm, image_path, figure_path, panel_index):
    """Read actual swatch geometry in a full-resolution legend crop once."""
    prompt = ("第一张为图例放大裁图，第二张为完整论文插图。图例来自第"
              +str(panel_index)+"个子图（从上到下、每行从左到右）。只输出 json：\n"
              '{"shared":true,"entries":[{"label":"图例原文","color":"#rrggbb",'
              '"draw":"line_only","marker_shape":null,"marker_fill":null}],"evidence":"样本笔画依据"}\n'
              "逐项看第一张图例样本：实线/虚线无标记填line_only；独立标记填markers_only；"
              "标记连线填markers_connected。marker_shape用circle/square/triangle/diamond/star，"
              "marker_fill用hollow/filled或null。不预设EXP、ML、Experiment、Simulation的画法。"
              "shared只在完整图明确为其它子图共用该图例且没有相互冲突的图例时填true。"
              "只读图例，不读取数据区，不给锚点。看不清的字段填null。")
    data,raw = vlm.ask_json([image_path,figure_path],prompt,system=NAME_SYSTEM,max_tokens=1500)
    return data if isinstance(data,dict) and isinstance(data.get('entries'),list) else {}


def spot_check(vlm, image_path, queries, figure_path=None):
    """Independent re-reading: the prompt carries NO axis range on purpose.

    Handing the model our own axis values would let it convert positions with our
    (possibly wrong) calibration, so its answer would agree with our data even when the
    calibration is off. By asking only "read the value at x on this chart" the answer
    comes from the chart's own printed ticks and is therefore a true cross-check.
    """
    lines = "\n".join(
        f"- id={q['id']}: 在{q.get('where') or ('横轴坐标约为 x=%g 处' % q['x'])}，"
        f"{q.get('series') or '图中对应曲线'} 的纵轴数值是多少？"
        + (f"（颜色={q.get('color')}，画法={q.get('draw')}）" if q.get('color') else "")
        + (f"（请按{'左' if q['axis'] == 'left' else '右'}侧纵轴的刻度读数）"
           if q.get("axis") else "")
        for q in queries)
    prompt = SPOT_PROMPT.format(queries=lines)
    if figure_path:
        image_path = [image_path,figure_path]
        prompt += ('\n第一张图是当前子图，所有数值只读第一张的刻度。第二张是完整原图，'
                   '用来核对共用图例里的系列名称、颜色与画法；不要从相邻子图读数。')
    data, raw = vlm.ask_json(image_path, prompt, system=SPOT_SYSTEM)
    data["_raw"] = raw
    return data


def shortlist_figures(vlm, want, index_text):
    """Text-only pass: narrow a long figure list down to what fits on a contact sheet."""
    data, raw = vlm.ask_json(
        None, SHORTLIST_PROMPT.format(want=want, index=index_text),
        system=SELECT_SYSTEM, retries=2)
    data["_raw"] = raw
    return data


def select_figures(vlm, want, index_text, image_path=None):
    """Pick the figures a natural-language request asks for.

    The contact sheet carries the part a caption cannot: what the axes actually plot.
    Users describe the chart they want ("velocity versus time at 800 bar"), which is
    usually written inside the figure, not in its caption.
    """
    prompt = SELECT_PROMPT.format(want=want, index=index_text)
    if image_path is None:                     # 没有缩略图可看时退化成纯文字
        prompt = prompt.replace("（2）一张**对照图**：把所有候选图的缩略图按编号排在一起，"
                                "每张缩略图左上角的红字就是它的编号。", "（2）没有缩略图，只能依据图注文字判断。")
    data, raw = vlm.ask_json(image_path, prompt, system=SELECT_SYSTEM, retries=2)
    data["_raw"] = raw
    return data


def select_figures_by_body(vlm, want, index_text, body_text):
    """Second opinion from the body text: only the paragraphs that mention a figure.

    A caption says what a figure *is*; the body says what it *shows* and under which
    conditions. That is the part a caption-only selection cannot resolve, and the
    relevant paragraphs are only a few percent of the paper.
    """
    data, raw = vlm.ask_json(
        None, BODY_SELECT_PROMPT.format(want=want, index=index_text, body=body_text),
        system=SELECT_SYSTEM, retries=2)
    data["_raw"] = raw
    return data


def inspect_figure(vlm, want, index, image_path):
    data, raw = vlm.ask_json(
        image_path, INSPECT_FIGURE_PROMPT.format(want=want, index=index)+PHYSICAL_MATCH_POLICY,
        system=SELECT_SYSTEM, max_tokens=3072, retries=2)
    if not isinstance(data, dict):
        raise ValueError("逐图语义读图返回的不是 JSON 对象")
    data["_raw"] = raw
    return data


def select_figures_by_evidence(vlm, want, index, body=""):
    data, raw = vlm.ask_json(
        None, EVIDENCE_SELECT_PROMPT.format(want=want, index=index, body=body or "（未提供）")+PHYSICAL_MATCH_POLICY,
        system=SELECT_SYSTEM, max_tokens=3072, retries=2)
    if not isinstance(data, dict) or not isinstance(data.get("selected"), list):
        raise ValueError("选图汇总缺少 selected 数组")
    data["_raw"] = raw
    return data


def check_panel_request(vlm, image_path, want, context=""):
    data, raw = vlm.ask_json(
        image_path, PANEL_REQUEST_PROMPT.format(want=want, context=context)+PHYSICAL_MATCH_POLICY,
        system=SELECT_SYSTEM, max_tokens=1024, retries=2)
    if not isinstance(data, dict):
        raise ValueError("子图需求核对返回的不是 JSON 对象")
    data["_raw"] = raw
    return data


FIND_SERIES_SYSTEM = "你是科研论文图表的读图助手，只输出 json。"

# 让模型**自己找出**数据曲线，而不是评价代码给出的候选清单。这样它的输出是一份
# "要点提取的曲线"清单（正向），不会出现"数据被判成非数据而消失"的情况；漏了哪条
# 也只是报告里对账得到，不会静默丢。
FIND_SERIES_PROMPT = """这是一张论文插图。请找出图里**所有需要提取数据的曲线**，并给出每条曲线的大致位置。
图例里读到的条目（供参考，可能不全）：{legend}
只输出 json：
{{"chart_family": "line", "series_count": 2,
 "exclude_regions": [{{"role": "legend", "box": [0.68, 0.03, 0.98, 0.25]}}],
 "series": [
  {{"label": "Series A", "y_axis": "left", "color": "#3366CC", "dark": false,
    "role": "data", "line_color": "#3366CC", "marker_color": "#3366CC",
    "marker_shape": "circle", "linestyle": "dashed", "draw": "markers_connected", "closed": false,
    "overlaps": ["Series B"], "occluded": [[0.62, 0.72], [0.78, 0.60]],
    "expected_x_extent": [0.06, 0.97], "occluded_ranges": [[0.06, 0.30]],
    "anchors": [[0.06, 0.02], [0.5, 0.55], [0.95, 0.78]],
    "note": "两条曲线在前半段几乎重合，重合段压在下面"}}],
 "ignore": [{{"what": "蓝色实线，旁边写着 S ∝ t^0.5", "why": "作者的斜率参考线，不是数据"}}]}}
字段要求：
- **series_count**：需要输出成独立数据文件的物理曲线总数，必须与 series 数组长度一致。
  同颜色的五条实验曲线就是 5，不得因为颜色相同或起点重合写成 1；同一条曲线上的
  标记和连线仍然只算 1
- chart_family：line|scatter|closed_loop|mixed，用来约束代码选择追踪范式
- **exclude_regions**：绘图区里绝不能参与追踪的区域，目前只报 legend 或 inset。
  box=[x0,y0,x1,y1] 使用与 anchors 相同的绘图区归一化笛卡尔坐标（左下 0,0，右上 1,1）。
  图例框必须整框给出，框里的样本线、marker、文字都不是数据；没有就填 []
- label：这条曲线对应图例里的哪个条目；图例里没有就填 null 并在 note 说明
- y_axis：读的是左轴还是右轴（只有一个纵轴时填 left）
- **role**：data|model|event|control|reference。只有 data/model 是连续数值曲线；event 是
  单个或少量事件点；control 是脉冲门控/触发窗口/工况方波；reference 是斜率/阈值辅助线。
  control/reference 不应出现在 series，放入 ignore
- **line_color**：连接数据的线条实际颜色，十六进制；没有连接线填 null
- **marker_color**：marker 实际颜色；没有 marker 填 null
- **marker_shape**：circle|square|triangle|diamond|cross|none
- **marker_fill**：hollow（空心轮廓）|filled（实心）|null。依据当前子图标记本身填写，用来识别被连线/其它颜色切开的标记。
- color：兼容字段，优先填 line_color；没有连接线时填 marker_color
- **dark**：数据连接线是不是黑色/深灰色。黑白论文图里必须填 true——代码会改用"深色
  笔画 + 你给的锚点"来追，不再依赖颜色
- linestyle：solid|dashed|dashdot|marker|band
- **draw**：这张图里**这条曲线**是怎么画出来的，四选一：
  * markers_only        只有散点，没有连线
  * markers_connected   散点用折线连起来（点即曲线；**不要**报成"点+线"两条）
  * line_only           只有线，没有标记
  * markers_plus_line   散点是实验值，另有一条**独立的**拟合/模型曲线（这才是两条）
  特别注意：有些图用**黑色连接线 + 彩色/不同形状 marker**区分多个系列。这时它仍是一条
  markers_connected 曲线：line_color="#000000"、dark=true，marker_color/marker_shape 写身份；
  绝不能只把零散彩色 marker 当完整曲线，也不能把所有黑线合成一条
- **closed**：这条曲线是不是首尾相接的闭合回线（P-V 图边界、喷雾包络、等值线那种
  围成一圈的）。普通从左到右的曲线填 false。**这条决定代码用哪种追踪方式**，填错会让
  曲线被追成一圈来回走的假轨迹
- **overlaps**：这条曲线与哪几条曲线有重合（颜色互相盖住）？填那些曲线的 label，
  没有就填 []。重合处只显示上面那条的颜色是**正常的**——不要因此漏掉被压住的那条，
  也不要把同一根线报两遍
- **occluded**：这条曲线**自己颜色看不见**的段落，每段给一个点 [x, y]（绘图区归一化），
  例如被另一条曲线压住、被图例框或文字盖住的位置。代码会在这些地方沿另一条曲线补全
- **expected_x_extent**：这条曲线按原图应覆盖的最左/最右 x 位置 [x0,x1]（相对**坐标轴
  围成的绘图区**归一化：左边框=0、右边框=1，不包括刻度文字和图例边距）。这是校验
  “只追到半条”的关键，不要按当前可见颜色的起止位置缩短
- **occluded_ranges**：自己颜色看不见的 x 区间 [[x0,x1], ...]；不知道精确边界可略放宽。
  与 expected_x_extent 一起供修复器决定哪里需要局部问图，不直接产生数值
- **anchors**：直线给 3 个可见点；弯曲、交叉或遮挡曲线给 5–9 个可见点，按 [x, y] 给。x、y 都是相对
  **坐标轴绘图区**的归一化笛卡尔坐标：**左下角是 (0,0)、右上角是 (1,1)**；
  不是整张图片，也不是横纵轴的真实数值。顺序为
  "从左数第一个可见点、前段转弯位置、中段、交叉附近、最右端点"。位置可有约 ±3% 误差，
  代码会以它们为种子去精确追踪。曲线被遮挡或断裂时，挑你确实看得见的位置；
  不要套用通用的对角线锚点，也不要用真实轴数值替代归一化位置。
- note：一句话说明（例如"与另一条曲线重叠"、"前半段被图例遮住"）
另外用 ignore 列出你**没有**当作数据的东西（作者画的斜率参考线、拟合辅助线、示意图、
标注文字等），各写一句理由。
注意：
1. 坐标轴、网格、图例框、纯标注文字都不是数据曲线。
   图例里即便有条目，若它表示事件门控/脉冲时序/触发窗口/阈值框等工况示意，而不是
   横纵轴所定义物理量的一组测量或模型数值，也放进 ignore，不输出 CSV。
2. 一条独立数据轨迹输出一份数据：同一根曲线的标记和连接线用 draw=markers_connected，
   不要拆成两条，也不要把遮挡前后拆开。图例条目与轨迹不一定一一对应：同一颜色/工况
   可同时表示两种物理量、两种模型或不同线型。必须逐条核对图内所有分开的轨迹，
   不能用图例条目数量限制 series_count；若一个条目对应两条轨迹，分别给独立锚点，
   用图上有依据的物理量/线型区分 label，无法确定身份时明确标记不确定，不得任选一条。
3. 颜色相同、靠得很近的两条线（例如同色虚线）请**分别**报出来，并在 note 里说明
   它们靠什么区分（上下位置 / 线型 / 哪一段分开）。
4. 输出前重新数一遍图例条目和图内独立轨迹；series_count 与 series 长度不一致时先修正。"""


def find_series(vlm, image_path, legend_text, context=None, figure_path=None, panel_label=None, frame=None):
    """Ask the model where the data curves are (coarse seeds), not what our blobs are."""
    prompt = FIND_SERIES_PROMPT.format(legend=legend_text or "（没读到图例）")
    if context:
        prompt += "\n上下文（只补充身份，不提供当前图的坐标；锚点、端点、图例框必须重新读当前子图）：\n" + context
    if frame is not None and len(frame) == 4:
        left,top,right,bottom = [int(v) for v in frame]
        prompt += (f"\n第一张图的已校准绘图区像素边界为 left={left}, top={top}, right={right}, bottom={bottom}。"
                   "像素原点在图像左上。anchors 仍用绘图区笛卡尔归一化："
                   "x=(px-left)/(right-left)，y=(bottom-py)/(bottom-top)。"
                   "轴标签和图例留白不算绘图区。按实际可见笔画的位置填写。")
    if figure_path:
        image_path = [image_path, figure_path]
        prompt += ("\n第一张图是当前子图 " + str(panel_label or "") +
                   "，第二张图是完整原图，供读取共用图例。所有 anchors/expected_x_extent/"
                   "exclude_regions 只相对第一张图的坐标框归一化。第二张图中的其它子图"
                   "不得加入 series。逐项读共用图例的样本笔画，核对 label→role→draw→"
                   "marker_shape/marker_fill；同色的实验标记与模型线不要互换。"
                   "只给当前图真正可见的数据位置，不能把其它子图的位置搬过来。")
    data, raw = vlm.ask_json(image_path, prompt, system=FIND_SERIES_SYSTEM, retries=2,
                             max_tokens=4000)
    data["_raw"] = raw
    return data


REVISE_SYSTEM = "你是科研论文图表的校对助手，只输出 json。"

# 复盘：模型像素上不准（所以不让它画线），但"这两条是不是同一条""这条追到图例上了"
# 这类判断正是它的强项。把代码追出来的结果编号画回原图给它看，让它当裁判——
# 算法在模型指导下收尾，只在有问题的面板上多问一次。
REVISE_PROMPT = """图上是代码从这张图里追出来的曲线：彩色描边 + 编号 #1 #2 ...，
红色 ✗ 是代码没追到的。请核对它们对不对，只输出 json：
{{"merge": [[1, 3]], "drop": [4], "add": [], "extend": [], "ok": false, "note": "一句话"}}
字段要求：
- merge：哪两个编号其实是**同一条曲线**被重复提取了（配对给出，保留编号小的那个）。
  颜色相同、位置重合、明显是同一根线的都算。没有填 []
- drop：哪个编号追错了（不是数据 / 只是碎段 / 追到别的曲线或图例上 / 追到了坐标轴上 /
  **把背景网格线或坐标框当成了曲线**——那种轨迹横跨整幅、几乎没有起伏）。没有填 []
- add：图上有数据曲线、而代码**一条都没追上**的（注意看图例里列了、但编号里没有的）。
  每条给：{{"label": 图例名, "color": "#rrggbb", "dark": false, "closed": false,
  "draw": "line_only", "anchors": [[x, y], [x, y], [x, y]], "why": "哪条"}}
  锚点必须是你能看见的、落在那条线上的点，使用绘图区归一化笛卡尔坐标
  （左下 0,0，右上 1,1）。没有填 []
- **extend**：已有编号对应的曲线只追到半条、起点太晚、终点太早或中段断裂时给修复指令：
  {{"id": 2, "coord_space": "data", "anchors": [[x,y],...], "why": "前半段被#1压住"}}。
  anchors 必须使用右侧重绘图坐标轴上的**真实数据坐标**（不是 0~1 归一化位置），给 4~8 个
  按 x 递增的锚点，覆盖缺失区并接到两侧已追到的部分。extend 只能补“没有轨迹”的范围，
  不能用来改写已有编号已经覆盖的区间（已有区间形状看起来可疑时在 note 说明即可）；
  完整曲线不要写。没有填 []
- issues：已有轨迹的形状、身份、标记数量或完整性仍有问题，但无法给出可靠修复指令时，
  给 {{"id": 编号, "why": "问题及原图证据"}}；没有填 []。
- ok：没有上述操作，也没有 issues 时才填 true；系列数量齐全不代表标记识别完整。
- **axis_fix**：只在**坐标轴映射明显不对**时才给（代码用的轴范围见下面的"轴的现状"）。
  典型的错法：曲线数值整体偏移、被压成一条平线、超出轴范围、或者漏了 ×10^-4 这种
  数量级标注。给的话写成
  {{"x": [min, max], "y": [min, max], "multiplier": 0.0001}}
  （multiplier 是数据要乘的倍数；分不清就只给 x/y）。没问题就不要这个字段
- note：一句话总结（例如"#2 与 #4 是同一条，另外漏了图例里的 Ethanol"）
判断只看图上画的描边和原图，注意：
* **重合处只显示上面那条的颜色是正常的**，不要因为下面那条看不见就说它多余；
* 散点用折线连起来的那种图是**一条**曲线，不是"点加线"两条；
* 先逐项核对“模型清单里的曲线数”和编号数，再核对每条的最左端、最右端和中间连续性；
* 一条编号轨迹如果突然回到横轴起点再重新上升，通常是多条同色曲线被错误拼接，不能判 ok；
* 拿不准的不要动（宁可 ok=true）。
模型最初读到的曲线清单（供参考）：{spec}
代码实际追出来的：
{found}
轴的现状（代码正在用的映射）：{axis}"""


def revise_traces(vlm, image_path, spec_text, found_text, axis_text="", original_path=None):
    """One review round: the model judges the traces the code produced."""
    prompt = REVISE_PROMPT.format(spec=spec_text or "（无）", found=found_text or "（无）",
                                  axis=axis_text or "（未记录）")
    if original_path:
        image_path = [original_path,image_path]
        prompt += ("\n第一张为未覆盖的原图，第二张为编号测量叠加图。独立散点只叠加已测标记中心，"
                   "不连接；请逐段对比原图的标记数量和位置，部分识别不能因系列数量齐全就判 ok。"
                   "无法确认的写入 note；不要把模型锚点当作已测数据。")
    data, raw = vlm.ask_json(image_path, prompt, system=REVISE_SYSTEM, retries=2,
                             max_tokens=900)
    data["_raw"] = raw
    return data


def compare_spot_check(checks, expected, tol_abs, tol_by_id=None):
    """Attach our own traced value and an agreement verdict to each VLM reading.

    `tol_abs` is an absolute tolerance, chosen by the caller from the chart's own tick
    spacing (half a tick is the best any reader - human or model - can be expected to
    achieve). A fixed percentage of the axis range is the wrong yardstick: 5% of a
    0-70 axis is 3.5 units, while 5% of a 0-2.5 axis is 0.125, which no eye can hit.
    """
    out = []
    for c in checks.get("checks", []):
        cid = c.get("id")
        ours = expected.get(cid)
        read = c.get("read_y")
        if ours is not None and isinstance(read, (int, float)):
            diff = abs(float(read) - float(ours))
            # a series reading a different y axis needs that axis' tolerance; the value
            # travels separately because the model's reply only echoes id/read_x/read_y
            tol = (tol_by_id or {}).get(cid) or c.get("tol") or tol_abs
            c["our_y"] = ours
            c["abs_diff"] = round(diff, 6)
            c["tol"] = tol
            c["agree"] = bool(diff <= tol)
        out.append(c)
    n = sum(1 for c in out if c.get("agree") is not None)
    ok = sum(1 for c in out if c.get("agree"))
    return {"checks": out, "n_compared": n, "n_agree": ok,
            "agreement": (ok / n) if n else None, "tol_abs": tol_abs}


def axis_tolerance(axis_range, tick_step, min_frac=0.02, tick_frac=0.6):
    """Absolute tolerance for a spot check: a fraction of a tick, never below 2% of range."""
    span = abs(axis_range[1] - axis_range[0]) if axis_range else 0.0
    tol = min_frac * span
    if tick_step and tick_step > 0:
        tol = max(tol, tick_frac * abs(tick_step))
    return round(tol, 6)


GAP_SYSTEM = "你是科研论文图表的读图助手，只输出 json。"

# 曲线被压住的那一段：颜色追踪到这里就断了（重合处只显示上面那条的颜色）。把这一小块
# 裁剪图交给模型，让它沿着两侧可见的走向说出这段的锚点——读图用它的能力，像素归代码。
GAP_PROMPT = """这张裁剪图来自一篇论文的曲线图。曲线「{label}」在{where}有一段没被识别出来，
可能被别的曲线压住、被图例/文字/阴影盖住。请沿着这条曲线的走向给出这段里的 4~8 个锚点：
{{"anchors": [[x, y], [x, y]], "note": "一句话"}}
要求：
- x、y 是**相对这张裁剪图**的归一化坐标（左上角 0,0，右下角 1,1）
- 锚点必须落在这条曲线该段的走向上（用两侧可见的部分推它的走势）
- 按 x 从左到右排列；弯曲处多给点，近似直线处少给点，并至少有一个点贴近已知端点
- 图里已识别的部分画成{color}线，灰色是别的曲线（用它判断这条线是不是贴着它走）
- 看不出走势就填空数组，**不要编**
可见端点/已知信息：{info}"""


def gap_anchors(vlm, crop_path, label, color, info=""):
    """让模型给"被盖住那一段"的锚点 -> {"anchors": [[x,y],...], "note": ...}。"""
    prompt = GAP_PROMPT.format(label=label or "该曲线", color=color or "彩色",
                               where="图中间", info=info or "（无）")
    data, raw = vlm.ask_json(crop_path, prompt, system=GAP_SYSTEM, retries=1,
                             max_tokens=700)
    data["_raw"] = raw
    return data
