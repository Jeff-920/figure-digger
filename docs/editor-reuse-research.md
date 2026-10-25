# 核对图编辑工具：复用调研

在实现移动、擦除、补充前，先查看了可复用的开源实现。结论是没有能直接嵌入当前 SVG 图像叠加层的现成编辑器，因此保留自有交互层，只借鉴交互和数据结构思路，没有复制第三方代码，也没有增加前端依赖。

| 项目 | 许可 | 能力 | 结论 |
| --- | --- | --- | --- |
| [WebPlotDigitizer](https://github.com/automeris-io/WebPlotDigitizer) | AGPL-3.0 | 图像上的手动点编辑、增删点、拖点 | 交互参考；AGPL 与当前 MIT 项目不兼容，不能复制代码；其手动模式也没有本需求里的“断线/孤立点清理/左右锚点接回”语义 |
| [chartjs-plugin-dragdata](https://github.com/artus9033/chartjs-plugin-dragdata) | MIT | Chart.js 折线图数据点拖动 | 只解决图表坐标系的拖点，不处理原图叠加、折线断开和补充连接；未引入 |
| [Paper.js](https://github.com/paperjs/paper.js) | MIT | 矢量路径、路径分割、擦除类操作 | 可做路径几何，但需要替换当前 SVG 核对层，且断线/接回仍需自定义；未引入 |
| [Paper.js Vector Erase gist](https://gist.github.com/beardicus/8cbe9511d43e3fb58d76e336f76f4eb2) | MIT | 矢量擦除示例 | 借鉴“擦除后拆分路径”的思路，未复制代码 |

实现上沿用现有轻量 SVG：

- 每条系列由若干有序 `segments` 组成，初始为一条完整折线。
- 擦除命中点后，按命中位置把 segment 拆成多个片段；不足两个点的片段删除。
- 拖动只改点坐标，segment 顺序不变，折线自然跟随。
- 补充先放置新点，再选左右两个锚点：同段相邻点直接插入；不同段的末端/起点则合并接回。
- 所有编辑都在浏览器内存中完成，可撤销；保存时写回各自的 CSV，断线用空行保留；新建曲线会追加 CSV 并登记到 `extract_results.json`。

这样可以避免为一个局部功能引入 Chart.js、Paper.js 或图编辑框架，同时保持现有的面板坐标映射和采样预览。
