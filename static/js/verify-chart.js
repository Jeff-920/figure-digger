(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const dataNode = byId("chartData");
  const svg = byId("chartOverlay");
  const stage = byId("chartStage");
  const slider = byId("chartResolution");
  const sliderValue = byId("chartResolutionValue");
  const selection = byId("chartSelection");
  const resetStart = byId("chartReset");
  const hint = byId("chartHint");
  const editStatus = byId("chartEditStatus");
  const undoButton = byId("chartUndo");
  const resetEdits = byId("chartResetEdits");
  const exportButton = byId("chartExport");
  const saveButton = byId("chartSave");
  const newCurveButton = byId("chartNewCurve");
  const finishCurveButton = byId("chartFinishCurve");
  const legendElement = byId("chartLegend");
  const plotRoot = byId("cleanPlot");
  const plotXInput = byId("plotXLabel");
  const plotYInput = byId("plotYLabel");
  const plotXFactor = byId("plotXFactor");
  const plotXOffset = byId("plotXOffset");
  const plotYFactor = byId("plotYFactor");
  const plotYOffset = byId("plotYOffset");
  const plotConfirm = byId("plotConfirm");
  const plotStatus = byId("plotStatus");
  const loupe = byId("chartLoupe");
  const loupeCanvas = byId("chartLoupeCanvas");
  const LOUPE_SIZE = 164;
  const LOUPE_ZOOM = 3;
  let loupeCtx = null;
  let loupeDpr = 1;
  let lastPointer = null;
  if (loupeCanvas) {
    loupeDpr = window.devicePixelRatio || 1;
    loupeCanvas.width = LOUPE_SIZE * loupeDpr;
    loupeCanvas.height = LOUPE_SIZE * loupeDpr;
    loupeCanvas.style.width = `${LOUPE_SIZE}px`;
    loupeCanvas.style.height = `${LOUPE_SIZE}px`;
    loupeCtx = loupeCanvas.getContext("2d");
  }
  if (!dataNode || !svg || !stage || !slider || !sliderValue || !selection
      || !resetStart || !hint || !editStatus || !undoButton || !resetEdits
      || !exportButton || !saveButton || !newCurveButton || !finishCurveButton
      || !legendElement) return;

  let data;
  try {
    data = JSON.parse(dataNode.textContent);
  } catch (error) {
    hint.textContent = "逐点核对数据读取失败，请查看静态核对图。";
    return;
  }
  if (!data || !Array.isArray(data.series) || !data.series.length) return;

  const NS = "http://www.w3.org/2000/svg";
  const [left, top, right, bottom] = data.frame.map(Number);
  let [xMin, xMax] = data.x_range.map(Number);
  const width = Number(data.width);
  const height = Number(data.height);
  const frameWidth = right - left;
  const frameHeight = bottom - top;
  const runId = stage.dataset.runId || "";
  const paperScope = String(data.paper_scope || "");
  const panelId = String(data.panel_id || "");
  const colors = ["#1d4ed8", "#dc2626", "#047857", "#b45309", "#7c3aed", "#0f766e"];
  let nextPointId = 1;
  let newCurveCount = 0;

  const makePoint = (x, y) => ({ id: nextPointId++, x: Number(x), y: Number(y) });
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const clonePoint = (point) => ({ id: point.id, x: point.x, y: point.y });

  const initialSeries = data.series.map((series, index) => {
    const source = (series.segments || []).map((segment) =>
      segment.map(([x, y]) => makePoint(x, y)));
    return {
      label: String(series.label || `系列 ${index + 1}`),
      color: series.color || colors[index % colors.length],
      y_range: (series.y_range || data.y_range).map(Number),
      file: series.file || null,
      sourceSegments: source,
      segments: clone(source),
      sampled: false,
      edited: false,
      modified: false,
      isNew: false,
    };
  }).filter((series) => series.segments.some((segment) => segment.length > 1));

  let seriesList = clone(initialSeries);
  let baselineSeries = clone(initialSeries);
  let activeSeries = -1;
  let start = null;
  let tool = "start";
  let pendingAdd = null;
  let drag = null;
  let erasing = null;
  let brush = null;
  let brushCircle = null;
  let undoStack = [];
  let dirty = false;
  let axisDirty = false;
  let saving = false;
  let plotStale = true;

  const mapX = (x) => left + ((x - xMin) / (xMax - xMin)) * frameWidth;
  const mapY = (series, y) => bottom - ((y - series.y_range[0]) / (series.y_range[1] - series.y_range[0])) * frameHeight;
  const toDataX = (px) => xMin + ((px - left) / frameWidth) * (xMax - xMin);
  const toDataY = (series, py) => series.y_range[0] + ((bottom - py) / frameHeight) * (series.y_range[1] - series.y_range[0]);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const pointPx = (series, point) => [mapX(point.x), mapY(series, point.y)];
  const countPoints = (series) => series.segments.reduce((total, segment) => total + segment.length, 0);
  const sameSegments = (a, b) => JSON.stringify(a) === JSON.stringify(b);

  const screenToImage = (pixels) => {
    const rect = svg.getBoundingClientRect();
    return rect.width > 0 ? pixels * (width / rect.width) : pixels;
  };
  const hitRadius = (pixels = 14) => Math.max(1, screenToImage(pixels));

  const clientToLocal = (clientX, clientY) => {
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const matrix = svg.getScreenCTM();
    if (matrix) {
      const local = point.matrixTransform(matrix.inverse());
      return [local.x, local.y];
    }
    const rect = svg.getBoundingClientRect();
    return [
      ((clientX - rect.left) / Math.max(1, rect.width)) * width,
      ((clientY - rect.top) / Math.max(1, rect.height)) * height,
    ];
  };

  const setStatus = (message, tone = "") => {
    editStatus.textContent = message;
    editStatus.dataset.tone = tone;
  };

  const findPoint = (segments, pointId) => {
    for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex += 1) {
      const segment = segments[segmentIndex];
      for (let pointIndex = 0; pointIndex < segment.length; pointIndex += 1) {
        if (segment[pointIndex].id === pointId) {
          return { segmentIndex, segment, pointIndex, point: segment[pointIndex] };
        }
      }
    }
    return null;
  };

  const nearestPoint = ([localX, localY], maxDistance = Infinity, onlySeries = null) => {
    let best = null;
    let bestDistance = maxDistance * maxDistance;
    seriesList.forEach((series, seriesIndex) => {
      if (onlySeries !== null && seriesIndex !== onlySeries) return;
      series.segments.forEach((segment, segmentIndex) => {
        segment.forEach((point, pointIndex) => {
          const [px, py] = pointPx(series, point);
          const dx = px - localX;
          const dy = py - localY;
          const distance = dx * dx + dy * dy;
          if (distance <= bestDistance) {
            best = { seriesIndex, series, segmentIndex, segment, pointIndex, point, px, py };
            bestDistance = distance;
          }
        });
      });
    });
    return best;
  };

  const chooseStartPoint = (local) => {
    if (activeSeries >= 0) {
      const active = nearestPoint(local, hitRadius(18), activeSeries);
      if (active) return active;
    }
    return nearestPoint(local);
  };

  const canEdit = (series) => series.sampled || series.isNew;

  const recomputeDirty = () => {
    dirty = seriesList.some((series) => series.modified);
  };

  const refreshActions = () => {
    undoButton.disabled = saving || undoStack.length === 0;
    resetEdits.disabled = saving || !dirty;
    saveButton.disabled = saving || (!dirty && !axisDirty);
    const active = seriesList[activeSeries];
    exportButton.disabled = saving || (!dirty && !axisDirty) || !active
      || !active.segments.some((segment) => segment.length >= 2);
    finishCurveButton.disabled = tool !== "draw";
  };

  const pushUndo = (state) => {
    undoStack.push(state);
    if (undoStack.length > 30) undoStack.shift();
    refreshActions();
  };

  const snapshot = () => ({
    series: clone(seriesList),
    activeSeries,
    start: start ? { ...start } : null,
    tool,
  });

  let lastLegendKey = "";
  const renderLegend = () => {
    legendElement.replaceChildren();
    seriesList.forEach((series, index) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "chart-legend-item";
      item.dataset.seriesIndex = String(index);
      item.setAttribute("aria-pressed", String(index === activeSeries));
      item.title = `点击切换并确认当前曲线：${series.label}`;
      if (index === activeSeries) item.classList.add("active");
      if (activeSeries >= 0 && index !== activeSeries) item.classList.add("dimmed");
      const dot = document.createElement("i");
      dot.className = "chart-legend-dot";
      dot.style.background = series.color;
      item.appendChild(dot);
      item.appendChild(document.createTextNode(series.label + (series.isNew ? " · 新" : "")));
      if (index === activeSeries) {
        const check = document.createElement("i");
        check.className = "fa-solid fa-check legend-check";
        check.setAttribute("aria-hidden", "true");
        item.appendChild(check);
      }
      item.addEventListener("click", () => selectCurve(index));
      legendElement.appendChild(item);
    });
    const key = seriesList.map((series) => series.label).join("\u0000");
    if (key !== lastLegendKey) {
      lastLegendKey = key;
      scheduleAlignPlots();
    }
  };

  const line = (points, color, opacity, strokeWidth, className) => {
    const element = document.createElementNS(NS, "polyline");
    element.setAttribute("points", points.map(([px, py]) => `${px},${py}`).join(" "));
    element.setAttribute("fill", "none");
    element.setAttribute("stroke", color);
    element.setAttribute("stroke-width", String(strokeWidth));
    element.setAttribute("stroke-linecap", "round");
    element.setAttribute("stroke-linejoin", "round");
    element.setAttribute("opacity", String(opacity));
    element.setAttribute("vector-effect", "non-scaling-stroke");
    if (className) element.setAttribute("class", className);
    return element;
  };

  const circle = (x, y, radius, fill, stroke, strokeWidth, className) => {
    const element = document.createElementNS(NS, "circle");
    element.setAttribute("cx", String(x));
    element.setAttribute("cy", String(y));
    element.setAttribute("r", String(radius));
    element.setAttribute("fill", fill);
    element.setAttribute("stroke", stroke);
    element.setAttribute("stroke-width", String(strokeWidth));
    if (className) element.setAttribute("class", className);
    return element;
  };

  const label = (x, y, value) => {
    const element = document.createElementNS(NS, "text");
    element.setAttribute("x", String(x));
    element.setAttribute("y", String(y));
    element.setAttribute("class", "chart-start-label");
    element.textContent = value;
    return element;
  };

  const formatNumber = (value) => {
    if (!Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    if (abs >= 1000) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(2).replace(/\.?0+$/, "");
    return value.toFixed(3).replace(/\.?0+$/, "");
  };

  const readAxisSettings = () => {
    const readNumber = (input, fallback) => {
      const raw = input ? String(input.value).trim() : "";
      if (!raw) return fallback;
      const value = Number(raw);
      return Number.isFinite(value) ? value : NaN;
    };
    const xFactor = readNumber(plotXFactor, 1);
    const xOffset = readNumber(plotXOffset, 0);
    const yFactor = readNumber(plotYFactor, 1);
    const yOffset = readNumber(plotYOffset, 0);
    const valid = Number.isFinite(xFactor) && xFactor > 0
      && Number.isFinite(yFactor) && yFactor > 0
      && Number.isFinite(xOffset) && Number.isFinite(yOffset);
    return {
      xFactor, xOffset, yFactor, yOffset, valid,
      converted: valid && (xFactor !== 1 || xOffset !== 0 || yFactor !== 1 || yOffset !== 0),
      xLabel: (plotXInput?.value || data.x_label || "X").trim() || "X",
      yLabel: (plotYInput?.value || data.y_label || "Y").trim() || "Y",
    };
  };

  const svgNode = (tag, attrs = {}, text = "") => {
    const element = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    if (text) element.textContent = text;
    return element;
  };

  const niceTicks = (min, max, target = 6) => {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    if (min === max) {
      const pad = min === 0 ? 1 : Math.abs(min) * 0.1;
      min -= pad;
      max += pad;
    }
    const span = max - min;
    const raw = span / target;
    const magnitude = 10 ** Math.floor(Math.log10(raw));
    const normalized = raw / magnitude;
    const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
    const start = Math.floor(min / step) * step;
    const end = Math.ceil(max / step) * step;
    const ticks = [];
    for (let value = start; value <= end + step * 1e-9; value += step) {
      ticks.push(Number(value.toFixed(12)));
    }
    return { min: start, max: end, step, ticks };
  };

  const tickText = (value, step) => {
    const decimals = step >= 1 ? 0 : step >= 0.1 ? 1 : step >= 0.01 ? 2 : step >= 0.001 ? 3 : 4;
    return Number(value).toFixed(decimals);
  };

  const markerShape = (shape, x, y, size, color) => {
    if (shape % 4 === 1) {
      return svgNode("rect", {
        x: x - size, y: y - size, width: size * 2, height: size * 2,
        rx: 0.6, fill: color, stroke: "#1f2937", "stroke-width": 0.8,
      });
    }
    if (shape % 4 === 2) {
      return svgNode("polygon", {
        points: `${x},${y - size - 1} ${x - size - 1},${y + size} ${x + size + 1},${y + size}`,
        fill: color, stroke: "#1f2937", "stroke-width": 0.8,
      });
    }
    if (shape % 4 === 3) {
      return svgNode("polygon", {
        points: `${x},${y - size - 1} ${x + size + 1},${y} ${x},${y + size + 1} ${x - size - 1},${y}`,
        fill: color, stroke: "#1f2937", "stroke-width": 0.8,
      });
    }
    return svgNode("circle", {
      cx: x, cy: y, r: size, fill: color, stroke: "#1f2937", "stroke-width": 0.8,
    });
  };

  const markPlotStale = (message = "左侧已修改，请重新确认绘制。") => {
    plotStale = true;
    if (plotStatus) {
      plotStatus.textContent = message;
      plotStatus.dataset.tone = "warn";
    }
    if (plotConfirm) plotConfirm.classList.add("attention");
  };

  const drawCleanPlot = () => {
    if (!plotRoot || !plotConfirm) return;
    const axis = readAxisSettings();
    const xFactor = axis.valid ? axis.xFactor : 1;
    const xOffset = axis.valid ? axis.xOffset : 0;
    const yFactor = axis.valid ? axis.yFactor : 1;
    const yOffset = axis.valid ? axis.yOffset : 0;
    const activeSeriesData = seriesList.map((series, index) => ({
      ...series,
      shape: index,
      plotColor: series.color,
      segments: series.segments
        .filter((segment) => segment.length > 1)
        .map((segment) => segment.map((point) => ({
          x: point.x * xFactor + xOffset,
          y: point.y * yFactor + yOffset,
        }))),
    })).filter((series) => series.segments.length);
    const allPoints = activeSeriesData.flatMap((series) => series.segments.flat());
    plotRoot.replaceChildren();
    if (!allPoints.length) {
      const empty = document.createElement("div");
      empty.className = "plot-placeholder";
      empty.textContent = "当前没有可绘制的曲线点。";
      plotRoot.appendChild(empty);
      plotStatus.textContent = "当前没有可绘制的曲线点。";
      plotStatus.dataset.tone = "warn";
      return;
    }
    let xMin = Math.min(...allPoints.map((point) => point.x));
    let xMax = Math.max(...allPoints.map((point) => point.x));
    let yMin = Math.min(...allPoints.map((point) => point.y));
    let yMax = Math.max(...allPoints.map((point) => point.y));
    if (xMin === xMax) { xMin -= 1; xMax += 1; }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    const xAxis = niceTicks(xMin, xMax, 6);
    const yAxis = niceTicks(yMin, yMax, 6);
    if (!xAxis || !yAxis) return;

    const width = 640;
    const height = 480;
    const margin = { left: 64, right: 18, top: 18, bottom: 54 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const plotRight = margin.left + plotWidth;
    const plotBottom = margin.top + plotHeight;
    const mapX = (value) => margin.left + ((value - xAxis.min) / (xAxis.max - xAxis.min)) * plotWidth;
    const mapY = (value) => plotBottom - ((value - yAxis.min) / (yAxis.max - yAxis.min)) * plotHeight;

    const chart = svgNode("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "确认绘制后的提取曲线",
    });
    chart.appendChild(svgNode("rect", { width, height, fill: "#ffffff" }));
    const defs = svgNode("defs", {});
    const clip = svgNode("clipPath", { id: "cleanPlotClip" });
    clip.appendChild(svgNode("rect", {
      x: margin.left, y: margin.top, width: plotWidth, height: plotHeight,
    }));
    defs.appendChild(clip);
    chart.appendChild(defs);

    const dataLayer = svgNode("g", { "clip-path": "url(#cleanPlotClip)" });
    activeSeriesData.forEach((series) => {
      series.segments.forEach((segment) => {
        const points = segment.map((point) => `${mapX(point.x)},${mapY(point.y)}`).join(" ");
        dataLayer.appendChild(svgNode("polyline", {
          points, fill: "none", stroke: series.plotColor, "stroke-width": 1.8,
          "stroke-linejoin": "round", "stroke-linecap": "round",
        }));
        const stride = Math.max(1, Math.ceil(segment.length / 60));
        const markerIndexes = [];
        for (let index = 0; index < segment.length; index += stride) markerIndexes.push(index);
        if (markerIndexes[markerIndexes.length - 1] !== segment.length - 1) {
          markerIndexes.push(segment.length - 1);
        }
        markerIndexes.forEach((index) => {
          const point = segment[index];
          dataLayer.appendChild(markerShape(series.shape, mapX(point.x), mapY(point.y), 4.2, series.plotColor));
        });
      });
    });
    chart.appendChild(dataLayer);

    chart.appendChild(svgNode("line", {
      x1: margin.left, y1: plotBottom, x2: plotRight, y2: plotBottom,
      stroke: "#111827", "stroke-width": 1.2,
    }));
    chart.appendChild(svgNode("line", {
      x1: margin.left, y1: margin.top, x2: margin.left, y2: plotBottom,
      stroke: "#111827", "stroke-width": 1.2,
    }));
    chart.appendChild(svgNode("line", {
      x1: margin.left, y1: margin.top, x2: plotRight, y2: margin.top,
      stroke: "#111827", "stroke-width": 1.2,
    }));
    chart.appendChild(svgNode("line", {
      x1: plotRight, y1: margin.top, x2: plotRight, y2: plotBottom,
      stroke: "#111827", "stroke-width": 1.2,
    }));
    xAxis.ticks.forEach((tick) => {
      const x = mapX(tick);
      chart.appendChild(svgNode("line", {
        x1: x, y1: plotBottom, x2: x, y2: plotBottom + 5, stroke: "#111827", "stroke-width": 1,
      }));
      chart.appendChild(svgNode("text", {
        x, y: plotBottom + 19, "text-anchor": "middle", "font-size": 11.5, fill: "#334155",
      }, tickText(tick, xAxis.step)));
    });
    yAxis.ticks.forEach((tick) => {
      const y = mapY(tick);
      chart.appendChild(svgNode("line", {
        x1: margin.left - 5, y1: y, x2: margin.left, y2: y, stroke: "#111827", "stroke-width": 1,
      }));
      chart.appendChild(svgNode("text", {
        x: margin.left - 9, y: y + 3.5, "text-anchor": "end", "font-size": 11.5, fill: "#334155",
      }, tickText(tick, yAxis.step)));
    });

    const xLabel = axis.xLabel;
    const yLabel = axis.yLabel;
    chart.appendChild(svgNode("text", {
      x: margin.left + plotWidth / 2, y: height - 12, "text-anchor": "middle",
      "font-size": 15, "font-weight": 600, fill: "#111827",
    }, xLabel));
    chart.appendChild(svgNode("text", {
      x: 15, y: margin.top + plotHeight / 2, "text-anchor": "middle",
      "font-size": 15, "font-weight": 600, fill: "#111827",
      transform: `rotate(-90 15 ${margin.top + plotHeight / 2})`,
    }, yLabel));

    const legendX = margin.left + 10;
    const legendY = margin.top + 10;
    const legendRows = activeSeriesData.map((series) => {
      const text = series.label.length > 26 ? `${series.label.slice(0, 24)}…` : series.label;
      return { series, text };
    });
    const legendWidth = Math.min(plotWidth - 20,
      Math.max(140, Math.max(...legendRows.map((row) => row.text.length)) * 6.8 + 40));
    const legendHeight = 10 + legendRows.length * 19;
    const legend = svgNode("g", {});
    legend.appendChild(svgNode("rect", {
      x: legendX, y: legendY, width: legendWidth, height: legendHeight,
      fill: "rgba(255,255,255,.94)", stroke: "#94a3b8", "stroke-width": 0.8, rx: 2,
    }));
    legendRows.forEach((row, index) => {
      const centerY = legendY + 14 + index * 19;
      legend.appendChild(markerShape(row.series.shape, legendX + 13, centerY, 4, row.series.plotColor));
      legend.appendChild(svgNode("text", {
        x: legendX + 25, y: centerY + 3.8, "font-size": 11.5, fill: "#111827",
      }, row.text));
    });
    chart.appendChild(legend);
    plotRoot.appendChild(chart);
    plotStale = false;
    plotConfirm.classList.remove("attention");
    const pointCount = allPoints.length;
    plotStatus.textContent = `已绘制 ${activeSeriesData.length} 条曲线 · ${pointCount} 个点`;
    plotStatus.dataset.tone = "ok";
  };

  const activeStart = () => {
    if (!start) return null;
    const series = seriesList[start.seriesIndex];
    if (!series) {
      start = null;
      return null;
    }
    const found = findPoint(series.segments, start.pointId);
    if (!found) {
      start = null;
      return null;
    }
    return { ...found, series, seriesIndex: start.seriesIndex };
  };

  const isLoupeTool = () => ["start", "move", "erase", "add", "draw"].includes(tool);

  const hideLoupe = () => {
    if (loupe) loupe.hidden = true;
    lastPointer = null;
  };

  const positionLoupe = (clientX, clientY) => {
    if (!loupe) return;
    const gap = 22;
    const stageRect = stage.getBoundingClientRect();
    let left = clientX + gap;
    let top = clientY + gap;
    if (left + LOUPE_SIZE > stageRect.right - 8) left = clientX - LOUPE_SIZE - gap;
    if (top + LOUPE_SIZE > stageRect.bottom - 8) top = clientY - LOUPE_SIZE - gap;
    left = Math.max(8, Math.min(window.innerWidth - LOUPE_SIZE - 8, left));
    top = Math.max(8, Math.min(window.innerHeight - LOUPE_SIZE - 8, top));
    loupe.style.left = `${left}px`;
    loupe.style.top = `${top}px`;
  };

  const drawLoupe = (local) => {
    if (!loupeCtx || !loupe) return;
    const ctx = loupeCtx;
    const size = LOUPE_SIZE;
    const zoom = LOUPE_ZOOM;
    const half = size / 2;
    const sourceSize = size / zoom;
    const sx = local[0] - sourceSize / 2;
    const sy = local[1] - sourceSize / 2;
    const image = stage.querySelector("img");
    ctx.setTransform(loupeDpr, 0, 0, loupeDpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    ctx.save();
    ctx.beginPath();
    ctx.arc(half, half, half - 1, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, size, size);
    if (image && image.complete && image.naturalWidth) {
      const ix = Math.max(0, sx);
      const iy = Math.max(0, sy);
      const rightEdge = Math.min(image.naturalWidth, sx + sourceSize);
      const bottomEdge = Math.min(image.naturalHeight, sy + sourceSize);
      const iw = rightEdge - ix;
      const ih = bottomEdge - iy;
      if (iw > 0 && ih > 0) {
        ctx.drawImage(
          image,
          ix, iy, iw, ih,
          (ix - sx) * zoom, (iy - sy) * zoom, iw * zoom, ih * zoom,
        );
      }
    }
    const lensPoint = (px, py) => [(px - sx) * zoom, (py - sy) * zoom];
    const inside = (x, y) => x >= 0 && y >= 0 && x <= size && y <= size;

    seriesList.forEach((series, seriesIndex) => {
      const active = seriesIndex === activeSeries;
      ctx.globalAlpha = active ? 1 : 0.62;
      series.segments.forEach((segment) => {
        if (segment.length < 2) return;
        ctx.beginPath();
        segment.forEach((point, index) => {
          const [px, py] = pointPx(series, point);
          const [x, y] = lensPoint(px, py);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = series.color;
        ctx.lineWidth = active ? 2.4 : 1.5;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke();
        segment.forEach((point) => {
          const [px, py] = pointPx(series, point);
          const [x, y] = lensPoint(px, py);
          if (!inside(x, y)) return;
          ctx.beginPath();
          ctx.arc(x, y, active ? 3 : 2.2, 0, Math.PI * 2);
          ctx.fillStyle = series.color;
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 0.8;
          ctx.stroke();
        });
      });
    });
    ctx.globalAlpha = 1;

    const resolved = activeStart();
    if (resolved) {
      const [px, py] = pointPx(resolved.series, resolved.point);
      const [x, y] = lensPoint(px, py);
      if (inside(x, y)) {
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 5.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    let candidate = null;
    if (tool === "start") {
      candidate = nearestPoint(local, hitRadius(24));
    } else if (["move", "erase", "add"].includes(tool) && activeSeries >= 0) {
      candidate = nearestPoint(local, hitRadius(24), activeSeries);
    }
    if (candidate) {
      const [x, y] = lensPoint(candidate.px, candidate.py);
      if (inside(x, y)) {
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 5.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    if (pendingAdd) {
      const [nx, ny] = lensPoint(pendingAdd.px, pendingAdd.py);
      if (inside(nx, ny)) {
        ctx.beginPath();
        ctx.arc(nx, ny, 6, 0, Math.PI * 2);
        ctx.strokeStyle = "#d97706";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      pendingAdd.anchors.forEach((anchor) => {
        const series = seriesList[anchor.seriesIndex];
        const found = series ? findPoint(series.segments, anchor.pointId) : null;
        if (!found) return;
        const [px, py] = pointPx(series, found.point);
        const [x, y] = lensPoint(px, py);
        if (!inside(x, y)) return;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.strokeStyle = "#d97706";
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }

    if (tool === "erase") {
      const radius = hitRadius(8) * zoom;
      ctx.beginPath();
      ctx.arc(half, half, radius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(220, 38, 38, .08)";
      ctx.fill();
      ctx.strokeStyle = "rgba(220, 38, 38, .85)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    ctx.strokeStyle = "rgba(255,255,255,.95)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(half - 12, half); ctx.lineTo(half - 4, half);
    ctx.moveTo(half + 4, half); ctx.lineTo(half + 12, half);
    ctx.moveTo(half, half - 12); ctx.lineTo(half, half - 4);
    ctx.moveTo(half, half + 4); ctx.lineTo(half, half + 12);
    ctx.stroke();
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(half - 12, half); ctx.lineTo(half - 4, half);
    ctx.moveTo(half + 4, half); ctx.lineTo(half + 12, half);
    ctx.moveTo(half, half - 12); ctx.lineTo(half, half - 4);
    ctx.moveTo(half, half + 4); ctx.lineTo(half, half + 12);
    ctx.stroke();
    ctx.restore();
  };

  const updateLoupe = (clientX, clientY, local) => {
    if (!loupe || !loupeCtx || !isLoupeTool()) {
      hideLoupe();
      return;
    }
    lastPointer = { clientX, clientY, local };
    loupe.hidden = false;
    positionLoupe(clientX, clientY);
    drawLoupe(local);
  };

  let alignTimer = 0;
  const alignReviewPlots = () => {
    if (!stage || !plotRoot) return;
    stage.style.marginTop = "";
    plotRoot.style.marginTop = "";
    if (window.matchMedia("(max-width: 1100px)").matches) return;
    const split = document.querySelector(".review-split");
    if (!split) return;
    const splitTop = split.getBoundingClientRect().top;
    const leftTop = stage.getBoundingClientRect().top - splitTop;
    const rightTop = plotRoot.getBoundingClientRect().top - splitTop;
    const difference = Math.round(rightTop - leftTop);
    if (Math.abs(difference) < 1) return;
    const stageExtra = difference > 0 ? difference : 0;
    const plotExtra = difference < 0 ? -difference : 0;
    stage.style.marginTop = `${stageExtra}px`;
    plotRoot.style.marginTop = `${plotExtra}px`;
    const correction = Math.round(
      (plotRoot.getBoundingClientRect().top - splitTop)
      - (stage.getBoundingClientRect().top - splitTop),
    );
    if (Math.abs(correction) >= 1) {
      if (correction > 0) stage.style.marginTop = `${stageExtra + correction}px`;
      else plotRoot.style.marginTop = `${plotExtra - correction}px`;
    }
  };

  const scheduleAlignPlots = () => {
    window.clearTimeout(alignTimer);
    alignTimer = window.setTimeout(alignReviewPlots, 40);
  };

  const sampleSeries = (series, pointId, sliderValueRaw) => {
    const source = findPoint(series.sourceSegments, pointId);
    if (!source) return null;
    const remaining = source.segment.length - source.pointIndex;
    const maxStep = Math.max(1, Math.ceil(remaining / 4));
    const ratio = Math.max(0, Math.min(100, Number(sliderValueRaw))) / 100;
    const step = Math.max(1, Math.round(1 + (maxStep - 1) * (1 - ratio)));
    const sampled = [];
    for (let index = source.pointIndex; index < source.segment.length; index += step) {
      sampled.push(clonePoint(source.segment[index]));
    }
    if (sampled.length && sampled[sampled.length - 1].id !== source.segment[source.segment.length - 1].id) {
      sampled.push(clonePoint(source.segment[source.segment.length - 1]));
    }
    const segments = series.sourceSegments.map((segment, index) => {
      if (index === source.segmentIndex) return sampled;
      return segment.map(clonePoint);
    });
    return { segments, step };
  };

  const applySampling = (series, pointId, sliderValueRaw) => {
    const sampled = sampleSeries(series, pointId, sliderValueRaw);
    if (!sampled) return false;
    series.segments = sampled.segments;
    series.sampled = true;
    series.edited = false;
    series.modified = !sameSegments(series.segments, series.sourceSegments);
    recomputeDirty();
    markPlotStale("采样已更新，请重新确认绘制。");
    return true;
  };

  const markEdited = (series, message) => {
    series.edited = true;
    series.modified = true;
    dirty = true;
    setStatus(`已修改 · ${message}`, "warn");
    markPlotStale("曲线已修改，请重新确认绘制。");
    refreshActions();
  };

  const validateStart = () => {
    if (!start) return;
    const series = seriesList[start.seriesIndex];
    if (!series || !findPoint(series.segments, start.pointId)) start = null;
  };

  const render = () => {
    validateStart();
    svg.replaceChildren();
    const defs = document.createElementNS(NS, "defs");
    const clip = document.createElementNS(NS, "clipPath");
    clip.setAttribute("id", "chartFrameClip");
    const clipRect = document.createElementNS(NS, "rect");
    clipRect.setAttribute("x", String(left));
    clipRect.setAttribute("y", String(top));
    clipRect.setAttribute("width", String(frameWidth));
    clipRect.setAttribute("height", String(frameHeight));
    clip.appendChild(clipRect);
    defs.appendChild(clip);
    svg.appendChild(defs);

    const base = document.createElementNS(NS, "g");
    base.setAttribute("clip-path", "url(#chartFrameClip)");
    seriesList.forEach((series, seriesIndex) => {
      const active = seriesIndex === activeSeries;
      series.segments.forEach((segment) => {
        if (segment.length < 2) return;
        base.appendChild(line(
          segment.map((point) => pointPx(series, point)),
          series.color,
          activeSeries >= 0 ? (active ? 0.92 : 0.14) : 0.3,
          active ? 2.2 : 1.2,
          "chart-series-line",
        ));
      });
    });

    const showHandles = ["move", "erase", "add", "draw"].includes(tool)
      || (tool === "start" && activeSeries >= 0 && seriesList[activeSeries]?.sampled)
      || (tool === "start" && seriesList[activeSeries]?.isNew);
    if (showHandles && activeSeries >= 0 && !drag && !erasing) {
      const series = seriesList[activeSeries];
      series.segments.forEach((segment) => {
        segment.forEach((point) => {
          const [px, py] = pointPx(series, point);
          base.appendChild(circle(px, py, 2.2, series.color, "#ffffff", 0.8, "chart-point-dot"));
        });
      });
    }
    svg.appendChild(base);

    if (pendingAdd) {
      const layer = document.createElementNS(NS, "g");
      layer.setAttribute("clip-path", "url(#chartFrameClip)");
      layer.appendChild(circle(pendingAdd.px, pendingAdd.py, 6, "#ffffff", "#d97706", 2, "chart-add-point"));
      layer.appendChild(label(pendingAdd.px + 9, pendingAdd.py - 9, "新点"));
      pendingAdd.anchors.forEach((anchor) => {
        const target = seriesList[anchor.seriesIndex];
        const found = target ? findPoint(target.segments, anchor.pointId) : null;
        if (!found) return;
        const [ax, ay] = pointPx(target, found.point);
        layer.appendChild(circle(ax, ay, 6, "rgba(255,255,255,.92)", "#d97706", 2, "chart-anchor"));
        layer.appendChild(label(ax + 8, ay - 8, "锚"));
      });
      svg.appendChild(layer);
    }

    const resolved = activeStart();
    const active = seriesList[activeSeries];
    if (resolved) {
      const [sx, sy] = pointPx(resolved.series, resolved.point);
      const layer = document.createElementNS(NS, "g");
      layer.appendChild(circle(sx, sy, 7, "rgba(255,255,255,.92)", "#0f172a", 2, "chart-start-ring"));
      layer.appendChild(circle(sx, sy, 2.8, resolved.series.color, resolved.series.color, 1, "chart-start-dot"));
      layer.appendChild(label(sx + 10, sy - 9, "起点"));
      svg.appendChild(layer);
      selection.textContent = `当前曲线：${resolved.series.label} · 起点 x=${formatNumber(resolved.point.x)} y=${formatNumber(resolved.point.y)}`;
    } else if (active) {
      selection.textContent = `当前曲线：${active.label} · 未选起点`;
    } else {
      selection.textContent = "当前曲线：未选择";
    }

    const activeCount = active ? countPoints(active) : 0;
    if (start && active && active.sampled) {
      sliderValue.textContent = active.edited
        ? `已锁定 · ${activeCount} 点`
        : `共 ${activeCount} 点`;
    } else if (active && active.sampled) {
      sliderValue.textContent = `已采样 · ${activeCount} 点`;
    } else {
      sliderValue.textContent = "未选择起点";
    }
    slider.disabled = !(start && active && active.sampled && !active.edited);
    resetStart.disabled = !start;

    if (drag) {
      const target = seriesList[drag.seriesIndex];
      const found = target ? findPoint(target.segments, drag.pointId) : null;
      if (found) {
        const [px, py] = pointPx(target, found.point);
        const layer = document.createElementNS(NS, "g");
        layer.appendChild(circle(px, py, 7, "rgba(255,255,255,.94)", "#0f172a", 2, "chart-drag-ring"));
        layer.appendChild(circle(px, py, 3, target.color, target.color, 1, "chart-drag-dot"));
        svg.appendChild(layer);
      }
    }

    brushCircle = null;
    if (tool === "erase") {
      const radius = hitRadius(8);
      brushCircle = circle(brush ? brush[0] : -radius * 2, brush ? brush[1] : -radius * 2,
        radius, "rgba(220,38,38,.08)", "#dc2626", 1.5, "chart-brush");
      svg.appendChild(brushCircle);
    }

    renderLegend();
    refreshActions();
    if (loupe && !loupe.hidden && lastPointer) drawLoupe(lastPointer.local);
  };

  const updateBrush = () => {
    if (!brushCircle) return;
    const radius = Number(brushCircle.getAttribute("r"));
    brushCircle.setAttribute("cx", brush ? String(brush[0]) : String(-radius * 2));
    brushCircle.setAttribute("cy", brush ? String(brush[1]) : String(-radius * 2));
  };

  const clearStart = () => {
    start = null;
    slider.value = "100";
    slider.disabled = true;
    resetStart.disabled = true;
    setStatus("起点已清除", dirty ? "warn" : "");
    render();
  };

  const selectCurve = (index) => {
    if (tool === "draw") finishDrawing();
    const series = seriesList[index];
    if (!series) return;
    const changed = activeSeries !== index;
    activeSeries = index;
    if (changed) {
      start = null;
      slider.value = "100";
      slider.disabled = true;
    }
    pendingAdd = null;
    hideLoupe();
    const state = series.isNew ? "新曲线"
      : series.edited ? "已修改，分辨率锁定"
        : series.sampled ? "已采样" : "未采样，请先选起点";
    setStatus(`当前曲线：${series.label} · ${state}`, "ok");
    render();
  };

  const selectStart = (found) => {
    const series = seriesList[found.seriesIndex];
    if (!series) return;
    if (series.edited || series.isNew) {
      activeSeries = found.seriesIndex;
      start = null;
      slider.value = "100";
      slider.disabled = true;
      setStatus(series.isNew ? "已切换到新曲线，可直接移动或补充。" : "已切换到已修改曲线；分辨率已锁定。", "warn");
      render();
      return;
    }
    activeSeries = found.seriesIndex;
    start = { seriesIndex: found.seriesIndex, pointId: found.point.id };
    slider.value = "100";
    const ok = applySampling(series, found.point.id, 100);
    if (!ok) {
      start = null;
      setStatus("该起点不在原始数据中，无法采样。", "warn");
      render();
      return;
    }
    setStatus(dirty ? "已修改 · 已应用起点采样" : "已应用起点采样", dirty ? "warn" : "");
    render();
  };

  const onSlider = () => {
    const series = seriesList[activeSeries];
    if (!start || !series || series.edited || !series.sampled) return;
    if (!applySampling(series, start.pointId, slider.value)) return;
    setStatus(dirty ? "已修改 · 采样已更新" : "采样已更新", dirty ? "warn" : "");
    render();
  };

  const movePoint = (found, local) => {
    found.point.x = toDataX(clamp(local[0], left, right));
    found.point.y = toDataY(found.series, clamp(local[1], top, bottom));
    drag.moved = true;
    render();
  };

  const eraseAt = (local) => {
    const series = seriesList[erasing.seriesIndex];
    if (!series) return 0;
    const radius = hitRadius(8);
    const radiusSq = radius * radius;
    let removed = 0;
    const nextSegments = [];
    series.segments.forEach((segment) => {
      const hits = new Set();
      segment.forEach((point) => {
        const [px, py] = pointPx(series, point);
        const dx = px - local[0];
        const dy = py - local[1];
        if (dx * dx + dy * dy <= radiusSq) hits.add(point.id);
      });
      if (!hits.size) {
        nextSegments.push(segment);
        return;
      }
      let run = [];
      let kept = 0;
      segment.forEach((point) => {
        if (hits.has(point.id)) {
          if (run.length >= 2) {
            nextSegments.push(run);
            kept += run.length;
          }
          run = [];
        } else {
          run.push(point);
        }
      });
      if (run.length >= 2) {
        nextSegments.push(run);
        kept += run.length;
      }
      removed += segment.length - kept;
    });
    if (!removed) return 0;
    if (!nextSegments.some((segment) => segment.length >= 2)) {
      setStatus("至少保留两个点；请缩小擦除范围。", "warn");
      return 0;
    }
    series.segments = nextSegments;
    series.modified = true;
    if (start && start.seriesIndex === erasing.seriesIndex
        && !findPoint(series.segments, start.pointId)) {
      start = null;
    }
    erasing.removed += removed;
    render();
    return removed;
  };

  const finishDrag = (event) => {
    if (!drag) return;
    const target = seriesList[drag.seriesIndex];
    const moved = drag.moved;
    const before = drag.before;
    drag = null;
    if (event) {
      try { stage.releasePointerCapture(event.pointerId); } catch (error) { /* no capture */ }
    }
    if (moved && target) {
      target.edited = true;
      target.modified = true;
      markEdited(target, "已移动数据点");
      pushUndo(before);
    } else {
      setStatus("未移动数据点");
    }
    render();
  };

  const finishErase = (event) => {
    if (!erasing) return;
    const target = seriesList[erasing.seriesIndex];
    const removed = erasing.removed;
    const before = erasing.before;
    erasing = null;
    if (event) {
      try { stage.releasePointerCapture(event.pointerId); } catch (error) { /* no capture */ }
    }
    if (removed > 0 && target) {
      target.edited = true;
      target.modified = true;
      markEdited(target, `已擦除 ${removed} 个点`);
      pushUndo(before);
    } else {
      setStatus("擦除未命中数据点");
    }
    render();
  };

  const connectAdd = () => {
    const anchors = pendingAdd.anchors
      .map((anchor) => {
        const series = seriesList[anchor.seriesIndex];
        const found = series ? findPoint(series.segments, anchor.pointId) : null;
        return found ? { ...found, series, seriesIndex: anchor.seriesIndex } : null;
      })
      .filter(Boolean);
    if (anchors.length !== 2) return { ok: false, message: "请选择左右两个数据点。" };
    if (anchors[0].seriesIndex !== anchors[1].seriesIndex) {
      return { ok: false, message: "左右锚点必须在同一条曲线上。" };
    }
    const series = anchors[0].series;
    if (!canEdit(series)) return { ok: false, message: "先为这条曲线设置起点并应用采样。" };
    if (anchors[0].point.id === anchors[1].point.id) {
      return { ok: false, message: "左右锚点不能是同一个点。" };
    }
    const [leftAnchor, rightAnchor] = [...anchors].sort((a, b) => a.point.x - b.point.x);
    if (leftAnchor.point.x === rightAnchor.point.x) {
      return { ok: false, message: "左右锚点不能在同一个 x 位置。" };
    }
    const next = makePoint(toDataX(pendingAdd.px), toDataY(series, pendingAdd.py));
    if (next.x < leftAnchor.point.x || next.x > rightAnchor.point.x) {
      return { ok: false, message: "新点应位于两个锚点之间。" };
    }
    const before = snapshot();
    if (leftAnchor.segmentIndex === rightAnchor.segmentIndex) {
      if (Math.abs(leftAnchor.pointIndex - rightAnchor.pointIndex) !== 1) {
        return { ok: false, message: "两个锚点必须是同一条折线上相邻的点。" };
      }
      pushUndo(before);
      series.segments[leftAnchor.segmentIndex].splice(Math.max(leftAnchor.pointIndex, rightAnchor.pointIndex), 0, next);
    } else {
      const leftSegment = series.segments[leftAnchor.segmentIndex];
      const rightSegment = series.segments[rightAnchor.segmentIndex];
      if (leftAnchor.pointIndex !== leftSegment.length - 1 || rightAnchor.pointIndex !== 0) {
        return { ok: false, message: "连接两段折线时，请选择左段末端和右段起点。" };
      }
      const merged = [...leftSegment, next, ...rightSegment];
      const indexes = [leftAnchor.segmentIndex, rightAnchor.segmentIndex].sort((a, b) => b - a);
      pushUndo(before);
      indexes.forEach((index) => series.segments.splice(index, 1));
      series.segments.splice(Math.min(leftAnchor.segmentIndex, rightAnchor.segmentIndex), 0, merged);
    }
    activeSeries = anchors[0].seriesIndex;
    start = null;
    pendingAdd = null;
    series.edited = true;
    markEdited(series, "已补充 1 个点");
    render();
    return { ok: true };
  };

  const isEndpoint = (found) => found
    && (found.pointIndex === 0 || found.pointIndex === found.segment.length - 1);

  const connectEndpoint = (found) => {
    const active = seriesList[activeSeries];
    if (!active || !canEdit(active)) {
      return { ok: false, message: "先为这条曲线应用采样，再接到端点。" };
    }
    const point = findPoint(active.segments, found.point.id);
    if (!point || !isEndpoint(point)) {
      return { ok: false, message: "请双击曲线的端点。" };
    }
    const next = makePoint(toDataX(pendingAdd.px), toDataY(active, pendingAdd.py));
    pushUndo(snapshot());
    if (point.pointIndex === 0) point.segment.unshift(next);
    else point.segment.push(next);
    activeSeries = found.seriesIndex;
    start = null;
    pendingAdd = null;
    active.edited = true;
    markEdited(active, "已接到端点并追加 1 个点");
    render();
    return { ok: true };
  };

  const handleAddClick = (local) => {
    const active = seriesList[activeSeries];
    if (!active || !canEdit(active)) {
      setStatus("先用“起点”选中一条曲线并应用采样，再补充点。", "warn");
      return;
    }
    if (!pendingAdd) {
      pendingAdd = {
        px: clamp(local[0], left, right),
        py: clamp(local[1], top, bottom),
        anchors: [],
        endpoint: null,
      };
      setStatus("新点已放置，请点击当前曲线上的锚点；如果接端点，直接双击端点。");
      render();
      return;
    }
    const found = nearestPoint(local, hitRadius(18), activeSeries);
    if (!found) {
      setStatus("未点中当前曲线的锚点，请点击折线上的数据点。", "warn");
      return;
    }
    if (pendingAdd.anchors.length === 1
        && pendingAdd.endpoint
        && pendingAdd.endpoint.pointId === found.point.id
        && isEndpoint(found)) {
      const elapsed = performance.now() - pendingAdd.endpoint.at;
      if (elapsed <= 650) {
        const result = connectEndpoint(found);
        if (!result.ok) {
          setStatus(result.message, "warn");
          const old = pendingAdd.anchors[0];
          const current = seriesList[old.seriesIndex];
          const currentFound = current ? findPoint(current.segments, old.pointId) : null;
          if (currentFound) pendingAdd.endpoint = { pointId: old.pointId, at: performance.now() };
          render();
        }
        return;
      }
      pendingAdd.endpoint = { pointId: found.point.id, at: performance.now() };
      setStatus("端点已选中；再次快速点击这个端点即可追加，或再选另一个点。");
      render();
      return;
    }
    if (pendingAdd.anchors.some((anchor) => anchor.pointId === found.point.id)) {
      setStatus("这个锚点已经选过，请选另一个点。", "warn");
      return;
    }
    if (pendingAdd.anchors.length === 0) {
      pendingAdd.anchors.push({ seriesIndex: activeSeries, pointId: found.point.id });
      pendingAdd.endpoint = isEndpoint(found)
        ? { pointId: found.point.id, at: performance.now() }
        : null;
      setStatus(isEndpoint(found)
        ? "已选端点；双击该端点可直接追加，或再点击另一个点。"
        : "已选择第一个锚点，请再点击相邻的另一个点。");
      render();
      return;
    }
    pendingAdd.anchors.push({ seriesIndex: activeSeries, pointId: found.point.id });
    const result = connectAdd();
    if (!result.ok) {
      setStatus(result.message, "warn");
      pendingAdd.anchors = [];
      pendingAdd.endpoint = null;
      render();
    }
  };

  const finishDrawing = () => {
    const series = seriesList[activeSeries];
    if (tool !== "draw" || !series || !series.isNew) return;
    const count = countPoints(series);
    if (count < 2) {
      seriesList.splice(activeSeries, 1);
      activeSeries = -1;
      start = null;
      recomputeDirty();
      setStatus("新曲线点数不足，已取消。", "warn");
    } else {
      series.edited = true;
      series.modified = true;
      dirty = true;
      setStatus(`新曲线已建立 · ${count} 点`, "warn");
    }
    markPlotStale("曲线已变化，请重新确认绘制。");
    tool = "move";
    stage.dataset.tool = tool;
    finishCurveButton.disabled = true;
    newCurveButton.classList.remove("active");
    document.querySelectorAll(".chart-tool[data-tool]").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === tool);
    });
    hint.innerHTML = '<i class="fa-solid fa-up-down-left-right" aria-hidden="true"></i> 拖动数据点到正确位置，折线会跟随移动。';
    render();
  };

  const startNewCurve = () => {
    if (tool === "draw") finishDrawing();
    pushUndo(snapshot());
    const index = seriesList.length;
    const color = colors.find((candidate) => !seriesList.some((series) => series.color === candidate))
      || colors[index % colors.length];
    seriesList.push({
      label: `新曲线 ${++newCurveCount}`,
      color,
      y_range: data.y_range.map(Number),
      file: null,
      sourceSegments: [],
      segments: [[]],
      sampled: true,
      edited: true,
      modified: true,
      isNew: true,
    });
    activeSeries = index;
    start = null;
    slider.value = "100";
    tool = "draw";
    stage.dataset.tool = tool;
    document.querySelectorAll(".chart-tool[data-tool]").forEach((button) => button.classList.remove("active"));
    newCurveButton.classList.add("active");
    finishCurveButton.disabled = false;
    pendingAdd = null;
    dirty = true;
    hint.innerHTML = '<i class="fa-solid fa-bezier-curve" aria-hidden="true"></i> 沿缺失曲线依次点击；至少两个点后点“完成”结束。';
    setStatus("新曲线绘制中 · 点击曲线位置", "warn");
    markPlotStale("新曲线尚未绘制完成，请完成后重新确认成图。");
    render();
  };

  const undo = () => {
    if (!undoStack.length || saving) return;
    const state = undoStack.pop();
    seriesList = clone(state.series);
    activeSeries = state.activeSeries;
    start = state.start ? { ...state.start } : null;
    tool = state.tool === "draw" ? "move" : state.tool;
    pendingAdd = null;
    drag = null;
    erasing = null;
    recomputeDirty();
    slider.value = "100";
    stage.dataset.tool = tool;
    document.querySelectorAll(".chart-tool[data-tool]").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === tool);
    });
    finishCurveButton.disabled = tool !== "draw";
    newCurveButton.classList.toggle("active", tool === "draw");
    markPlotStale("已撤销修改，请重新确认绘制。");
    setStatus(dirty ? "已撤销上一步 · 仍有修改" : "已撤销 · 已回到原始数据", dirty ? "warn" : "");
    updateHintForTool();
    render();
  };

  const restoreOriginal = () => {
    if (!dirty || saving) return;
    pushUndo(snapshot());
    seriesList = clone(baselineSeries);
    activeSeries = -1;
    start = null;
    pendingAdd = null;
    tool = "start";
    stage.dataset.tool = tool;
    slider.value = "100";
    slider.disabled = true;
    recomputeDirty();
    markPlotStale("已还原数据，请重新确认绘制。");
    setStatus("已还原全部修改", "ok");
    updateHintForTool();
    render();
  };

  const exportCurrent = () => {
    const series = seriesList[activeSeries];
    if (!series) return;
    const axis = readAxisSettings();
    if (!axis.valid) {
      setStatus("轴倍率必须大于 0，倍率和偏移必须是数字。", "warn");
      return;
    }
    const rows = ["x,y"];
    series.segments.forEach((segment, index) => {
      if (!segment.length) return;
      if (index > 0) rows.push("");
      segment.forEach((point) => rows.push(
        `${point.x * axis.xFactor + axis.xOffset},${point.y * axis.yFactor + axis.yOffset}`));
    });
    const blob = new Blob([`${rows.join("\n")}\n`], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    const name = (series.label || "series").replace(/[^\w\u4e00-\u9fa5.-]+/g, "_");
    link.href = URL.createObjectURL(blob);
    link.download = `${name}_converted.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    setStatus(axis.converted
      ? "已导出当前系列换算 CSV · 原文件未改"
      : "已导出当前系列修正 CSV · 原文件未改", "ok");
  };

  const saveCurves = async () => {
    if (saving) return;
    if (tool === "draw") finishDrawing();
    const axisSettings = readAxisSettings();
    if (!axisSettings.valid) {
      setStatus("轴倍率必须大于 0，倍率和偏移必须是数字。", "warn");
      return;
    }
    const targets = [];
    seriesList.forEach((series, index) => {
      if ((axisSettings.converted || series.modified)
          && series.segments.some((segment) => segment.length >= 2)) {
        targets.push({ series, index });
      }
    });
    if (!targets.length && !axisDirty) {
      setStatus("没有需要保存的曲线或轴设置。", "warn");
      return;
    }
    if (!runId) {
      setStatus("缺少任务标识，无法保存。", "warn");
      return;
    }
    saving = true;
    refreshActions();
    setStatus(targets.length
      ? `正在保存 ${targets.length} 条曲线${axisSettings.converted ? "并换算单位" : ""}…`
      : "正在保存轴设置…");
    const payload = {
      paper_scope: paperScope,
      panel_id: panelId,
      axis: {
        x_label: axisSettings.xLabel,
        y_label: axisSettings.yLabel,
        x_factor: axisSettings.xFactor,
        x_offset: axisSettings.xOffset,
        y_factor: axisSettings.yFactor,
        y_offset: axisSettings.yOffset,
      },
      series: targets.map(({ series }) => ({
        file: series.file,
        label: series.label,
        color: series.color,
        is_new: series.isNew,
        segments: series.segments.map((segment) => segment.map((point) => [point.x, point.y])),
      })),
    };
    try {
      const response = await fetch(`/runs/${encodeURIComponent(runId)}/save-curves`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok || !result.ok) {
        throw new Error(result.error || "保存失败。");
      }
      (result.saved || []).forEach((saved, index) => {
        const target = targets[index];
        if (!target) return;
        const series = target.series;
        series.file = saved.file;
        series.label = saved.label || series.label;
        series.isNew = false;
      });
      if (axisSettings.converted) {
        seriesList.forEach((series) => {
          series.segments = series.segments.map((segment) => segment.map((point) => ({
            ...point,
            x: point.x * axisSettings.xFactor + axisSettings.xOffset,
            y: point.y * axisSettings.yFactor + axisSettings.yOffset,
          })));
          series.sourceSegments = series.sourceSegments.map((segment) => segment.map((point) => ({
            ...point,
            x: point.x * axisSettings.xFactor + axisSettings.xOffset,
            y: point.y * axisSettings.yFactor + axisSettings.yOffset,
          })));
          series.y_range = [
            series.y_range[0] * axisSettings.yFactor + axisSettings.yOffset,
            series.y_range[1] * axisSettings.yFactor + axisSettings.yOffset,
          ];
          series.sampled = true;
          series.edited = false;
          series.modified = false;
        });
        xMin = xMin * axisSettings.xFactor + axisSettings.xOffset;
        xMax = xMax * axisSettings.xFactor + axisSettings.xOffset;
        data.x_range = [xMin, xMax];
        data.y_range = data.y_range.map(
          (value) => value * axisSettings.yFactor + axisSettings.yOffset);
      } else {
        seriesList.forEach((series) => {
          if (!targets.some((target) => target.series === series)) return;
          series.sourceSegments = clone(series.segments);
          series.sampled = true;
          series.edited = false;
          series.modified = false;
        });
      }
      data.x_label = axisSettings.xLabel;
      data.y_label = axisSettings.yLabel;
      if (plotXFactor) plotXFactor.value = "1";
      if (plotXOffset) plotXOffset.value = "0";
      if (plotYFactor) plotYFactor.value = "1";
      if (plotYOffset) plotYOffset.value = "0";
      start = null;
      slider.value = "100";
      slider.disabled = true;
      undoStack = [];
      dirty = false;
      axisDirty = false;
      saving = false;
      baselineSeries = clone(seriesList);
      const savedCount = (result.saved || []).length;
      setStatus(savedCount === 0 ? "轴标签已保存" : (axisSettings.converted
        ? `已保存 ${savedCount} 条曲线，CSV 已按单位换算覆盖`
        : `已保存 ${savedCount} 条曲线，CSV 已覆盖`), "ok");
      drawCleanPlot();
      render();
    } catch (error) {
      saving = false;
      setStatus(`保存失败：${error.message}`, "warn");
      refreshActions();
    }
  };

  const updateHintForTool = () => {
    if (tool === "start") {
      hint.innerHTML = '<i class="fa-solid fa-location-crosshairs" aria-hidden="true"></i> 先点起点，再拖采样间隔。采样确定后，再用移动/擦除/补充修改这些点。';
    } else if (tool === "move") {
      hint.innerHTML = '<i class="fa-solid fa-up-down-left-right" aria-hidden="true"></i> 拖动已采样的数据点；编辑后该曲线的分辨率锁定。';
    } else if (tool === "erase") {
      hint.innerHTML = '<i class="fa-solid fa-eraser" aria-hidden="true"></i> 在已采样的曲线上按住拖动；删除后折线断开，孤立的单点会一并移除。';
    } else if (tool === "add") {
      hint.innerHTML = '<i class="fa-solid fa-plus" aria-hidden="true"></i> 先放新点，再点左右两个相邻锚点；如果接端点，直接双击端点即可。';
    } else if (tool === "draw") {
      hint.innerHTML = '<i class="fa-solid fa-bezier-curve" aria-hidden="true"></i> 沿缺失曲线依次点击；至少两个点后点“完成”结束。';
    }
  };

  const setTool = (nextTool) => {
    if (tool === "draw" && nextTool !== "draw") finishDrawing();
    tool = nextTool;
    pendingAdd = null;
    brush = null;
    drag = null;
    erasing = null;
    document.querySelectorAll(".chart-tool[data-tool]").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === tool);
    });
    newCurveButton.classList.toggle("active", tool === "draw");
    finishCurveButton.disabled = tool !== "draw";
    stage.dataset.tool = tool;
    updateHintForTool();
    setStatus(dirty || axisDirty ? "已修改 · 未保存" : "未修改",
      dirty || axisDirty ? "warn" : "");
    render();
  };

  stage.addEventListener("pointerdown", (event) => {
    if (event.button !== undefined && event.button !== 0) return;
    const local = clientToLocal(event.clientX, event.clientY);
    if (event.pointerType !== "touch") updateLoupe(event.clientX, event.clientY, local);
    if (tool === "start") {
      const found = chooseStartPoint(local);
      if (found) selectStart(found);
      return;
    }
    if (tool === "draw") {
      const series = seriesList[activeSeries];
      if (!series || !series.isNew) return;
      series.segments[0].push(makePoint(
        toDataX(clamp(local[0], left, right)),
        toDataY(series, clamp(local[1], top, bottom)),
      ));
      series.modified = true;
      dirty = true;
      setStatus(`新曲线绘制中 · ${series.segments[0].length} 点`, "warn");
      render();
      event.preventDefault();
      return;
    }
    if (tool === "move") {
      const active = seriesList[activeSeries];
      if (!active || !canEdit(active)) {
        setStatus("先用“起点”选中一条曲线并应用采样，再修改点。", "warn");
        return;
      }
      const found = nearestPoint(local, hitRadius(16), activeSeries);
      if (!found) {
        setStatus("未点中当前曲线的数据点，请靠近数据点再拖动。", "warn");
        return;
      }
      drag = {
        seriesIndex: activeSeries,
        pointId: found.point.id,
        before: snapshot(),
        moved: false,
      };
      try { stage.setPointerCapture(event.pointerId); } catch (error) { /* no capture */ }
      setStatus("正在移动数据点…");
      render();
      event.preventDefault();
      return;
    }
    if (tool === "erase") {
      const active = seriesList[activeSeries];
      if (!active || !canEdit(active)) {
        setStatus("先用“起点”选中一条曲线并应用采样，再擦除。", "warn");
        return;
      }
      const found = nearestPoint(local, hitRadius(18), activeSeries);
      if (!found) {
        setStatus("未点中当前曲线的数据点，请靠近数据点再擦除。", "warn");
        return;
      }
      brush = local;
      erasing = { before: snapshot(), removed: 0, seriesIndex: activeSeries };
      try { stage.setPointerCapture(event.pointerId); } catch (error) { /* no capture */ }
      eraseAt(local);
      render();
      event.preventDefault();
      return;
    }
    if (tool === "add") {
      handleAddClick(local);
      event.preventDefault();
    }
  });

  stage.addEventListener("pointermove", (event) => {
    const local = clientToLocal(event.clientX, event.clientY);
    if (tool === "erase") {
      brush = local;
      if (erasing) eraseAt(local);
      updateBrush();
    }
    if (drag) {
      const target = seriesList[drag.seriesIndex];
      const found = target ? findPoint(target.segments, drag.pointId) : null;
      if (found) movePoint({ ...found, series: target }, local);
    }
    if (event.pointerType === "touch") hideLoupe();
    else updateLoupe(event.clientX, event.clientY, local);
  });

  const release = (event) => {
    if (drag) finishDrag(event);
    if (erasing) finishErase(event);
  };

  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  stage.addEventListener("pointerleave", () => {
    hideLoupe();
    if (drag || erasing) return;
    brush = null;
    updateBrush();
  });
  window.addEventListener("scroll", hideLoupe, { passive: true });
  window.addEventListener("blur", hideLoupe);
  window.addEventListener("resize", scheduleAlignPlots);
  if (document.fonts?.ready) document.fonts.ready.then(scheduleAlignPlots);

  slider.addEventListener("input", onSlider);
  resetStart.addEventListener("click", clearStart);
  undoButton.addEventListener("click", undo);
  resetEdits.addEventListener("click", restoreOriginal);
  saveButton.addEventListener("click", saveCurves);
  exportButton.addEventListener("click", exportCurrent);
  newCurveButton.addEventListener("click", startNewCurve);
  finishCurveButton.addEventListener("click", finishDrawing);
  document.querySelectorAll(".chart-tool[data-tool]").forEach((button) => {
    button.addEventListener("click", () => setTool(button.dataset.tool));
  });
  if (plotConfirm) plotConfirm.addEventListener("click", drawCleanPlot);
  [plotXInput, plotYInput, plotXFactor, plotXOffset, plotYFactor, plotYOffset].forEach((input) => {
    if (!input) return;
    input.addEventListener("input", () => {
      axisDirty = true;
      plotStale = true;
      plotConfirm?.classList.add("attention");
      if (plotStatus) {
        plotStatus.textContent = "轴设置已修改，请确认绘制并保存。";
        plotStatus.dataset.tone = "warn";
      }
      refreshActions();
    });
  });

  setStatus("未修改");
  stage.dataset.tool = "start";
  updateHintForTool();
  render();
  window.requestAnimationFrame(scheduleAlignPlots);
})();
