(() => {
  "use strict";

  const formatBytes = (bytes) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / (1024 ** index);
    return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
  };

  const autoSubmit = () => {
    document.querySelectorAll("[data-auto-submit]").forEach((control) => {
      control.addEventListener("change", () => {
        const form = control.closest("form");
        if (!form) return;
        if (typeof form.requestSubmit === "function") {
          form.requestSubmit();
        } else {
          form.submit();
        }
      });
    });
  };

  const initNewRun = () => {
    const form = document.getElementById("new-run-form");
    if (!form) return;

    const pdfInput = document.getElementById("pdfs");
    const fileList = document.getElementById("pdf-file-list");
    const wantInput = document.getElementById("want");
    const pagesInput = document.getElementById("pages");
    const dpiInput = document.getElementById("dpi");
    const summaryFiles = document.getElementById("summary-files");
    const summaryMode = document.getElementById("summary-mode");
    const summaryDpi = document.getElementById("summary-dpi");
    const summaryWant = document.getElementById("summary-want");
    const summaryPages = document.getElementById("summary-pages");
    const submitButton = document.getElementById("submit-run");
    const infoError = document.getElementById("pdf-info-error");
    let infoAbort = null;
    let infoToken = 0;

    const mode = () => {
      const checked = form.querySelector('input[name="mode"]:checked');
      return checked ? checked.value : "auto";
    };

    const syncControls = () => {
      const isAuto = mode() === "auto";
      form.querySelectorAll("[data-auto-only]").forEach((element) => {
        if ("disabled" in element) {
          element.disabled = !isAuto;
        }
      });
    };

    const setPageText = (index, text, title) => {
      if (!fileList) return;
      const pages = fileList.querySelector(
        `[data-file-index="${index}"] .file-pages`);
      if (!pages) return;
      pages.textContent = text;
      pages.title = title || "";
    };

    const loadPageCounts = (files) => {
      if (!fileList) return;
      if (!files.length) {
        if (infoError) infoError.textContent = "";
        return;
      }
      const url = fileList.dataset.infoUrl;
      if (!url) {
        files.forEach((file, index) => setPageText(index, "页数未读取"));
        return;
      }
      if (infoAbort) infoAbort.abort();
      infoAbort = new AbortController();
      const token = ++infoToken;
      const body = new FormData();
      files.forEach((file) => body.append("pdfs", file, file.name));
      fetch(url, { method: "POST", body, signal: infoAbort.signal })
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then((data) => {
          if (token !== infoToken) return;
          const entries = Array.isArray(data.files) ? data.files : [];
          const messages = [];
          entries.forEach((info, index) => {
            if (info.error) {
              setPageText(index, "页数未读取", info.error);
              messages.push(`${info.name || "文件"}：${info.error}`);
            } else {
              setPageText(index, `${info.pages} 页`);
            }
          });
          files.forEach((file, index) => {
            if (index >= entries.length) setPageText(index, "页数未读取");
          });
          if (infoError) infoError.textContent = messages.join("；");
        })
        .catch((error) => {
          if (error.name === "AbortError" || token !== infoToken) return;
          files.forEach((file, index) => setPageText(index, "页数未读取"));
          if (infoError) infoError.textContent = "页数读取失败，提交时服务器仍会校验。";
        });
    };

    const renderFiles = () => {
      if (!fileList || !pdfInput) return;
      const files = Array.from(pdfInput.files || []);
      fileList.replaceChildren();
      if (infoError) infoError.textContent = "";
      if (!files.length) {
        const empty = document.createElement("p");
        empty.className = "file-empty";
        empty.textContent = "选择后显示文件名、页数和大小。";
        fileList.appendChild(empty);
        return;
      }
      files.forEach((file, index) => {
        const row = document.createElement("div");
        row.className = "file-item";
        row.dataset.fileIndex = String(index);

        const icon = document.createElement("i");
        icon.className = "fa-solid fa-file-pdf";
        icon.setAttribute("aria-hidden", "true");

        const name = document.createElement("span");
        name.className = "file-name";
        name.textContent = file.name;

        const pages = document.createElement("span");
        pages.className = "file-pages";
        pages.textContent = "读取页数…";

        const size = document.createElement("span");
        size.className = "file-size";
        size.textContent = formatBytes(file.size);

        row.append(icon, name, pages, size);
        fileList.appendChild(row);
      });
      loadPageCounts(files);
    };

    const updateSummary = () => {
      const files = Array.from(pdfInput ? pdfInput.files || [] : []);
      const totalSize = files.reduce((sum, file) => sum + file.size, 0);
      if (summaryFiles) {
        summaryFiles.textContent = files.length
          ? `${files.length} 篇 PDF · ${formatBytes(totalSize)}`
          : "尚未选择";
      }
      if (summaryMode) {
        summaryMode.textContent = mode() === "auto" ? "自动提取" : "仅分析，稍后确认坐标";
      }
      if (summaryDpi) {
        summaryDpi.textContent = `${dpiInput ? dpiInput.value : "300"} DPI`;
      }
      if (summaryWant) {
        const text = wantInput ? wantInput.value.trim() : "";
        summaryWant.textContent = text || "全部数据图";
      }
      if (summaryPages) {
        summaryPages.textContent = pagesInput && pagesInput.value.trim()
          ? pagesInput.value.trim() : "全部";
      }
    };

    form.addEventListener("change", (event) => {
      syncControls();
      if (event.target === pdfInput) renderFiles();
      updateSummary();
    });
    form.addEventListener("input", (event) => {
      if (event.target === wantInput || event.target === pagesInput) {
        updateSummary();
      }
    });
    form.addEventListener("submit", () => {
      if (!submitButton) return;
      window.setTimeout(() => {
        submitButton.disabled = true;
        submitButton.setAttribute("aria-busy", "true");
        submitButton.textContent = "正在提交…";
      }, 0);
    });

    syncControls();
    renderFiles();
    updateSummary();
  };

  const initRunPolling = () => {
    const log = document.getElementById("run-log");
    if (!log || !log.dataset.statusUrl) return;

    const statusUrl = log.dataset.statusUrl;
    const statusLabel = document.querySelector("[data-status-label]");
    const elapsed = document.getElementById("run-elapsed");
    const errorBox = document.getElementById("log-error");
    let stopped = false;
    let reloading = false;
    let failures = 0;
    let timer = null;

    const schedule = (delay) => {
      if (stopped) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(poll, delay);
    };

    const renderLog = (text) => {
      const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 48;
      log.textContent = typeof text === "string" && text ? text : "等待处理程序输出…";
      if (nearBottom) log.scrollTop = log.scrollHeight;
    };

    const poll = async () => {
      if (stopped) return;
      if (document.hidden) {
        schedule(1500);
        return;
      }
      try {
        const response = await fetch(statusUrl, {
          cache: "no-store",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        failures = 0;
        if (errorBox) errorBox.classList.add("d-none");
        renderLog(data.log);
        if (statusLabel && data.status_label) statusLabel.textContent = data.status_label;
        if (elapsed && data.elapsed_display) elapsed.textContent = data.elapsed_display;
        if (data.finished) {
          stopped = true;
          window.clearTimeout(timer);
          if (!reloading) {
            reloading = true;
            window.location.reload();
          }
          return;
        }
        schedule(1500);
      } catch (error) {
        failures += 1;
        if (errorBox) {
          errorBox.textContent = "状态更新暂时失败，稍后自动重试。";
          errorBox.classList.remove("d-none");
        }
        schedule(Math.min(10000, 1500 * (2 ** Math.min(failures, 3))));
      }
    };

    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && !stopped) {
        window.clearTimeout(timer);
        poll();
      }
    });
    window.addEventListener("pagehide", () => {
      stopped = true;
      window.clearTimeout(timer);
    });
    schedule(1200);
  };

  const initBusyLinks = () => {
    document.querySelectorAll("a[data-busy]").forEach((link) => {
      link.addEventListener("click", () => {
        if (link.getAttribute("aria-disabled") === "true") return;
        link.setAttribute("aria-disabled", "true");
        link.classList.add("disabled");
        const icon = link.querySelector("i");
        if (icon) icon.className = "fa-solid fa-spinner fa-spin";
        const label = link.querySelector("span");
        if (label) label.textContent = "正在打包…";
      });
    });
  };

  const init = () => {
    autoSubmit();
    initNewRun();
    initRunPolling();
    initBusyLinks();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.addEventListener("pageshow", (event) => {
    if (event.persisted) window.location.reload();
  });
})();
