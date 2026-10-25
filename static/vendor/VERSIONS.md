# 本地静态资源

这些文件随项目提供，页面运行不依赖 CDN。版本固定如下：

| 资源 | 版本 | 来源 | 许可证 |
| --- | --- | --- | --- |
| Bootstrap CSS / bundle JS | 5.3.8 | https://cdn.jsdelivr.net/npm/bootstrap@5.3.8/ | MIT，见 `bootstrap/LICENSE` |
| Font Awesome Free | 6.7.2 | https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.7.2/ | 图标 CC BY 4.0、字体 SIL OFL 1.1、代码 MIT，见 `fontawesome/LICENSE.txt` |

页面只使用 Font Awesome 的 solid 图标，因此只保留 `fa-solid-900.woff2`。
Bootstrap bundle 已包含 Popper，页面不使用 jQuery，也没有 Node 构建步骤。
