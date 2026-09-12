# ML Atlas · 机器学习手记

基于仓库中 185 份已经执行验证的 Jupyter Notebook 构建的静态学习网站。包含课程总览、20 个章节、学习笔记搜索、公式排版、代码高亮与复制、长代码展开、运行结果与图形放大。

## 本地开发

需要 Node.js 22 或更高版本：

```bash
npm ci
npm run build
npm run check
npm run dev
```

打开终端打印的 `/MachineLearningNotebooks/` 地址。修改 `src/` 后重新执行 `npm run build` 并刷新页面。默认从上一级仓库目录读取 Notebook，也支持原始工作区中的 `../机器学习_Notebooks`。其他位置可通过 `NOTEBOOKS_DIR` 指定。

## 内容与阅读方式

- 所有课程与章节由现有 Notebook 生成，保留代码、公式、表格及保存的执行输出。
- 数学公式在构建时由 MathJax 转成 SVG，网页阅读不需要加载外部公式服务。
- 代码使用 highlight.js 高亮，保持原始缩进；复制按钮复制原始 Python 内容。
- 原文参考图与当前执行结果有不同标记。预处理兼容性提示折叠展示。
- 三个 Plotly 图形按需加载，支持原有交互。
- 网页展示保存的结果，不提供 Python 内核。需运行时，按仓库根目录 README 准备完整数据与环境。
- 首页按算法名称、章节、简介和代码关键词检索，英文算法名也可搜索。
- 对线性回归原文“优化后性能提升”的结论，网页补充了与实际保存指标一致的阅读提示；原始 Notebook 未修改。

## GitHub Pages

网站源代码位于仓库 `website/`，自动发布配置为 `.github/workflows/pages.yml`。

工作流在 GitHub Actions 中读取仓库现有的 Notebook 和附件，构建 `website/dist/`，完成内容完整性检查后发布至 GitHub Pages。图片和内容文件在构建时生成，不重复提交到 Git 仓库。

使用 hash 路由，课程与章节链接在 GitHub Pages 的项目子路径下均可直接打开或刷新。
