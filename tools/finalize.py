"""汇总执行记录、数据依赖、文件结构和公式检查，生成学习目录与最终报告。"""
from pathlib import Path
from urllib.parse import quote, unquote
import ast, json, re, hashlib, collections, platform
import nbformat
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'reports/conversion_manifest.json').read_text())
paths=sorted(ROOT.glob('[0-9]*/*.ipynb'),key=lambda p:(int(p.parent.name.split('.')[0]),p.name))
issues=[];results=[];image_refs=set();used_data=set();code_count=0;figure_count=0;warning_counts=collections.Counter()
for path in paths:
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    rel=str(path.relative_to(ROOT));meta=nb.metadata.conversion
    sig=hashlib.sha256(json.dumps([(c.cell_type,c.source) for c in nb.cells],ensure_ascii=False).encode()).hexdigest()
    validation=nb.metadata.get('validation',{})
    if validation.get('status')!='passed' or validation.get('signature')!=sig or sig!=meta.cell_sha256:
        issues.append(rel+': missing or outdated execution')
    code_cells=[c for c in nb.cells if c.cell_type=='code'];code_count+=len(code_cells)
    for cell in code_cells:
        if cell.execution_count is None:issues.append(rel+': unexecuted cell')
        if any(o.output_type=='error' for o in cell.outputs):issues.append(rel+': error output')
        code='\n'.join(l for l in cell.source.splitlines() if not l.startswith('%'))
        tree=ast.parse(code)
        if any(isinstance(n,ast.Constant) and n.value is Ellipsis for n in ast.walk(tree)):issues.append(rel+': executable ellipsis')
        for o in cell.outputs:
            if any(k.startswith('image/') or k=='application/vnd.plotly.v1+json' for k in o.get('data',{})):figure_count+=1
            if o.get('name')=='stderr':
                for name in re.findall(r'\b(\w+Warning):',o.get('text','')):warning_counts[name]+=1
    for cell in nb.cells:
        if cell.cell_type=='markdown':
            for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',cell.source):
                if not target.startswith('http'):
                    p=(path.parent/unquote(target)).resolve()
                    if not p.is_file():issues.append(rel+': missing image '+target)
                    image_refs.add(p)
    for data in meta.data_files:
        if not (ROOT/'dataset'/data).is_file():issues.append(rel+': missing data '+data)
        used_data.add(data)
    results.append({'notebook':rel,'status':validation.get('status'),'code_cells':len(code_cells),
                    'figures':validation.get('figures',0),'seconds':validation.get('seconds'),
                    'source':meta.source,'data':list(meta.data_files),'changes':list(meta.changes)})
# 清除本次生成过程中留下但最终未引用的附件；不接触原笔记。
removed=[]
for path in ROOT.glob('[0-9]*/attachments/*'):
    if path.is_file() and path.resolve() not in image_refs:
        path.unlink();removed.append(str(path.relative_to(ROOT)))
actual_data={str(p.relative_to(ROOT/'dataset')) for p in (ROOT/'dataset').rglob('*') if p.is_file()}
if actual_data!=used_data:issues.append('data set mismatch: '+str(actual_data^used_data))
source_count=len(manifest['included'])+len(manifest['excluded'])
if len(paths)!=len(manifest['included']) or len(paths)!=185 or source_count!=201:issues.append('coverage mismatch')
math=json.loads((ROOT/'reports/math_rendering.json').read_text())
if math['errors']:issues.append('MathJax errors')
if json.loads((ROOT/'reports/math_delimiter_issues.json').read_text()):issues.append('math delimiter issues')
size=sum(p.stat().st_size for p in (ROOT/'dataset').rglob('*') if p.is_file())
summary={'notebooks':len(paths),'source_markdown':source_count,'excluded':len(manifest['excluded']),
         'code_cells':code_count,'figures':figure_count,'formulas':math['total'],
         'data_files':len(actual_data),'data_bytes':size,'image_attachments':len(image_refs),
         'warning_counts':dict(warning_counts),'issues':issues,'results':results}
(ROOT/'reports/validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
if issues:
    print(json.dumps(issues,ensure_ascii=False,indent=2));raise SystemExit(1)
readme=f'''# 机器学习学习笔记 · Jupyter Notebook

由原目录的 201 份 Markdown 转写而来，共 **185 份 Notebook**，按原来的 20 个章节存放。每份均保留理论、公式与可运行案例，附有实际执行输出。

## 从 GitHub 下载完整内容

本仓库包含 185 份 Notebook、必要数据、图片附件、运行依赖和验证报告。`dataset/creditcard.csv` 使用 Git LFS 保存，其余文件直接保存在 Git 仓库中。

先安装 [Git LFS](https://git-lfs.com/)，再执行：

```bash
git lfs install
git clone https://github.com/fxt-gw-pb/MachineLearningNotebooks.git
cd MachineLearningNotebooks
git lfs pull
```

如果此前下载的是 ZIP，`creditcard.csv` 可能只是 LFS 指针；请按上面的方式获取完整数据。数据 SHA-256 校验值见 [数据清单](reports/data_manifest.json)。

## 开始学习

macOS 可双击 **`启动Notebook.command`**，或在终端运行 `./启动Notebook.command`。首次启动会建立本文件夹中的 `.venv` 并联网安装 `requirements.txt` 中的依赖，此后复用该环境。启动器优先使用本机已有 Anaconda Python，也可通过 `NOTEBOOK_PYTHON` 指定 Python 3.11–3.13。

其他系统可在本文件夹中运行：

```bash
python -m venv .venv
# macOS / Linux：source .venv/bin/activate
# Windows PowerShell：.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name python3 --display-name "机器学习笔记"
python -m jupyterlab
```

从下方目录选择一篇笔记，用 **Restart Kernel and Run All Cells** 从头执行。请将本文件夹作为 Jupyter 的工作目录，保留 `notebook_support.py` 和 `dataset/`。Notebook 的图片附件在各章节的 `attachments/` 中。

依赖安装后，全部案例数据均可离线读取；无需原 Markdown 目录或外部模型下载。换电脑时复制整个文件夹，重新建立 Python 环境；不要复制已有 `.venv`。

## 内容范围

已按确认范围排除整个 **21. 神经网络**（14 篇）、**12. 时间序列分析方法/09. LSTM** 和 **2. 分类算法/12. 多层感知机分类器**，共 16 篇。

Adam、网格搜索、数据增强、MDS 等通用内容保留；夹带的神经网络训练或 BERT 示例已换为 NumPy 或传统机器学习案例。高斯过程使用 scikit-learn。每篇开头列出了具体调整。`2. 分类算法` 中原有的 XGBoost、LightGBM 案例实际是房价回归，本次保持原目录并明确说明。

## 数据与实验规模

仅复制实际使用的 **{len(actual_data)} 个数据文件，约 {size/1024/1024:.1f} MiB**，原数据保持不变。House Prices 压缩包仅提取 train.csv 与 test.csv，未复制其他内容。MNIST 仅保留用于传统降维的训练图片与标签 gzip。完整来源、字节数、SHA-256 校验值见 [数据清单](reports/data_manifest.json)。

为便于个人电脑学习，加州房价默认固定抽样 2500 行、信用卡交易分层抽样 30000 行、MNIST 分层抽样 2000 张。`load_local_housing`、`load_creditcard`、`load_local_mnist` 均可设置 `sample_size=None` 使用全部有效数据。较大网格已缩小并在笔记中说明；每个保留候选都实际训练。

加州房价目标单位为十万美元。本地 CSV 缺失卧室数的行被删除，所以结果与 sklearn 在线版本不必完全相同。图片和原文静态输出标为“原文参考”，当前实验以代码输出为准。设置了随机种子，但不同平台或底层库仍可能出现小的数值差异。

## 验证与维护

本次在 Python {platform.python_version()} 上，使用 [固定依赖版本](requirements.txt) 独立启动内核逐份验证。**{len(paths)} / {len(paths)} 份通过，{code_count} 个代码单元全部执行，无错误输出；{math['total']} 个公式片段通过 MathJax 排版检查。** 还检查了图片链接、文件格式、可执行占位符及数据依赖。

- [逐份执行报告](reports/验证报告.md)
- [机器可读汇总](reports/validation_summary.json)
- [转写与排除清单](reports/conversion_manifest.json)
- [公式检查结果](reports/math_rendering.json)
- [迁移后运行验证](reports/portability.json)
- [独立环境依赖解析](reports/dependency_resolution.json)

库弃用提示、模型可识别性提示或统计诊断提示会如实保留；它们不等同于执行失败。实验仅用于学习，性能高低与模型是否适用于当前数据需要通过对应指标判断。

复验全部 Notebook：

```bash
python tools/validate_notebooks.py --workers 3
```

只复验修改或失败的文件：

```bash
python tools/validate_notebooks.py --retry --workers 3
```

`tools/build_notebooks.py` 和 `tools/overrides.py` 保存了可追溯的转写规则，重新生成时需要原 Markdown 仍位于本文件夹的上一级；日常阅读、运行和复验 Notebook 不需要原目录。

接口适配参考：[scikit-learn OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)、[XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)、[PyMC sample](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html)。

## 学习目录

'''
chapter=None
for result in results:
    p=Path(result['notebook'])
    if chapter!=p.parent.name:
        chapter=p.parent.name;readme+='\n### '+chapter+'\n\n'
    readme+='- ['+p.stem+']('+quote(result['notebook'])+')\n'
(ROOT/'README.md').write_text(readme)
report=f'''# Notebook 验证报告

最终验证：{len(paths)} 份全部通过；{code_count} 个代码单元均有执行序号；{figure_count} 个图像或交互图输出；{math['total']} 个公式排版检查通过。

每篇从新内核启动，按单元顺序执行；遇到异常即标记失败，修复后从头复验。网格搜索设置 error_score='raise'，不会把非法参数或失败评分静默当成有效结果。

| Notebook | 状态 | 代码单元 | 图形输出 | 执行秒数 |
|---|---|---:|---:|---:|
'''
for r in results:report+=f"| [{Path(r['notebook']).stem}](../{quote(r['notebook'])}) | 通过 | {r['code_cells']} | {r['figures']} | {r['seconds']} |\n"
report+='\n## 转写中的实质修复\n\n'
for r in results:
    important=[c for c in r['changes'] if not c.startswith(('修复或适配代码','大规模参数网格','从本地 housing','信用卡数据默认','读取本地 MNIST'))]
    if important:report+='- **'+r['notebook']+'**：'+' '.join(important)+'\n'
report+='\n## 保留的提示\n\n'+('、'.join(f'{k}: {v}' for k,v in sorted(warning_counts.items())) or '无警告输出。')+'\n'
(ROOT/'reports/验证报告.md').write_text(report)
print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False,indent=2))
