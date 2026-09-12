"""从相邻的原始 Markdown 生成 Notebook；已执行且源单元未变化的文件不会覆盖。
运行：python tools/build_notebooks.py
"""
from pathlib import Path
import re, ast, json, shutil, hashlib, textwrap
import nbformat
from overrides import OVERRIDES, NOTES
from prose_repairs import repair_prose

OUT = Path(__file__).resolve().parents[1]
SOURCE = OUT.parent
FENCE = re.compile(r'^```([^\n]*)\n(.*?)^```[^\n]*$', re.M | re.S)
EXCLUDED = []
RECORDS = []
BOOTSTRAP = '''# 从当前目录向上寻找本套 Notebook 的根目录，移动整个文件夹后仍可运行。
from pathlib import Path
import os, sys, tempfile, random
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "notebook_support.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("请从这套 Notebook 文件夹内启动 Jupyter，并保留 notebook_support.py。")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ml_notebooks_matplotlib"))
from notebook_support import DATA_DIR, load_local_housing, load_local_boston, load_local_mnist, load_creditcard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
random.seed(42)
np.random.seed(42)
# 按实际安装的字体选择，兼容 macOS / Windows / Linux。
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
CJK_FONTS = [f for f in ["Arial Unicode MS", "PingFang SC", "Heiti TC", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
             if f in _available_fonts] + ["DejaVu Sans"]
plt.rcParams.update({"font.sans-serif": CJK_FONTS, "axes.unicode_minus": False,
                     "figure.dpi": 90, "figure.max_open_warning": 30})
%matplotlib inline
'''


def fix_math(text):
    text = text.replace('\\*\\*', '**')
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: '\n\n$$\n' + m[1].strip() + '\n$$\n\n', text, flags=re.S)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: '$' + m[1].strip() + '$', text, flags=re.S)
    # 同一行上的 $a$$b$ 是两个相邻行内公式，不是 display math 的开头。
    for _ in range(10):
        previous = text
        text = re.sub(r'(?<!\$)\$([^$\n]+)\$\$(?=[^$\n]+\$)', lambda m: '$'+m[1].strip()+'$；$', text)
        if previous == text: break
    def formula(m):
        body = m[1].strip()
        return '\n\n$$\n' + body + '\n$$\n\n'
    text = re.sub(r'\$\$(.*?)\$\$', formula, text, flags=re.S)
    text = re.sub(r'(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)', lambda m: '$' + m[1].strip() + '$', text)
    return re.sub(r'\n{4,}', '\n\n\n', text).strip()


def common_code(code, key, index, changes):
    original = code
    if (key, index) in OVERRIDES:
        code = OVERRIDES[(key, index)]
    # 旧接口、字体以及 Notebook 内联显示。
    code = re.sub(r'^(\s*)plt\.rcParams\[[\'"]font\.(?:sans-serif|family)[\'"]\]\s*=.*$', r"\1plt.rcParams['font.sans-serif'] = CJK_FONTS", code, flags=re.M)
    code = re.sub(r"font(?:name|family)\s*=\s*['\"](?:SimHei|Microsoft YaHei|Arial Unicode MS)['\"]", 'fontfamily=CJK_FONTS[0]', code)
    code = code.replace("'seaborn-darkgrid'", "'seaborn-v0_8-darkgrid'")
    code = re.sub(r'\bbase_estimator\b', 'estimator', code).replace('base_estimator__', 'estimator__')
    code = re.sub(r'\bsparse\s*=\s*False', 'sparse_output=False', code)
    code = code.replace('.get_feature_names(', '.get_feature_names_out(')
    code = code.replace('shade=True', 'fill=True').replace('use_line_collection=True,', '').replace(', use_line_collection=True', '')
    code = re.sub(r',?\s*algorithm=[\'"]SAMME.R[\'"]', '', code)
    code = re.sub(r'use_label_encoder=False,\s*', '', code)
    code = code.replace("loss='log'", "loss='log_loss'")
    code = re.sub(r"(SGDRegressor\([^\n]*?)loss=['\"]squared_loss['\"]", r"\1loss='squared_error'", code)
    code = code.replace('affinity=\'euclidean\'', "metric='euclidean'")
    code = code.replace('sm.families.links.log()', 'sm.families.links.Log()')
    code = re.sub(r'\bn_jobs\s*=\s*-1', 'n_jobs=1', code)
    code = code.replace('"n_jobs": -1', '"n_jobs": 1')
    code = re.sub(r'^.*warnings\.filterwarnings\([\'"]ignore[\'"].*\n?', '', code, flags=re.M)
    code = re.sub(r'^.*ssl\._create_default_https_context\s*=.*\n?', '', code, flags=re.M)
    # 本地 CSV 读取，所有路径均指向独立数据目录。
    code = re.sub(r"pd\.read_csv\(['\"]creditcard\.csv['\"]\)", 'load_creditcard()', code)
    code = re.sub(r'^from sklearn.datasets import fetch_california_housing\s*$', 'from notebook_support import load_local_housing', code, flags=re.M)
    code = code.replace('fetch_california_housing(', 'load_local_housing(')
    code = re.sub(r'^from sklearn.datasets import load_boston\s*$', 'from notebook_support import load_local_boston', code, flags=re.M)
    code = code.replace('load_boston(', 'load_local_boston(')
    code = re.sub(r'^from sklearn.datasets import fetch_openml\s*$', 'from notebook_support import load_local_mnist', code, flags=re.M)
    code = code.replace("fetch_openml('mnist_784', version=1)", 'load_local_mnist()')
    code = code.replace("'stock_prices.csv'", "'stock_data.csv'")
    code = re.sub(r"pd\.read_csv\((['\"])([^'\"]+\.(?:csv|tsv))\1", lambda m: "pd.read_csv(DATA_DIR / " + repr(m[2].removeprefix('./')) , code)
    if key == '2/10':
        code = code.replace("DATA_DIR / 'train.csv'", "DATA_DIR / 'house-prices-advanced-regression-techniques/train.csv'")
        code = code.replace("DATA_DIR / 'test.csv'", "DATA_DIR / 'house-prices-advanced-regression-techniques/test.csv'")
    code = re.sub(r"((?:train_path|test_path|image_path|url)\s*=\s*)(['\"])([^'\"]+\.(?:csv|tsv|jpg))\2", lambda m: m[1]+'DATA_DIR / '+repr(m[3].removeprefix('./')), code)
    # 保存模型/预测结果至输出文件夹中，而不弹出外部应用。
    code = code.replace("output.to_csv('submission.csv',", "output.to_csv(ROOT / 'reports' / 'lightgbm_submission.csv',")
    code = code.replace('joblib.dump(xgb_clf_best, "xgb_model_optimized.pkl")', 'joblib.dump(xgb_clf_best, ROOT / "reports" / "xgb_model_optimized.pkl")')
    # 大网格保留待选参数含义，每维默认最多两个候选值。
    if 'param_grid' in code:
        try:
            tree = ast.parse(code)
            lines = code.splitlines(keepends=True)
            edits = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Dict):
                    parent_assign = next((a for a in ast.walk(tree) if isinstance(a, ast.Assign) and a.value is n and any(isinstance(t, ast.Name) and 'param_grid' in t.id for t in a.targets)), None)
                    if parent_assign:
                        for value in n.values:
                            if isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) > 2:
                                if value.lineno == value.end_lineno:
                                    vals = value.elts[:2]
                                    repl = '[' + ', '.join(ast.get_source_segment(code, v) for v in vals) + ']'
                                    edits.append((value.lineno - 1, value.col_offset, value.end_col_offset, repl))
            for line, start, end, repl in sorted(edits, reverse=True):
                # 当前行前缀为 ASCII 参数名，字节偏移与字符偏移一致。
                raw = lines[line].encode(); lines[line] = (raw[:start] + repl.encode() + raw[end:]).decode()
            if edits:
                code = ''.join(lines)
                changes.add('大规模参数网格缩为每维至多两个候选值；所有候选仍真实训练，未跳过运行。')
        except SyntaxError:
            pass
    if key in ['1/07', '1/08', '6/03']:
        code = re.sub(r"(['\"]max_features['\"]\s*:\s*\[[^\]]*?)['\"]auto['\"]", r'\1None', code)
    if key == '1/13':
        code = code.replace("pd.read_csv(DATA_DIR / 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')", "pd.read_csv(DATA_DIR / 'AmesHousing.csv')[['Gr Liv Area', 'Overall Qual', 'House Style', 'Central Air', 'SalePrice']].rename(columns={'Gr Liv Area': 'GrLivArea', 'Overall Qual': 'OverallQual', 'House Style': 'HouseStyle', 'Central Air': 'CentralAir'})")
        for a,b in [('charges','SalePrice'),('smoker','CentralAir'),('sex','HouseStyle'),('age','GrLivArea')]:
            code=re.sub(r'([\'"])'+a+r'\1', lambda m: m[1]+b+m[1],code)
        code=code.replace('Medical Charges by Sex and Smoking Status','Ames House Prices by House Style and Central Air').replace('年龄 vs 医疗费用（含吸烟者标记）','居住面积与房价').replace('Charges','SalePrice').replace("'Age'", "'GrLivArea'")
        changes.add('本地没有原文的 insurance.csv；Gamma GLM 改为 AmesHousing 的正值 SalePrice，使用居住面积、质量、房屋类型与中央空调作为特征。')
    if key == '2/04':
        code=code.replace('import graphviz','import matplotlib.pyplot as plt')
        start=code.index('def visualize_tree(');end=code.index('# 6. 案例演示',start)
        code=code[:start]+'''def visualize_tree(node, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(14, 7))
    def draw(current, x, y, dx):
        if current.value is not None:
            label = str(class_names[current.value])
        else:
            label = f"{feature_names[current.feature_index]} <= {current.threshold:.2f}"
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='gray'))
        if current.value is None:
            for child, sign in [(current.left, -1), (current.right, 1)]:
                nx, ny = x + sign * dx, y - 1
                ax.plot([x, nx], [y, ny], color='gray')
                draw(child, nx, ny, dx / 2)
    draw(node, 0, 0, 4)
    ax.axis('off'); plt.tight_layout(); plt.show()

''' + code[end:]
        code=code.replace('    dot = visualize_tree(tree, feature_names, class_names)\n    dot.render("c45_tree", view=True)', '    visualize_tree(tree, feature_names, class_names)')
        changes.add('保留增益率决策树教学实现；树图使用 Matplotlib 内联绘制，不需要系统 Graphviz。此简化实现使用预剪枝，不实现完整 C4.5 的缺失值分配及后剪枝。')
    if key=='5/06':
        code=code.replace('import pymc3 as pm','import pymc as pm\nimport arviz as az')
        code=code.replace("trace = pm.sample(2000, chains=1, cores=1, return_inferencedata=False)", "idata = pm.sample(1000, tune=1000, chains=2, cores=1, random_seed=42, target_accept=0.9, progressbar=False)\n    trace = {name: idata.posterior[name].values.ravel() for name in ['mu_a', 'mu_b']}")
        code=code.replace('pm.traceplot(trace)', "az.plot_trace(idata, var_names=['mu_a', 'mu_b'])\nprint(az.summary(idata, var_names=['mu_a', 'mu_b']))")
        code=code.replace('bf_10', 'posterior_prob').replace('计算贝叶斯因子', '计算后验概率').replace('Bayes Factor (BF10)', 'Posterior probability P(mu_A > mu_B | data)')
        changes.add('迁移至 PyMC 5，以两条链检查采样；P(μA > μB | 数据) 是后验概率，不能称为贝叶斯因子。')
    if key=='7/08':
        code=code.replace('from openTSNE import TSNE', 'from sklearn.manifold import TSNE').replace('"n_iter": 1000', '"max_iter": 750').replace('tsne.fit(X_scaled)','tsne.fit_transform(X_scaled)').replace('openTSNE', 'scikit-learn')
    if key=='7/11': code=code.replace("TSNE(n_components=2, metric='cosine', random_state=42)", "TSNE(n_components=2, perplexity=3, metric='cosine', random_state=42)")
    if key=='10/02':
        code=re.sub(r'^import torch.*\n|^import torch\.nn.*\n|^import torch\.optim.*\n','',code,flags=re.M)
        code=code.replace('torch.zeros(', 'np.zeros(').replace('torch.argmax(', 'np.argmax(')
    if key in ['9/04','9/06']:
        code=re.sub(r'image_url = .*', "image_path = DATA_DIR / 'lenna.jpg'", code)
        code=code.replace('io.imread(image_url)', 'io.imread(image_path)')
        code=re.sub(r"io.imread\(['\"]https://[^'\"]+['\"]\)", "io.imread(DATA_DIR / 'lenna.jpg')",code)
        changes.add('网络图片改为本地 lenna.jpg，用于像素聚类演示。')
    if key=='18/03' and index==2:
        code='data = df.to_dict(orient="list")\n# 上面完整案例中的六个特征，不再用省略号占位。'
    # 原文已经提供完整代码时，后面的省略式复述也补成可执行语句。
    if '...' in code:
        code=code.replace('train_test_split(...)', 'train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)')
        code=re.sub(r',\s*\.\.\.', '', code)
    if key=='3/01' and index==1:
        return None
    # 确保绘图主题不覆盖中文字体；模型搜索中不接受静默的 NaN 分数。
    code = re.sub(r'^(\s*)sns\.(?:set|set_style|set_theme)\([^\n]*\)\s*$', lambda m: m[0].rstrip() + "\n" + m[1] + "plt.rcParams['font.sans-serif'] = CJK_FONTS", code, flags=re.M)
    try:
        parsed=ast.parse(code)
        raw=code.encode(); starts=[0]
        for line in code.splitlines(keepends=True): starts.append(starts[-1]+len(line.encode()))
        edits=[]
        for call in ast.walk(parsed):
            if isinstance(call,ast.Call):
                name=call.func.id if isinstance(call.func,ast.Name) else call.func.attr if isinstance(call.func,ast.Attribute) else ''
                kwargs={kw.arg for kw in call.keywords}
                additions=[]
                if name in ['GridSearchCV','BayesSearchCV'] and 'error_score' not in kwargs:
                    additions.append("error_score='raise'")
                if name in ['XGBRegressor','XGBClassifier','LGBMRegressor','LGBMClassifier'] and 'n_jobs' not in kwargs:
                    additions.append('n_jobs=1')
                if name in ['LGBMRegressor','LGBMClassifier'] and 'verbosity' not in kwargs:
                    additions.append('verbosity=-1')
                if additions:
                    at=starts[call.end_lineno-1]+call.end_col_offset-1
                    previous=raw[:at].rstrip()[-1:]
                    prefix='' if previous in [b'(',b','] else ', '
                    edits.append((at,(prefix+', '.join(additions)).encode()))
        for at,repl in sorted(edits,reverse=True):raw=raw[:at]+repl+raw[at:]
        code=raw.decode()
    except SyntaxError:
        pass
    if original != code:
        changes.add('修复或适配代码；原讲解性输出保留为 Markdown，Python 单元按顺序执行。')
    return code.strip()


def split_code(code):
    # 长代码只在完整图形 show 后分段，避免内联后端提前关闭正在绘制的图。
    if len(code.splitlines()) < 90: return [code]
    tree=ast.parse(code); lines=code.splitlines(); cuts=[]; start=0
    for node in tree.body:
        complete_plot = (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'show')
        if complete_plot and node.end_lineno-start>=50:
            cuts.append('\n'.join(lines[start:node.end_lineno]).strip());start=node.end_lineno
    if start<len(lines): cuts.append('\n'.join(lines[start:]).strip())
    return [c for c in cuts if c]


def build():
    for path in sorted(SOURCE.glob('[0-9]*/*.md'), key=lambda p:(int(p.parent.name.split('.')[0]),p.name)):
        relative=path.relative_to(SOURCE)
        if path.parent.name.startswith('21.') or 'LSTM' in path.name or '多层感知机分类器' in path.name:
            EXCLUDED.append(str(relative));continue
        key=str(int(path.parent.name.split('.')[0]))+'/'+path.name.split('.')[0]
        dest=OUT/relative.with_suffix('.ipynb');dest.parent.mkdir(exist_ok=True)
        raw=path.read_text();changes=set(NOTES.get(key,[]));cells=[];pos=0;python_index=0
        for m in FENCE.finditer(raw):
            prose=raw[pos:m.start()].strip()
            if prose: cells.append(nbformat.v4.new_markdown_cell(fix_math(prose)))
            lang=m[1].strip().lower(); code=m[2].rstrip()
            if lang in ['python','py','python3']:
                transformed=common_code(code,key,python_index,changes)
                if transformed is None:
                    cells.append(nbformat.v4.new_markdown_cell('原文参考输出（实际结果以上方运行输出为准）：\n\n```text\n'+code+'\n```'))
                else:
                    try:
                        chunks=split_code(transformed)
                        for c in chunks:
                            ast.parse(c)
                            cells.append(nbformat.v4.new_code_cell(c,metadata={'source_block':python_index}))
                    except SyntaxError as e:
                        raise RuntimeError(f'{relative} Python block {python_index}: {e}') from e
                python_index+=1
            else:
                cells.append(nbformat.v4.new_markdown_cell('原文参考输出或命令说明（无需在 Python 内核执行）：\n\n```text\n'+code+'\n```'))
            pos=m.end()
        if raw[pos:].strip():cells.append(nbformat.v4.new_markdown_cell(fix_math(raw[pos:])))
        cells = repair_prose(cells, key)
        # 同步本地图片，只迁移本笔记实际引用的附件。
        data_deps=set()
        for cell in cells:
            if cell.cell_type=='markdown':
                for img in re.findall(r'!\[[^\]]*\]\((attachments/[^)]+)\)',cell.source):
                    source=path.parent/img
                    if not source.is_file():raise FileNotFoundError(source)
                    target=dest.parent/img;target.parent.mkdir(exist_ok=True)
                    if not target.exists():shutil.copy2(source,target)
                # 原文生成图不等于当前 Notebook 实际结果，明确标注。
                cell.source=re.sub(r'!\[\]\((attachments/[^)]+)\)', r'![原文参考图，当前结果见代码输出](\1)',cell.source)
            else:
                s=cell.source
                data_deps.update(re.findall(r"DATA_DIR / ['\"]([^'\"]+)['\"]",s))
                if 'load_local_housing(' in s:
                    data_deps.add('housing.csv');changes.add('从本地 housing.csv 构造加州房价特征，删除缺失行，固定抽样 2500 行；load_local_housing(sample_size=None) 可运行全量有效数据。')
                if 'load_local_mnist(' in s:
                    data_deps.update(['MNIST/raw/train-images-idx3-ubyte.gz','MNIST/raw/train-labels-idx1-ubyte.gz']);changes.add('读取本地 MNIST IDX 文件，固定分层抽取 2000 张图像做传统降维；可调整 sample_size。')
                if 'load_creditcard(' in s:
                    data_deps.add('creditcard.csv');changes.add('信用卡数据默认固定分层抽样 30000 行并保持原类别比例；load_creditcard(sample_size=None) 可读全量数据。')
                if 'load_local_boston(' in s:data_deps.add('BostonHousing.csv')
        intro='## Notebook 使用说明\n\n建议先阅读理论，再从上到下运行全部单元；使用 Python 3.11–3.13 内核。数据与辅助模块均来自本文件夹，安装依赖后无需联网获取案例数据。\n\n'
        if data_deps:intro+='本篇数据：'+ '、'.join('`dataset/'+d+'`' for d in sorted(data_deps))+'。\n\n'
        else:intro+='本篇使用代码生成的模拟数据或 scikit-learn 随包附带的小数据集，无需外部数据文件。\n\n'
        if changes:intro+='转写调整：\n\n'+'\n'.join('- '+c for c in sorted(changes))+'\n\n'
        if key in ['2/09','2/10']:
            intro+='原文件位于“分类算法”目录，但案例实际是房价回归；本篇保留原目录和回归案例，评估使用回归指标。\n\n'
        intro+='原文参考图及静态数值仅用于对照；重跑后的代码输出为本次实验结果。依赖版本与验证记录见根目录 `README.md` 和 `reports/`。'
        # 标题和运行说明在前，理论及案例的顺序保持不变。
        title=raw.splitlines()[0]
        if cells and cells[0].cell_type=='markdown' and cells[0].source.startswith(title):
            cells[0].source=cells[0].source[len(title):].strip()
        cells=[nbformat.v4.new_markdown_cell(title+'\n\n'+intro),nbformat.v4.new_code_cell(BOOTSTRAP,metadata={'tags':['setup']})]+[c for c in cells if c.source.strip()]
        nb=nbformat.v4.new_notebook(cells=cells,metadata={
            'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
            'language_info':{'name':'python','version':'3.13.9'},
            'conversion':{'source':str(relative),'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'changes':sorted(changes),'data_files':sorted(data_deps)}
        })
        signature=hashlib.sha256(json.dumps([(c.cell_type,c.source) for c in cells],ensure_ascii=False).encode()).hexdigest()
        nb.metadata.conversion['cell_sha256']=signature
        write=True
        if dest.exists():
            previous=nbformat.read(dest,as_version=4)
            if previous.metadata.get('conversion',{}).get('cell_sha256')==signature:write=False
        if write:nbformat.write(nb,dest)
        RECORDS.append({'source':str(relative),'notebook':str(dest.relative_to(OUT)),'python_blocks':python_index,'code_cells':sum(c.cell_type=='code' for c in cells),'data':sorted(data_deps),'changes':sorted(changes),'updated':write})
    (OUT/'reports'/'conversion_manifest.json').write_text(json.dumps({'included':RECORDS,'excluded':EXCLUDED},ensure_ascii=False,indent=2))
    print('Notebooks:',len(RECORDS),'excluded:',len(EXCLUDED),'updated:',sum(r['updated'] for r in RECORDS))

if __name__=='__main__':build()
