"""每篇使用独立内核，真实执行全部单元并保留输出；出现异常立即失败。
python tools/validate_notebooks.py --workers 3
支持 --retry（只重跑失败或修改过的笔记）、--only '章/文件片段'。
"""
from pathlib import Path
import os, sys, time, json, argparse, tempfile, traceback, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(tempfile.gettempdir())/'ml_notebooks_validation'
CACHE.mkdir(exist_ok=True)
for key,value in {'MPLCONFIGDIR':str(CACHE/'matplotlib'), 'IPYTHONDIR':str(CACHE/'ipython'),
                  'JUPYTER_RUNTIME_DIR':str(CACHE/'runtime'), 'XDG_CACHE_HOME':str(CACHE/'cache'),
                  'NUMBA_CACHE_DIR':str(CACHE/'numba'), 'PYTENSOR_FLAGS':'base_compiledir='+str(CACHE/'pytensor'),
                  'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                  'VECLIB_MAXIMUM_THREADS':'1','NUMBA_NUM_THREADS':'1','MPLBACKEND':'Agg'}.items():
    os.environ[key]=value
KERNELS=CACHE/'kernels'; (KERNELS/'ml-notebooks').mkdir(parents=True,exist_ok=True)
(KERNELS/'ml-notebooks'/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'ML notebooks validation','language':'python'}))


def run(path, timeout):
    start=time.monotonic();nb=nbformat.read(path,as_version=4)
    result={'notebook':str(path.relative_to(ROOT)), 'signature':nb.metadata.get('conversion',{}).get('cell_sha256'),
            'status':'failed','seconds':0,'code_cells':sum(c.cell_type=='code' for c in nb.cells)}
    try:
        nbformat.validate(nb)
        km=KernelManager(kernel_name='ml-notebooks',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(KERNELS)]))
        client=NotebookClient(nb,km=km,timeout=timeout,startup_timeout=60,allow_errors=False,
                              resources={'metadata':{'path':str(path.parent)}})
        client.execute(cleanup_kc=True)
        for c in nb.cells:
            if c.cell_type=='code':
                assert c.execution_count is not None, 'Unexecuted code cell'
                assert all(o.output_type!='error' for o in c.outputs)
        result['status']='passed'
        result['outputs']=sum(len(c.get('outputs',[])) for c in nb.cells)
        result['figures']=sum(any(k.startswith('image/') or k=='application/vnd.plotly.v1+json' for k in o.get('data',{})) for c in nb.cells for o in c.get('outputs',[]))
    except Exception as e:
        result['error']=str(e)
        result['traceback']=traceback.format_exc()
    result['seconds']=round(time.monotonic()-start,2)
    nb.metadata['validation']={k:v for k,v in result.items() if k not in ['traceback','error']}
    nbformat.write(nb,path)
    report=ROOT/'reports'/'execution'/path.parent.name/(path.stem+'.json')
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k in ['notebook','status','seconds','figures']}),flush=True)
    if result['status']=='failed':
        print('ERROR:',result['error'][-1800:],flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--timeout',type=int,default=240)
    ap.add_argument('--retry',action='store_true');ap.add_argument('--only',default='');args=ap.parse_args()
    paths=sorted(ROOT.glob('[0-9]*/*.ipynb'),key=lambda p:(int(p.parent.name.split('.')[0]),p.name))
    paths=[p for p in paths if args.only in str(p.relative_to(ROOT))]
    if args.retry:
        paths=[p for p in paths if (lambda n:n.metadata.get('validation',{}).get('status')!='passed' or n.metadata.get('validation',{}).get('signature')!=n.metadata.get('conversion',{}).get('cell_sha256'))(nbformat.read(p,as_version=4))]
    print('Executing',len(paths),'notebooks',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results=list(executor.map(lambda p:run(p,args.timeout),paths))
    print('Batch finished:',sum(r['status']=='passed' for r in results),'passed;',sum(r['status']!='passed' for r in results),'failed',flush=True)

if __name__=='__main__':main()
