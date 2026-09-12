const BASE=new URL('./',import.meta.url);
const REPO='https://github.com/fxt-gw-pb/MachineLearningNotebooks';
const app=document.querySelector('#app');
const icon=(name,cls='')=>`<i data-lucide="${name}" class="${cls}" aria-hidden="true"></i>`;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const asset=p=>new URL(p,BASE).href;
let catalog;let currentLesson;let requestId=0;let lessonCache=new Map();
let viewMode='full';let searchQuery='';let activeHeadingObserver;let plotObserver;let plotlyPromise;
const mobileMenu=matchMedia('(max-width:760px)');
const loadJson=async p=>{const r=await fetch(asset(p));if(!r.ok)throw new Error('内容暂时无法加载');return r.json();};
const icons=()=>window.lucide?.createIcons({attrs:{'stroke-width':1.7}});
const lessonLink=l=>`#/lesson/${l.id}`;
function sourceUrl(l){return REPO+'/blob/main/'+l.path.split('/').map(encodeURIComponent).join('/');}
function sidebar(chapterId){
  return `<aside class="sidebar" id="sidebar">
    <a class="brand" href="#/"><img src="${asset('favicon.svg')}" alt=""/><span>ML <b>Atlas</b><small>机器学习手记</small></span></a>
    <button class="sidebar-close icon-button" data-close-menu-button aria-label="关闭章节目录">${icon('x')}</button>
    <div class="sidebar-caption">YOUR LEARNING ATLAS</div>
    <a class="overview-link ${!chapterId?'active':''}" href="#/">${icon('layout-grid')}<span>课程总览</span>${icon('arrow-up-right')}</a>
    <div class="nav-title">学习目录 <span>20 CHAPTERS</span></div>
    <nav class="chapter-nav" aria-label="课程章节">
      ${catalog.chapters.map(ch=>`<a class="chapter-link ${chapterId===ch.id?'active':''}" href="#/chapter/${ch.id}" ${chapterId===ch.id?'aria-current="page"':''}>
        <span class="chapter-index">${ch.id}</span><span>${esc(ch.title)}</span><small>${ch.lessons.length}</small>
      </a>${chapterId===ch.id&&currentLesson?`<div class="lesson-subnav">${ch.lessons.map(id=>{const l=catalog.lessons.find(l=>l.id===id);return `<a href="${lessonLink(l)}" class="${id===currentLesson.id?'active':''}">${esc(l.title)}</a>`;}).join('')}</div>`:''}`).join('')}
    </nav>
    <a class="sidebar-bottom" href="${REPO}" target="_blank" rel="noopener noreferrer">${icon('github')}<span>开源学习，共同进步<small>查看 GitHub 仓库</small></span>${icon('arrow-up-right')}</a>
  </aside>`;
}
function shell(body,chapterId=null,crumb='课程总览'){
  app.innerHTML=`${sidebar(chapterId)}<div class="sidebar-backdrop" data-close-menu></div><div class="workspace">
    <header class="topbar"><div class="breadcrumb"><button class="icon-button menu-button" aria-label="打开章节目录" data-menu>${icon('menu')}</button><a href="#/">学习空间</a>${icon('chevron-right')}<span>${esc(crumb)}</span></div>
      <div class="topbar-right"><span class="edition">THE NOTEBOOK COLLECTION <b>01—20</b></span><a class="github-link" href="${REPO}" target="_blank" rel="noopener noreferrer">${icon('github')}<span>GitHub</span>${icon('arrow-up-right')}</a></div>
    </header><main id="main-content" tabindex="-1">${body}</main><footer class="site-footer"><a href="#/">ML Atlas <span>机器学习手记</span></a><span>从原理到实践 · 用代码理解算法</span><a href="${REPO}" target="_blank" rel="noopener noreferrer">在 GitHub 查看源文件 ${icon('arrow-up-right')}</a></footer></div>`;
  icons();bindShell();
}
function bindShell(){
  document.querySelector('#sidebar').toggleAttribute('inert',mobileMenu.matches);
  document.querySelector('#sidebar').setAttribute('aria-hidden',String(mobileMenu.matches));
  document.querySelector('[data-menu]')?.addEventListener('click',()=>{
    document.body.classList.add('menu-open');document.querySelector('#sidebar').removeAttribute('inert');
    document.querySelector('#sidebar').setAttribute('aria-hidden','false');
    document.querySelector('.workspace').setAttribute('inert','');document.querySelector('.overview-link')?.focus();
  });
  document.querySelector('[data-close-menu]')?.addEventListener('click',closeMenu);
  document.querySelector('[data-close-menu-button]')?.addEventListener('click',()=>{closeMenu();document.querySelector('[data-menu]')?.focus();});
}
function closeMenu(){document.body.classList.remove('menu-open');const side=document.querySelector('#sidebar'),work=document.querySelector('.workspace');if(side){side.toggleAttribute('inert',mobileMenu.matches);side.setAttribute('aria-hidden',String(mobileMenu.matches));}if(work)work.removeAttribute('inert');}
mobileMenu.addEventListener('change',closeMenu);
function chapterCard(ch){
  const names=ch.lessons.slice(0,3).map(id=>catalog.lessons.find(l=>l.id===id).title);
  return `<a class="chapter-card" href="#/chapter/${ch.id}"><div class="card-top"><span class="chapter-symbol tone-${Number(ch.id)%4}">${icon(ch.icon)}</span><span class="chapter-label">CHAPTER ${ch.id}</span></div><h3>${esc(ch.title)}</h3><p>${esc(ch.description)}</p><div class="chapter-topics">${names.map(t=>`<span>${esc(t)}</span>`).join('')}</div><div class="card-bottom"><span>${ch.lessons.length} 份学习笔记</span><span class="round-arrow">${icon('arrow-up-right')}</span></div></a>`;
}
function renderHome(){
  currentLesson=null;viewMode='full';document.title='ML Atlas · 机器学习手记';
  const first=catalog.lessons[0];
  shell(`<div class="home-page">
    <section class="home-intro"><div class="eyebrow"><span class="eyebrow-line"></span>LEARN. UNDERSTAND. BUILD.</div><div class="intro-title"><div><h1>机器学习，<span>从理解到实践。</span></h1><p>每一个算法，都有迹可循。<br class="mobile-break"/>在原理、代码与结果之间，建立自己的理解。</p></div><div class="intro-mark" aria-hidden="true">M<span>L</span><small>FIELD NOTES / 2026</small></div></div>
      <div class="collection-stats"><span><b>${catalog.stats.lessons}</b> 份可运行笔记</span><span><b>${catalog.stats.chapters}</b> 个主题章节</span><span>${icon('code-2')} Python 实践</span><span>${icon('chart-no-axes-combined')} 真实运行结果</span></div>
    </section>
    <section class="featured-lesson" aria-labelledby="featured-title"><div class="featured-copy"><span class="featured-kicker"><span>START HERE</span> 你的第一份学习笔记</span><h2 id="featured-title">从一条直线，<br/>理解预测的起点。</h2><p>用线性回归建立第一个模型，看懂特征、参数与误差之间的关系。</p><a class="primary-button" href="${lessonLink(first)}">学习线性回归 ${icon('arrow-right')}</a><div class="featured-meta">01 · 回归算法 <span>约 ${first.minutes} 分钟</span></div></div>
      <a class="featured-notebook" href="${lessonLink(first)}" aria-label="打开线性回归笔记"><div class="notebook-window-bar"><span><i></i><i></i><i></i></span><span>linear_regression.ipynb</span>${icon('code-2')}</div><div class="featured-code"><span class="line-no">01</span><code><b>model</b> = LinearRegression()</code><span class="line-no">02</span><code>model.<em>fit</em>(X_train, y_train)</code><span class="line-no">03</span><code>y_pred = model.<em>predict</em>(X_test)</code></div><div class="featured-output"><span class="output-mini-label">OUTPUT <span>真实值与预测值</span></span><img src="${asset(first.cover)}" alt="线性回归笔记中真实生成的预测散点图"/></div></a>
    </section>
    <section class="catalog-section" aria-labelledby="catalog-title"><div class="section-header"><div><div class="eyebrow small">EXPLORE THE COLLECTION</div><h2 id="catalog-title">选择一个主题，开始探索<span>20</span></h2></div><label class="search-field">${icon('search')}<input id="catalog-search" type="search" placeholder="搜索算法、模型或关键词…" aria-label="搜索学习笔记" autocomplete="off"/><kbd>/</kbd></label></div>
      <div class="catalog-tabs" role="group" aria-label="主题分类"><button data-category="all" class="active">全部主题 <span>20</span></button><button data-category="basics">建模与基础</button><button data-category="methods">方法与实践</button><button data-category="evaluation">度量与诊断</button><span class="catalog-view-note">沿着好奇心，找到下一页 ${icon('arrow-down')}</span></div>
      <div id="catalog-results" class="chapter-grid">${catalog.chapters.map(chapterCard).join('')}</div>
    </section></div>`);
  let selected='all';const input=document.querySelector('#catalog-search');
  const update=()=>{searchQuery=input.value.trim();renderCatalogResults(searchQuery,selected);};
  input.addEventListener('input',update);
  document.querySelectorAll('[data-category]').forEach(btn=>btn.addEventListener('click',()=>{selected=btn.dataset.category;document.querySelectorAll('[data-category]').forEach(b=>b.classList.toggle('active',b===btn));update();}));
}
function renderCatalogResults(query,category){
  const inCategory=ch=>category==='all'||category==='basics'&&Number(ch)<=6||category==='methods'&&Number(ch)>6&&Number(ch)<=13||category==='evaluation'&&Number(ch)>=14;
  const target=document.querySelector('#catalog-results');
  if(!query){target.className='chapter-grid';target.innerHTML=catalog.chapters.filter(ch=>inCategory(ch.id)).map(chapterCard).join('');}
  else {
    const normalized=query.toLowerCase().replace(/\s+/g,'');
    const rank=l=>l.title.toLowerCase().replace(/\s+/g,'').includes(normalized)?100:l.description.toLowerCase().includes(normalized)?30:10;
    const results=catalog.lessons.filter(l=>inCategory(l.chapter)&&`${l.title} ${l.chapterTitle} ${l.description} ${l.keywords??''}`.toLowerCase().replace(/\s+/g,'').includes(normalized)).sort((a,b)=>rank(b)-rank(a));
    target.className='search-results';target.innerHTML=`<div class="search-summary">找到 ${results.length} 份相关笔记</div>${results.length?results.map(l=>lessonRow(l,'search')).join(''):`<div class="empty-state">${icon('search')}<h3>还没有找到相关笔记</h3><p>试试「回归」「PCA」「概率」等关键词，或切换到全部主题。</p></div>`}`;
    document.querySelector('#announcement').textContent=`找到 ${results.length} 份笔记`;
  }
  icons();
}
function lessonRow(l,context){return `<a class="lesson-row" href="${lessonLink(l)}"><span class="row-number">${l.id.split('-')[1]}</span><div>${context==='search'?`<span class="search-chapter">CH ${l.chapter} · ${esc(l.chapterTitle)}</span>`:''}<h3>${esc(l.title)}</h3><p>${esc(l.description||l.chapterTitle)}</p></div><div class="row-metadata"><span>${icon('code-2')}${l.codeCount} 段代码</span><span>${icon('clock-3')}${l.minutes} 分钟</span></div>${icon('arrow-up-right','row-arrow')}</a>`;}
function renderChapter(id){
  currentLesson=null;viewMode='full';const chapter=catalog.chapters.find(c=>c.id===id);
  if(!chapter)return notFound();document.title=chapter.title+' · ML Atlas';
  shell(`<div class="chapter-page"><a class="text-link back-link" href="#/">${icon('arrow-left')} 返回全部主题</a><div class="chapter-hero"><div><div class="eyebrow"><span class="eyebrow-line"></span>CHAPTER ${chapter.id}</div><h1>${esc(chapter.title)}</h1><p>${esc(chapter.description)}</p><div class="chapter-hero-meta"><span>${chapter.lessons.length} 份学习笔记</span><span>理论推导 + Python 实践</span></div></div><span class="giant-chapter-number" aria-hidden="true">${chapter.id}</span></div><div class="chapter-list-header"><h2>本章学习笔记</h2><span>按顺序阅读，也可以从感兴趣的主题开始</span></div><div class="lesson-list">${chapter.lessons.map(id=>lessonRow(catalog.lessons.find(l=>l.id===id))).join('')}</div></div>`,id,chapter.title);
}
function renderOutput(out){
  if(out.type==='notice')return `<details class="library-notice"><summary>${icon('info')}库提示与诊断信息</summary><pre>${esc(out.text)}</pre></details>`;
  if(out.type==='image')return `<figure class="result-figure"><img src="${asset(out.src)}" loading="lazy" width="${out.width}" height="${out.height}" alt="本代码单元的运行图形"/><button data-zoom="${esc(out.src)}" class="figure-expand" aria-label="放大查看运行图形">${icon('maximize-2')}</button></figure>`;
  if(out.type==='html')return `<div class="output-table" tabindex="0" role="region" aria-label="运行结果数据表">${out.html}</div>`;
  if(out.type==='plotly')return `<div class="plotly-output" data-plotly="${esc(out.src)}"><span class="plot-loading">正在加载交互图形…</span></div>`;
  return `<pre tabindex="0" class="text-output ${out.type==='error'?'error-output':''}">${esc(out.text)}</pre>`;
}
function codeCell(cell,index){
  const lineNumbers=Array.from({length:cell.lines},(_,i)=>i+1).join('\n');
  const regular=cell.outputs.filter(o=>o.type!=='notice');const notices=cell.outputs.filter(o=>o.type==='notice');
  const content=`<section class="code-cell" id="code-${cell.number}"><div class="code-block"><div class="code-header"><div><span class="python-dot"></span><b>PYTHON</b><span class="cell-label">${cell.setup?'环境准备':String(cell.number).padStart(2,'0')}</span></div><button class="copy-code" data-copy="${index}" aria-label="复制代码">${icon('copy')}<span>复制代码</span></button></div><div class="code-scroll ${cell.lines>20?'is-collapsed':''}"><div class="code-lines" aria-hidden="true">${lineNumbers}</div><pre tabindex="0"><code>${cell.highlight}</code></pre></div>${cell.lines>20?`<button class="expand-code" data-expand-code>展开完整代码 <span>${cell.lines} 行</span>${icon('chevron-down')}</button>`:''}</div>${regular.length?`<div class="cell-outputs"><div class="output-header"><span>${icon('terminal')}运行结果</span><small>Out [${cell.execution??cell.number}]</small></div>${regular.map(renderOutput).join('')}</div>`:''}${notices.length?`<details class="library-notice"><summary>${icon('info')}库提示与诊断信息 <span>${notices.length}</span></summary>${notices.map(o=>`<pre>${esc(o.text)}</pre>`).join('')}</details>`:''}</section>`;
  return cell.setup?`<details class="environment-cell"><summary>${icon('settings-2')}运行环境与公共依赖<span>展开查看</span>${icon('chevron-down')}</summary>${content}</details>`:content;
}
function lessonCells(lesson){return lesson.cells.map((c,i)=>c.type==='markdown'?`<div class="prose-cell">${c.html}</div>`:c.type==='notes'?`<details class="conversion-note"><summary>${icon('info')}这份笔记的数据与运行说明${icon('chevron-down')}</summary><div class="prose">${c.html}</div></details>`:codeCell(c,i)).join('');}
async function renderLesson(id){
  const sequence=requestId;const summary=catalog.lessons.find(l=>l.id===id);if(!summary)return notFound();
  currentLesson=summary;viewMode='full';document.title=summary.title+' · ML Atlas';
  if(!lessonCache.has(id)){
    shell(`<div class="lesson-loading"><span class="loading-spinner"></span><p>正在打开「${esc(summary.title)}」…</p></div>`,summary.chapter,summary.title);
    lessonCache.set(id,await loadJson('content/'+id+'.json'));
  }
  if(sequence!==requestId)return;
  const lesson=lessonCache.get(id);currentLesson=lesson;
  const position=catalog.lessons.findIndex(l=>l.id===id);const next=catalog.lessons[position+1],prev=catalog.lessons[position-1];
  const toc=lesson.headings.filter(h=>h.title!=='Notebook 使用说明');
  shell(`<div class="reader-layout"><article class="lesson-article"><header class="lesson-header"><a class="text-link back-link" href="#/chapter/${lesson.chapter}">${icon('arrow-left')}${esc(lesson.chapterTitle)}</a><div class="eyebrow small">CHAPTER ${lesson.chapter} <span>/</span> NOTE ${id.split('-')[1]}</div><h1>${esc(lesson.title)}</h1><div class="lesson-metadata"><span>${icon('clock-3')}约 ${lesson.minutes} 分钟</span><span>${icon('code-2')}${lesson.codeCount} 段代码</span><span>${icon('chart-no-axes-combined')}${lesson.imageCount} 个图形结果</span></div><div class="reader-toolbar"><div class="reading-tabs" role="group" aria-label="阅读模式"><button class="active" data-mode="full">${icon('book-open')}完整笔记</button><button data-mode="code">${icon('code-2')}代码与结果</button></div><a class="notebook-source-link" href="${sourceUrl(lesson)}" target="_blank" rel="noopener noreferrer">原始 Notebook ${icon('arrow-up-right')}</a></div></header><div class="lesson-content prose" id="lesson-content">${lessonCells(lesson)}</div><nav class="lesson-pagination" aria-label="相邻学习笔记">${prev?`<a href="${lessonLink(prev)}"><span>${icon('arrow-left')}上一篇</span><strong>${esc(prev.title)}</strong></a>`:'<div></div>'}${next?`<a href="${lessonLink(next)}" class="next"><span>继续学习${icon('arrow-right')}</span><strong>${esc(next.title)}</strong></a>`:'<div></div>'}</nav></article><aside class="reader-toc"><div class="toc-inner"><div class="toc-title">${icon('list')}本页目录</div><nav aria-label="本页目录">${toc.map(h=>`<button data-heading="${h.id}" class="level-${h.level}">${esc(h.title)}</button>`).join('')}</nav><div class="reader-tip"><span>边读，边验证</span><p>对照代码与真实输出理解算法。图形支持放大查看。</p><a href="${sourceUrl(lesson)}" target="_blank" rel="noopener noreferrer">在 Jupyter 中实践 ${icon('arrow-up-right')}</a></div></div></aside></div>`,lesson.chapter,lesson.title);
  // Markdown images use build-relative paths; keep GitHub Pages subpaths intact.
  document.querySelectorAll('.prose-cell img').forEach(img=>{if(img.getAttribute('src').startsWith('assets/'))img.src=asset(img.getAttribute('src'));});
  const sourceLink=document.querySelector('.notebook-source-link');
  sourceLink.insertAdjacentHTML('beforebegin',`<button class="download-notebook" data-download>${icon('download')}下载 Notebook</button>`);
  sourceLink.innerHTML=icon('github');sourceLink.setAttribute('aria-label','在 GitHub 查看原始 Notebook');sourceLink.title='在 GitHub 查看源文件';
  document.querySelector('.reader-tip').innerHTML=`<span>边读，边验证</span><p>这里展示已保存的运行结果。运行笔记前，请先准备仓库中的数据与公共依赖。</p><a href="${REPO}#开始学习" target="_blank" rel="noopener noreferrer">查看完整运行指南 ${icon('arrow-up-right')}</a>`;
  document.querySelectorAll('.prose-cell img').forEach(img=>{
    img.tabIndex=0;img.setAttribute('role','button');img.setAttribute('aria-label','放大查看原文参考图');
    img.addEventListener('click',()=>showFigure(img.src));
    img.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();showFigure(img.src);}});
    const caption=document.createElement('span');caption.className='reference-caption';caption.textContent='原文参考图 · 当前实验结果见代码输出';img.after(caption);
  });
  icons();
  bindLesson(lesson);
}
function bindLesson(lesson){
  bindToc();
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{
    viewMode=b.dataset.mode;document.querySelector('.lesson-content').classList.toggle('code-only',viewMode==='code');
    document.querySelectorAll('[data-mode]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',x===b?'true':'false');});
    const entries=viewMode==='code'?lesson.cells.filter(c=>c.type==='code'&&!c.setup).map(c=>({id:'code-'+c.number,level:2,title:`代码 ${String(c.number).padStart(2,'0')} · ${c.source.split('\n').find(l=>l.startsWith('# '))?.slice(2,24)||'Python 实践'}`})):lesson.headings;
    document.querySelector('.reader-toc nav').innerHTML=entries.map(h=>`<button data-heading="${h.id}" class="level-${h.level}">${esc(h.title)}</button>`).join('');
    bindToc();icons();
  }));
  document.querySelectorAll('[data-expand-code]').forEach(b=>{b.setAttribute('aria-expanded','false');b.addEventListener('click',()=>{const panel=b.previousElementSibling;const collapsed=panel.classList.toggle('is-collapsed');b.setAttribute('aria-expanded',String(!collapsed));b.innerHTML=`${collapsed?'展开完整代码':'收起代码'}${icon(collapsed?'chevron-down':'chevron-up')}`;icons();});});
  document.querySelectorAll('[data-copy]').forEach(b=>b.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(lesson.cells[Number(b.dataset.copy)].source);b.innerHTML=icon('check')+'<span>已复制</span>';icons();document.querySelector('#announcement').textContent='代码已复制';setTimeout(()=>{b.innerHTML=icon('copy')+'<span>复制代码</span>';icons();},1800);}catch{document.querySelector('#announcement').textContent='请选中代码后复制。';}}));
  document.querySelectorAll('[data-zoom]').forEach(b=>b.addEventListener('click',()=>showFigure(b.dataset.zoom)));
  document.querySelector('[data-download]')?.addEventListener('click',async e=>{
    const button=e.currentTarget;button.disabled=true;button.textContent='正在准备…';
    try{
      const url='https://raw.githubusercontent.com/fxt-gw-pb/MachineLearningNotebooks/main/'+lesson.path.split('/').map(encodeURIComponent).join('/');
      const response=await fetch(url);if(!response.ok)throw new Error('Notebook download failed');
      const blob=await response.blob();const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=lesson.path.split('/').pop();document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(link.href),5000);
      button.innerHTML=icon('check')+'已准备下载';
    }catch{button.textContent='下载失败，点击重试';document.querySelector('#announcement').textContent='无法下载，您也可以使用旁边的 GitHub 链接获取原文件。';}
    finally{button.disabled=false;icons();}
  });
  plotObserver?.disconnect();
  plotObserver=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){plotObserver.unobserve(e.target);renderPlotly(e.target);}}),{rootMargin:'350px'});
  document.querySelectorAll('[data-plotly]').forEach(el=>plotObserver.observe(el));
}
function bindToc(){
  activeHeadingObserver?.disconnect();
  document.querySelectorAll('[data-heading]').forEach(b=>b.addEventListener('click',()=>document.getElementById(b.dataset.heading)?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth',block:'start'})));
  activeHeadingObserver=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){document.querySelectorAll('[data-heading]').forEach(b=>b.classList.toggle('active',b.dataset.heading===e.target.id));}},{rootMargin:'-90px 0px -65% 0px'});
  document.querySelectorAll('[data-heading]').forEach(b=>{const el=document.getElementById(b.dataset.heading);if(el)activeHeadingObserver.observe(el);});
}
function loadPlotly(){
  return plotlyPromise??=(new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=asset('vendor/plotly.js');script.onload=()=>resolve(window.Plotly);script.onerror=()=>{plotlyPromise=null;reject(new Error('无法加载交互图形库'));};document.head.append(script);}));
}
async function renderPlotly(element){
  try{
    const [Plotly,figure]=await Promise.all([loadPlotly(),loadJson(element.dataset.plotly)]);
    if(!element.isConnected)return;
    const layout={...figure.layout,autosize:true,height:mobileMenu.matches?450:540,font:{...figure.layout?.font,family:'Manrope, PingFang SC, sans-serif'}};
    delete layout.width;
    if(mobileMenu.matches){layout.legend={...layout.legend,orientation:'h',x:0,y:-0.08,font:{size:11}};layout.margin={...layout.margin,l:22,r:12,t:64,b:55};layout.title={...layout.title,font:{...layout.title?.font,size:14}};}
    element.innerHTML='';await Plotly.newPlot(element,figure.data,layout,{responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['sendChartToCloud']});
  }catch(e){element.innerHTML='<p class="plot-loading">交互图暂时无法加载，请刷新后重试。</p>';console.error(e);}
}
function showFigure(src){const dialog=document.querySelector('#figure-dialog');document.querySelector('#figure-dialog-body').innerHTML=`<img src="${asset(src)}" alt="完整运行图形"/>`;dialog.showModal();}
document.querySelector('#close-figure').addEventListener('click',()=>document.querySelector('#figure-dialog').close());
document.querySelector('#figure-dialog').addEventListener('click',e=>{if(e.target===e.currentTarget)e.currentTarget.close();});
function notFound(){currentLesson=null;shell(`<div class="empty-state"><h1>这一页暂时不在目录中</h1><p>可以返回课程总览，选择一份笔记继续学习。</p><a href="#/" class="primary-button">返回课程总览 ${icon('arrow-right')}</a></div>`);}
async function route(){
  const routeSequence=++requestId;closeMenu();activeHeadingObserver?.disconnect();plotObserver?.disconnect();
  document.querySelectorAll('.js-plotly-plot').forEach(el=>window.Plotly?.purge(el));
  window.scrollTo({top:0,behavior:'instant'});
  const [kind,id]=(location.hash.replace(/^#\/?/,'')).split('/');
  try{if(kind==='lesson')await renderLesson(id);else if(kind==='chapter')renderChapter(id);else renderHome();}catch(error){if(routeSequence!==requestId)return;shell(`<div class="empty-state"><h2>内容暂时无法加载</h2><p>请检查网络后刷新页面。</p><button class="primary-button" onclick="location.reload()">重新加载</button></div>`);console.error(error);}
}
window.addEventListener('hashchange',route);
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('menu-open')){closeMenu();document.querySelector('[data-menu]')?.focus();}if(e.key==='/'&&!/INPUT|TEXTAREA/.test(document.activeElement.tagName)){const input=document.querySelector('#catalog-search');if(input){e.preventDefault();input.focus();}}});
document.querySelector('.skip-link').addEventListener('click',e=>{e.preventDefault();document.querySelector('#main-content')?.focus();});
try{catalog=await loadJson('content/index.json');await route();}catch(e){app.innerHTML='<main class="empty-state"><h1>学习目录暂时无法加载</h1><p>请检查网络并刷新页面。</p></main>';console.error(e);}
