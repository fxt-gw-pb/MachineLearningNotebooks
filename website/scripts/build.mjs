import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {marked} from 'marked';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import sanitizeHtml from 'sanitize-html';

const require = createRequire(import.meta.url);
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {AllPackages} = require('mathjax-full/js/input/tex/AllPackages.js');
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DIST = path.join(ROOT, 'dist');
const candidates = [process.env.NOTEBOOKS_DIR, path.join(ROOT, '..'), path.join(ROOT, '../机器学习_Notebooks')].filter(Boolean);
const source = candidates.map(p => path.resolve(p)).find(p => fs.existsSync(path.join(p,'1. 回归算法/01. 线性回归.ipynb')));
if (!source) throw new Error('没有找到原始 Notebook；请设置 NOTEBOOKS_DIR。');
fs.mkdirSync(path.join(DIST,'content'),{recursive:true});
fs.mkdirSync(path.join(DIST,'assets'),{recursive:true});
fs.mkdirSync(path.join(DIST,'vendor'),{recursive:true});
fs.mkdirSync(path.join(DIST,'fonts'),{recursive:true});
hljs.registerLanguage('python',python);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text=x=>Array.isArray(x)?x.join(''):String(x??'');
const stripAnsi=s=>s.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g,'');
const adaptor=liteAdaptor();RegisterHTMLHandler(adaptor);
const mathErrors=[];
const tex=new TeX({packages:AllPackages,formatError:(jax,e)=>{mathErrors.push(e.message);return jax.formatError(e);}});
const mathDoc=mathjax.document('',{InputJax:tex,OutputJax:new SVG({fontCache:'none'})});
const mathCache=new Map();
function renderMath(source,display){
  const key=display+'|'+source;
  if(!mathCache.has(key)){
    const svg=adaptor.outerHTML(mathDoc.convert(source,{display}));
    mathCache.set(key,`<span class="math ${display?'math-display':'math-inline'}" role="math" aria-label="${esc(source)}">${svg}</span>`);
  }
  return mathCache.get(key);
}
const assets=new Map();const dimensions=new Map();
function asset(buffer,extension='png'){
  const hash=crypto.createHash('sha256').update(buffer).digest('hex').slice(0,24);
  const name=`assets/${hash}.${extension}`;
  if(!assets.has(name)){fs.writeFileSync(path.join(DIST,name),buffer);assets.set(name,buffer.length);}
  if(extension==='png'&&buffer.length>24)dimensions.set(name,{width:buffer.readUInt32BE(16),height:buffer.readUInt32BE(20)});
  return name;
}
let headings=[];let headingNumber=0;
const renderer=new marked.Renderer();
renderer.heading=function({tokens,depth}){
  const inner=this.parser.parseInline(tokens);
  const label=inner.replace(/<[^>]+>/g,'').replace(/MLMATHPLACEHOLDER\d+END/g,'公式');
  const id=`section-${headingNumber++}`;
  if(depth<=3)headings.push({id,title:label,level:depth});
  return `<h${Math.max(2,depth)} id="${id}">${inner}</h${Math.max(2,depth)}>`;
};
marked.use({renderer,gfm:true,breaks:false});
function markdown(sourceText,directory){
  const math=[];
  let s=sourceText.replace(/!\[([^\]]*)\]\((attachments\/[^)]+)\)/g,(_,alt,file)=>{
    const filename=path.join(directory,file);
    if(!fs.existsSync(filename))throw new Error('Missing image '+filename);
    return `![${alt}](${asset(fs.readFileSync(filename),path.extname(filename).slice(1))})`;
  });
  // Protect fenced/inline code; turn math into placeholders before parsing Markdown tables.
  s=s.replace(/```[\s\S]*?```|`[^`\n]+`|\$\$([\s\S]*?)\$\$|(?<!\\)\$([^$\n]+)\$/g,(whole,block,inline)=>{
    if(whole.startsWith('`'))return whole;
    const index=math.length;math.push(renderMath((block??inline).trim(),block!==undefined));
    return `MLMATHPLACEHOLDER${index}END`;
  });
  let html=marked.parse(s);
  html=sanitizeHtml(html,{allowedTags:[...sanitizeHtml.defaults.allowedTags,'img','details','summary'],
    allowedAttributes:{...sanitizeHtml.defaults.allowedAttributes,'*':['class','id'],'img':['src','alt','title','width','height'],'a':['href','title','target','rel']},
    allowedSchemes:['https','http','mailto'],
    transformTags:{a:(tag,attrs)=>({tagName:tag,attribs:{...attrs,...(attrs.href?.startsWith('http')?{target:'_blank',rel:'noopener noreferrer'}:{})}})},
  });
  html=html.replace(/MLMATHPLACEHOLDER(\d+)END/g,(_,i)=>math[Number(i)]);
  html=html.replace(/<img ([^>]*src="([^"]+)"[^>]*)>/g,(_,attrs,src)=>{const d=dimensions.get(src);return `<img loading="lazy" decoding="async" ${d?`width="${d.width}" height="${d.height}" `:''}${attrs}>`;});
  return html;
}
const descriptions={
  1:'从一条直线开始，学习连续数值的预测。',2:'找到数据之间的边界，把观察转化为判断。',
  3:'组合多个模型，理解更稳健的预测从何而来。',4:'认识随机性，为统计学习建立概率直觉。',
  5:'用数据检验假设，让结论有据可依。',6:'找到真正有用的特征，减少冗余与噪声。',
  7:'把高维数据展开，看见隐藏的结构。',8:'理解参数如何一步步走向更好的解。',
  9:'无需预先给定标签，发现数据中的自然分组。',10:'在行动与反馈中，学习更好的决策。',
  11:'从原始数据出发，为可靠建模打好基础。',12:'学习趋势、季节性与随时间变化的规律。',
  13:'通过相似性，在新的特征空间中学习。',14:'理解模型如何衡量错误，以及优化什么。',
  15:'控制模型复杂度，让学习适可而止。',16:'用熵和互信息衡量不确定性与关联。',
  17:'选对评价指标，判断模型是否学得有效。',18:'调整数值尺度，让特征在合适的范围内工作。',
  19:'从有限的观测，估计总体与模型参数。',20:'理解异常值，并选择有依据的处理方式。'
};
const chapterIcons=['chart-no-axes-combined','shapes','layers','chart-column','flask-conical','scan-line','orbit','sliders-horizontal','network','route','filter','chart-line','waypoints','square-function','shield-check','binary','badge-check','ruler','sigma','scan-search'];
const chapters=[];const lessons=[];const outputCounts={};
for(const directory of fs.readdirSync(source).filter(d=>/^\d+\. /.test(d)).sort((a,b)=>parseInt(a)-parseInt(b))){
  const chapterId=String(parseInt(directory)).padStart(2,'0');
  const files=fs.readdirSync(path.join(source,directory)).filter(f=>f.endsWith('.ipynb')).sort();
  if(!files.length)continue;
  const chapter={id:chapterId,title:directory.replace(/^\d+\. /,''),description:descriptions[parseInt(directory)],icon:chapterIcons[parseInt(directory)-1],lessons:[]};
  for(const filename of files){
    const nb=JSON.parse(fs.readFileSync(path.join(source,directory,filename),'utf8'));
    const lessonId=chapterId+'-'+filename.match(/^\d+/)[0];
    const title=filename.replace(/^\d+\. /,'').replace(/\.ipynb$/,'');
    headings=[];headingNumber=0;
    let firstImage;let paragraph='';let codeNumber=0;let imageCount=0;let codeLines=0;
    const cells=[];
    for(const [index,cell] of nb.cells.entries()){
      let raw=text(cell.source);
      if(lessonId==='01-01'&&cell.cell_type==='markdown')raw=raw.replace('最后，通过特征选择和标准化，我们进一步优化了模型，提升了其预测性能。','> **对照运行结果**：这里保存的输出中，测试集 MSE 从约 0.5021 变为 0.5064，略有增大。特征选择和标准化是值得比较的步骤，但并不保证提升预测性能，应以独立测试集指标为准。');
      if(cell.cell_type==='markdown'){
        if(index===0){cells.push({type:'notes',html:markdown(raw.replace(/^# .*\n/,'').replace('## Notebook 使用说明',''),path.join(source,directory))});continue;}
        if(!paragraph){
          const lines=raw.split('\n').map(l=>l.trim().replace(/\*\*|`/g,'')).filter(l=>l.length>16&&!/^(#|!|\||-|\$|\d+\.)/.test(l));
          paragraph=lines.find(l=>!/(咱们|本文|本篇|今天|今儿|大家|聊聊|下面|\$|\\)/.test(l))||lines[0]||'';
        }
        cells.push({type:'markdown',html:markdown(raw,path.join(source,directory))});
      }else if(cell.cell_type==='code'){
        const setup=cell.metadata?.tags?.includes('setup')||raw.includes('CJK_FONTS =')&&raw.includes('notebook_support');
        if(!setup){codeNumber++;codeLines+=raw.split('\n').length;}
        const sourceHighlighted=hljs.highlight(raw,{language:'python',ignoreIllegals:true}).value;
        const block={type:'code',number:setup?0:codeNumber,execution:cell.execution_count,source:raw,highlight:sourceHighlighted,lines:raw.split('\n').length,setup,outputs:[]};
        for(const output of cell.outputs??[]){
          const data=output.data??{};
          if(data['image/png']){
            const img=asset(Buffer.from(text(data['image/png']),'base64'));
            firstImage??=img;imageCount++;block.outputs.push({type:'image',src:img,...dimensions.get(img)});
          }else if(data['application/vnd.plotly.v1+json']){
            const figure=data['application/vnd.plotly.v1+json'];
            const name=`content/${lessonId}-plot-${imageCount}.json`;
            fs.writeFileSync(path.join(DIST,name),JSON.stringify(figure));
            imageCount++;block.outputs.push({type:'plotly',src:name});
          }else if(data['text/html']&& !text(data['text/html']).includes('<script')){
            block.outputs.push({type:'html',html:sanitizeHtml(text(data['text/html']),{allowedTags:[...sanitizeHtml.defaults.allowedTags,'table','thead','tbody','tr','td','th'],allowedAttributes:{'*':['class','colspan','rowspan']}})});
          }else if(output.output_type==='stream'){
            const value=stripAnsi(text(output.text));
            if(value.trim())block.outputs.push({type:output.name==='stderr'?'notice':'text',text:value});
          }else if(data['text/plain']&&!/^<Figure|^<Axes|^<matplotlib\.|^Text\(|^<seaborn\.|^<IPython\./.test(text(data['text/plain']))){
            block.outputs.push({type:'text',text:stripAnsi(text(data['text/plain']))});
          }else if(output.output_type==='error'){
            block.outputs.push({type:'error',text:stripAnsi((output.traceback??[]).join('\n'))});
          }
        }
        for(const out of block.outputs)outputCounts[out.type]=(outputCounts[out.type]??0)+1;
        cells.push(block);
      }
    }
    const plain=nb.cells.filter(c=>c.cell_type==='markdown').map(c=>text(c.source)).join('\n');
    const keywords=[...new Set(cells.filter(c=>c.type==='code'&&!c.setup).flatMap(c=>c.source.match(/\b[A-Za-z][A-Za-z_0-9]{2,}\b/g)??[]))].slice(0,200).join(' ');
    const summary={id:lessonId,title,chapter:chapterId,chapterTitle:chapter.title,description:paragraph.replace(/\\([~_*])/g,'$1').replace(/\*|`|\$/g,'').slice(0,110),keywords,
      codeCount:codeNumber,imageCount,minutes:Math.max(3,Math.ceil(plain.length/650+codeLines/70)),path:directory+'/'+filename,cover:firstImage??null,
      dataFiles:nb.metadata?.conversion?.data_files??[],status:nb.metadata?.validation?.status??'unknown'};
    lessons.push(summary);chapter.lessons.push(lessonId);
    fs.writeFileSync(path.join(DIST,'content',lessonId+'.json'),JSON.stringify({...summary,headings:headings.filter(h=>h.level===2||h.level===3),cells}));
  }
  chapters.push(chapter);
}
for(const filename of fs.readdirSync(path.join(ROOT,'src'))) {
  fs.copyFileSync(path.join(ROOT,'src',filename),path.join(DIST,filename));
}
const vendor=[['lucide/dist/umd/lucide.min.js','lucide.js'],['plotly.js-dist-min/plotly.min.js','plotly.js']];
for(const [from,to] of vendor){const p=path.join(ROOT,'node_modules',from);if(fs.existsSync(p))fs.copyFileSync(p,path.join(DIST,'vendor',to));}
for(const family of ['manrope','jetbrains-mono']){
  const dir=path.join(ROOT,'node_modules','@fontsource-variable',family,'files');
  const font=fs.readdirSync(dir).find(x=>x===`${family}-latin-wght-normal.woff2`);
  if(font)fs.copyFileSync(path.join(dir,font),path.join(DIST,'fonts',family+'.woff2'));
}
const catalog={chapters,lessons,stats:{lessons:lessons.length,chapters:chapters.length,codes:lessons.reduce((a,l)=>a+l.codeCount,0),figures:lessons.reduce((a,l)=>a+l.imageCount,0)}};
fs.writeFileSync(path.join(DIST,'content/index.json'),JSON.stringify(catalog));
fs.writeFileSync(path.join(DIST,'.nojekyll'),'');
fs.writeFileSync(path.join(DIST,'build-info.json'),JSON.stringify({lessons:lessons.length,chapters:chapters.length,assets:assets.size,assetBytes:[...assets.values()].reduce((a,b)=>a+b,0),mathExpressions:mathCache.size,mathErrors,outputs:outputCounts},null,2));
if(mathErrors.length)throw new Error('公式构建错误: '+mathErrors.slice(0,3).join('; '));
if(lessons.length!==185)throw new Error('Expected 185 lessons, got '+lessons.length);
console.log(`Built ${lessons.length} lessons, ${chapters.length} chapters, ${assets.size} images and ${mathCache.size} unique math expressions.`);
