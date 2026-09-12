import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../dist');
const index=JSON.parse(fs.readFileSync(path.join(ROOT,'content/index.json'),'utf8'));
assert.equal(index.lessons.length,185);assert.equal(index.chapters.length,20);
const ids=new Set(index.lessons.map(l=>l.id));assert.equal(ids.size,185);
let codes=0,images=0,plots=0,math=0;
for(const summary of index.lessons){
  const lesson=JSON.parse(fs.readFileSync(path.join(ROOT,'content',summary.id+'.json'),'utf8'));
  assert.equal(lesson.id,summary.id);assert.equal(lesson.status,'passed');
  assert(lesson.cells.some(c=>c.type==='markdown'));assert(lesson.cells.some(c=>c.type==='code'));
  const headingIds=new Set();
  for(const cell of lesson.cells){
    if(cell.html){
      assert(!cell.html.includes('MLMATHPLACEHOLDER'));assert(!/<script\b/i.test(cell.html));
      math+=(cell.html.match(/role="math"/g)||[]).length;
      for(const m of cell.html.matchAll(/id="(section-\d+)"/g))headingIds.add(m[1]);
      for(const m of cell.html.matchAll(/<img [^>]*src="([^"]+)"/g)){
        if(m[1].startsWith('assets/'))assert(fs.existsSync(path.join(ROOT,m[1])),m[1]);
      }
    }
    if(cell.type==='code'){
      codes++;assert(cell.source.length>0);assert(cell.highlight.length>0);
      for(const output of cell.outputs){
        assert.notEqual(output.type,'error');
        if(output.src)assert(fs.existsSync(path.join(ROOT,output.src)),output.src);
        if(output.type==='image')images++;
        if(output.type==='plotly')plots++;
        if(output.type==='html')assert(!/<script\b/i.test(output.html));
      }
    }
  }
  for(const heading of lesson.headings)assert(headingIds.has(heading.id),`${lesson.id} broken heading ${heading.id}`);
}
for(const chapter of index.chapters)for(const id of chapter.lessons)assert(ids.has(id));
for(const file of ['index.html','app.js','style.css','favicon.svg','vendor/lucide.js','vendor/plotly.js','fonts/manrope.woff2','fonts/jetbrains-mono.woff2'])assert(fs.statSync(path.join(ROOT,file)).size>0,file);
assert.equal(codes,866);assert.equal(images,539);assert.equal(plots,3);assert(math>8000);
console.log(`PASS: 185 lessons; ${codes} code cells; ${images} rendered figures; ${plots} interactive figures; ${math} math expressions; all content assets and anchors resolve.`);
