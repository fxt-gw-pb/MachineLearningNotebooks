from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
formulas=[];issues=[]
for path in ROOT.glob('[0-9]*/*.ipynb'):
 nb=json.loads(path.read_text())
 for i,cell in enumerate(nb['cells']):
  if cell['cell_type']!='markdown':continue
  source=''.join(cell['source'])
  text=re.sub(r'```.*?```','',source,flags=re.S)
  text=re.sub(r'`[^`]*`','',text)
  delimiters=list(re.finditer(r'(?<!\\)(\$\$|\$)',text));j=0
  while j<len(delimiters):
   start=delimiters[j]
   if j+1>=len(delimiters) or delimiters[j+1][0]!=start[0]:
    issues.append({'notebook':str(path.relative_to(ROOT)),'cell':i,'issue':'unmatched math delimiter','context':text[max(0,start.start()-80):start.end()+180]});j+=1;continue
   end=delimiters[j+1];tex=text[start.end():end.start()]
   formulas.append({'notebook':str(path.relative_to(ROOT)),'cell':i,'tex':tex,'display':start[0]=='$$'})
   if start[0]=='$' and '\n\n' in tex:issues.append({'notebook':str(path.relative_to(ROOT)),'cell':i,'issue':'inline math crosses paragraphs','context':tex})
   j+=2
(ROOT/'reports'/'formulas.json').write_text(json.dumps(formulas,ensure_ascii=False))
(ROOT/'reports'/'math_delimiter_issues.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2))
print('Formulas:',len(formulas),'delimiter issues:',len(issues))
for e in issues[:30]:print(e)
