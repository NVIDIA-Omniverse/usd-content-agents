#!/usr/bin/env python3
"""Fetch exact public original bytes and verify SHA256; never execute source."""
import argparse,concurrent.futures,hashlib,json,shutil,tempfile,urllib.request,zipfile
from pathlib import Path
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
def fetch(row,root):
 target=(root/row['destination']).resolve();target.relative_to(root)
 if target.is_file() and sha(target)==row['sha256']:return {'path':row['destination'],'status':'already_verified'}
 target.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.source-fetch-',dir=target.parent) as tmp:
  downloaded=Path(tmp)/'download'
  req=urllib.request.Request(row['url'],headers={'User-Agent':'Astra-Content-Value-Reproducer/1.0'})
  with urllib.request.urlopen(req,timeout=180) as response,downloaded.open('wb') as out:shutil.copyfileobj(response,out,1024*1024)
  if 'zip_member' in row:
   if sha(downloaded)!=row['archive_sha256']:raise ValueError('Original archive digest changed: '+row['case_id'])
   extracted=Path(tmp)/'extracted'
   with zipfile.ZipFile(downloaded) as archive:
    with archive.open(row['zip_member']) as src,extracted.open('wb') as out:shutil.copyfileobj(src,out,1024*1024)
   downloaded=extracted
  if downloaded.stat().st_size!=row['bytes'] or sha(downloaded)!=row['sha256']:raise ValueError('Source bytes do not match the frozen original: '+row['destination'])
  downloaded.replace(target)
 return {'path':row['destination'],'status':'downloaded_and_verified'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--case',action='append');p.add_argument('--workers',type=int,default=4);p.add_argument('--verify-only',action='store_true');a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
 rows=json.loads(a.manifest.read_text())['files'];rows=[r for r in rows if not a.case or r['case_id'] in a.case]
 if a.verify_only:
  bad=[r['destination'] for r in rows if not (root/r['destination']).is_file() or sha(root/r['destination'])!=r['sha256']]
  print(json.dumps({'verified_files':len(rows)-len(bad),'mismatches':bad}));raise SystemExit(bool(bad))
 with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(a.workers,8))) as pool:
  for result in pool.map(lambda row:fetch(row,root),rows):print(json.dumps(result),flush=True)
if __name__=='__main__':main()
