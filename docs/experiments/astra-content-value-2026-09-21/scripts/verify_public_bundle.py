#!/usr/bin/env python3
"""Verify the explicit published byte manifest without requiring USD or a GPU."""
import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 data=json.loads((root/'publication_manifest.json').read_text());bad=[];seen=set()
 for item in data['files']:
  relative=item['path'];target=root/relative
  if relative in seen or not target.resolve().is_relative_to(root) or target.is_symlink():bad.append(relative);continue
  seen.add(relative)
  if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:bad.append(relative)
  if not item['projection'] and item['sha256']!=item['original_retained_sha256']:bad.append(relative+' original digest')
 print(json.dumps({'checked_files':len(data['files']),'mismatches':bad}));raise SystemExit(bool(bad))
if __name__=='__main__':main()
