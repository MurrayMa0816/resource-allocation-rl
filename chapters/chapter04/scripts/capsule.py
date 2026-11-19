"""Validate and temporarily reconstruct this chapter's selected source tree."""
from pathlib import Path
import argparse, hashlib, json
from source_layout import manifest, source_capsule, verify

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--check',action='store_true',help='Check source hashes and temporary ordinary-file reconstruction only')
 parser.parse_args()
 count=verify()
 with source_capsule() as root:
  for relative,record in manifest()['files'].items():
   path=root/relative
   if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
    raise ValueError('Source capsule differs: '+relative)
 print(json.dumps({'chapter':manifest()['chapter'],'status':'PASS','selected_source_files':count,'temporary_source_tree_removed':True,'model_loading':0,'environment_steps':0},indent=2))
if __name__=='__main__':main()
