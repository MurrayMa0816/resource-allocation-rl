"""Current chapter source/data entry. Importing this module executes no science."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from portable_paths import final_data_root,chapter_source_paths,chapter_environment
CHAPTER=4
METHODS=['TRoC/F1', 'PROPS', 'SAC+ReDo', 'MR.Q', 'BRO(Fast)', 'MAD-TD']

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','python'],nargs='?',default='inspect')
    args,rest=parser.parse_known_args(argv)
    if args.command=='inspect':
        if rest:parser.error('inspect takes no arguments')
        print(json.dumps({'chapter':CHAPTER,'methods':METHODS,'code_root':str(ROOT),'data_root':str(final_data_root()),'source_paths':list(map(str,chapter_source_paths(CHAPTER))),'filesystem_links':False,'automatic_queue':False},ensure_ascii=False,indent=2));return 0
    if rest[:1]==['--']:rest=rest[1:]
    if not rest:parser.error('python requires explicit Python arguments')
    return subprocess.run([sys.executable,'-B',*rest],env=chapter_environment(CHAPTER),check=True).returncode

if __name__=='__main__':raise SystemExit(main())
