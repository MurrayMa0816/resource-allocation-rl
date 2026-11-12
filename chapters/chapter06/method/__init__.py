"""C6 algorithm source lookup; no learner, tensor or model is imported."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[3]
if str(PROJECT)not in sys.path:sys.path.insert(0,str(PROJECT))
from portable_paths import project_root,chapter_data_root

def source_paths():
 root=project_root()
 return {'SeDVal':root/'frozen/chapter06/reasonable89_20261109/method','PASTA':root/'chapters/chapter05/_frozen/formal285/baselines/pasta','COLA':root/'chapters/chapter05/_frozen/formal285/baselines/cola','C6_adapters':root/'applications/irrigation/learning_comparison_20261109/methods'}
