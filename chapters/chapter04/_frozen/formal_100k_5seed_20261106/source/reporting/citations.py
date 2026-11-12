"""Primary-source identities read during preparation; existing keys are reused."""
from __future__ import annotations
import re
from pathlib import Path

SOURCES={
 'props':dict(key='corrado2026props',title='PROPS',source_identity='Reuse publisher URL in existing canonical thesis bibliography entry.',read='Local published author version props_tmlr_author_v3.pdf, method pages5--7; existing reading_audit.rst retained.'),
 'pgr':dict(key='wang2025pgr',title='Prioritized Generative Replay',source_identity='Reuse publisher URL in existing canonical thesis bibliography entry.',read='Local ICLR2025 pgr_iclr2025.pdf method pages3--5; existing reading_audit.rst retained.'),
 'redo':dict(key='sokar2023dormant',title='The Dormant Neuron Phenomenon in Deep Reinforcement Learning',url='https://proceedings.mlr.press/v202/sokar23a.html',pdf='https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf',read='Publisher full paper sections3--4 including Algorithm1 actually read; implementation-specific threshold and recycling units kept separate.'),
 'mrq':dict(key='fujimoto2025mrq',title='Towards General-Purpose Model-Free Reinforcement Learning',url='https://proceedings.iclr.cc/paper_files/paper/2025/hash/cf6501108fced72ee5c47e2151c4e153-Abstract-Conference.html',pdf='https://proceedings.iclr.cc/paper_files/paper/2025/file/cf6501108fced72ee5c47e2151c4e153-Paper-Conference.pdf',read='Official ICLR proceedings full-text sections4 and4.2 including equations13--15 actually retrieved/read; initial direct PDF fetch timed out, cached publisher full text read.'),
 'bro':dict(key='nauman2024bro',title='Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control',url='https://papers.neurips.cc/paper_files/paper/2024/file/cd3b5d2ed967e906af24b33d6a356cac-Paper-Conference.pdf',read='Official NeurIPS full paper title and section2.2 actually read; Fast RR2 differs from default RR10.')}

ENTRIES={
 'redo':r'''@inproceedings{sokar2023dormant,
 author = {Sokar, Ghada and Agarwal, Rishabh and Castro, Pablo Samuel and Evci, Utku},
 title = {The Dormant Neuron Phenomenon in Deep Reinforcement Learning},
 booktitle = {Proceedings of the 40th International Conference on Machine Learning},
 year = {2023}, volume = {202}, pages = {32145--32168}, publisher = {PMLR},
 url = {https://proceedings.mlr.press/v202/sokar23a.html}
}''',
 'mrq':r'''@inproceedings{fujimoto2025mrq,
 author = {Fujimoto, Scott and D'Oro, Pierluca and Zhang, Amy and Tian, Yuandong and Rabbat, Michael},
 title = {Towards General-Purpose Model-Free Reinforcement Learning},
 booktitle = {International Conference on Learning Representations}, year = {2025},
 url = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/cf6501108fced72ee5c47e2151c4e153-Abstract-Conference.html}
}''',
 'bro':r'''@inproceedings{nauman2024bro,
 author = {Nauman, Michal and Ostaszewski, Mateusz and Jankowski, Krzysztof and Mi{\l}o{\'s}, Piotr and Cygan, Marek},
 title = {Bigger, Regularized, Optimistic: scaling for compute and sample-efficient continuous control},
 booktitle = {Advances in Neural Information Processing Systems}, year = {2024}, volume = {37},
 url = {https://papers.neurips.cc/paper_files/paper/2024/file/cd3b5d2ed967e906af24b33d6a356cac-Paper-Conference.pdf}
}'''}

def norm(s):return re.sub('[^a-z0-9]','',s.lower())
def resolve(thesis):
    """Find existing canonical entries before appending a genuinely missing work."""
    entries=[]
    for p in (Path(thesis)/'chapters').rglob('*.bib'):
        text=p.read_text()
        for m in re.finditer(r'@\w+\s*\{\s*([^,\s]+)\s*,',text):
            end=text.find('\n@',m.end());chunk=text[m.start():end if end!=-1 else len(text)]
            title=re.search(r'title\s*=\s*\{(.*?)\}\s*[,\n]',chunk,re.I|re.S)
            entries.append((m.group(1),norm(title.group(1))if title else'',p,chunk))
    keys={e[0]for e in entries};mapping={};append=[]
    for method,source in SOURCES.items():
        if method in ('props','pgr'):
            if source['key']not in keys:raise ValueError('Expected existing literature missing: '+source['key'])
            mapping[method]=source['key'];continue
        matches=[e for e in entries if e[1]==norm(source['title'])]
        if len(matches)>1:raise ValueError('Duplicate canonical bibliography work: '+method)
        if matches:mapping[method]=matches[0][0]
        elif source['key']in keys:raise ValueError('Bibliography key collision: '+source['key'])
        else:mapping[method]=source['key'];append.append(ENTRIES[method])
    return mapping,'\n\n'.join(append)
