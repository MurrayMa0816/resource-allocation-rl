"""Pinned primary sources; no imports of simulation or learning frameworks."""
from pathlib import Path
import hashlib
import subprocess

ROOT=Path(__file__).resolve().parent
SOURCES={
    'props':dict(name='PROPS',venue='TMLR',year=2026,
        title='On-Policy Policy Gradient Reinforcement Learning Without On-Policy Sampling',
        paper='https://openreview.net/forum?id=nCoyFp8uO1',
        official='https://github.com/NicholasCorrado/PPOROS',
        commit='a0c8708529047f85dcf98f90d5ea264dd43aeba2',license='not declared',
        role='adaptive behavior sampling and recent-data policy-gradient learning'),
    'bro':dict(name='BRO',venue='NeurIPS',year=2024,
        title='Bigger, Regularized, Optimistic: scaling for compute and sample efficient continuous control',
        paper='https://proceedings.nips.cc/paper_files/paper/2024/hash/cd3b5d2ed967e906af24b33d6a356cac-Abstract-Conference.html',
        official='https://github.com/naumix/BiggerRegularizedOptimistic',
        commit='5870a13d4bf4f66eeddfc291a59df6d2009de600',license='MIT',
        role='regularized large critics, high replay ratio and optimistic exploration',
        reserve_only=True,reason='bro_full has an actual historical chapter-3 readiness experiment'),
    'fog':dict(name='FoG',venue='ICML',year=2025,
        title='A Forget-and-Grow Strategy for Deep Reinforcement Learning Scaling in Continuous Control',
        paper='https://proceedings.mlr.press/v267/kang25c.html',
        official='https://github.com/nothingbutbut/FoG',
        commit='de74db3a295971c9edbdda31efae3b35b8111c4f',license='not declared',
        role='experience decay, critic network growth and adaptive replay learning'),
    'addq':dict(name='ADDQ (ADQRSAC)',venue='ICML',year=2025,
        title='ADDQ: Adaptive distributional double Q-learning',
        paper='https://proceedings.mlr.press/v267/doring25a.html',
        official='https://github.com/BommeHD/ADDQ',
        commit='60f91bcab1fc53d602d8631b98bc639556393166',license='MIT',
        role='adaptive value-bias control; value-learning comparator, not a dedicated replay method'),
    'tdmpc2':dict(name='TD-MPC2',venue='ICLR',year=2024,
        title='TD-MPC2: Scalable, Robust World Models for Continuous Control',
        paper='https://proceedings.iclr.cc/paper_files/paper/2024/file/cf73d57b6dcda32b293df7c2d5341f49-Paper-Conference.pdf',
        official='https://github.com/nicklashansen/tdmpc2',
        commit='332f19e4aba4bffe2dc736f45ea7f7bdc585903a',license='MIT',
        role='learned latent dynamics and model predictive control',
        version_note='Official maintenance revision dated 2026-10-20 includes episodic extension'),
    'pgr':dict(name='PGR',venue='ICLR (Oral)',year=2025,
        title='Prioritized Generative Replay',paper='https://openreview.net/forum?id=5IkDAfabuo',
        official='https://github.com/renwang435/pgr',
        commit='b52b41b681f67608f8a5678211ce6cd5d933f9cc',license='MIT',
        role='priority-conditioned generative replay',
        submodule_commit='7b5d1bff39291a57325a2836bd397a55728960bb'),
    'crossq':dict(name='CrossQ',venue='ICLR',year=2024,
        title='CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity',
        paper='https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf',
        official='https://github.com/adityab/CrossQ',
        commit='a318a266679bec3f9d93c44e029860a6da077c89',license='MIT',
        reserve_only=True,reason='Historical chapter-3 model/diagnostic records; excluded to avoid repetition')
}
FINAL_METHODS=('props',)  # Current legacy-PROPS entry only; current other methods live in replacement_candidates.


def verify(method):
    if method != 'props':
        raise ValueError('This maintained legacy entry exposes PROPS only; use current replacement_candidates for BRO/MR.Q/ReDo/MAD-TD.')
    from chapters.chapter04.source_provenance import verify_props_source
    ticket = verify_props_source(ROOT/'vendor'/'props', SOURCES['props']['commit'])
    return dict(SOURCES['props'], actual_commit=ticket['actual_sha'], tracked_source_changes=[],
                local_repository=str(ROOT/'vendor'/'props'), source_ticket=ticket)
