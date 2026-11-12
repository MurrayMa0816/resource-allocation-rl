"""One corrected learner, one prespecified learning gate, one conditional study."""
from allocation_efficiency.common import ROOT,write_json
from allocation_efficiency.methods import METHODS

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'protocol.json').exists():raise FileExistsError('Registered protocol is immutable')
 p=dict(schema='calibrated-snapshot-v5',study_chapter=4,task_chapter=3,
  domains=['radio','v5'],methods=list(METHODS),
  purpose='Correct V4 learning foundation before testing snapshot control variates; no added method families, no changed task',
  environment='unchanged frozen chapter23_final, same A joint quota/spatial Gaussian, public state and feasible decoder',
  feedback='native terminalR only, all intermediate rewards0; no privileged model, simulator labels or reward shaping',
  device='cuda:0 float64',
  warmup=dict(episodes=32,collection=8,learner='identical native MC-PPO4epoch',charged=True),
  learners=dict(b1='unchanged strengthened A current8 x8epoch',f1='unchanged fixed4new+previous4 generalizedPPO/Vtrace',
   p2='calibrated current-trajectory natural PG; same reference16/inner4x8 schedule and frozen baseline as S2',
   s2='same calibrated learner, finite snapshot reference gradient plus exact cross-policy difference'),
  calibrated=dict(reference_episodes=16,inner_episodes=4,inner_updates=8,
   baseline='state-only critic frozen immediately after warm32 and after each cycle final critic fit, before next collection; shared throughout each reference and inner cycle',
   live_critic='four fullbatch MC epochs after each postwarm fresh collection, never used for the same cycle baseline',
   h='sum_active grad_log_pi(raw|state)*(terminalR-frozen_state_value); identical frozen baseline for both policies; no batch normalization',
   importance='full trajectory reference/current density ratio from full raw joint Gaussian; inactive events ratio1; no silent clipping or self normalization',
   reference_error='finite reference-gradient error retained; no conditional unbiasedness or theorem claims',
   update='one natural-gradient parameter proposal per complete reference or inner batch, exact visited-state KL backtracking',
   fisher='active current-state complete joint Gaussian KL Hessian plus0.01I',cg_iterations=10,damping=.01,kl_target=.01,
   acceptance='finite and empirical visited-state KL only, no fresh-MC-return ascent gate',
   actor_optimizer='native Adam during commonwarm32, manual NPG thereafter; Adam and manual steps accounted separately',
   uncertainty='visited-state KL and sample variance are empirical diagnostics, not true gradient MSE or global trust-region guarantees'),
  engineering=dict(methods=['p2','s2'],jobs=4,seeds=[0],episodes=96,evaluation_interval=48,dev_paths=4,holdout_paths=0,
   train_master=202611019301,test_master=202611019399),
  pilot=dict(methods=['b1','f1','p2'],jobs=6,seeds=[0],episodes=240,evaluation_interval=48,dev_paths=16,holdout_paths=0,
   train_master=202611019501,test_master=202611019599),
  learning_gate=dict(reference_episode=48,late_episodes=[192,240],minimum_gain=1.,baseline_gain_fraction=.5,
   require_both_late_above_reference=True,
   formula='G_p2=mean(dev192,dev240)-dev48; eachdomain G_p2>=max(1,0.5*max(G_b1,0)) AND dev192>dev48 AND dev240>dev48',
   interpretation='prespecified descriptive learning-ability gate, not superiority or statistical confirmation',
   action='any domain fails => stop this backend, do not run S2 formal, do not change parameters/thresholds or repeat pilot'),
  formal=dict(methods=list(METHODS),jobs=24,seeds=[90,91,92],episodes=480,evaluation_interval=48,dev_paths=32,holdout_paths=64,
   train_master=202611019101,test_master=202611019199,
   condition='engineering audit and both-domain pilot learning gate PASS',
   readout='all24 complete before formal efficacy read; no outcomes-driven tuning, new seeds or extensions'),
  threshold=dict(radio=30.,v5=65.,rule='two consecutive fixed development checks; actual confirmation at second point',
   censor='never-confirmed retains right censoring; no successful-only average; capped costs are not achievement times'),
  outcomes=dict(primary='new training environment steps to same-quality actual confirmation',
   process='normalized fixed-grid stochastic development AUC',quality='independent stochastic480 score, descriptive1point guard only',
   mechanism='frozen-baseline gradient/difference sample variance, reference-error proxies, NPG step acceptance/CG/FVP/KL; no oracle true-gradient labels',
   cost='train/development/holdout, reference refresh, Adam attempts/commits, manualNPG attempts/commits, critic, CG/FVP and timing separate'),
  comparisons=[['s2','p2'],['s2','b1'],['s2','f1']],
  analysis=dict(type='one pilot gate then conditional exploratory3seed study',no_confirmatory_tests=True,no_prior_data_pooled=True,
   success='same-quality meaningful fresh-interaction benefit vs both strong references, with endpoint quality retained; matched-control win alone insufficient',
   compute='P2/S2 match sampling/critic/NPG algorithm; cross-framework actual compute differs and must be shown'),
  boundaries=['baseline and NPG corrections are standard learning foundations, not claimed innovations',
   'pilot never enters formal effect estimates',
   'pilot failure rejects this configured backend for further investment, not all variance reduction',
   'finite horizon terminal condition does not itself prove a sparse-reward bottleneck',
   'freeze before real evaluation; no threshold search, extra families or positive-result chasing'])
 write_json(ROOT/'protocol.json',p)
 write_json(ROOT/'active_task.json',dict(status='running',stage='implementation',pending=['synthetic','math_audit','engineering4','pilot6','conditional_formal24','report']))
 print(ROOT/'protocol.json')
if __name__=='__main__':main()
