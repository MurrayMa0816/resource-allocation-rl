"""Register a bounded new family before any learner performance is observed."""
from allocation_reuse.common import ROOT,write_json
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'protocol.json').exists():raise FileExistsError('Immutable protocol exists')
    p=dict(schema='chapter4-prefix-reuse-v1',objective='Fewer new physical interactions to fixed allocation quality by reliable completed-trajectory reuse',
        hypothesis='Weight old-state learning by frozen exclusive history likelihood ratios; compare its variance tradeoff with a bounded self-normalized form',
        domains=['radio','v5'],methods=['f1','p1','p2'],device='cuda:0',inference='uniform CPU frozen mirrors; unchanged CUDA learner and random stream',dtype='float64',
        environment='frozen native chapter23_final; same A_FULL joint quota+spatial Gaussian and feasible decoder',
        feedback='only native terminal return; intermediate zero; no added private/model/oracle labels',
        arms=dict(f1='unchanged FrequentPPO current4+immediately previous4',p1='F1 with raw exclusive-prefix state weights',
            p2='F1 with prefix weights preclipped[.5,2] then normalized over old4 at each physical period; fresh4 exactly1'),
        weighting=dict(formula='h[j,t]=exp(sum(u<t) mask[j,u]*(log pi_anchor-log mu)); h[j,0]=1',
            anchor='frozen pre-update policy; weights detached once',fresh='four fresh trajectories have weight exactly1',
            inactive='prefix ratio1; all four old paths remain in per-time normalization; original mask on learning, never terminal',
            p1='raw, no self-normalization; log numeric guard[-40,40] counted',
            p2='clip h to[.5,2], divide each period by old4 mean, final range[4/13,16/7]',
            hooks=['weighted advantage mean/std','actor surrogate with native fixed batch denominator','value MSE with weighted mask denominator',
                'macroKL and empirical TV controller','analytic KL diagnostics'],
            unchanged=['full raw event likelihood','generalized clip center','frozen finite Vtrace suffix targets','behavior record/validation','Adam/gradient clipping']),
        warmup=dict(episodes=32,fresh=8,learner='original raw-event MC-PPO4epochs, identical allarms',charged=True),
        postwarm=dict(fresh=4,old=4,epochs=4,minibatch=64,grad_clip=.5,gamma=1.,trace_lambda=1.,rho_cap=1.,c_cap=1.,
            epsilon=.2/1.5,actor_lr_initial=.0003,value_lr=.001,TV_alpha=.03,TV_beta=.5,actor_lr_bounds=[.00001,.001],macroKL_stop=.12),
        engineering=dict(methods=['f1','p1','p2'],jobs=6,seeds=[0],episodes=48,evaluation_interval=48,dev_paths=4,holdout_paths=0,
            train_master=202611021701,test_master=202611021799),
        formal=dict(methods=['f1','p1','p2'],jobs=18,seeds=[120,121,122],episodes=480,evaluation_interval=48,dev_paths=32,holdout_paths=64,
            train_master=202611021501,test_master=202611021599,condition='tests, independent source review and engineering PASS',
            readout='all18 complete; no tuning/seed replacement/outcome-dependent extension'),
        threshold=dict(radio=30.,v5=65.,rule='two consecutive fixed stochastic development means >=target; Nq at second confirmation',
            censor='unconfirmed by480 right-censored; never average only successful seeds or call budget an achieved Nq'),
        outcomes=dict(primary='Nq new TRAIN physical steps to fixed quality confirmation',process='normalized trapezoid AUC on fixed stochastic development grid; curve x new steps',
            quality='independent64 stochastic plus64 mean-control, reported separately',quality_guard='descriptive paired1point endpoint guard, not statistical noninferiority',
            costs='all train/dev/holdout interactions and wall/collection/update time, SGD count, GPU memory',
            mechanism='prefix ESS by time, per-event ESS, max weights/cap counts, TV/clip/LR/gradient/KL; mechanism is not proof of efficacy'),
        analysis=dict(seeds='exploratory3; paired each candidate versus same-round F1; no prior cohort pooling',
            selection='requires consistent interaction savings with retained independent quality; isolated mechanism gain is insufficient',
            stop='if neither candidate improves this tradeoff, stop the prefix family and retain F1; do not relabel failure as new problem evidence'),
        parallel=dict(max_workers=18,threads_per_process=1,existing_tasks='left untouched',timing='shared GPU/CPU, not exclusive throughput'),
        limitations=['GePPO mixed-state surrogate is a deliberate valid design, not a bug',
            'prefix is a path IS estimator of state expectations, not conditional marginal occupancy ratio',
            'latent nullspace likelihood variation can exist without state change',
            'PPO clipping, clipped Vtrace and regularization remain biased; no unbiased whole-learner or convergence claim',
            'state correction and prefix IS are prior ideas; this is an incremental task-specific test',
            'terminal-only feedback retained; no terminal-specific advantage claimed'])
    write_json(ROOT/'protocol.json',p)
    write_json(ROOT/'active_task.json',dict(status='running',stage='code_and_mechanism',pending=['code_review','engineering6','formal18','analysis','report']))
    print(ROOT/'protocol.json')
if __name__=='__main__':main()
