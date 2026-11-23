#!/bin/bash
# wait for SI-SDE+corr seed 1 to finish, stop the rest of that loop (seeds 2,3,0), then run run.sh
L1="/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/results/_runs/si_corr_seeds/full/logs/32_2_128_2__M100__traj11__ours_shared_jac10__fc25_8797d2ae__si_sde__seed1.log"
until grep -q '^exit ' "$L1" 2>/dev/null; do sleep 30; done
tmux kill-session -t rb_sicorr 2>/dev/null; sleep 5
pkill -f "si_corr_seeds/run.sh" 2>/dev/null; pkill -f "si_corr_seeds/full/metrics/.*seed[230]" 2>/dev/null; sleep 10
echo "seed-1 finished; stopped seeds 2,3,0 at $(date)" >> "/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/results/_runs/si_corr_seeds/README.md"
bash "/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/results/_runs/tic16_m100/run.sh" > "/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/results/_runs/tic16_m100/queue.log" 2>&1
