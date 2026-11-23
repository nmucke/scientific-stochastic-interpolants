#!/usr/bin/env bash
# start each main 3-sequence SI run when the quick run on the same cores has finished
PE="/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"; D="/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/results/_runs/abs1_quick"
pending="si_shared si_fc si_jacfree"
while [ -n "$pending" ]; do
  left=""
  for k in $pending; do
    if tmux has-session -t rb_abs_q_$k 2>/dev/null; then left="$left $k"
    else tmux new-session -d -s rb_abs_m_$k "bash \"$PE/results/_runs/run_job.sh\" abs3_$k 'bash \"$D/cmd_main_${k}_defer.sh\"'"; echo "$(date -Is) started main $k"; fi
  done
  pending="$left"; [ -n "$pending" ] && sleep 30
done
