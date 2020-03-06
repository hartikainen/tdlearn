#!/bin/bash


EXPERIMENT_NAMES=(
    "baird"
    "boyan"
    # "budget_disc"
    # "corrupted_chain"

    "disc_random_off"
    # "disc_random_off_conv"
    "disc_random_on"
    # "disc_random_on_conv"

    "link20_imp_offpolicy"
    "link20_imp_onpolicy"

    "lqr_full_offpolicy"
    "lqr_full_onpolicy"
    "lqr_gauss_onpolicy"

    "lqr_imp_offpolicy"
    "lqr_imp_onpolicy"
    "lqr_imp_onpolicy_unnorm"

    "swingup_gauss_offpolicy"
    "swingup_gauss_onpolicy"
    # "swingup_regtd"
)

for experiment in ${EXPERIMENT_NAMES[@]}; do
    echo "experiment: ${experiment}"
    python -m "experiments.${experiment}"
done
