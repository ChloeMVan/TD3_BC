#!/bin/bash

# Script to reproduce results

# Fast - 2, Mid - 2.3, Slow - 2.5

envs=(
	# "halfcheetah-random-v0"
	# "hopper-random-v0"
	# "walker2d-random-v2"
	"halfcheetah-medium-v2"
	# "hopper-medium-v0"
	# "walker2d-medium-v2"
	# "halfcheetah-expert-v2"
	# "hopper-expert-v0"
	# "walker2d-expert-v2"
	# "halfcheetah-medium-expert-v0"
	# "hopper-medium-expert-v0"
	# "walker2d-medium-expert-v2"
	# "halfcheetah-medium-replay-v0"
	# "hopper-medium-replay-v0"
	# "walker2d-medium-replay-v0"
	)

for ((i=0;i<1;i+=1))
do 
	for env in ${envs[*]}
	do
		python o2o_BC.py \
        --file_tag 'BC_Fast' \
		--env $env \
		--seed $i \
		--root_num 2
	done
done
