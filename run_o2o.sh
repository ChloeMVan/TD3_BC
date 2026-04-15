#!/bin/bash

# Script to reproduce results

envs=(
	"halfcheetah-expert-v2"
	)

for ((i=0;i<1;i+=1))
do 
	for env in ${envs[*]}
	do
		python o2o.py \
		--env $env \
		--seed $i
	done
done
