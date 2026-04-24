#!/bin/bash

# nohup bash -c 'cd /home/chloe/TD3_BC; source /home/chloe/miniconda3/etc/profile.d/conda.sh; conda activate td3_env; export TMPDIR=/home/chloe/tmp; bash manual_runs.sh' > test_run.log 2>&1 &

# test

# python3 o2o_BC.py --file_tag 'First_BC_Exp' --function_type 'exp' --env 'halfcheetah-medium-v2' --param_num 2.3

# python3 o2o_BC.py --file_tag 'First_BC_Mid_Linear' --function_type 'linear' --env 'halfcheetah-medium-v2'

# python3 o2o_BC.py --file_tag 'First_BC_Slow_Log' --function_type 'log' --env 'halfcheetah-medium-v2'

# python3 o2o_BC.py --file_tag 'First_BC_Exp' --function_type 'exp' --env 'halfcheetah-expert-v2' --param_num 2.3

# python3 o2o_BC.py --file_tag 'First_BC_Mid_Linear' --function_type 'linear' --env 'halfcheetah-expert-v2'

# python3 o2o_BC.py --file_tag 'First_BC_Slow_Log' --function_type 'log' --env 'halfcheetah-expert-v2'


python3 o2o_BC.py --file_tag 'Test_record_BC_Exp' --function_type 'exp' --env 'hopper-medium-v2' --param_num 2.3 --max_timesteps 20000

python3 o2o_BC.py --file_tag 'First_BC_Mid_Linear' --function_type 'linear' --env 'hopper-medium-v2'

python3 o2o_BC.py --file_tag 'First_BC_Slow_Log' --function_type 'log' --env 'hopper-medium-v2'

python3 o2o_BC.py --file_tag 'First_BC_Exp' --function_type 'exp' --env 'hopper-expert-v2' --param_num 2.3

python3 o2o_BC.py --file_tag 'First_BC_Mid_Linear' --function_type 'linear' --env 'hopper-expert-v2'

python3 o2o_BC.py --file_tag 'First_BC_Slow_Log' --function_type 'log' --env 'hopper-expert-v2'



