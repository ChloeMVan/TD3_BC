#!/bin/bash

# test
python3 o2o_BC.py --file_tag 'Test_BC_Fast' --env 'halfcheetah-expert-v2' --root_num 0.5

python3 o2o_BC.py --file_tag 'Test_BC_Mid' --env 'halfcheetah-expert-v2' --root_num 1.5

python3 o2o_BC.py --file_tag 'Test_BC_Slow' --env 'halfcheetah-expert-v2' --root_num 2.5


python3 o2o_BC.py --file_tag 'Test_BC_Mid_Linear' --function_type 'linear' --env 'halfcheetah-expert-v2'

python3 o2o_BC.py --file_tag 'Test_BC_Slow_Log' --function_type 'log' --env 'halfcheetah-expert-v2'

nohup bash -c 'cd /home/chloe/TD3_BC; source /home/chloe/miniconda3/etc/profile.d/conda.sh; conda activate td3_env; export TMPDIR=/home/chloe/tmp; python3 o2o_BC.py --file_tag "Test_BC_Mid_Linear" --function_type "linear" --env "halfcheetah-expert-v2"; python3 o2o_BC.py --file_tag "Test_BC_Slow_Log" --function_type "log" --env "halfcheetah-expert-v2"' > test_run.log 2>&1 &

# prior
python3 o2o_BC.py --file_tag 'BC_Fast' --env 'halfcheetah-expert-v2' --root_num 2

python3 o2o_BC.py --file_tag 'BC_Mid' --env 'halfcheetah-expert-v2' --root_num 2.3

python3 o2o_BC.py --file_tag 'BC_Slow' --env 'halfcheetah-expert-v2' --root_num 2.5

python3 o2o_BC.py --file_tag 'BC_Fast' --env 'halfcheetah-medium-expert-v2' --root_num 2

python3 o2o_BC.py --file_tag 'BC_Mid' --env 'halfcheetah-medium-expert-v2' --root_num 2.3

python3 o2o_BC.py --file_tag 'BC_Slow' --env 'halfcheetah-medium-expert-v2' --root_num 2.5

python3 o2o_BC.py --file_tag 'BC_Fast' --env 'halfcheetah-medium-v2' --root_num 2

python3 o2o_BC.py --file_tag 'BC_Mid' --env 'halfcheetah-medium-v2' --root_num 2.3

python3 o2o_BC.py --file_tag 'BC_Slow' --env 'halfcheetah-medium-v2' --root_num 2.5

python3 o2o.py --file_tag 'o2o' --env 'halfcheetah-expert-v2'

python3 online.py --file_tag 'org' --env 'walker2d-random-v2' 

