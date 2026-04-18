#!/bin/bash

# test
python3 o2o_BC.py --file_tag 'Test_BC_Fast' --env 'halfcheetah-expert-v2' --root_num 0.5

python3 o2o_BC.py --file_tag 'Test_BC_Fast_No_term' --env 'halfcheetah-expert-v2' --root_num 0

python3 o2o_BC.py --file_tag 'Test_BC_Mid' --env 'halfcheetah-expert-v2' --root_num 1.5

python3 o2o_BC.py --file_tag 'Test_BC_Slow' --env 'halfcheetah-expert-v2' --root_num 2.5

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

