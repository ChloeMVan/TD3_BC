#!/bin/bash
python3 o2o_BC.py --file_tag 'BC_Fast' --env 'halfcheetah-expert-v2' --root_num 2

python3 o2o_BC.py --file_tag 'BC_Mid' --env 'halfcheetah-expert-v2' --root_num 2.3

python3 o2o_BC.py --file_tag 'BC_Slow' --env 'halfcheetah-expert-v2' --root_num 2.5

python3 o2o_BC.py --file_tag 'BC_Fast' --env 'halfcheetah-medium-expert-v2' --root_num 2

python3 o2o_BC.py --file_tag 'BC_Mid' --env 'halfcheetah-medium-expert-v2' --root_num 2.3

python3 o2o_BC.py --file_tag 'BC_Slow' --env 'halfcheetah-medium-expert-v2' --root_num 2.5

python3 o2o_BC.py --file_tag 'BC_Fast' --env 'halfcheetah-medium-v2' --root_num 2

python3 o2o_BC.py --file_tag 'BC_Mid' --env 'halfcheetah-medium-v2' --root_num 2.3

python3 o2o_BC.py --file_tag 'BC_Slow' --env 'halfcheetah-medium-v2' --root_num 2.5

