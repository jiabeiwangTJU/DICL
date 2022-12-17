#!/bin/bash
config_name='prw_dicl'
num_epoch='20'
python ../../tools/test.py ../../configs/dicl/${config_name}.py work_dirs/${config_name}/epoch_${num_epoch}.pth --eval bbox --out work_dirs/${config_name}/results_1000.pkl >work_dirs/${config_name}/log_tmp_${num_epoch}.txt 2>&1
python ../../tools/test_results_prw.py ${config_name} >work_dirs/${config_name}/result_${config_name}_${num_epoch}.txt 2>&1
