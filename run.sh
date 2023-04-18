MODEL=SM_MFEA_SBX
cp pyMSOO/RunModel/$MODEL/run.py run.py
cp pyMSOO/RunModel/$MODEL/cfg.yaml cfg.yaml
python run.py --lr 0.1 \
              --nb_generations 100 \
              --name_benchmark CEC17 \
              --number_tasks 10 \
              --ls_id_run '1-1-1' \
              --save_path './RESULTS/' \


        