"""
Evaluates a driving model on a set of CARLA routes wherein each route is evaluated on a separate machine in parallel.
This script generates the necessary shell files to run this on a SLURM cluster.
It also monitors the evaluation and resubmits crashed routes.
At the end all results files are aggregated and parsed.
Best run inside a tmux terminal.
"""

import subprocess
import time
from pathlib import Path
import os
import fnmatch
import ujson
import sys

# Our centOS is missing some c libraries.
# Usually miniconda has them, so we tell the linker to look there as well.
newlib = '/mnt/qb/work/geiger/gwb710/software/anaconda3/lib'
# export LD_LIBRARY_PATH='/mnt/qb/work/geiger/gwb710/software/anaconda3/lib'
if not newlib in os.environ['LD_LIBRARY_PATH']:
  os.environ['LD_LIBRARY_PATH'] += ':' + newlib


def create_run_eval_bash(bash_save_dir, results_save_dir, route_path, route, checkpoint, logs_save_dir,
                         carla_tm_port_start, benchmark, carla_root, only_vehicle_bb=0, perc_bb=0, perc_light=0, perc_stop=0):
  Path(f'{results_save_dir}').mkdir(parents=True, exist_ok=True)
  with open(f'{bash_save_dir}/eval_{route}.sh', 'w', encoding='utf-8') as rsh:
    rsh.write(f'''\
export CARLA_ROOT={carla_root}
export CARLA_SERVER=${{CARLA_ROOT}}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${{CARLA_ROOT}}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export PYTHONPATH="${{CARLA_ROOT}}/PythonAPI/carla/":"${{SCENARIO_RUNNER_ROOT}}":"${{LEADERBOARD_ROOT}}":${{PYTHONPATH}}
echo "CARLA_ROOT:" $CARLA_ROOT
echo "CARLA_SERVER:" $CARLA_SERVER
''')
    rsh.write(f"""
export PORT=$1
echo 'World Port:' $PORT
export TM_PORT=`comm -23 <(seq {carla_tm_port_start} {carla_tm_port_start+49} | sort) <(ss -Htan | awk '{{print $4}}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT
export ROUTES={route_path}{route}.xml
export SCENARIOS=leaderboard/data/scenarios/eval_scenarios.json
export TEAM_AGENT=team_code/perception_plant_agent.py
export TEAM_CONFIG=team_code/checkpoints/{checkpoint}/
export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=1
export CHECKPOINT_ENDPOINT={results_save_dir}/{route}.json
export DEBUG_CHALLENGE=0
export DATAGEN=0
export SAVE_PATH={logs_save_dir}
export DIRECT=1
export UNCERTAINTY_WEIGHT=1
export UNCERTAINTY_THRESHOLD=0.5
export HISTOGRAM=0
export BLOCKED_THRESHOLD=180
export TMP_VISU=0
export VISU_PLANT=0
export SLOWER=1
export STOP_CONTROL=1
export TP_STATS=0
export BENCHMARK={benchmark}
export ONLY_VEHICLE_BB={only_vehicle_bb}
export PERC_BB={perc_bb}
export PERC_LIGHT={perc_light}
export PERC_STOP={perc_stop}
""")
    rsh.write('''
python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=0 \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--timeout=600 \
--trafficManagerPort=${TM_PORT} \
--track MAP
''')


def make_jobsub_file(commands, job_number, exp_name, exp_root_name, partition):
  os.makedirs(f'evaluation/{exp_root_name}/{exp_name}/run_files/logs', exist_ok=True)
  os.makedirs(f'evaluation/{exp_root_name}/{exp_name}/run_files/job_files', exist_ok=True)
  job_file = f'evaluation/{exp_root_name}/{exp_name}/run_files/job_files/{job_number}.sh'
  qsub_template = f"""#!/bin/bash
#SBATCH --job-name={exp_name}{job_number}
#SBATCH --partition={partition}
#SBATCH -o evaluation/{exp_root_name}/{exp_name}/run_files/logs/qsub_out{job_number}.log
#SBATCH -e evaluation/{exp_root_name}/{exp_name}/run_files/logs/qsub_err{job_number}.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10gb
#SBATCH --time=00-06:00
#SBATCH --gres=gpu:1
"""
  for cmd in commands:
    qsub_template = qsub_template + f"""
{cmd}

"""

  with open(job_file, 'w', encoding='utf-8') as f:
    f.write(qsub_template)
  return job_file


def get_num_jobs(job_name, benchmark, username):
  len_usrn = len(username)
  num_running_jobs = int(
      subprocess.check_output(
          f"SQUEUE_FORMAT2='username:{len_usrn},name:130' squeue --sort V | grep {username} | grep {job_name}.*{benchmark} | wc -l",
          shell=True,
      ).decode('utf-8').replace('\n', ''))
  with open('max_num_jobs.txt', 'r', encoding='utf-8') as f:
    try:
      max_num_parallel_jobs = int(f.read())
    except:
      max_num_parallel_jobs = 1
  return num_running_jobs, max_num_parallel_jobs


def main():
  num_repetitions = 3
  experiment = 'EVAL' 
  model_dir = '/mnt/qb/work/geiger/gwb710/carla_garage/pretrained_models/longest6/plant_all_1'
  code_root = '/mnt/qb/work/geiger/gwb710/carla_garage'
  carla_root = '/mnt/qb/work/geiger/gwb710/carla_garage/carla'
  partition = 'gpu-2080ti-preemptable  '
  username = 'gwb710'
  epochs = ["model_0046"]# ,'model_0030']
  benchmarks = ["longest6"]# ["lav", "longest6"]
  grid_only_vehicle_bb = [0] #[0,1]
  grid_perc_bb = [1] #[0,1]
  grid_perc_light = [1]
  grid_perc_stop = [0]
  benchmark_length_dict = {"lav": 16*num_repetitions, "longest6": 36*num_repetitions}
  benchmark_length = sum([benchmark_length_dict[benchmark] for benchmark in benchmarks])
  total_num_evals_runs = len(benchmarks) * len(grid_only_vehicle_bb) * len(grid_perc_bb) * len(grid_perc_light) * len(grid_perc_stop)   
  total_num_evaluations = int(benchmark_length * (total_num_evals_runs/len(benchmarks)))
  num_evaluations = 0
  num_eval_runs = 0
  raw_experiment = experiment
  meta_jobs = {}
    
  for benchmark in benchmarks:
    for only_vehicle_bb in grid_only_vehicle_bb:
      for perc_bb in grid_perc_bb:
        for perc_light in grid_perc_light:
          for perc_stop in grid_perc_stop:
            experiment = raw_experiment
            num_eval_runs += 1

            print("\n\n\n")
            print(f"Starting Evaluation Run {num_eval_runs}/{total_num_evals_runs}")
            print("\n\n\n")

            if only_vehicle_bb:
              experiment += "_PURE_VEHICLE_BB"
            if perc_bb:
              experiment += "_PRED_BB"         
            if perc_light:
              experiment += "_PRED_LIGHT"    
            if perc_stop:
              experiment += "_PRED_STOP"      
            if raw_experiment == experiment or raw_experiment == f"{experiment}_PURE_VEHICLE_BB":
              experiment += "_FULL_PRIV"

            raw_experiment_name_stem = f"{raw_experiment}"
            experiment_name_stem = f'{experiment}_{benchmark}'
            exp_names_tmp = []
            for i in range(num_repetitions):
              exp_names_tmp.append(experiment_name_stem + f'_e{i}')
            route_path = f'leaderboard/data/{benchmark}_split/'
            route_pattern = '*.xml'

            carla_world_port_start = 10000
            carla_streaming_port_start = 20000
            carla_tm_port_start = 30000
            job_nr = 0

            for epoch in epochs:
              # Root folder in which each of the evaluation seeds will be stored
              experiment_name_root = experiment_name_stem + '_' + epoch
              exp_names = []
              for name in exp_names_tmp:
                exp_names.append(name + '_' + epoch)

              checkpoint = experiment
              checkpoint_new_name = checkpoint + '_' + epoch

              # Links the model file into team_code
              copy_model = True

              if copy_model:
                # copy checkpoint to my folder
                cmd = f'mkdir -p team_code/checkpoints/{checkpoint_new_name}'
                os.system(cmd)
                # cmd = f'cp {model_dir}/{checkpoint}/config.pickle team_code/checkpoints/{checkpoint_new_name}/'
                cmd = f'cp {model_dir}/config.pickle team_code/checkpoints/{checkpoint_new_name}/'
                os.system(cmd)
                cmd = f'ln -sf {model_dir}/{epoch}.pth team_code/checkpoints/{checkpoint_new_name}/model.pth'
                os.system(cmd)

              route_files = []
              for root, _, files in os.walk(route_path):
                for name in files:
                  if fnmatch.fnmatch(name, route_pattern):
                    route_files.append(os.path.join(root, name))

              for exp_name in exp_names:
                bash_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/run_bashs')
                results_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/results')
                logs_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/logs')
                bash_save_dir.mkdir(parents=True, exist_ok=True)
                results_save_dir.mkdir(parents=True, exist_ok=True)
                logs_save_dir.mkdir(parents=True, exist_ok=True)

              

              for exp_name in exp_names:
                for route in route_files:
                  route = Path(route).stem

                  bash_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/run_bashs')
                  results_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/results')
                  logs_save_dir = Path(f'evaluation/{experiment_name_root}/{exp_name}/logs')

                  commands = []

                  # Finds a free port
                  commands.append(
                      f'FREE_WORLD_PORT=`comm -23 <(seq {carla_world_port_start} {carla_world_port_start + 49} | sort) '
                      f'<(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
                  commands.append("echo 'World Port:' $FREE_WORLD_PORT")
                  commands.append(
                      f'FREE_STREAMING_PORT=`comm -23 <(seq {carla_streaming_port_start} {carla_streaming_port_start + 49} '
                      f'| sort) <(ss -Htan | awk \'{{print $4}}\' | cut -d\':\' -f2 | sort -u) | shuf | head -n 1`')
                  commands.append("echo 'Streaming Port:' $FREE_STREAMING_PORT")
                  commands.append(
                      f'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 {carla_root}/CarlaUE4.sh '
                      f'-carla-rpc-port=${{FREE_WORLD_PORT}} -nosound -carla-streaming-port=${{FREE_STREAMING_PORT}} -opengl &')
                  commands.append('sleep 180')  # Waits for CARLA to finish starting
                  create_run_eval_bash(bash_save_dir,
                                      results_save_dir,
                                      route_path,
                                      route,
                                      checkpoint_new_name,
                                      logs_save_dir,
                                      carla_tm_port_start,
                                      benchmark=benchmark,
                                      carla_root=carla_root,
                                      only_vehicle_bb=only_vehicle_bb,
                                      perc_bb=perc_bb,
                                      perc_light=perc_light,
                                      perc_stop=perc_stop)
                  commands.append(f'chmod u+x {bash_save_dir}/eval_{route}.sh')
                  commands.append(f'./{bash_save_dir}/eval_{route}.sh $FREE_WORLD_PORT')
                  commands.append('sleep 2')

                  carla_world_port_start += 50
                  carla_streaming_port_start += 50
                  carla_tm_port_start += 50

                  job_file = make_jobsub_file(commands=commands,
                                              job_number=job_nr,
                                              exp_name=experiment_name_stem,
                                              exp_root_name=experiment_name_root,
                                              partition=partition)
                  result_file = f'{results_save_dir}/{route}.json'

                  # Wait until submitting new jobs that the #jobs are at below max
                  num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=raw_experiment_name_stem, benchmark=benchmark, username=username)
                  print(f'{num_running_jobs}/{max_num_parallel_jobs} jobs are running...')
                  while num_running_jobs >= max_num_parallel_jobs:
                    num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=raw_experiment_name_stem, benchmark=benchmark, username=username)
                  time.sleep(0.05)
                  print(f'Submitting job {job_nr}/{len(route_files) * num_repetitions}: {job_file}, eval_run_number {num_eval_runs}/{total_num_evals_runs} eval_number {num_evaluations+1}/{total_num_evaluations}')
                  jobid = subprocess.check_output(f'sbatch {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                                  maxsplit=1)[-1]
                  meta_jobs[jobid] = (False, job_file, result_file, 0)

                  job_nr += 1
                  num_evaluations += 1

  training_finished = False
  while not training_finished:
    num_running_jobs, max_num_parallel_jobs = get_num_jobs(job_name=raw_experiment_name_stem, benchmark=benchmark, username=username)
    print(f'{num_running_jobs} jobs are running...')
    time.sleep(10)

    # resubmit unfinished jobs
    for k in list(meta_jobs.keys()):
      job_finished, job_file, result_file, resubmitted = meta_jobs[k]
      need_to_resubmit = False
      if not job_finished and resubmitted < 5:
        # check whether job is running
        if int(subprocess.check_output(f'squeue | grep {k} | wc -l', shell=True).decode('utf-8').strip()) == 0:
          # check whether result file is finished?
          if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f_result:
              evaluation_data = ujson.load(f_result)
            progress = evaluation_data['_checkpoint']['progress']

            if len(progress) < 2 or progress[0] < progress[1]:
              need_to_resubmit = True
            else:
              for record in evaluation_data['_checkpoint']['records']:
                if record['status'] == 'Failed - Agent couldn\'t be set up':
                  need_to_resubmit = True
                  print('Resubmit - Agent not setup')
                elif record['status'] == 'Failed':
                  need_to_resubmit = True
                elif record['status'] == 'Failed - Simulation crashed':
                  need_to_resubmit = True
                elif record['status'] == 'Failed - Agent crashed':
                  need_to_resubmit = True

            if not need_to_resubmit:
              # delete old job
              print(f'Finished job {job_file}')
              meta_jobs[k] = (True, None, None, 0)
          else:
            need_to_resubmit = True

      if need_to_resubmit:
        # Remove crashed results file
        if os.path.exists(result_file):
          print('Remove file: ', result_file)
          Path(result_file).unlink()
        print(f'resubmit sbatch {job_file}')
        jobid = subprocess.check_output(f'sbatch {job_file}', shell=True).decode('utf-8').strip().rsplit(' ',
                                                                                                         maxsplit=1)[-1]
        meta_jobs[jobid] = (False, job_file, result_file, resubmitted + 1)
        meta_jobs[k] = (True, None, None, 0)

    time.sleep(10)

    if num_running_jobs == 0:
      training_finished = True

  print('Evaluation finished. Start parsing results.')
  for benchmark in benchmarks:
    for only_vehicle_bb in grid_only_vehicle_bb:
      for perc_bb in grid_perc_bb:
        for perc_light in grid_perc_light:
          for perc_stop in grid_perc_stop:
            for epoch in epochs:
              experiment = raw_experiment
              if only_vehicle_bb:
                experiment += "_PURE_VEHICLE_BB"
              if perc_bb:
                experiment += "_PRED_BB"         
              if perc_light:
                experiment += "_PRED_LIGHT"    
              if perc_stop:
                experiment += "_PRED_STOP"      
              if raw_experiment == experiment:
                experiment += "_FULL_PRIV"
              experiment_name_root = f"{experiment}_{benchmark}_{epoch}"
              eval_root = f'{code_root}/evaluation/{experiment_name_root}'
              subprocess.check_call(
                  f'python {code_root}/tools/result_parser.py --xml {code_root}/leaderboard/data/{benchmark}.xml '
                  f'--results {eval_root} --log_dir {eval_root} --town_maps {code_root}/leaderboard/data/town_maps_xodr '
                  f'--map_dir {code_root}/leaderboard/data/town_maps_tga --device cpu '
                  f'--map_data_folder {code_root}/tools/proxy_simulator/map_data --subsample 1 --strict --visualize_infractions',
                  stdout=sys.stdout,
                  stderr=sys.stderr,
                  shell=True)
              subprocess.check_call(
                  f'python {code_root}/tools/result_parser_no_stop_infractions.py --results {eval_root}' ,
                  stdout=sys.stdout,
                  stderr=sys.stderr,
                  shell=True)
              subprocess.check_call(
                  f'python {code_root}/tools/result_parser_katrin.py --results {eval_root}' ,
                  stdout=sys.stdout,
                  stderr=sys.stderr,
                  shell=True)

if __name__ == '__main__':
  main()
