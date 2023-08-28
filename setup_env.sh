pwd_root=$(pwd)

cd $CONDA_PREFIX/envs/garage
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

echo "export CARLA_ROOT=${pwd_root}/carla" >> ./etc/conda/activate.d/env_vars.sh
echo "export WORK_DIR=${pwd_root}" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\${CARLA_ROOT}/PythonAPI/carla" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=\$PYTHONPATH:\$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" >> ./etc/conda/activate.d/env_vars.sh
echo "export SCENARIO_RUNNER_ROOT=\${WORK_DIR}/scenario_runner" >> ./etc/conda/activate.d/env_vars.sh
echo "export LEADERBOARD_ROOT=\${WORK_DIR}/leaderboard" >> ./etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH="\${CARLA_ROOT}/PythonAPI/carla/":"\${SCENARIO_RUNNER_ROOT}":"\${LEADERBOARD_ROOT}":\${PYTHONPATH}" >> ./etc/conda/activate.d/env_vars.sh

echo "unset CARLA_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset CARLA_SERVER" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset PYTHONPATH" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset SCENARIO_RUNNER_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset LEADERBOARD_ROOT" >> ./etc/conda/deactivate.d/env_vars.sh

cd $pwd_root

