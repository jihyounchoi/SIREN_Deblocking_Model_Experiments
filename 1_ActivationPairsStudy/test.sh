#!/bin/bash

#SBATCH --job-name Siren_TEST
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=5G
#SBATCH --time 1-0
#SBATCH --partition batch_ugrad
#SBATCH --nodelist=aurora-g4
#SBATCH -o logs/slurm-%A-%x.out

current_time="$(date '+%Y-%m-%d %H:%M:%S')"
start_time=$SECONDS

echo "Shell Script Starts Execution : $current_time"
echo -e "\n\n"

python test.py

end_time=$SECONDS
current_time="$(date '+%Y-%m-%d %H:%M:%S')"

echo "Shell Script Finishes Execution : $current_time"
echo "Total Execution Time : $((end_time - start_time)) seconds"
echo "End of Log File"

exit 0