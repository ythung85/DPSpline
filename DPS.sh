#!/bin/bash
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --cpus-per-task=4 # Number of CPU cores per task
#SBATCH --mem=600mb # Total memory limit
#SBATCH --time=00:05:00 # Time limit hrs:min:sec
#SBATCH --gres=gpu:V100:1 # 1 V100 GPU (32 GB VRAM)
#SBATCH --mail-type=END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yhung7@gsu.edu # Where to send mail
#SBATCH --account=math1581s250 # Project name (RS00000, ECON0001, MAT0001 ...)
#SBATCH --partition=partition # partition requested (qBF, qTRD , qECON ...)
#SBATCH --output=output_%j.txt # nameof the output file, see following section for pattern switches
#SBATCH --error=error_%j.txt # name of the output file, see the following section for pattern switchest

mkdir -p $SCRATCH
cd $SCRATCH

# copying test_python.py from irods projects directory to /scratch directory
cp $IRODS_PROJECT/DPS.py $SCRATCH

module purge
module load python
module load torch
python3 DPS.py --data A --trainsize 400 --testsize 200 --Fin 2 --Fout 1 --nk 15 --nm 50 --rep 1 --nl 2 --lr 1e-1 --nepoch 10000 --fine_tune_epoch 1000

# copying output(results.txt) to the irods projects directory
cp simulation_output.npy $IRODS_PROJECT/
rm -rf $SCRATCH