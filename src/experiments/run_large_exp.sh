#!/bin/bash
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=2 # Run a single task
#SBATCH --cpus-per-task=4 # Number of CPU cores per task
#SBATCH --mem=32gb # Total memory limit
#SBATCH --time=07:00:00 # Time limit hrs:min:sec
#SBATCH --gres=gpu:V100:1 # 1 V100 GPU (32 GB VRAM)
#SBATCH --mail-type=END # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yhung7@gsu.edu # Where to send mail
#SBATCH --account=math1581s250 # Project name (RS00000, ECON0001, MAT0001 ...)
#SBATCH --partition=qGPU24 # partition requested (qBF, qTRD , qECON ...)
##SBATCH --output=output_%j.npy # nameof the output file, see following section for pattern switches
#SBATCH --error=error_%j.txt # name of the output file, see the following section for pattern switchest
#SBATCH --array=0-2
#kdir -p $SCRATCH
#cd $SCRATCH


source /userapp/virtualenv/SDAM/venv/bin/activate

module purge
module load python
module load pytorch


PARAM_LIST=(
  "--hc 512 256 128 64 --nk 15 --fine_tune_lr 1e-6 --hp A --dropout 0.2 --case year"
  "--hc 256 128 64 32 --nk 15 --fine_tune_lr 1e-6 --hp B --dropout 0.2 --case year"
  "--hc 128 64 32 16 --nk 15 --fine_tune_lr 1e-6 --hp C --dropout 0.2 --case year"
)


PARAMS="${PARAM_LIST[$SLURM_ARRAY_TASK_ID]}"
python3 run_large_exp.py $PARAMS

# copying output(results.txt) to the irods projects directory
#cp simulation_output.npy $IRODS_PROJECT/
#rm -rf $SCRATCH