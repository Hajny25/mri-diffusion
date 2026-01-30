for i in {1,2,4,8,16,32,64}
do
    echo "Submitting job with $i workers"
    sbatch --partition=normal --time=00:30:00 --cpus-per-task=$i --wrap="source venv/bin/activate && python experiments/ddpm_25d/test_dataloader.py --workers $i"
done