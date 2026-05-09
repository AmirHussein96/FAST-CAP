import os
import json
import argparse
import glob
from subprocess import Popen, PIPE
from pprint import pprint
import re

PARTITION="batch_singlenode,polar,polar2,polar3,polar4"
DURATION="4.0"
MOUNTS="/lustre/fsw/portfolios/adlr/users/sanggill"
OUTPUT_ROOT_DIR="/lustre/fsw/portfolios/adlr/users/sanggill/experiments/ae_bigvgan"

def is_running(job_name, job='train.py'):
    with Popen("list_jobs --long", stdout=PIPE, stderr=PIPE, shell=True) as p:
        stdout, stderr = p.communicate()
    stdout = stdout.decode('utf-8')
    pattern = re.compile(r'\b' + re.escape(job_name) + r'_\d{8}-\d{6}\b')
    is_job_in_list_jobs = bool(pattern.search(stdout))
    
    return is_job_in_list_jobs


def load_filepaths(filename, split="|"):
    with open(filename) as f:
        filepaths = [line.strip().split(split) for line in f]
    return filepaths


def main(filepaths, image):
    script = 'train.py'
    for job_data in load_filepaths(filepaths):
        if job_data[0] == '#':
            continue
        if isinstance(job_data, list) and job_data[0][0] == '#':
            continue

        job_name, config_filepath, dataset_config_filepath, launch_params, params = job_data
        if job_name.startswith("#"):
           continue
        
        launch_params = '--gpu 8 --nodes 1' if launch_params == '' else launch_params
        
        # map --outfile
        output_dir = os.path.join(OUTPUT_ROOT_DIR, job_name)
        if not os.path.exists(output_dir): # if this is the first launch of this job, do not include backfill since warmup phase should not be preemptable
            print(f"first time launching the job, will exclude backfill_singlenode: {job_name}")
            partition = PARTITION
            os.makedirs(output_dir, exist_ok=True)
        else:
            # partition = "backfill_singlenode," + PARTITION
            partition = PARTITION

        if not is_running(job_name, script):
            print("##############################################################")
            print("launching {}".format(job_name))
            print(f"Using image {image}")
            print("##############################################################")
            # assuming single node training only, replace --mplaunch -c 'python {} \ with -c 'python {} \
            cmd = (
                f"submit_job --email_mode never "
                f"--mounts {MOUNTS} --partition {partition} --name {job_name} --duration {DURATION} --image {image} {launch_params} "
                f"--outfile {output_dir}/job.log --logdir {output_dir}/clusterinfo "
                f"-c 'python {script} "
                f"--config {config_filepath} --dataset_config {dataset_config_filepath} --checkpoint_path {output_dir} --params {params}'"
            )
            with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True) as p:
                stdout, stderr = p.communicate()
            print(cmd)
            print(stdout.decode('ascii'))
            print(stderr.decode('ascii'))
        else:
            print(f"already running: {job_name}")
            
    print("##############################################################")
    print("submit done")
    print("##############################################################")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepaths', type=str, help='Experiments file')
    parser.add_argument('-i', '--image', type=str, help='Docker image path')
    args = parser.parse_args()
    main(args.filepaths, args.image)
