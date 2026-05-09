#!/bin/bash

while true; do
    # Forward the command line arguments to submit_jobs.sh
    sh submit_jobs.sh "$@"
	echo "Sleeping for 31 minutes before next submit_jobs.sh"
	sleep 1860  # Sleep for 31min
done