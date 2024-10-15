import os
import sys
import subprocess
from subprocess import PIPE

os.environ['KUBECONFIG'] = os.path.expanduser('~/Downloads/config-files/kubedevus04.config')

# Run command on response and get only il-dealcentre pods, print only 1st column, describe each pod and only select "Service Account" is il-dealcentre

output_data = subprocess.getoutput('kubectl get pods -n deals-int | grep il-dealcentre | awk \'{print $1}\'')


# list_of_services = subprocess.call([response, 'awk '{print $3}')'])
print(output_data)
