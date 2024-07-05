import sys
import yaml
from utils.arguments import handle_args, modify_config
import subprocess

with open(sys.argv[1], 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))

subprocess.run(["python", "spot_inference.py"] + sys.argv[1:])
subprocess.run(["python", "eval.py"] + sys.argv[1:])
