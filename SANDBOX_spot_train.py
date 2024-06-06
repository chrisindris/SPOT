import yaml
import sys

with open("./configs/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)

print(sys.argv)
print(type(config))
