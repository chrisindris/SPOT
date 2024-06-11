import yaml
import sys
import copy


def handle_args(args=sys.argv):
    """ cli arguments in the form <key1>.<key2>.<key3>...=<value> are parsed into [<key1>, ...] and [<value>], 
    and these lists are added to keys and new_values for convenient usage for modifying the config dictionary.
        
    Example:
    dataset.name='new_name' -> ['dataset', 'name'], ['new_name'] -> these lists are appended to keys, new_values
    """
    keys = []
    new_values = []
    for arg in args[1:]:
        key, value = arg.split('=')
        key = key.split('.')
        keys.append(key)
        new_values.append(value)
    return keys, new_values

#keys, new_values = handle_args()


"""
print('iterate through the config')
a = copy.deepcopy(config)
for i in ['dataset', 'num_classes']:
    a = copy.deepcopy(a[i])
    print(type(a))
    print('')
"""

def modify_config(config, keys, new_values):
    """ Given a dictionary read from a config, this modifies it based on command-line arguments.

    """

    for key in range(len(keys)):
        a = copy.deepcopy(config)
        lst = [copy.deepcopy(config)]

        for i in keys[key]:
            lst.append(copy.deepcopy(lst[-1][i]))
            a = copy.deepcopy(a[i]) 
        
        lst[-1] = new_values[key]   

        # type correction
        if isinstance(a, list):
            lst[-1] = [float(i) for i in lst[-1].replace(' ', '')[1:-1].split(',')]
            pass
        else:
            a = type(a)
            lst[-1] = a(lst[-1])


        for i in range(1, len(keys[key]) + 1):
            lst[-1-i][keys[key][::-1][i-1]] = lst[-i]

        config = copy.deepcopy(lst[0])
    return config


with open("./configs/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = modify_config(yaml.load(tmp, Loader=yaml.FullLoader), *handle_args(sys.argv))
config = modify_config(config, *handle_args())
#print(config)

"""
print('\n')

twigs = [] # keys whose value is not a dictionary
def is_twig(dic, parents=''):
    
    curr_parents = parents

    for key in dic.keys():
        parents = copy.deepcopy(curr_parents) 
        parents = parents + ('.' * int(len(parents) != 0))  + key
        if isinstance(dic[key], dict):
            is_twig(dic[key], parents)
        else:
            twigs.append(parents) # this is a twig

print('\n')
is_twig(config)
print(twigs)
"""

