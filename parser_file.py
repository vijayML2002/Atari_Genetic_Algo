from more_itertools import iterate

#k(0) - single int value
#k(1) - multiple string value
#k(2) - multiple int value
#k(Other) - cross,mutate value

def clean(pre_str):
    new_str = pre_str.rstrip()
    return new_str

def process(value, type_value):
    k = value.split(",")
    if type_value == 0:
        return int(k[0])
    elif type_value == 1:
        return k
    elif type_value == 2:
        layer = []
        for val in k:
            layer.append(int(val))
        return layer
    else:
        cross = []
        mutate = []
        for val in k:
            q = val.split("&")
            if len(q) == 1:
                mutate.append(int(q[0]))
            else:
                cross.append([int(q[0]), int(q[1])])
        return cross, mutate


def text_value():
    arr = []
    with open("info.txt") as input_file:
        for line in input_file:
            if line.isspace():
                continue
            arr.append(line.split("="))

    for i in range(len(arr)):
        arr[i][1] = clean(arr[i][1])
    return arr


def layer_constant():
    arr = text_value()
    layer = process(arr[0][1], 0)
    layer_detail = process(arr[1][1], 2)
    activation_detail = process(arr[2][1], 1)
    return layer, layer_detail, activation_detail

def no_iter():
    arr = text_value()
    iteration = process(arr[3][1], 0)
    return iteration

def genetic_value():
    arr = text_value()
    generation_no = process(arr[4][1], 0)
    genome_no = process(arr[5][1], 0)
    return genome_no, generation_no

def cross_mutate_value():
    arr = text_value()
    cross, mutate = process(arr[6][1], 3)
    return cross, mutate
"""
layer, layer_detail, activation_detail = layer_constant()
iteration = no_iter()
genome_no, generation_no = genetic_value()
cross, mutate = cross_mutate_value()

print("Layer : ", layer)
print("Layer_detail : ", layer_detail)
print("Activation_detail : ", activation_detail)

print("Iteration : ", iteration)

print("Genome_no : ", genome_no)
print("Generation_no : ", generation_no)

print("Cross : ", cross)
print("Mutate : ", mutate)
"""