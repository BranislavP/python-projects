import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import copy
import sys
from tesserocr import PyTessBaseAPI, PSM
import multiprocessing
import time
from PIL import Image

from Tesseract.Precision import overall_precision as pre


def get_bounds(index):
    if index <= 140:
        return 0, 1
    elif index <= 142:
        return -75, -10
    elif index <= 148:
        return 0, 255
    elif index <= 153:
        return 0, 100
    elif index <= 185:
        return 1, 8
    elif index <= 211:
        return 1, 20
    elif index <= 219:
        return 1, 40
    elif index <= 227:
        return 10, 60
    elif index <= 235:
        return 40, 135
    elif index <= 236:
        return 125, 750
    elif index <= 238:
        return 750, 3500
    elif index <= 240:
        return 7500, 15000
    elif index <= 246:
        return 0.0, 1.5
    elif index <= 256:
        return 0.0, 1.0
    elif index <= 265:
        return 0.25, 2.5


def randomize(index):
    if index <= 240:
        func = random.randint
    else:
        func = random.uniform
    bounds = get_bounds(index)
    return func(bounds[0], bounds[1])


def generate_list(ind):
    indiv = [randomize(i) for i in range(266)]
    ind = ind(indiv)
    return ind


def evaluate_tesseract(individual):
    with PyTessBaseAPI(lang="slk", psm=PSM.SINGLE_BLOCK) as api:
        for value, parameter in zip(vals, par):
            api.SetVariable(parameter, str(value))
        strings = list()
        for image, end in zip(files, ends):
            with Image.open("images/" + image + end) as im:
                api.SetImage(im)
                strings.append(api.GetUTF8Text())
    precision = 0
    for string, valid in zip(strings, valids):
        precision += pre(string, valid)
    return precision,


def mutation(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            if i <= 140:
                individual[i] = 0 if individual[i] == 1 else 1
            else:
                bounds = get_bounds(i)
                func = random.randint if i <= 240 else random.uniform
                if random.random() < 0.5:
                    individual[i] -= func(0, individual[i] - bounds[0])
                else:
                    individual[i] += func(0, bounds[1] - individual[i])
    return individual,

out_file = sys.argv[1]
with open("Parameters/parameters.txt") as params:
    par = params.read().splitlines()
valids = list()
for file in files:
    with open("valid/" + file + "_valid.txt") as valid_file:
        valids.append(valid_file.read())

creator.create("FitnessMax", base.Fitness, weights=(-1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("individual", generate_list, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_tesseract)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutation, indpb=0.05)
toolbox.register("select", tools.selBest)

if __name__ == "__main__":
    with open("my_config.txt", "w") as c_f:
        for i in range(len(par)):
            c_f.write("%s\t%s\n" %(par[i], vals_my[i]))
    start = time.time()
    pool = multiprocessing.Pool(processes=8, maxtasksperchild=None)
    toolbox.register("map", pool.map)
    pop = toolbox.population(n=40)
    
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("best", lambda x: copy.deepcopy(tools.selBest(pop, 1)[0]))
    with open(out_file, "w") as f:
        sys.stdout = f
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100,
                                       stats=stats, halloffame=hof, verbose=True)
        print(hof[0].fitness)
        print(hof)
        print(time.time() - start)
    pool.close()

