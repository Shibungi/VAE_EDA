from cec2013single_master.cec2013single.cec2013 import Benchmark
import numpy as np
import eda_bench
import pandas as pd
import sys

bench = Benchmark()

def make_csv(name,list,maxitr):
    df = pd.DataFrame(list).T
    ave = df.loc[:,0:maxitr-1].mean(axis=1)
    var = df.loc[:,0:maxitr-1].var(axis=1)
    df['ave'] = ave
    df['var'] = var
    df.to_csv(name)

def test(EDA,PROBLEM_NUM,maxitr):

    def PROBLEM(x):
        y = np.array(x)*200.0 - 100.0
        p = bench.get_function(PROBLEM_NUM)
        return p(np.array(y))

    bests = []
    aves = []
    divs = []
    input = 10
    generation = 200
    population = 500
    training = 50
    elite = 0
    intermediate = 20
    latent = 3
    epochs = 10

    for i in range(maxitr):
        print("itr : " + str(i+1))
        b,a,d = EDA(input,PROBLEM,generation,population,training,elite,intermediate,latent,epochs)
        bests.append(b)

    s = 'data/'+ EDA.__name__ + '_P' + str(PROBLEM_NUM) + '_'
    s += str(input)+'_'+str(generation)+'_'+str(population)+'_'+str(training)+'_'+str(elite)+'_'+str(intermediate)+'_'+str(latent)+'_'+str(epochs)
    make_csv(s+'_bests',bests,maxitr)


p = int(sys.argv[1])

test(eda_bench.EDA1,p,3)
