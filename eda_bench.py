from math import sqrt
import numpy as np

import vae

# 通常のVAE-EDA
def EDA1(input_size,evaluate,maxitr,pop_size,train_size,elite_size,intermediate_dim,latent_dim,epochs):
    best_in_samples = [0]*(maxitr+1)
    average_in_samples = [0]*(maxitr+1)
    diversity_in_samples = [0]*(maxitr+1)
    sample_size = pop_size - elite_size
    
    population = generate_populaton(pop_size,input_size,evaluate)
    model = vae.VAE(input_size,intermediate_dim,latent_dim,epochs)
    
    best_in_samples[0] = best_in_pop(population)
    average_in_samples[0] = average_in_pop(population)
    diversity_in_samples[0] = calc_divesity(population)

    for gen in range(maxitr):
        print("gen : " + str(gen+1))

        train_data = make_train_data(population,train_size)
        model.train(train_data)
        
        elite = elite_select(population,elite_size)

        samples = sample_from_model(model,input_size,sample_size,evaluate)
        
        print("best in samples = " + str(best_in_samples[gen]))
        print("average in samples = " + str(average_in_samples[gen]))

        population = elite + samples
        best_in_samples[gen+1] = best_in_pop(population)
        average_in_samples[gen+1] = average_in_pop(population)
        diversity_in_samples[gen+1] = calc_divesity(population)
        print("diversity = " + str(calc_divesity(population)))
        print("")

    return best_in_samples, average_in_samples, diversity_in_samples

# E-VAE-EDA
def EDA2(input_size,evaluate,maxitr,pop_size,train_size,elite_size,intermediate_dim,latent_dim,epochs):
    best_in_samples = [0]*(maxitr+1)
    average_in_samples = [0]*(maxitr+1)
    diversity_in_samples = [0]*(maxitr+1)
    sample_size = pop_size - elite_size
    
    population = generate_populaton(pop_size,input_size,evaluate)
    model = vae.EVAE(input_size,intermediate_dim,latent_dim,epochs)
    
    best_in_samples[0] = best_in_pop(population)
    average_in_samples[0] = average_in_pop(population)
    diversity_in_samples[0] = calc_divesity(population)

    for gen in range(maxitr):
        print("gen : " + str(gen+1))
        
        train_data,train_score = make_train_data_score(population,train_size)
        model.train(train_data,train_score)
        
        elite = elite_select(population,elite_size)

        samples = sample_from_model(model,input_size,sample_size,evaluate)
        
        print("best in samples = " + str(best_in_samples[gen]))
        print("average in samples = " + str(average_in_samples[gen]))
        print("")

        population = elite + samples
        best_in_samples[gen+1] = best_in_pop(population)
        average_in_samples[gen+1] = average_in_pop(population)
        diversity_in_samples[gen+1] = calc_divesity(population)
        print("diversity = " + str(calc_divesity(population)))
        print("")

    return best_in_samples, average_in_samples, diversity_in_samples

# CE-VAE-EDA
def EDA3(input_size,evaluate,maxitr,pop_size,train_size,elite_size,intermediate_dim,latent_dim,epochs):
    best_in_samples = [0]*(maxitr+1)
    average_in_samples = [0]*(maxitr+1)
    diversity_in_samples = [0]*(maxitr+1)
    sample_size = pop_size - elite_size
    
    population = generate_populaton(pop_size,input_size,evaluate)
    model = vae.CEVAE(input_size,intermediate_dim,latent_dim,epochs)

    best_in_samples[0] = best_in_pop(population)
    average_in_samples[0] = average_in_pop(population)
    diversity_in_samples[0] = calc_divesity(population)
    

    for gen in range(maxitr):
        print("gen : " + str(gen+1))
        
        train_data,train_score = make_train_data_score(population,train_size)
        model.train(train_data,train_score)
        
        elite = elite_select(population,elite_size)

        samples = sample_from_cevae(model,input_size,sample_size,evaluate,best_in_pop(population))
        
        print("best in samples = " + str(best_in_samples[gen]))
        print("average in samples = " + str(average_in_samples[gen]))
        print("")

        population = elite + samples
        best_in_samples[gen+1] = best_in_pop(population)
        average_in_samples[gen+1] = average_in_pop(population)
        diversity_in_samples[gen+1] = calc_divesity(population)
        print("diversity = " + str(calc_divesity(population)))
        print("")

    return best_in_samples, average_in_samples, diversity_in_samples

# 提案手法
# VAEからのサンプリングに用いる正規分布の分散を制御する
def EDA4(input_size,evaluate,maxitr,pop_size,train_size,elite_size,intermediate_dim,latent_dim,epochs):
    best_in_samples = [0]*(maxitr+1)
    average_in_samples = [0]*(maxitr+1)
    diversity_in_samples = [0]*(maxitr+1)
    sample_size = pop_size - elite_size

    population = generate_populaton(pop_size,input_size,evaluate)
    
    best_in_samples[0] = best_in_pop(population)
    average_in_samples[0] = average_in_pop(population)
    diversity_in_samples[0] = calc_divesity(population)

    model1 = vae.VAE3(input_size,intermediate_dim,latent_dim,epochs)
    model2 = vae.VAE3(input_size,intermediate_dim,latent_dim,epochs)

    c = 1
    c_max = 10
    c_inc = 1.2

    for gen in range(maxitr):
        print("gen : " + str(gen+1))

        # 分散を制御するモデルと制御しないモデルの2つを用意する
        train_data1 = make_train_data(population,train_size)
        train_data2 = make_train_data(population,train_size)

        model1.train(train_data1)
        model2.train(train_data2)
        
        elite = elite_select(population,elite_size)

        samples1 = sample_from_vae3(model1,input_size,sample_size//2,evaluate,c)
        samples2 = sample_from_vae3(model2,input_size,sample_size//2,evaluate,1)
        

        population = elite + samples1 + samples2
        population = sorted(population,key=lambda p: p.fitness)

        print("samples1 best = " + str(best_in_pop(samples1)))
        print("samples2 best = " + str(best_in_pop(samples2)))
        print("c = " + str(c))
        print("")

        best_in_samples[gen+1] = best_in_pop(population)
        average_in_samples[gen+1] = average_in_pop(population)
        diversity_in_samples[gen+1] = calc_divesity(population)
        print("best in samples = " + str(best_in_samples[gen+1]))
        print("average in samples = " + str(average_in_samples[gen+1]))
        print("diversity = " + str(calc_divesity(population)))
        print("")

        if best_in_samples[gen+1] > best_in_samples[gen]:
            c *= c_inc
            c = min(c,c_max)
        else:
            c /= c_inc
            c = max(c,1)

    return best_in_samples, average_in_samples, diversity_in_samples

# VAE-EDA-Q
def EDAQ1(input_size,evaluate,maxitr,pop_size,train_size,elite_size,intermediate_dim,latent_dim,epochs):
    queue_size = 5
    best_in_samples = [0]*(maxitr-queue_size+1+1)
    average_in_samples = [0]*(maxitr-queue_size+1+1)
    diversity_in_samples = [0]*(maxitr-queue_size+1+1)
    sample_size = pop_size - elite_size
    
    population = []
    pop_queue = generate_populaton(pop_size*queue_size,input_size,evaluate)
    model = vae.VAE(input_size,intermediate_dim,latent_dim,epochs)
    
    best_in_samples[0] = best_in_pop(pop_queue)
    average_in_samples[0] = average_in_pop(pop_queue)
    diversity_in_samples[0] = calc_divesity(pop_queue)

    for gen in range(maxitr-queue_size+1):
        print("gen : " + str(gen+1))

        train_queue = sorted(pop_queue,key=lambda p: p.fitness)
        train_data = make_train_data(train_queue,train_size)
        model.train(train_data)
        
        if gen == 0:
            elite = elite_select(pop_queue,elite_size)
        else:
            elite = elite_select(population,elite_size)
        
        samples = sample_from_model(model,input_size,sample_size,evaluate)
        
        print("best in samples = " + str(best_in_samples[gen]))
        print("average in samples = " + str(average_in_samples[gen]))

        population = elite + samples
        pop_queue = pop_queue[pop_size:] + population
    
        best_in_samples[gen+1] = best_in_pop(population)
        average_in_samples[gen+1] = average_in_pop(population)
        diversity_in_samples[gen+1] = calc_divesity(population)
        print("diversity = " + str(calc_divesity(population)))
        print("")

    return best_in_samples, average_in_samples, diversity_in_samples

#-------------------------------------------------------------------------#

class Problem():
    def __init__(self,input_size,evaluate):
        self.solution = [np.random.rand() for i in range(input_size)]
        self.fitness = evaluate(self.solution)

# functon for EDA
def generate_populaton(pop_size,input_size,evaluate):
    pop = [Problem(input_size,evaluate) for _ in range(pop_size)]
    return pop

def make_train_data(pop,train_size):
    # 上位から
    train_pop = sorted(pop,key=lambda p: p.fitness)[:train_size]
    train_data = [s.solution[:] for s in train_pop]
    return np.array(train_data)

def make_train_data_score(pop,train_size):
    # 上位から
    train_pop = sorted(pop,key=lambda p: p.fitness)[:train_size]
    train_data = [s.solution[:] for s in train_pop]
    train_score = [[s.fitness] for s in train_pop]
    return np.array(train_data),np.array(train_score)

def elite_select(pop,elite_size):
    # 上位から
    elite = sorted(pop,key=lambda p: p.fitness)
    return elite[:elite_size]

def best_in_pop(pop):
    return min(pop,key=lambda p: p.fitness).fitness

def average_in_pop(pop):
    pop_fitness = [p.fitness for p in pop]
    return sum(pop_fitness)/len(pop_fitness)

def sample_from_model(model,input_size,sample_size,evaluate):
    samples = [Problem(input_size,evaluate) for _ in range(sample_size)]
    sample_np = model.sample(sample_size)
    for i in range(sample_size):
        sample_solution = sample_np[i].tolist()
        samples[i].solution = sample_solution[:]
        samples[i].fitness = evaluate(sample_solution)
    return samples

def sample_from_vae3(model,input_size,sample_size,evaluate,c):
    samples = [Problem(input_size,evaluate) for _ in range(sample_size)]
    sample_np = model.sample(sample_size,c)
    for i in range(sample_size):
        sample_solution = sample_np[i].tolist()
        samples[i].solution = sample_solution[:]
        samples[i].fitness = evaluate(sample_solution)
    return samples

def sample_from_cevae(model,input_size,sample_size,evaluate,best_f):
    samples = [Problem(input_size,evaluate) for _ in range(sample_size)]
    a = 0.95
    b = 1.05
    aim_f = [best_f * ((b-a)*np.random.rand()+a) for _ in range(sample_size)]
    sample_np = model.sample(sample_size,aim_f)
    for i in range(sample_size):
        sample_solution = sample_np[i].tolist()
        samples[i].solution = sample_solution[:]
        samples[i].fitness = evaluate(sample_solution)
    return samples

def calc_divesity(pop):
    P = len(pop)
    N = len(pop[0].solution)
    L = sqrt(N)
    s = np.array([x.solution[:] for x in pop])
    s_ave = np.average(s, axis = 0)
    diversity = 0
    for i in range(P):
        A = 0
        for j in range(N):
            A += (s[i][j] - s_ave[j])**2
        diversity += sqrt(A)
    diversity = diversity/(P*L)
    return diversity
