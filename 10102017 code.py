#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 00:25:01 2017

@author: laukaki
"""

import numpy as np
from numpy import linspace
from pylab import imshow, plot,show, ylim, xlabel, ylabel, subplots, legend


def NumberofStock(T, C, Porfoliolist, StockH, StockL, GreedFactor):
    NumberofStockH = []
    NumberofStockL = []
    for i in range(0,50):
        if GreedFactor[T][i] == 0:
            Portfolio = Porfoliolist[0]
        else: 
            Portfolio = Porfoliolist[1]
        NumberofStockH.append(C[T][i]*Portfolio[0]/StockH[T])
        NumberofStockL.append(C[T][i]*Portfolio[1]/StockL[T])
    return NumberofStockH, NumberofStockL
    
def getCapital(t, C, Porfoliolist, StockH, StockL, GreedFactor):
    Captialatt = []
#    print(GreedFactor)
#    print(C[t-1])
#    print(StockH[t])
    for i in range(0,50):
        CH = StockH[t] * NumberofStock(t-1, C, Porfoliolist, StockH, StockL, GreedFactor)[0][i]
        CL = StockL[t] * NumberofStock(t-1, C, Porfoliolist, StockH, StockL, GreedFactor)[1][i]
        Captialatt.append(CH + CL)
#    print(CH)
    return Captialatt
    
def Memory(t, i, GreedFactor, Winhowmuch, Connect):
    
    L = list(zip(Winhowmuch[t], GreedFactor[t-1]))
#    print(L)
#    print(len(L))
#    print(Winhowmuch[t])
#    print(GreedFactor[t-1])
    MemoryList = [L[x % len(L)] for x in range(i-Connect, i+Connect+2)]
#    print(MemoryList)
    return MemoryList
    

def FW(i, memory):
    Winner = max(memory)
    Next = Winner[1]
    return Next

def AL(i, memory):
    Loser = min(memory)
    if memory[1] == Loser:
        if memory[1][1] == 0:
            Next = 1
        if memory[1][1] == 1:
            Next = 0
    else:
        Next = memory[1][1]
    return Next
    
def StockH(T):
#    C1 = np.random.random()
    C1 = 0.6713081033987339
    print("C1 is:")
    print(C1)
#    C2 = np.random.random()
    C2 = 0.3863530740707609
    print("C2 is:")
    print(C2)
#    wlist = linspace(5, 10, 100)
#    w = np.random.choice(wlist)
    w = 6.81818181818
    print("w is:")
    print(w)
    StockH = [2 + C1*np.sin(w*t/30) + C2*np.cos(w*t/30) for t in T]
#    StockH = [1 + t for t in T]
    return StockH
    
def Simulation(T, Stock, StockL, N, PFW, Porfoliolist, Connect):
    C = [[100.0]*50]
    GreedFactor = [[0, 1]*25]
    Winhowmuch = []
    Strategy = []
    Winhowmuch.append([0]*50)
    Strategy.append([""]*50)
    
    for t in T[1:]:
        C.append(getCapital(t, C, Porfoliolist, Stock, StockL, GreedFactor))
#        print(C)
        The_value_win = [C[t][i]-C[t-1][i] for i in range(len(C[t-1]))]
        Winhowmuch.append(The_value_win)   
        Strategy.append([])
        for i in range(0,50):
            New = np.random.choice(["FW","AL"],p=[PFW,1-PFW])
            Strategy[t].append(New)
        GreedFactor.append([])
        for i in range(0,50):
            memory = Memory(t, i, GreedFactor, Winhowmuch, Connect)
            if Strategy[t][i]=="FW":
                Next = FW(i, memory)
            if Strategy[t][i]=="AL":
                Next = AL(i, memory)
            GreedFactor[t].append(Next)
    return C, GreedFactor, Strategy, Winhowmuch
    
def TotalC(Clist):
    TotalC = []
    for x in Clist:
        Total = sum(x)
        TotalC.append(Total)
    return TotalC
    
def SimulationManyTimesSameStock(T, Porfoliolist, Stock, StockL, PFW, Connect):
#    print(StockL)    
    N = 0
    averageY = [[0]*50]*50
    GreedFactor = []
    Strategy = []
    Yineverysimulation = []
    while N < 70:
        Simulate = Simulation(T, Stock, StockL, N, PFW, Porfoliolist, Connect)
        Y = Simulate[0]
        Yineverysimulation.append(Y)

        averageY = [[x[i]*N +y[i] for i in range(0,50)] for x,y in zip(averageY, Y)]
        N = N+1
        averageY = [[x[i]/N for i in range(0,50)] for x in averageY]
        
        Z = TotalC(averageY)
        GreedFactor.append(Simulate[1])
        Strategy.append(Simulate[2])
    return averageY, Z, GreedFactor, Strategy ,Yineverysimulation

def Exhausive(T, Porfoliolist, Stock, StockL):
    Sequence = [[0, 1]*25]
    Porfoliolist = [(0.1, 0.9), (0.9, 0.1)]
    Capital = [[100.0]*50]
    for t in T[1:-1]:
#        print(t)
        Capital.append(getCapital(t, Capital, Porfoliolist, Stock, StockL, Sequence))
        Sequence.append([])
        for i in range(0,50):
            if Stock[t+1] > Stock[t]:
                GreedFactor = 1
            if Stock[t+1] < Stock[t]:
                GreedFactor = 0
            Sequence[t].append(GreedFactor) 
    Z = TotalC(Capital)
    return Sequence, Z, Z[-1], Capital

"""
This part find best PFW for this stock
"""

T =[int(t) for t in linspace(0, 30, 31)]
Stock = StockH(T)
StockL = [1]*31      
Porfoliolist = [(0.1, 0.9), (0.9, 0.1)]
   
Best = Exhausive(T, Porfoliolist, Stock, StockL)
BestSequence = Best[0]
Bestpath = Best[1]
BestResult = Best[2]    
print("The greatest possible result of this sequence in ratio:")
print(BestResult/5000)

Connect = 2
while Connect < 3:
    print("The connection is:")
    print(Connect)

    PFW = linspace(0, 1, 11)
    R1 = []
    R2 = []
    R3 = []
    R4 = []
    R5 = []
    R6 = []


    for p in PFW:
        R0 = SimulationManyTimesSameStock(T, Porfoliolist, Stock, StockL, p, Connect)
        R1.append(R0[0])
        R2.append(R0[1])
        R3.append(R0[2])
        R4.append(R0[3])
        R5.append(R0[4])
        R6.append(R0[1][-1])

    Result = list(zip(R6, PFW))
    OptPFW = max(Result)
    print("The result:")
    print(Result)
    print("The best result and PFW is:")
    print(OptPFW)
    print("The ratio of initial captital:")
    print(OptPFW[0]/5000)

    """
    This part compare best result with result of best PFW. 
    Then find out difference of that particular Stock H.
    """

    HowGoodIsGroup = BestResult/5000 - OptPFW[0]/5000
    print("How good is the group? (The smaller the better)")
    print(HowGoodIsGroup)

#    plot(Stock,"b-",linewidth=1)
#    xlabel("Time t")
#    ylabel("Value of stock")
#    show()
    plot(PFW, R6,"g-",linewidth=1)
    xlabel("The probability of FW")
    ylabel("Final Capital")
    show()  
    Connect = Connect + 1
    

 

"""
Graphs
"""
#import matplotlib
#print(R3[][0])
#fig = matplotlib.pyplot.figure()
#ax = fig.add_subplot(111)
#xlabel("Players")
#ylabel("Portfolio")
#img = ax.imshow(R3[][0], aspect='auto', cmap=matplotlib.pyplot.cm.gray, interpolation='nearest')
#img
#
#print(R2[])
#plot(R2[],"b-",linewidth=1)
#xlabel("Time t")
#ylabel("Total capital")
#show()
#
#R7 = []
#for i in range(0, 140):
#    R7.append(sum(R5[][i][-1]))
#
#print(R7)
#matplotlib.pyplot.hist(R7, bins=[5000, 6000, 7000, 8000, 9000, 10000])  # arguments are passed to np.histogram
#matplotlib.pyplot.title("Histogram of final results in different simulations")
#xlabel("Range of results")
#ylabel("Count")
#matplotlib.pyplot.show()

import matplotlib.pyplot as plt
Connectlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24, 24.5]
ResultC = [0.305008342848, 0.21423043697, 0.233427226981, 0.208447301838, 0.183920489017, 0.170993762608, 0.160799235214, 0.15411932321, 0.158481255932, 0.156769825085, 0.154584887582, 0.151416325674, 0.145907380647, 0.14485527226, 0.135111437454, 0.148030894198, 0.156137803419, 0.152048811412, 0.157706956877, 0.144418248371, 0.156008115612, 0.142236981728, 0.143290925347, 0.163900718468, 0.148973818972]
plt.plot(Connectlist, ResultC, 'ro')
plt.xlabel("Number of connections on each side")
plt.ylabel("Difference")
plt.show()

"""
Repeat for 10 different stocks. obtain the average of the use of a ring.
"""
