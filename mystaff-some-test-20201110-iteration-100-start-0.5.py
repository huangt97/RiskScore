import random
import subprocess 
import numpy
from scipy import sparse
import time
from collections import defaultdict
import csv
import networkx as nx
import pandas as pd
import datetime
import math
from math import exp
import csv
import pickle

all_hash=pd.read_csv('./data/all_hash_1191809_with_train_642.csv',names = ['index','address','label'])
all_index=pd.read_csv('./data/all_index_adj_4136463_with_count_with_all.csv')
nodes = all_hash["index"]
edges = all_index["Unnamed: 0"]
print("ethereum network has %d nodes and %d edges" % (len(nodes), len(edges)))

reliable = dict()
trust = dict()
confidence = dict()
score = dict()
max_to_count = max(all_index["to_count"])
max_from_count = max(all_index["from_count"])
max_all_count = max(all_index["all_count"])
print("max_to_count",max_to_count)
print("max_from_count",max_from_count)
print("max_all_count",math.log(max_to_count,10))
for edge in edges:
    confidence[edge] = 0.5
    score[edge] = random.uniform(-1,1)
#     score[edge] = ((2*math.log(all_index["from_count"][edge],10)-math.log(max_from_count,10))/math.log(max_from_count,10) + (2*math.log(all_index["to_count"][edge],10)-math.log(max_to_count,10))/math.log(max_to_count,10))/2
    if all_hash["label"][all_index["index_from"][edge]] == 0:
        confidence[edge] = 1 
    if all_hash["label"][all_index["index_from"][edge]] == 1:
        confidence[edge] = 1
        
for node_t in nodes:
    trust[node_t] = 0.5
    reliable[node_t] = 0.5
    if all_hash["label"][node_t] == 0:
        reliable[node_t] = 1 
    if all_hash["label"][node_t] == 1:
        reliable[node_t] = 0 
#     elif all_hash["label"][node_t] == 0:
#         all_hash["reliable"][node_t] = 0
#     else:
#         all_hash["reliable"][node_t] = 0

iter = 0

du = 0
dp = 0
dr = 0
beta1 = 1
beta2 = 1

##### ITERATIONS START ######
incount_for_node = dict()
outcount_for_node = dict()
trust_value_all = dict()
reliable_value_all = dict()
for node_t in nodes:
    incount_for_node[node_t] = 0
    outcount_for_node[node_t] = 0
    trust_value_all[node_t] = 0
    reliable_value_all[node_t] = 0
while iter < 50:
    print('-----------------')
    print("Epoch number %d with du = %f, dp = %f, dr = %f, for (%d,%d)" % (iter, du, dp, dr, beta1, beta2))
    if numpy.isnan(du) or numpy.isnan(dp) or numpy.isnan(dr):
        break
    
    du = 0
    dp = 0
    dr = 0
    ############################################################
    for node_t in nodes:
        incount_for_node[node_t] = 0
        outcount_for_node[node_t] = 0
        trust_value_all[node_t] = 0
        reliable_value_all[node_t] = 0
    print("Calculating Relialbe and trust of accounts by all the edges")
    for edge in edges:
        incount_for_node[all_index["index_to"][edge]] += 1
        outcount_for_node[all_index["index_from"][edge]] += 1
        trust_value_all[all_index["index_to"][edge]] += score[edge]*confidence[edge]
        reliable_value_all[all_index["index_from"][edge]] += confidence[edge]
     
    ############################################################
    ############################################################
    
    print('Updating trust of account')

#     current_trust_vals = []
#     for node in nodes:
#         current_trust_vals.append(trust[node])
    
#     median_trust_vals = numpy.median(current_trust_vals) # Alternatively, we can use mean here, intead of median
#     print("median_trust_vals:",median_trust_vals)

    for node in nodes:
        
#         inedges = G.in_edges(node,  data=True)
        ftotal = incount_for_node[node]
        trust_total = trust_value_all[node]
        
        #kl_timestamp = ((1 - full_birdnest_product[product_map[node]]) - 0.5)*2

        if ftotal > 0.0:
            trust_for_node = trust_total / ftotal
        else:
            trust_for_node = 0.5
        
        x = trust_for_node
        
        if x < 0:
            x = 0
        if x > 1.0:
            x = 1.0
        dp += abs(trust[node] - x)
        trust[node] = x
    
    ############################################################
    
    print("Updating Relialbe of accounts")
#     current_reliable_vals = []
#     for node in nodes:
#         current_reliable_vals.append(reliable[node])
#     median_reliable_vals = numpy.median(currentfvals) # Alternatively, we can use mean here, intead of median

    for node in nodes:
        if all_hash["label"][node] == 0 or all_hash["label"][node] == 1:
            continue
        rtotal = outcount_for_node[node]
        reliable_total = reliable_value_all[node]
        
        if rtotal > 0.0:
            reliable_for_node = reliable_total / rtotal
        else:
            reliable_for_node = 0.5
        
        x = reliable_for_node
        if x < 0.00:
            x = 0.0
        if x > 1.0:
            x = 1.0

        du += abs(reliable[node] - x)
        reliable[node] = x
    
    ############################################################
    
    print("Updating confidence of transactions")
#     current_trans_confidence = []
#     for edge in edges:
#         current_trans_confidence.append(confidence[edge])
    
#     median_trans_confidence = numpy.median(current_trans_confidence) # Alternatively, we can use mean here, intead of median
    for edge in edges:
        if all_hash["label"][all_index["index_from"][edge]] == 0 or all_hash["label"][all_index["index_from"][edge]] == 1:
            continue
        account_from_reliable = reliable[all_index["index_from"][edge]]
        account_to_trust = trust[all_index["index_to"][edge]]
        transaction_score = score[edge]
        x = (beta1*account_from_reliable + beta2*(1-abs(transaction_score-account_to_trust)))/(beta1+beta2)

        if x < 0.00:
            x = 0.0
        if x > 1.0:
            x = 1.0
        
        dr += abs(confidence[edge] - x)
        confidence[edge] = x
    
    ###########################################################
    
    iter += 1
    if  dp < 0.01 and dr < 0.01 and du < 0.01:
        print("The propagation equation reaches convergence after "+str(iter)+" more iterations!")
        break
        
### SAVE THE RESULT

# current_trust_vals = []
# for node in nodes:
#     current_trust_vals.append(trust[node])
    
# median_trust_vals = numpy.median(current_trust_vals)
# print(len(current_trust_vals), median_trust_vals)

# current_reliable_vals = []
# for node in nodes:
#     current_reliable_vals.append(reliable[node])
    
# median_reliable_vals = numpy.median(current_reliable_vals)
# print(len(current_reliable_vals), median_reliable_vals)

# all_edge_confidence_vals = []
# for edge in edges:
#     all_edge_confidence_vals.append(confidence[edge])
# median_edge_confidence = numpy.median(all_edge_confidence_vals)
# print(len(all_edge_confidence_vals), median_edge_confidence)

import operator
# sort users based on their scores

fw = open("./results/20201113-in-out-account-trust-reliable-%d-%d-iteration-50-random.csv" % (beta1, beta2),"w")
fw.write("address,incount,outcount,trust,reliable\n")

for node in nodes:
    fw.write("%s,%s,%s,%s,%s\n" %(all_hash["address"][node],str(incount_for_node[node]),str(outcount_for_node[node]),str(trust[node]), str(reliable[node])))
fw.close()

fw_confidence = open("./results/20201113-transasction-confidence-score-%d-%d-iteration-50-random.csv" % (beta1, beta2),"w")
fw_confidence.write("confidence,score\n")
for edge in edges:
    fw_confidence.write("%s,%s\n" %(str(confidence[edge]),str(score[edge])))
fw_confidence.close()