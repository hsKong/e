import numpy as np
import math

import torch
from torch.autograd import Variable

from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from utils import batchify


def doc_to_line(cluster_doc, vocab, maxlen):

    dropped = 0
    linecount = 0
    lines = []
    for line in cluster_doc:
        linecount += 1        
        words = line[:-1].strip().split(" ")
        
        if len(words) > maxlen:
            dropped += 1
            continue
        words = ['<sos>'] + words
        words += ['<eos>']

        # vectorize
        unk_idx = vocab['<oov>']
        indices = [vocab[w] if w in vocab else unk_idx for w in words]
#         print(indices)
        lines.append(indices)
        
    return lines, vocab


def make_code(lines, autoencoder):
    eval_batch_size = 1
    test_data = batchify(lines, eval_batch_size, shuffle=False)
     
    code = []
    for i, batch in enumerate(test_data):
            source, target, lengths = batch
            # output: batch x seq_len x ntokens
            output = autoencoder.encode(Variable(source), lengths, noise=False)
            code.append(output)
            code_cat=torch.cat(code,0)
    return code_cat
	
	
def cosine_title(cluster_doc_rawdata,docno_cluster, autoencoder, vocab, maxlen):
#	title_raw_eucli={}
	title_raw_cosine={}

#	num_title_raw_eucli={}
	num_title_raw_cosine={}

	for c in cluster_doc_rawdata.keys() : 
	    
	    lines, vocab = doc_to_line(cluster_doc_rawdata[c], vocab, maxlen)
	#     code = make_code(lines)
	#     code_cat=torch.cat(code,0)
	    code_cat = make_code(lines, autoencoder)
	    
	    #code 평균
	    code_mean=torch.mean(code_cat,0).view(1,-1)
	    
	    #code 평균과 길이가 최소인 vector 구하기
	    input1 = code_cat.data.numpy()
	    mean_np = code_mean.data.numpy()
	    eucli = euclidean_distances(input1, mean_np).squeeze()
	    cosine = cosine_distances(input1, mean_np).squeeze()
	    
	    #{clusterno : title}
	#     title_raw_eucli[c]=cluster_doc_rawdata[c][np.argmin(eucli)]
	    title_raw_cosine[c]=cluster_doc_rawdata[c][np.argmin(cosine)]
	    
	    #{clusterno : title num}
	#     num_title_raw_eucli[c]=docno_cluster[c][np.argmin(eucli)]
	    num_title_raw_cosine[c]=docno_cluster[c][np.argmin(cosine)]
		
	return title_raw_cosine, num_title_raw_cosine
	
	
def entropy(docno_cluster, parsingdict):
	pair_dic={}
	## {rawclusterno : [[parsing_clusterno, count]]}
	for rawclno in docno_cluster.keys():
	    li = []
	    for docno in docno_cluster[rawclno]:
	        if parsingdict[docno] !=[]:
	            li.append(parsingdict[docno][0])	    
	    unique, counts = np.unique(li, return_counts=True)
	    k=np.stack((unique,counts),axis=1).tolist()	    
	    pair_dic[rawclno]=k
	## calculate_entropy
	entropy_list=[]
	for i in pair_dic.keys():
	    entropy=0
	    for j in range(len(pair_dic[i])):
	        p_i=(int(pair_dic[i][j][1])/len(docno_cluster[i]))
	        entropy += p_i * math.log(1/p_i)
	    entropy_list.append([i, entropy])
	## Entorpy 기준 : 2.75
	clu_entro=np.array(entropy_list)
	select_entropy = clu_entro[clu_entro[:,1].astype(np.float)>2.75]
	select_cluster=select_entropy[:,0]
	return select_cluster
	
	
	
def jaccard_select_title(docno_cluster, select_cluster, num_title_raw_cosine, title_raw_cosine, raw_docno_token, parsingdict, parse_num_title_raw_cosine, parse_docno_token, parse_title_raw_cosine):
	#{clusternum(raw기준):[[docno(raw기준), selection]]}
	result={}


	for clusterno in docno_cluster.keys(): #모든 cluster에 대해서
	    
	    if clusterno in select_cluster: #Entropy 기준에 해당하는 cluster에 대해서
	    
	        raw_titleno = num_title_raw_cosine[clusterno]
	        raw_title_token = raw_docno_token[raw_titleno] # title token (raw data)
	        
	        result_docno_titleselect=[]

	        for docno in docno_cluster[clusterno]:
	            rawdoc_token=raw_docno_token[docno] # rawdoc token (원문 token)
	            parse_clusternum=parsingdict[docno] # parsing data cluster number
	            all_parse_title_token=[]
	            docno_titleselect=[]
	            #print(docno)
	            
	            flag = 0
	            docno_titleselect.append(docno)
	            for parse_clno in parse_clusternum:
	                parse_select_result=[]
	                parse_titleno=parse_num_title_raw_cosine[parse_clno]
	                all_parse_title_token = parse_docno_token[parse_titleno] #각 doc별 다중코딩 총 title token
	                parse_select_result.append(parse_clno)
	                parse_select_result.append(parse_title_raw_cosine[parse_clno])

	                raw_matchcount=0
	                parse_matchcount=0
	                #print(list(set(rawdoc_token)))
	                #print(list(set(raw_title_token)))
	                #print(list(set(all_parse_title_token)))

	                
	                #기본 매칭 개수 --------------method 1-------------
	                for i in list(set(rawdoc_token)):
	                    raw_matchcount+=list(set(raw_title_token)).count(i)
	                    parse_matchcount+=list(set(all_parse_title_token)).count(i)
	                #print("매칭 개수")
	                #print(raw_matchcount,parse_matchcount)
	                

	                #분수형태: -------------method 2-------------
	                #if len(raw_title_token) != 0:
	                #    raw_matchscore = raw_matchcount/len(set(raw_title_token))
	                #else:
	                #    raw_matchscore=0
	                #if len(all_parse_title_token) != 0:
	                #    parse_matchscore = parse_matchcount/len(set(all_parse_title_token))
	                #else:
	                #    parse_matchscore=0
	#                 print("분수")
	#                 print(raw_matchscore,parse_matchscore)
	                
	                #자카드: -------------method 3-------------  
	                if len(raw_title_token) != 0:
	                    raw_matchscore_jaccard = raw_matchcount/len(list(set(list(set(raw_title_token))+list(set(rawdoc_token)))))
	                else:
	                    raw_matchscore_jaccard=0
	                if len(all_parse_title_token) != 0:
	                    parse_matchscore_jaccard = parse_matchcount/len(list(set(list(set(all_parse_title_token))+list(set(rawdoc_token)))))
	                else:
	                    parse_matchscore_jaccard=0               
	                
	                #print("jaccard")
	                #print(list(set(list(set(raw_title_token))+list(set(rawdoc_token)))))
	                #print(list(set(list(set(all_parse_title_token))+list(set(rawdoc_token)))))
	                #print(raw_matchscore_jaccard,parse_matchscore_jaccard)
	                

	        #         if raw_matchcount >= parse_matchcount :  # -------method1
	#                 if raw_matchscore >= parse_matchscore :  #----------method2
	                if raw_matchscore_jaccard >= parse_matchscore_jaccard :  #--------method3
	                    selection=0 #raw title선택
	#                     docno_titleselect.append(selection)
	#                     docno_titleselect.append(title_raw_cosine[clusterno])
	                else : 
	                    selection=1 #parsing title선택
	                    if flag == 0 :
	                        docno_titleselect.append(selection)
	                   #print(parse_select_result)
	                    docno_titleselect.extend(parse_select_result)
	                    flag+=1
	            
	            if flag == 0:
	                docno_titleselect.append(0)
	                docno_titleselect.append(title_raw_cosine[clusterno])

	           #print(docno_titleselect)
	            result_docno_titleselect.append(docno_titleselect)

	        result[clusterno] = result_docno_titleselect
	    	
	    else: # Entropy에서 기준이상 아닌 cluster
	        result_docno_titleselect=[]
	        for docno in docno_cluster[clusterno]:
	            docno_titleselect=[]
	            docno_titleselect.append(docno)
	            selection=0
	            docno_titleselect.append(selection)
	            docno_titleselect.append(title_raw_cosine[clusterno])
	            result_docno_titleselect.append(docno_titleselect)
	        result[clusterno] = result_docno_titleselect
	return result