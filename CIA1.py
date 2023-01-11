
#Generating random string with the given genomic sequence A T G C 

import random
import string


sequence=['A','T','G','C']



def randStr(chars = string.ascii_uppercase + string.digits, N=16):
	return ''.join(random.choice(chars) for _ in range(N))
string_1=randStr(chars='ATGC')
string_2=""
print(string_1)

string_2=''.join(random.sample(string_1,len(string_1)))
print(string_2)


match_score=5
mismatch_score=-4

l1=len(string_1)
l2=len(string_2)

#Matrix creation initialised to zeros



import numpy as np

alignment_mat=np.zeros((5,5))
print(alignment_mat)

max_score=0 
