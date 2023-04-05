
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



seq1 = 'ATCGATCGATCG'
seq2 = 'ATCGATAG'
#Matrix creation initialised to zeros

#Scoring function

def score(i, j):
    if i == 0 and j == 0:
        return 0
    elif i == 0:
        return j * mismatch_score
    elif j == 0:
        return i * mismatch_score
    else:
        match = score(i-1, j-1) + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
        delete = score(i-1, j) + mismatch_score
        insert = score(i, j-1) + mismatch_score
        return max(match, delete, insert)


m, n = len(seq1), len(seq2)
matrix = np.zeros((m+1, n+1), dtype=int)
for i in range(m+1):
    for j in range(n+1):
        matrix[i,j] = score(i, j)
	
	
#Backtracking
def backtrack(i, j):
    if i == 0 and j == 0:
        return '', ''
    elif i == 0:
        s1, s2 = backtrack(i, j-1)
        return s1 + '-', s2 + seq2[j-1]
    elif j == 0:
        s1, s2 = backtrack(i-1, j)
        return s1 + seq1[i-1], s2 + '-'
    else:
        match = score(i-1, j-1) + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
        delete = score(i-1, j) + mismatch_score
        insert = score(i, j-1) + mismatch_score
        if matrix[i,j] == match:
            s1, s2 = backtrack(i-1, j-1)
            return s1 + seq1[i-1], s2 + seq2[j-1]
        elif matrix[i,j] == delete:
            s1, s2 = backtrack(i-1, j)
            return s1 + seq1[i-1], s2 + '-'
        else:
            s1, s2 = backtrack(i, j-1)
            return s1 + '-', s2 + seq2[j-1]
alignment1, alignment2 = backtrack(m, n)	
#import numpy as np

#alignment_mat=np.zeros((5,5))
#print(alignment_mat)
print("Alignment score:", matrix[m,n])
print("Alignment:")
print(alignment1)
print(alignment2)
