import collections
import math

TRAIN_T = "DATASET/train/truthful.txt"
TRAIN_D = "DATASET/train/deceptive.txt"
TEST     = "DATASET/test/test.txt"
VAL_T   = "DATASET/validation/truthful.txt"
VAL_D   = "DATASET/validation/deceptive.txt"


def make_unigram(filepath):
    """
    Build a smoothed unigram language model from a corpus file.
    Uses Laplace-style smoothing with an <unk> token for unseen words.
    Returns a dict of word probabilities and the total token count.
    """
    smoothing = 0.5
    d = collections.defaultdict(int)
    word_count = 0

    with open(filepath) as f:
        for line in f:
            for word in line.split():
                d[word] += 1
                word_count += 1

    d['<unk>'] = smoothing * len(d)
    word_count += d['<unk>']

    for k in d:
        d[k] = (d[k] + 0.6) / (word_count + 0.6 * len(d))

    return d, word_count


def make_bigram(filepath):
    """
    Build a smoothed bigram language model from a corpus file.
    Uses add-1 smoothing with per-word <unk> backoff.
    Bigram counts reset at the start of each line (review boundary).
    Returns a nested dict of bigram probabilities and the total token count.
    """
    smoothing = 1
    word_count = 0
    d = collections.defaultdict(lambda: collections.defaultdict(int))

    with open(filepath) as f:
        for line in f:
            last_word = '<START>'
            for word in line.split():
                word_count += 1
                if last_word == '<START>':
                    last_word = word
                    continue
                d[last_word][word] += 1
                last_word = word

    num_bigrams = sum(len(bigrams) for bigrams in d.values())
    d['<unk>']['<unk>'] = num_bigrams * smoothing

    for k in d:
        d[k]['<unk>'] = len(d[k]) * smoothing
        acc = sum(d[k][i] + 1 for i in d[k])
        for j in d[k]:
            d[k][j] = (d[k][j] + 1) / acc

    return d, word_count


def perplexity(d, filepath, ngram):
    """
    Compute per-review perplexity of a language model on a corpus file.
    Returns a list of perplexity scores, one per line (review).
    Unseen words and bigrams are handled via <unk> backoff.
    """
    plist = []

    with open(filepath) as f:
        for line in f:
            perplex = 0
            word_count = 0
            last_word = '<START>'

            for word in line.split():
                word_count += 1

                if ngram == 1:
                    prob = d[word] if word in d else d['<unk>']
                    perplex -= math.log(prob)
                else:
                    if last_word == '<START>':
                        last_word = word
                        continue
                    if last_word in d:
                        prob = d[last_word][word] if word in d[last_word] else d[last_word]['<unk>']
                    else:
                        prob = d['<unk>']['<unk>']
                    perplex -= math.log(prob)
                    last_word = word

            plist.append(math.exp(perplex / word_count))

    return plist


# --- Train language models ---
deceptive_unigram, _ = make_unigram(TRAIN_D)
truthful_unigram, _  = make_unigram(TRAIN_T)
deceptive_bigram, _  = make_bigram(TRAIN_D)
truthful_bigram, _   = make_bigram(TRAIN_T)

# --- Validation: unigram ---
print("Unigram validation perplexity (model on matching class):")
print(sum(perplexity(deceptive_unigram, VAL_D, 1)) / len(perplexity(deceptive_unigram, VAL_D, 1)),
      sum(perplexity(truthful_unigram,  VAL_T, 1)) / len(perplexity(truthful_unigram,  VAL_T, 1)))

# --- Validation: bigram ---
Dp = perplexity(deceptive_bigram, VAL_D, 2)
Tp = perplexity(truthful_bigram,  VAL_T, 2)
Dp_cross = perplexity(deceptive_bigram, VAL_T, 2)
Tp_cross = perplexity(truthful_bigram,  VAL_D, 2)

print("Bigram validation perplexity (model on matching class):")
print(sum(Dp) / len(Dp), sum(Tp) / len(Tp))
print("Bigram validation perplexity (model on opposite class):")
print(sum(Dp_cross) / len(Dp_cross), sum(Tp_cross) / len(Tp_cross))

# --- Test predictions ---
Dp_test = perplexity(deceptive_bigram, TEST, 2)
Tp_test = perplexity(truthful_bigram,  TEST, 2)

with open('test.csv', 'w') as f:
    f.write('Id,Prediction\n')
    for i in range(len(Dp_test)):
        # Classify as deceptive (0) if deceptive model assigns lower perplexity
        label = 0 if Dp_test[i] < Tp_test[i] else 1
        f.write(f"{i},{label}\n")
    d['<unk>'] = smoothing*len(d)
    word_count += d['<unk>']

    for k in d:

        d[k] = (d[k]+0.6)/(word_count+0.6*len(d))#corpus length includes punctation?


    return d, word_count

def make_Bigram(filepath):
    smoothing = 1
    acc = 0;
    last_word = '<LASTWORDINITIALIZE>'
    word_count = 0
    d = collections.defaultdict(lambda:collections.defaultdict(int))
    with open(filepath) as f:
        for line in f:
            last_word = '<LASTWORDINITIALIZE>'
            for word in line.split():
                #word = word.lower() #keeping capitalization 
                word_count +=1
                if last_word == '<LASTWORDINITIALIZE>' :
                    last_word = word
                    continue
                
                d[last_word][word]+=1
                last_word = word

    NUM_BIGRAMS = sum(len(bigrams) for bigrams in d.values())

    d['<unk>']['<unk>'] = NUM_BIGRAMS * smoothing


    # len(d) is just number of unique tokens 
    unk = 0
    for k in d:
        d[k]['<unk>'] =  len(d[k]) * smoothing 
 
        acc = 0 
        for i in d[k]:

            acc+= (d[k][i] + 1)
        for j in d[k]:
            
            d[k][j] = (d[k][j]+1)/acc

    return d, word_count

def perplexity(d, filepath, ngram):
    perplex = 0
    plist = []
    word_count1 = 0
    last_word = '<LASTWORDINITIALIZE>'
    with open(filepath) as f:
        for line in f:
            perplex = 0
            word_count1 = 0
            last_word = '<LASTWORDINITIALIZE>'
            for word in line.split():
                #word = word.lower()
                word_count1 += 1
                if ngram == 1:
                    if word in d:
                        perplex -= math.log(d[word])
                    else:
                        perplex -= math.log(d['<unk>'])

                ##BIGRAM
                else:
                    if last_word == '<LASTWORDINITIALIZE>' :
                        last_word = word
                        continue
                    if last_word in d:
                        if word in d[last_word]:
                            perplex += -math.log(d[last_word][word])
                        else:
                            perplex += -math.log(d[last_word]['<unk>'])
                    else:
                        perplex += -math.log(d['<unk>']['<unk>'])
                    last_word = word
            perplex = math.exp(perplex/word_count1)
            plist.append(perplex)   
    ## look at keys from the testing 
##    for k in d:
##        perplex += -math.log(d[k])*d[k]*word_count1
                ## this is wrong bc its not total tokens, it has
                ## it has smoothing
                

    
##    print(perplex) 

    return plist 
    


lastminD, x = make_Unigram(train_Dpath)
lastminT, y = make_Unigram(train_Tpath)

LDP = perplexity(lastminD, val_Dpath, 1)
LTP = perplexity(lastminT, val_Tpath, 1)

print ("deceptive on deceptive, T on T UNIGRAM")
LDPA = sum(LDP)/len(LDP)
LTPA = sum(LTP)/len(LTP)
print(LDPA,LTPA)

print("deceptive on truth, T on D UNIGRAM")
LDP1 = perplexity(lastminD, val_Tpath, 1)
LTP1 = perplexity(lastminT, val_Dpath, 1)
LDPA1 = sum(LDP1)/len(LDP1)
LTPA1 = sum(LTP1)/len(LTP1)
print(LDPA1,LTPA1)


decept_test, word_count1 = make_Bigram(train_Dpath)
truth_test, word_count2 = make_Bigram(train_Tpath)
Dp = perplexity(decept_test, val_Dpath , 2)
Tp = perplexity(truth_test, val_Tpath, 2)


print("deceptive on deceptive, T on T BIGRAM") 
##print (Dp, Tp)
Dp_avg = sum(Dp)/len(Dp)
Tp_avg = sum(Tp)/len(Tp)
print("avg d on d, agv t on t")
print(Dp_avg, Tp_avg)

Dp1 = perplexity(decept_test, val_Tpath , 2)
Tp1 = perplexity(truth_test, val_Dpath, 2)

Dp1_avg = sum(Dp1)/len(Dp1)
Tp1_avg = sum(Tp1)/len(Tp1)
print("avg d on t, agv t on d")
print(Dp1_avg, Tp1_avg)
##print("D on T, T on D")

##print (numpy.var(Dp))
##print (numpy.var(Dp1))
##print (numpy.var(Tp))
##print (numpy.var(Tp1))

Dp2 = perplexity(decept_test, train_Tpath , 2)
Tp2 = perplexity(truth_test, train_Dpath , 2)

Dp_test = perplexity(decept_test, test_path , 2)
Tp_test = perplexity(truth_test, test_path , 2)
Dpt_avg = sum(Dp_test)/len(Dp_test)
Tpt_avg = sum(Tp_test)/len(Tp_test)


##
with open('test.csv', 'w') as f:
    f.write('Id,Prediction\n')
    x = 1
    for i in range(len(Dp_test)):
        x = 1
        if (Dp_test[i]< Tp_test[i]):
            x = 0
        else:
            x = 1
##        if (Tp_test[i] > 200) & (Dp_test[i] > 200) & (abs(Tp_test[i]-Dp_test[i]) <Tp_test[1]/3):
##            x = 1
        f.write("%d,%d\n" %(i,x))
              
#this was validation testing & experimentation 
##with open('DECvaltest.csv', 'w') as f:
##    f.write('Id,Prediction\n')
##    x = 1
##    for i in range(len(Tp)):
##        x = 1
####        if abs(Dp1[i] - Tp[i]) < 0.5*Dp1_avg :
####            x = 1
##
##        if (Dp[i]< Tp1[i]):
##            x = 0
##        else:
##            x = 1
####        if ((Tp1[i] > Dp_avg) | (Dp[i] > Dp_avg)) & (abs(Tp1[i]-Dp[i]) <Tp1[1]/3):
####            x = 1
####        if (abs(Dp1[i] - Tp[i]) > 0.5*Dp1_avg):
####            x = 1
##        f.write("%d,%d,%d,%d\n"%(i, x, Dp[i],Tp1[i]))

##print("testD on TestT, and reverse")
##print(Dp2, Tp2)
##
##testP = perplexity(decept_test, train_Dpath, 2)
##print(testP)
##
##testP2 = perplexity(truth_test, train_Tpath, 2)
##print(testP2)



