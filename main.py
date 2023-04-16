import math, string
import requests

import nltk
nltk.download('stopwords')
nltk.download('punkt')


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import itertools

import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import sys

stop_words = set(stopwords.words('english'))
tfidf = TfidfVectorizer()


it = 0
iterations = 2

precision = 0

# # receive user input, the search
# query = input('What\'s your search? ')

if len(sys.argv) < 5:
    sys.exit("Usage: [API Key] [Search Engine ID] [Precision] [Query]")

api_key = sys.argv[1]
search_engine_id = sys.argv[2]
target_precision = float(sys.argv[3])

query = " ".join(sys.argv[4:])


while it < iterations and precision <= target_precision:
    query = query.strip('\'\"”“')
    print("Parameters:")
    print("Client key  = " + api_key)
    print("Engine key  = " + search_engine_id)
    print("Query       = " + query)
    print("Precision   = " + str(target_precision))
    print("Google Search Results:")
    print("======================")    
    # make search request with api
    response = requests.get(f'https://www.googleapis.com/customsearch/v1?q={query}&cx={search_engine_id}&key={api_key}')

    if response.status_code == 200: # 200 means response was a success
        # print(response.json()['snippet'])
        results = response.json()['items']

        relevancies = [None for result in results]

        sents = []
        for i in range(10): # print the top 10 results
            print(f'Result: {i + 1}.')
            print('[')
            print(f' URL: {results[i]["link"]}\n')
            print(f' Title: {results[i]["title"]}')
            print(f' Summary: {results[i]["snippet"]}\n')
            print(']\n')

            def remove_stopwords(string):
                word_tokens = word_tokenize(string)
                # converts the words in word_tokens to lower case and then checks whether 
                #they are present in stop_words or not
                filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
                #with no lower case conversion
                filtered_sentence = []           
                for w in word_tokens:
                    if w not in stop_words:
                        filtered_sentence.append(w)
                return ' '.join(filtered_sentence)

            sents.append(remove_stopwords(results[i]["title"]) + remove_stopwords(" " + results[i]["snippet"]))
            # print("sent\n",sents)
            answer = ""
            while answer not in ['y', 'n']:
                answer = input('Relevant (Y/N)? ').lower()
                relevancies[i] = answer

        precision = relevancies.count('y') / len(relevancies)
        #cv = CountVectorizer(ngram_range=(2,2)) for bigrams
        print("Precision " + str(precision))
        if precision >= target_precision:
            print("Desired precision reached, done")
            break
        else:
            print("Still below the desired precision of " + str(target_precision))
            
            
        
        
        sents.append(query)
        print("Indexing results ....")
        # df = pd.DataFrame(tfidf_vector[0].T.todense(),
        # 	index=tfidf.get_feature_names_out(), columns=["TF-IDF"])
        # df = df.sort_values('TF-IDF', ascending=False) 

        tfidf_vector = tfidf.fit_transform(sents)
        tfidf_array = tfidf_vector.toarray()
        #tdidf_array: document vector and query vector

        # Rocchio algorithm
        alpha = 1
        beta = 0.75
        gamma = 0.15
        query_vector = tfidf_array[10]
        sum_relevant = 0
        sum_irrelevant = 0
        R = 0.0
        NR = 0.0
        # calculate sum of doc vec that is relevant
        for i in range(10):
            if relevancies[i] == 'y':
                sum_relevant += tfidf_array[i]
                R += 1
            elif relevancies[i] == 'n':
                sum_irrelevant += tfidf_array[i]
                NR += 1
        if R == 0 and it == 0: # terminate program
            it = -1
            break

            

        
        if R != 0:  
            sum_relevant = sum_relevant/R
        else:
            sum_relevant = 0
        if NR != 0:
            sum_irrelevant = sum_irrelevant/NR 
        else: 
            sum_irrelevant = 0

        query_vector_new = alpha * query_vector + beta * sum_relevant + gamma * sum_irrelevant

        words_list = tfidf.get_feature_names_out() 
        #print(words_list)
        cnt = 0
        new_words = []
        query_list = query.split()
        


        word_tuple = []
        def take2(elem):
            return elem[1]
        #store words and corresponding weight in type list[tuple] for sorting
        for i in range(len(words_list)):
            word_tuple.append((words_list[i],query_vector_new[i]))
        word_tuple.sort(key = take2, reverse = True)

        # print(word_tuple)
        for i in range(len(word_tuple)):
            if cnt < 2:
                translator = str.maketrans('', '', string.punctuation) # need to remove punctuation from words, to prevent same word from being indexed

                if word_tuple[i][0].translate(translator) not in query_list:
                    cnt = cnt + 1
                    new_words.append(word_tuple[i][0])
                # elif word_tuple[i][0] in query_list:
                #     new_words.append(word_tuple[i][0])
        query = ' '.join(new_words)


        if (it == 0 and precision == 0) or it >= iterations:
            print("Below desired precision, but can no longer augment the query")
            break
        print("Augmenting by " + new_words[0] + " " + new_words[1] )

        # reordering query
        # first, extract weights of query words
        new_query = query_list + new_words
        new_weights = []

        for word in new_query:
            for word_weight in word_tuple:
                if word_weight[0] == word:
                    new_weights.append(word_weight[1])

        # make lang model
        
        n = 3  # size of ngrams
        sentences = [sent.split() for sent in sents]
        train_data, padded_sents = padded_everygram_pipeline(n, sentences)
        vocab = set(itertools.chain.from_iterable(sentences))
        
        model = MLE(n)
        model.fit(train_data, vocab)

        # function to calc the perplexity of each of the ngrams we make
        # docs used https://www.nltk.org/api/nltk.lm.html
        def find_perplexity(model, word_list, n):
            # pad sentence and convert to n-grams
            pad_words = ['<s>'] * (n-1) + list(word_list) + ['</s>'] # have to make sure that beginnings and ends of sentences are handled
            ngrams = list(nltk.ngrams(pad_words, n))
            
            
            log_prob = 0 # calculate perp using log
            for ngram in ngrams:
                context = tuple(ngram[:-1])
                word = ngram[-1]
                prob = model.score(word, context)
                log_prob += prob
                perplexity = math.exp(-log_prob / len(word_list) * math.log(2))
            return perplexity

        # get list of all permutations of our potential new query
        new_query = query_list + new_words
        perms = list(itertools.permutations(new_query))

        # calc perpl of each perm
        perplexities = []
        for perm in perms:
            perplexity = find_perplexity(model, perm, n)
            perplexities.append((perm, perplexity))

        # get best(smalled) perplexity perm
        best_permutation = min(perplexities, key=lambda x: x[1])[0]
        # print(best_permutation)
        query = ' '.join(best_permutation)

    else: # failed search response
        print(f'Request failed.')
