from sklearn.metrics.pairwise import cosine_similarity
import operator
    
def recommend(word, data, word_vectors, positivity):
    vals = list(data.values())
    maxAvg=0
    maxInd = -1
    ind = 0
    indices = dict()
    for val in vals:
        avg = 0
        vals_length = 0
        for value in val.items():
            try:
                avg = avg + (value[1] * cosine_similarity([word_vectors[word]], [word_vectors[value[0]]])[0])
                vals_length = vals_length + value[1]
            except:
                continue
        avg = avg / vals_length
        indices[ind] = float(avg)
        if avg > maxAvg:
            maxInd = ind
            maxAvg = avg
        ind = ind + 1
    sorted_indices = dict(sorted(indices.items(), key=operator.itemgetter(1), reverse=False if positivity == False else True))
    indices = {key: sorted_indices[key] for key in list(sorted_indices)[:3]}
    # print("The top 3 indices are")
    # print(indices)
    
    
    return indices
    # return maxInd

def recommend_categories(word, data, word_vectors, positivity):
    keys = list(data.keys())
    indices = dict()
    ind = 0
    for key in keys:
        indices[ind] = float(cosine_similarity([word_vectors[key.lower()]], [word_vectors[word]]))
        ind = ind + 1
    
    if positivity == True:
        indices = dict(sorted(indices.items(), key=operator.itemgetter(1), reverse=True))
    else:
        indices = dict(sorted(indices.items(), key=operator.itemgetter(1), reverse=False))
    
    # print("Indices in recommend_categories after sorting")
    # print(indices)
    
    indices = {key: indices[key] for key in list(indices)[:3]}

    return indices