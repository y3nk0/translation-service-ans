import numpy as np

import string
import math
import pickle


def getScore(prevToken, possibleToken, nextToken, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    pseudoCount = 5.0
    
    #Get Unigram Score
    nominator = uniDist[possibleToken]+pseudoCount    
    denominator = 0    
    for alternativeToken in wordCasingLookup[possibleToken.lower()]:
        denominator += uniDist[alternativeToken]+pseudoCount
        
    unigramScore = nominator / denominator
        
        
    #Get Backward Score  
    bigramBackwardScore = 1
    if prevToken != None:  
        nominator = backwardBiDist[prevToken+'_'+possibleToken]+pseudoCount
        denominator = 0    
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += backwardBiDist[prevToken+'_'+alternativeToken]+pseudoCount
            
        bigramBackwardScore = nominator / denominator
        
    #Get Forward Score  
    bigramForwardScore = 1
    if nextToken != None:  
        nextToken = nextToken.lower() #Ensure it is lower case
        nominator = forwardBiDist[possibleToken+"_"+nextToken]+pseudoCount
        denominator = 0    
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += forwardBiDist[alternativeToken+"_"+nextToken]+pseudoCount
            
        bigramForwardScore = nominator / denominator
        
        
    #Get Trigram Score  
    trigramScore = 1
    if prevToken != None and nextToken != None:  
        nextToken = nextToken.lower() #Ensure it is lower case
        nominator = trigramDist[prevToken+"_"+possibleToken+"_"+nextToken]+pseudoCount
        denominator = 0    
        for alternativeToken in wordCasingLookup[possibleToken.lower()]:
            denominator += trigramDist[prevToken+"_"+alternativeToken+"_"+nextToken]+pseudoCount
            
        trigramScore = nominator / denominator
        
    result = math.log(unigramScore) + math.log(bigramBackwardScore) + math.log(bigramForwardScore) + math.log(trigramScore)
  
  
    return result

def getTrueCase(tokens, outOfVocabularyTokenOption, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
    """
    Returns the true case for the passed tokens.
    @param tokens: Tokens in a single sentence
    @param outOfVocabulariyTokenOption:
        title: Returns out of vocabulary (OOV) tokens in 'title' format
        lower: Returns OOV tokens in lower case
        as-is: Returns OOV tokens as is
    """
    tokensTrueCase = []
    for tokenIdx in range(len(tokens)):
        token = tokens[tokenIdx]
        if token in string.punctuation or token.isdigit():
            tokensTrueCase.append(token)
        else:
            if token in wordCasingLookup:
                if len(wordCasingLookup[token]) == 1:
                    tokensTrueCase.append(list(wordCasingLookup[token])[0])
                else:
                    prevToken = tokensTrueCase[tokenIdx-1] if tokenIdx > 0  else None
                    nextToken = tokens[tokenIdx+1] if tokenIdx < len(tokens)-1 else None
                    
                    bestToken = None
                    highestScore = float("-inf")
                    
                    for possibleToken in wordCasingLookup[token]:
                        score = getScore(prevToken, possibleToken, nextToken, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
                           
                        if score > highestScore:
                            bestToken = possibleToken
                            highestScore = score
                        
                    tokensTrueCase.append(bestToken)
                    
                if tokenIdx == 0:
                    tokensTrueCase[0] = tokensTrueCase[0].title();
                    
            else: #Token out of vocabulary
                if outOfVocabularyTokenOption == 'title':
                    tokensTrueCase.append(token.title())
                elif outOfVocabularyTokenOption == 'lower':
                    tokensTrueCase.append(token.lower())
                else:
                    tokensTrueCase.append(token) 
    
    return tokensTrueCase

    f = open('english_distributions.obj', 'rb')
    uniDist = pickle.load(f)
    backwardBiDist = pickle.load(f)
    forwardBiDist = pickle.load(f)
    trigramDist = pickle.load(f)
    wordCasingLookup = pickle.load(f)
    f.close()

def truecasing_by_stats(input_text):
    truecase_text = ''
    sentences = sent_tokenize(input_text, language='english')
    for s in sentences:
        tokens = [token.lower() for token in nltk.word_tokenize(s)]
        tokensTrueCase = getTrueCase(tokens, 'lower', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
        sTrueCase = re.sub(" (?=[\.,'!?:;])", "", ' '.join(tokensTrueCase))
        truecase_text = truecase_text + sTrueCase + ' '
    return truecase_text.strip()


def um8_unit_replace(string):
    """Rule um8
    The letter μ meaning micro in a unit is replaced by the letter u.
    """
    if 'mol / L' in string:
        string = string.replace("μ","u")
    return string


def or1_orthographe_replace(string):
    """
    """
    string = string.replace("aigüe", "aiguë")
    return string


def check_for_mg(string):
    check = string.split("m")
    print(check)
    if check[len(check)-1] == "g":
        print("already in mg")
    else:
        step = string.split(" ")
        step2 = step[len(step)-1].split("mg")
        newstring = " ".join(step)
        step3 = step2[0].split("g")
        step4 = float(step3[0])
        newstring = newstring.replace(step2[0], str(step4*10000)+"mg")
        print(newstring)


def gr1_replace_greek_letter_with_long(string):
    """Rule gr1
    Greek letters are written in long form.
    """
    greek_letters = {}
    greek_letters['α'] = 'alpha'
    greek_letters["β"] = 'beta'
    greek_letters["γ"] = 'gamma'
    greek_letters['δ'] = 'delta'
    greek_letters['ε'] = 'epsilon'

    greek_letters['μ'] = 'mu'
    greek_letters['σ'] = 'sigma'

    for word, initial in greek_letters.items():
        string = string.replace(string.lower(), initial)
    return string


def disorder_fix():
    """
    "The word 'disorder' appearing in the preferred English term is translated as appropriate by:
      - trouble (if the object is a function),
      - affection (if the object is a body structure),
      - disorder or anomaly or nothing in other cases"

      Eating disorder -> trouble de l’alimentation
      Sleep disorder -> trouble du sommeil
      Developmental disorder -> trouble du développement Disorder of skin -> affection de la peau
      Lung disorder -> affection du poumon
      Rectal disorder -> affection rectale
      Disorder of electrolytes -> désordre hydroélectrolytique
      Chromosomal disorder -> anomalie chromosomique
      Disorder of acid-base balance -> déséquilibre acidobasique"
    """
    functions = ['Eating', 'sleep', 'Development']
    structures = ['Lung', 'rectal']
    if disorder in string:
        found = False
        for function in functions:
            if function in string:
                trans = "trouble"
                found = True
                break

        if found = False:
            for structure in structures:
                if structure in string:
                    trans = 'affection'
                    found = True
                    break

        if found = False:
            trans = 'anomalie'

    return string, trans


def pa2_disorder_fix_with_embeddings():
    """
    check if function or structure with topics from word embeddings or with distance to keyword
    """
