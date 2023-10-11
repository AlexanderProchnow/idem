import os
import sys
disc_path = os.path.join(os.getcwd() + r'\DISC')
sys.path.insert(0, disc_path)

import pandas as pd
from idiom_finder import IdiomFinder
from DISC.disc_idiom_finder import DISCIdiomFinder

class EnsembleFinder:
    def __init__(self):
        self.finder = IdiomFinder()
        self.disc = DISCIdiomFinder() 
        self.slide = pd.read_csv('./idiomLexicon.tsv', sep='\t')
        self.lemm_idis = pd.read_csv('./idiomLexicon_lemmatized.csv', sep=';')
        self.slide['Idiom'] = self.slide['Idiom'].str.lower()
        print(self.lemm_idis.head())
    
    
    def find_idioms(self, sentence):
        final_results = [] # elements look like: [idioms, probs, links], one element for each input sentence
        
        if type(sentence) == type("string"):
            sentence = [sentence]
            
        finder_results = self.finder.find_idioms(sentence)
        for fres in finder_results:
            curr_res = [[], [], []]
            if fres == None:
                final_results.append([[None], [None], [None]])
                continue
            
            for idiom, prob in zip(fres[0], fres[1]):
                idiom = idiom.lower() # just for safety reasons idk
                link_vals = self.slide.loc[self.slide['Idiom'] == idiom, 'WiktionaryURL'].values
                
                if len(link_vals) == 0: # Idiom not found in idiom vocab 
                    continue
                
                curr_res[0].append(idiom) # idiom
                curr_res[1].append(prob[2]) # prob
                curr_res[2].append(link_vals[0]) # link
            final_results.append(curr_res)
            
        disc_results = self.disc.find_idioms(sentence)
        for i, dres in enumerate(disc_results):
            if dres == None:
                continue
            
            for idiom, prob in zip(dres[0], dres[1]):
                idiom = self.find_idiom_in_dataset(idiom.lower())
                link_vals = self.slide.loc[self.slide['Idiom'] == idiom, 'WiktionaryURL'].values
                
                if len(link_vals) == 0: # Idiom not found in idiom vocab 
                    continue
                
                if idiom not in final_results[i][0]:
                    final_results[i][0].append(idiom)
                    final_results[i][1].append(prob)
                    final_results[i][2].append(link_vals[0])
                
        return final_results
                
                
    def find_idiom_in_dataset(self, idiom):
        lemm_idi = self.finder.full_lemmatize(idiom)
        corpus_idiom = self.lemm_idis.loc[self.lemm_idis['lemmatized'] == lemm_idi, 'idiom']
        if len(corpus_idiom.values) > 0:
            return str(corpus_idiom.values[0]).lower()
        return None
    
    
if __name__ == '__main__':
    finder = EnsembleFinder()
    result = finder.find_idioms([
        "She wanted to visit Paris in the worst way, filled with longing for the romantic city.",
        "Even after winning the lottery, Jane returned to her job, keeping business as usual to avoid attention."])
    for res in result:
        print(res)