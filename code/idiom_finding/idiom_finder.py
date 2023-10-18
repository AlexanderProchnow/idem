import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
# Download lemmatization libraries
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class IdiomFinder:
    def __init__(self, idiom_path='./idiomLexicon_lemmatized.csv'):
        """Initialize the class by loading in all idioms from a given file path.
        The loaded idiom file must contain the two columns "idiom" with the plain idiomatic expression
        and "lemmatized" which is the lemmatized version of the idiomatic expression.
        These files can be created by using and adjusting the "lemmatize_idioms.ipynb" file in this directory.

        Args:
            idiom_path (string): Path to .csv file of idioms
        """
        self.idioms, self.lemm_idioms = self.load_idioms(idiom_path)
        self.lemmatizer = WordNetLemmatizer()
        
        
    def load_idioms(self, idiom_path):
        """Load the idioms from file in pandas DataFrame

        Args:
            idiom_path (string): Path to .csv file of idioms 

        Returns:
            (list, list): One list containing the raw idioms and one list containing the lemmatized idioms
        """
        idioms = pd.read_csv(idiom_path, sep=';')
        return idioms['idiom'].to_list(), idioms['lemmatized'].to_list()
    
    
    def find_idioms(self, sentence):
        """The base function for finding idioms from a given sentence or list of sentences. 
        This function is public and supposed to be called from outside

        Args:
            sentence (string, list<string>): Either a string of list of strings with sentences or a single sentence

        Raises:
            TypeError: The given sentence must be of type <string> or type <list>. 
                TODO: It is not checked whether the contents of the list are of type <string>

        Returns:
            list: List of found idioms together with their score values (gap_count, valid_order, fbeta_score)
        """
        if type(sentence) == list: 
            return [self._find_idioms(s) for s in sentence]
        elif type(sentence) == str:
            return self._find_idioms(sentence)
        else:
            raise TypeError(f'The sentence "{sentence}" is {type(sentence)} but must be {type("")} or {type([])}')
    
    
    def _find_idioms(self, sentence):
        """The real function responsible for finding the idioms present in one given sentence. 
        This function is called after it was determined whether or not the original input is of type <string> or <list>.

        Args:
            sentence (string): A string with the sentence that should be investigated for idiomic expressions

        Returns:
            list: List of found idioms together with their score values (gap_count, valid_order, fbeta_score)
        """
        idioms_found = []
        sentence = self.full_lemmatize(sentence)
        
        # Check for each idiom, if all of it's words are also present in the given sentence.
        # Both idiom and sentence are lemmatized at this point.
        for idiom_id, idiom in enumerate(self.lemm_idioms):
            for idiom_word in idiom.split():
                # if one word is not present -> the idiom is not present in the sentence -> go to next idiom
                if idiom_word not in sentence.split():
                    break 
                
            else: # if the loop was not broken -> all idiom words are present in this sentence -> add to results
                idioms_found.append({
                    'idiom': self.idioms[idiom_id],
                    'idiom_lemm': idiom,
                })
                
        if len(idioms_found) == 0:
            return None
        
        # Remove found idioms, that are not present in the sentence, even though all of their words are
        # Example: After finding out that my money was stolen I am in a pickle now -> "in a pickle" is valid
        # Example: After coming in, my friend offered me a pickle -> "in a pickle" is not valid
        return self._validate_idioms(idioms_found, sentence)
        
        
    def full_lemmatize(self, sentence):
        """Lemmatize each word in the given sentence and concatenate them back together.
        This function is public and can also be called from outside.

        Args:
            sentence (string): The setence to be lemmatized

        Returns:
            string: The lemmatized sentence
        """
        return ' '.join([self._full_lemmatize_word(word) for word in sentence.split()])        
        
        
    def _full_lemmatize_word(self, word):
        """Lemmatize the given word.
        TODO: Check part-of-speech for more accurate lemmatizations.

        Args:
            word (string): The word to be lemamtized

        Returns:
            string: The lemmatized word
        """
        word = word.replace('-', ' ') # win-win -> win win
        word = re.sub('[a-zA-Z]*self', 'self', word) # yourself -> self, oneself -> self, xyzself -> self
        word = re.sub('[^0-9a-zA-Z ]+', '', word) # delete all non-alphanumeric characters
        word = word.lower() 
        
        # Lemmatize the word by using nltk.stem.WordNetLemmatizer()
        # The word will be treated as every part of speech once, to get all posibilities
        # v: verb, n: noun, a: adjective, r: adverbs, s: satellite adjective
        for pos in ['v', 'n', 'a', 'r', 's']: # this is bad! but idk deal with it i guess
            word = self.lemmatizer.lemmatize(word, pos=pos)
            
        # If nothing is left. For example with input ":( --> :)", all character will get deleted
        if len(word) == 0:
            return '' 
        
        # If last letter of word is "s", remove it
        if word[-1] == 's':
            return word[:-1]
        
        return word
        
        
    def _validate_idioms(self, idioms_found, sentence):
        """Validate that the found idiom is actually in the sentence and that the words are not there by accident.
        Currently, the order of the idiom's words 20% more important then the gaps in between them.
        All idioms above a certainty of 90% will be counted as valid.

        Args:
            idioms_found (list): List of found idioms, given in raw and lemmatized form
            sentence (string): Sentence in which these idioms have been found

        Returns:
            (list, list): List of found idioms together with their score values (gap_count, valid_order, fbeta_score)
        """
        res_list = []
        score_list = []
        for idiom in idioms_found:
            idi = idiom['idiom_lemm']

            # If f_beta score above 90% -> idiom is valid
            if self.get_fi_beta_score(1.2, idi, sentence) > 0.9:
                
                # Add raw idiom to results
                res_list.append(idiom['idiom'])
                
                # Add all score values to results       
                score_list.append([
                    self.get_word_gap_score(idi, sentence),
                    self.get_order_validity_score(idi, sentence),
                    self.get_fi_beta_score(1.2, idi, sentence),
                ])
                
        return res_list, score_list
    
    
    def get_word_gap_score(self, idiom, sentence):
        """Count the size of the gaps between words that belong to the idiomatic expression.
        The result is the accumulated size of the gaps divided by the number of non-idiom words.
        This function is not perfect, but good luck finding the problem ;)
        
        In the best case, the words are all next to each other.
        Example: "The broken car was a quick fix for me" -> quick fix -> Score: 1.0
        
        In the worst case, the words are spread all over the sentence, from start to finish, the largest gap possible
        Example: "Quick, I need someone for a fix" -> quick fix -> Score: 0.0 

        Args:
            idiom (string): The idiom that was found in
            sentence (string): The sentence to investigate

        Returns:
            float: A value between 0.0 and 1.0, describing the size of the gap.
        """
        idiom_word_collector = []
        idiom_split = idiom.split()
        gap_counter = 0
        num_non_idiom_words = 0

        for word in sentence.split():
            if word in idiom_split:
                idiom_word_collector.append([word, gap_counter]) # Word + number of words since last idiom word appeared
                gap_counter = 0 # reset gap counter everytime a idiom word appear in the sentence
                continue
            
            # if not in idiom.split() -> is a non-idiom word and therefore a gap!
            num_non_idiom_words += 1
            gap_counter += 1
            
        # Create bigrams for each appearing idiom-word-pair, together with their distance
        # Example: "Quick I need a fix" yields the bigram ['quick', 3, 'fix'], because the two words have a gap of three
        # Do that for all sequential pairs of idiom words
        dist_bigrams = []
        for i in range(len(idiom_word_collector) - 1):
            if idiom_word_collector[i][0] != idiom_word_collector[i+1][0]:
                dist_bigrams.append([
                    idiom_word_collector[i][0],
                    idiom_word_collector[i+1][1],
                    idiom_word_collector[i+1][0]])
                
        # Sort List of bigrams by gap size. If the same bigram is found multiple times, we can choose the one
        # with the smaller gap size
        dist_bigrams.sort(key=lambda x: x[1])
        
        # Delete all bigrams that include double-occuring words
        new_dist_bigrams = []
        word_occurances = {}
        for db in dist_bigrams:
            if db[0] not in word_occurances.keys():
                word_occurances[db[0]] = 0
            if db[2] not in word_occurances.keys():
                word_occurances[db[2]] = 0
            if word_occurances[db[0]] < 2 * idiom_split.count(db[0]) and word_occurances[db[2]] < 2 * idiom_split.count(db[2]):
                word_occurances[db[0]] += 1
                word_occurances[db[2]] += 1
                new_dist_bigrams.append(db)

        # Sum together all gaps that were found
        res_dist = sum([x[1] for x in new_dist_bigrams])
        
        # If no idiom words where found -> there are no gaps between them -> perfect scores
        if num_non_idiom_words == 0:
            return 1.0
        
        return 1 - (res_dist / num_non_idiom_words)

    
    def get_order_validity_score(self, idiom, sentence):
        """Check the order of the idiom's words as they appear in the sentence.
        This is done by creating bigrams for all idiom words and all idioms words found in the sentence.
        Then it is compared how many bigrams are the same. E.g. How many idiom words have the right neighbour word.
        In the best case, all words are in perfect order.
        In the worst case, all words are in reversed order.

        Args:
            idiom (string): The idiom that was found in
            sentence (string): The sentence to investigate

        Returns:
            float: A value between 0.0 and 1.0, describing percentual how many word pairs are ordered correctly.
        """
        # Collect all idiom words that appear in the sentence in their order
        idiom_word_collector = []
        idiom_split = idiom.split()
        for word in sentence.split():
            if word in idiom_split:
                idiom_word_collector.append(word)

        # Create bigrams for all words in the sentece
        sent_bigrams = []
        for i in range(len(idiom_word_collector) - 1):
            sent_bigrams.append([idiom_word_collector[i], idiom_word_collector[i+1]])
        
        # If the sentence is empty, the order is wrong
        if len(sent_bigrams) == 0:
            return 0.0

        # Create bigrams for all words in the idiom
        idiom_bigrams = []
        for i in range(len(idiom_split) - 1):
            idiom_bigrams.append([idiom_split[i], idiom_split[i+1]])
        
        # If the idiom is empty, the order is wrong
        if len(idiom_bigrams) == 0:
            return 0.0

        # Count the number of bigrams that overlap between the idiom and the sentence
        occ_counter = 0
        for ib in idiom_bigrams:
            if ib in sent_bigrams:
                occ_counter += 1

        # If all bigrams from the idiom are also in the sentence bigrams the order is perfect
        return occ_counter / len(idiom_bigrams)
    
    
    def get_fi1_score(self, idiom, sentence):
        """Classic F1 score. But instead of percision and recall we use word gap and order validity score

        Args:
            idiom (string): The idiom that was found in
            sentence (string): The sentence to investigate

        Returns:
            float: A value between 0.0 and 1.0
        """
        wg = self.get_word_gap_score(idiom, sentence)
        ov = self.get_order_validity_score(idiom, sentence)
        if wg == 0.0 and ov == 0.0: return 0.0
        return 2 * ((wg * ov) / (wg + ov))
    
    
    def get_fi_beta_score(self, beta, idiom, sentence):
        """Weighted F score. A larger beta leads to more weight on the right order of the words.
        A smaller beta leads to more weight on the gap size between the words.

        Args:
            beta (float): Beta value for weighting each score
            idiom (string): The idiom that was found in
            sentence (string): The sentence to investigate

        Returns:
            float: A value between 0.0 and 1.0, describing percentual how many word pairs are ordered correctly.
        """
        wg = self.get_word_gap_score(idiom, sentence)
        ov = self.get_order_validity_score(idiom, sentence)
        if wg == 0.0 and ov == 0.0: return 0.0
        return ((beta**2 + 1) * wg * ov) / (beta**2 * wg + ov)
    
    
# Try on all idioms from idioment dataset, can be switched for SLIDE dataset
if __name__ == '__main__':
    IF = IdiomFinder(idiom_path='./idiomLexicon_lemmatized.csv')
    sentences = pd.read_csv('../../../dataset/idem_test.csv', index_col=0)
    
    for sentence in sentences['sentence']:
        res, scores = IF.find_idioms(sentence)

        print(sentence)
        for r, s in zip(res, scores):
            print(f'"{r}" ({round(100 * s[2])}%)')
        print()