import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(instructions: str, prompt: str) -> str:
    """Try to get a completion from the API. If it fails, wait a bit and try again."""
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


class GPT4Experiment():
    """Class for running experiments with GPT-4.

    Parameters
    ----------
    name : str
        Name of the experiment to run. Must be one of:
        'no_info', 'idiom_info', 'idiom_def', 'idiom_sent', 'idiom_def_sent', 'demonstrations'
    data_path : Path
        Path to the csv file containing the sentences to evaluate.
    """
    
    def __init__(self, name: str, data_path: Path=Path('sentence_eval_1k_gpt4.csv')):
        experiments = [
            'no_info', 'idiom_info', 'idiom_def', 'idiom_sent', 'idiom_def_sent', 'demonstrations'
        ]
        if name not in experiments:
            raise ValueError(f'Invalid experiment name {name}. Must be one of: {experiments}')
        self.name = name
        self.data_path = data_path

        self.df = pd.read_csv(data_path, index_col=0)

        # the 36 defined target emotions
        self.emotion_list = [
            'Anger', 'Resentment', 'Frustration', 'Hate', 'Disgust', 'Boredom',
            'Reluctance', 'Sadness', 'Pity', 'Loneliness', 'Humiliation', 'Longing',
            'Envy', 'Guilt', 'Regret', 'Shame', 'Fear', 'Anxiety', 'Doubt',
            'Desperation', 'Confusion', 'Shock', 'Pleasure', 'Serenity', 'Relief',
            'Happiness', 'Lust', 'Affection', 'Gratitude', 'Admiration', 'Pride',
            'Determination', 'Fascination', 'Surprise', 'Excitement', 'Hope'
        ]

        self.instructions = f'You identify the emotion expressed in a sentence ' \
                          + f'and respond with one of: {", ".join(self.emotion_list)}.'
        
        self.idiom_prompt = lambda idx : f' This sentences contains the idiom "' \
                          + f'{self.df.at[idx, "idiom"]}".'
        
        self.idiom_lexicon = pd.read_csv('../idiom_recognition/idiom_lexicon_scraped.csv')

        
        # experimental settings
        if self.name == 'no_info':
            self._experimental_settings()

        elif self.name == 'idiom_info':
            self._experimental_settings(True)

        elif self.name == 'idiom_def':
            self._experimental_settings(True, True)

        elif self.name == 'idiom_sent':
            self._experimental_settings(True, False, True)

        elif self.name == 'idiom_def_sent':
            self._experimental_settings(True, True, True)

        elif self.name == 'demonstrations':
            self._experimental_settings(demonstrations=True)


    def _experimental_settings(self,
            include_idiom_information = False,
            include_def = False, 
            include_sent = False,
            demonstrations = False
        ):
        
        self.include_idiom_information = include_idiom_information
        self.include_def = include_def
        self.include_sent = include_sent
        self.demonstrations = demonstrations


    def _def_prompt(self, idiom_id):
        """Get the definition of the idiom from the idiom lexicon."""
        definition = self.idiom_lexicon[self.idiom_lexicon['idiom_id'] == idiom_id]
        if definition.empty:
            return None
        definition = definition.iloc[0]['definition']
        
        return f' The definition of this idiom is: "{definition}"'
    
    def _sent_prompt(self, idiom_id):
        """Get the sentiment of the idiom from the idiom lexicon."""
        sentiment = self.idiom_lexicon[self.idiom_lexicon['idiom_id'] == idiom_id]
        if sentiment.empty:
            return None
        sentiment = sentiment.iloc[0]['sentiment']
        return f' The sentiment of this idiom is: {sentiment}'

    def _add_demonstrations(self, n: int=5):
        """Include n demonstrations to the prompt chosen from our full dataset."""
        dems = pd.read_csv('../Sentence Generation/Datasets/full_dataset.csv', index_col=0)
        dems = dems[['sentence', 'emotion']]
        if n <= 5:
            # Using handpicked examples that do not contain the emotion in the sentence
            dems = dems.iloc[[1, 11, 18, 25, 51]].reset_index()[:n]
        else:
            print('Using random samples as demonstrations.')
            dems = dems.sample(n)

        dem_prompt = '\n'.join([
            f'{sen} {em}' for (sen, em) in zip(dems['sentence'], dems['emotion'])
        ])
        return dem_prompt + '\n'

    
    def run(self, print_prompt: bool=False):
        # iterate over idioment sentences and pass prompt to gpt
        # for index, row in self.df.iterrows(): # for testing
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            # skip the already completed sentences
            if not pd.isna(self.df.at[index, self.name]):
                continue
            
            prompt = ''

            if self.demonstrations:
                prompt += self._add_demonstrations()


            prompt += row['sentence']

            if self.include_idiom_information:

                prompt += self.idiom_prompt(index)

                if self.include_def:
                    def_prompt = self._def_prompt(row['index'])
                    # skip if no definition is available
                    if def_prompt is None:
                        continue
                    prompt += def_prompt

                if self.include_sent:
                    sent_prompt = self._sent_prompt(row['index'])
                    if sent_prompt is None:
                        continue
                    prompt += sent_prompt

            pred = completion_with_backoff(self.instructions, prompt)
            pred = pred.replace('Emotion: ', '')

            self.df.at[index, self.name] = pred

            self.save()

            if print_prompt:
                print(prompt)
                print(f'=>{pred}')


    def save(self, path: Path=None):
        # using data_path as default:
        if path is None:
            path = self.data_path

        self.df.to_csv(path)


    def calculate_metrics(self):
        from sklearn.metrics import f1_score, accuracy_score

        df_no_na = self.df.dropna(subset=[self.name])
        print('Acc:', accuracy_score(df_no_na['emotion'], df_no_na[self.name]))
        print('F1: ', f1_score(df_no_na['emotion'], df_no_na[self.name], average='weighted'))
        print(
            'OOV emotions: ',
            1-(df_no_na[self.name].isin(self.emotion_list).sum()/len(df_no_na))
        )


    def confusion_matrix(self, i: int=0, j: int=36, scale: float=2):
        """Display confusion matrix for the predictions of the model.
        
        Parameters:
        -----------
        i, j: int
            i and j are the indices of the emotions to be included in the matrix.
        scale: float
            Scale of the plot.
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 6*scale})

        df_no_na = self.df.dropna(subset=[self.name])
        cm = confusion_matrix(df_no_na['emotion'], df_no_na[self.name], normalize='true')[i:j, i:j]

        plt.figure(figsize=(6.4*scale, 4.8*scale))
        sns.heatmap(cm, xticklabels=self.emotion_list[i:j], yticklabels=self.emotion_list[i:j])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


    def reset(self):
        self.df[self.name] = pd.NA