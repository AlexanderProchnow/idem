import numpy as np
from src.utils.model_util import load_model_from_checkpoint
from src.model.read_comp_triflow import ReadingComprehensionDetector as DetectorMdl
from config import Config as config
from demo_helper.data_processor import DataHandler
from tqdm import tqdm
import torch
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# PLEASE RUN THIS FILE IN THE DISC DIRECTORY
# otherwise the loading of certain directories will fail

class DISCIdiomFinder:
    def __init__(self) -> None:        
        # Use config.py to change data and model settings
        self.data_handler = DataHandler(config)
        self.detector_model= load_model_from_checkpoint(DetectorMdl, self.data_handler.config)

    def find_idioms(self, sentence):
        """
        Docstring and whatnot
        """

        # MAKE BATCHES
        if type(sentence) == list:
            # Need to make batches, otherwise it crashes when doing inference (memory error or something, so if it crashes, maybe reduce batch size)
            bs = 16 # config.BATCH_SIZE
            batches = [sentence[b:b + bs] for b in range(0, len(sentence), bs)]
            assert len(batches) == np.ceil(len(sentence) / bs)
        elif type(sentence) == str:
            batches = [[sentence]]
        else: 
            raise TypeError(f'The sentence "{sentence}" is {type(sentence)} but must be {type("")} or {type([])}')
        
        # PREPARE DATA
        data = []
        for b in batches:
            d = self.data_handler.prepare_input(b)
            data.append(d)
        assert len(batches) == len(data)

        # PERFORM INFERENCE
        outputs = []
        for batch in tqdm(data):
            try:
                outputs.append(self.inference(batch))
            except:
                print("Something went wrong in a batch")
                outputs.append(([], [], []))

        # EXTRACT IDIOMS
        idioms = self.extract_idioms(outputs)

        return idioms

    def inference(self, batch):
        with torch.no_grad():
            ys_ = self.detector_model(batch)
            probs = torch.nn.functional.softmax(ys_, dim=-1)
        ys_ = ys_.cpu().detach().numpy()
        probs = probs.cpu().detach().numpy()
        # idiom_class_probs = probs[:, :, -1].tolist()
        predicts = np.argmax(ys_, axis=2)
        sentences_tkns = batch["xs_bert"].cpu().detach().numpy().tolist()
        sentences_tkns = [self.data_handler.tokenizer.convert_ids_to_tokens(s) for s in sentences_tkns]
        return probs, predicts, sentences_tkns

    def extract_idioms(self, outputs):
        """
        Extracts the idioms and simultaneously unbatches the samples. 
        ...
        returns the idioms in the form [([sentence 1 idioms], [sentence 1 scores]), ([sentence 2 idioms], [sentence 2 scores]), ...]
        """

        idioms = []
        for batch in outputs:
            probs, predicts, sentences_tkns = batch
            idiom_class_probs = probs[:, :, -1].tolist()
            assert len(predicts) == len(probs) == len(sentences_tkns) == len(idiom_class_probs)
            for i in range(len(predicts)):
                assert len(predicts[i]) == len(probs[i]) == len(sentences_tkns[i]) == len(idiom_class_probs[i])
                padding_start = np.where(np.array(sentences_tkns[i]) == self.data_handler.tokenizer.pad_token)[0]
                padding_start = padding_start[0] if len(padding_start) > 0 else len(predicts[i])
                # The line below extracts the indices where a value of 4 is predicted. This value corresponds to idiomatic usage of a token
                idiom_indices = np.where(predicts[i][:padding_start] == 4)[0]
                idioms_for_one_sample = []
                scores_for_one_sample = []
                current_idiom = []
                current_score = 1
                pointer = None
                for idiom_idx in idiom_indices:
                    if pointer is None or idiom_idx == pointer + 1:
                        # We count consecutive idiom tags as the same idiom
                        current_idiom.append(sentences_tkns[i][idiom_idx])
                        # We calculate the score as the product of the probabilities of each token being used idiomatically
                        current_score *= idiom_class_probs[i][idiom_idx]
                    else:
                        # If a token in between is tagged as literal, then we begin a new idiom
                        current_idiom = self.data_handler.tokenizer.convert_tokens_to_string(current_idiom)
                        idioms_for_one_sample.append(current_idiom)
                        scores_for_one_sample.append(current_score)
                        current_idiom = [sentences_tkns[i][idiom_idx]]
                        current_score = idiom_class_probs[i][idiom_idx]
                    pointer = idiom_idx
                if pointer is not None:
                    current_idiom = self.data_handler.tokenizer.convert_tokens_to_string(current_idiom)
                    idioms_for_one_sample.append(current_idiom)
                    scores_for_one_sample.append(current_score)
                idioms.append((idioms_for_one_sample, scores_for_one_sample))
        return idioms

# if __name__ == "__main__":
#     import pandas as pd
#     df = pd.read_csv('../../Sentence Generation/Datasets/gpt4_generated_sentences_negated_clean.csv', encoding = 'unicode_escape', engine ='python')
#     sents = [s.replace('"', '').replace("'", '').strip() for s in df['Sentence'].tolist()]
#     f = DISCIdiomFinder()
#     # sents = df['sentence'].tolist()
#     # sents = [
#     #     "When I got a bad grade, I totally went bananas",
#     #     "She got very sick and kicked the bucket shortly after",
#     #     "I took a nice walk in the park with my dog",
#     #     'If you’re head over heels, you’re completely in love.',
#     #     "It's not all there yet.",
#     #     "How plumptious I feel and tickled pink"
#     # ]
#     out = f.find_idioms(sents)
#     found_idioms = []
#     for r in out:
#         if not r or len(r[0]) == 0:
#             found_idioms.append(None)
#             continue
        
#         idioms, _ = r 
#         found_idioms.append('; '.join(idioms))
        
#     comp_df = pd.DataFrame()
#     # comp_df['dataset'] = df['idiom']
#     comp_df['disc'] = found_idioms
#     # comp_df.to_csv('1k_recog_comp_disc.csv')
#     comp_df.to_excel('1k_recog_comp_disc.xlsx')
#     # print(out)