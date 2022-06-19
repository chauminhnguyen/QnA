import os
import re
from collections import Counter

class Document:
    '''
    Store the embedding of each paragraph in the document.
    '''
    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.load_data()
    
    def load_data(self):
        from haystack.document_stores import FAISSDocumentStore
        from haystack.utils import convert_files_to_docs

        if os.path.exists('./faiss_document_store.db'):
            os.remove(os.path.join('./faiss_document_store.db'))

        self.document_store = FAISSDocumentStore()
        dicts = convert_files_to_docs(dir_path=self.doc_path, split_paragraphs=True)
        self.document_store.write_documents(dicts)
    
    def get_paras(self):
        paras = self.document_store.get_all_documents()
        paras = [paras.to_dict()['content'].lower() for paras in paras]
        return paras
    
    def get_document_store(self):
        return self.document_store
    
    def update_embedding(self, embedding):
        self.document_store.update_embeddings(embedding)


class QA_Measure:
    def __init__(self):
        pass

    def compareAnswer(self, predicts, ground_truth):
        '''
        Calculate the accuracy of the model. By couting the common words between the answer and the prediction. If the number of common words is greater than or equal to the length of the answer, the model is correct.
        answer (type: str): answer string
        predict (type: List): List of predict string
        '''

        max = 0.0
        score = 0.0
        ground_truth = re.sub("(^\W*|\W*$)", "", ground_truth)
        for ans in predicts:
            ans = re.sub("(^\W*|\W*$)", "", ans)
            a1 = re.split("\W+", ground_truth.lower())
            a2 = re.split("\W+", ans.lower())
            count = 0
            if len(a1) > 0 and len(a2) > 0:
                # for i in range(len(a2)):
                #   if a2[i] != "" and a2[i] in a1:
                #     count += 1
                # lenM = len(a1)
                # if lenM < len(a2):
                #   lenM = len(a2)
                # score = 1.0 * count / lenM
                # if score > max:
                #   max = score
                max_len = ""
                min_len = ""
                if len(a1) > len(a2):
                    max_len = a1
                    min_len = a2
                else:
                    max_len = a2
                    min_len = a1
                for i in range(len(min_len)):
                    if min_len[i] in max_len:
                        count += 1
                score = 1.0 * count / len(min_len)
                if score > max:
                    max = score
        return max

    def compareAnswerByF1(self, predicts, ground_truth):
        '''
        Calculate the accuracy of the model. Using F1 score
        gt (type: str): answer string
        predicts (type: List): List of predict string
        '''

        _max = 0
        ground_truth = re.sub("(^\W*|\W*$)", "", ground_truth)
        for predict in predicts:
            predict = re.sub("(^\W*|\W*$)", "", predict)
            gt_toks = re.split("\W+", ground_truth.lower())
            pred_toks = re.split("\W+", predict.lower())
            if len(gt_toks) == 0 or len(pred_toks) == 0:
                return int(gt_toks == pred_toks)
            common = Counter(gt_toks) & Counter(pred_toks)
            num_same = sum(common.values())
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gt_toks)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            if f1 > _max:
                _max = f1
        
        return _max