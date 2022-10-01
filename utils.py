import os
import re
from collections import Counter
from typing import List
import numpy as np
import pandas as pd
from datetime import date

from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs

class Exporter():
    def __init__(self, questions: List, answers: List, data_name):
        self.questions = questions
        self.answers = answers
        self.data_name = data_name
        self.final_predict_l = []
        self.final_scores_l = []
        self.comparison_scores_l = []
    
    def update(self, final_predict, final_scores, sc_l):
        if len(final_predict) == 2:
            self.final_predict_l.extend([final_predict[0], final_predict[1], " "])
            self.final_scores_l.extend([final_scores[0], final_scores[1], " "])
            self.comparison_scores_l.extend([sc_l[0], sc_l[1], 0])
        elif len(final_predict) == 1:
            self.final_predict_l.extend([final_predict[0], " ", " "])
            self.final_scores_l.extend([final_scores[0], " ", " "])
            self.comparison_scores_l.extend([sc_l[0], 0, 0])
        else:
            self.final_predict_l.extend(final_predict)
            self.final_scores_l.extend(final_scores)
            self.comparison_scores_l.extend(sc_l)
    
    def export(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        output_path = os.path.join(path, str(self.data_name) + "_output_" + str(date.today())[5:] + ".xlsx")

        # Build a dataframe
        df = pd.DataFrame()
        df['Question'] = self.list1_2_list3(self.questions)
        df['Ground Truth'] = self.list1_2_list3(self.answers)
        df['Predict'] = self.final_predict_l
        df['Score'] = self.final_scores_l
        df['Longest Common Substring (LCS)'] = self.LCSSen_list(self.list1_2_list3(self.answers, "dup"), self.final_predict_l)
        df['Comparison Score (%)'] = np.array(self.comparison_scores_l)*100.0
        df['Note'] = self.take_note(self.list1_2_list3(self.answers, 'dup'), self.final_predict_l, self.comparison_scores_l)

        # Filter all rows where there are no predicts
        df = df.loc[df["Predict"] != " "].reset_index(drop=True)

        # Color 'Notes' column
        df = df.style.applymap(self.color_notes, subset=['Note'])

        df.to_excel(output_path, index=False)
        self.summary(output_path)
    
    def summary(self, path):
        df = pd.read_excel(path)

        with pd.ExcelWriter(path, mode="a", if_sheet_exists='new') as writer:
            df_summary = self.summary_df(df)
            df_summary.to_excel(writer)
    
    def summary_df(self, df):
        '''df(pandas DataFrame)
        ----------
        Return(pandas DataFrame): number of each type of notes in df
        '''

        df1 = df.copy()
        df2 = pd.DataFrame(index = ['Matched', "Unmatched", "Ground Truth", 'Predict', "Partially Matched", 'summary'],\
                        columns = ['Sum'])
        notes = list(df1['Note'])
        df2['Sum'][0] = np.sum([1 for x in notes if x == 'Matched'])
        df2['Sum'][1] = np.sum([1 for x in notes if x == 'Unmatched'])
        df2['Sum'][2] = np.sum([1 for x in notes if x == 'Ground Truth'])
        df2['Sum'][3] = np.sum([1 for x in notes if x == 'Predict'])
        df2['Sum'][4] = np.sum([1 for x in notes if x == 'Partially Matched'])
        df2['Sum'][5] = np.sum(list(df2['Sum'][:5]))
        
        return df2
    
    def list1_2_list3(self, list1, mode='nan'):
        list3 = []

        if mode == 'nan':
            for i in range(len(list1)):
                list3.extend([list1[i], np.nan, np.nan])
        else:
            for i in range(len(list1)):
                list3.extend([list1[i], list1[i], list1[i]])
            
        return list3

    def LCSubSen(self, X, Y): 
        '''This code is contributed by rutvik_56 and edited by me'''
        X = X.lower().split()
        Y += "."
        Y = Y.split()

        m = len(X)
        n = len(Y)
        if m < n:
            temp = X
            X = Y
            Y = temp

            temp2 = m
            m = n
            n = temp2

        result = 0  
        end = 0
        length = [[0 for _ in range(m+1)]
                    for _ in range(2)]
    
        currRow = 0
    
        for i in range(0, m + 1):
            for j in range(0, n + 1):
                if (i == 0 or j == 0):
                    length[currRow][j] = 0
                
                elif (X[i - 1] == Y[j - 1]):
                    length[currRow][j] = length[1 - currRow][j - 1] + 1
                    
                    if (length[currRow][j] > result):
                        result = length[currRow][j]
                        end = i - 1
                else:
                    length[currRow][j] = 0
            currRow = 1 - currRow

        if (result == 0):
            return "-1"

        return X[end - result + 1 : end + 1]

    def LCSSen_list(self, X, Y):
        '''X (list): Ground Truth
        Y (list): Predict
        ---------
        Return: list
        '''

        lcss = list(map(self.LCSubSen, X, Y))
        listt = []
        for i in range(len(lcss)):
            listt.append(" ".join(lcss[i]))
        return listt

    def take_note(self, X, Y, score_list):
        '''X (list): Ground Truth
        Y (list): Predict
        score_list(list): compareAnswer()
        ---------
        Return (list): notes from X and Y
        '''

        LCS_list = self.LCSSen_list(X, Y)

        notes = []
        for i in range(len(LCS_list)):

            m = len(X[i].split())
            n = len(Y[i].split())

            mem = {
                "X": "Ground Truth",
                "Y": "Predict"
            }

            if m < n:
                temp = X[i]
                X[i] = Y[i]
                Y[i] = temp

                temp2 = m
                m = n
                n = temp2
                mem = {
                    "X": "Predict",
                    "Y": "Ground Truth"
                }

            len_LCSs = len(LCS_list[i].split())

            if score_list[i] == 1.0 or len_LCSs == m:
                notes.append("Matched")
            elif len_LCSs == n:
                notes.append(mem['Y'])
            elif len_LCSs < 5 or LCS_list[i] == "-1" or LCS_list[i] == "- 1":
                notes.append("Unmatched")
            elif len_LCSs < n:
                notes.append("Partially Matched")
            else:
                notes.append("-1")
        
        return notes

    def color_notes(self, val):
        """
        Takes a scalar and returns a string with
        the css property
        """
        colours_hex = {
        "red": "#ff5454",
        "green": "#6bff81",
        "blue": "#5edfff",
        "yellow": "#faff6b",
        "gray": "#808080"
        }
        if val == 'Matched':
            color = colours_hex['green']
        elif val == 'Unmatched':
            color = colours_hex['red']
        elif val == 'Ground Truth' or val == 'Predict':
            color = colours_hex['yellow']
        elif val == 'Partially Matched':
            color = colours_hex['blue']
        else:
            color = ''
        
        return 'background-color : ' + str(color)


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


class Sentence_Similarity:
    def __init__(self, sim_sen_model, device):
        self.sen_sim_model = sim_sen_model
        self.device = device
        self.load_model()
    
    def load_model(self):
        '''
        Load asymetric semantic search model
        Source code: https://www.sbert.net/docs/pretrained-models/msmarco-v3.html
        '''
        from sentence_transformers import SentenceTransformer, util
        if self.device == 'cpu':
            self.sen_trans = SentenceTransformer(self.sen_sim_model)
        else:
            self.sen_trans = SentenceTransformer(self.sen_sim_model, device=int(self.device))
        self.dot_score_measure = util.dot_score
        self.cos_score_measure = util.cos_sim
    
    def similarity(self, question, answers):
        '''
        Calculate the similarity between question and answers
        Input:
            question (type: str): question string
            answers (type: List): List of answer string
        Output:
            similarity (type: float): similarity between question and answers
        '''
        query_embedding = self.sen_trans.encode(question)
        sen_scores = []
        for answer in answers:
            passage_embedding = self.sen_trans.encode(answer)
            score = self.dot_score_measure(query_embedding, passage_embedding)
            sen_scores.append(score)
        
        return sen_scores


class Snapshot:
    def __init__(self):
        self.question = None
        self.answer = None
        self.top_model_infer_results = []
        self.model_infer_results = []
        self.top_retriever_results = []
        self.retriever_results = []
    
    def set_question(self, question : str):
        self.question = question
    
    def set_answer(self, answer : str):
        self.answer = answer

    def append_model_infer_results(self, predict, real_predict, score):
        dic = {
            "Predict": predict,
            "Real_predict": real_predict,
            "Score": score,
        }
        self.model_infer_results.append(dic)

    def append_retriever_results(self, predict, real_predict, score, sen_score):
        dic = {
            "Predict": predict,
            "Real_predict": real_predict,
            "Score": score,
            "Sen_score": sen_score
        }
        self.retriever_results.append(dic)
    
    def get_question(self):
        return self.question
    
    def get_answer(self):
        return self.answer

    def get_model_results(self):
        return self.model_infer_results
    
    def get_retriever_results(self):
        return self.retriever_results


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
    
    def list_compareAnswerByF1(self, predicts, ground_truth):
        '''
        Calculate the accuracy of the model. Using F1 score
        gt (type: str): answer string
        predicts (type: List): List of predict string
        '''

        score_list = []
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
            score_list.append(f1)
        
        return score_list