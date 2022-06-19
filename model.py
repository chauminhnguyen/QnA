import numpy as np
import time
from retriever import N_Gram_Retriever, DPR_Retriever
from stuff import QA_Measure, Document
import pandas as pd

class Sentence_Similarity:
    def __init__(self, device):
        self.device = device
        self.load_model()
    
    def load_model(self):
        '''
        Load asymetric semantic search model
        Source code: https://www.sbert.net/docs/pretrained-models/msmarco-v3.html
        '''
        from sentence_transformers import SentenceTransformer, util
        self.sen_trans = SentenceTransformer('sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco', device=self.device)
        self.dot_score_measure = util.dot_score
    
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

        indexes = np.argsort(np.array(sen_scores).flatten())
        
        return indexes


class QuestionAnswering:
    def __init__(self, doc_path, model_name, device):
        self.device = device
        self.doc_path = doc_path
        self.LIMIT = 6
        self.LIMIT_RR = 3
        self.load_model(model_name, self.device)
    
    def load_model(self, model_name, device):
        # Load sentence segmentation model
        from spacy.lang.en import English
        self.nlpSeg = English()  # just the language with no model
        self.nlpSeg.add_pipe("sentencizer")
        # Load data
        document_store = Document(self.doc_path)
        # Load Retriever model
        self.lexical_retriever = N_Gram_Retriever(document_store)
        self.sementic_retriever = DPR_Retriever(document_store, self.device)
        # Load QA model
        from transformers import pipeline
        self.model = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device)
        # Load Sentence Similarity model
        self.sen_sim = Sentence_Similarity(self.device)
        
    
    def getlongAns(self, para, res, nlpSeg):
        # Get a full sentence from paragraph in which contains the <res>
        doc = nlpSeg(para)
        for sent in doc.sents:
            if res in sent.text:
                return sent.text
        return res
    
    def load_qa_from_excel(self, question_path, data_name):
        xl_file = pd.ExcelFile(question_path, engine='openpyxl')

        dfs = {sheet_name: xl_file.parse(sheet_name)
                for sheet_name in xl_file.sheet_names}

        questions = []
        answers = []

        if data_name == 'NewYork':
            for index, question in enumerate( dfs['NY_Milestone_2']['Question']):
                questions.append(question.replace(u'\xa0', u' '))
                answers.append(dfs['NY_Milestone_2']['Answer 1'][index].replace(u'\xa0', u' '))
        elif data_name == 'BitCoin':
            for index, question in enumerate( dfs['Bitcoin_Milestone_2']['Question']):
                if dfs['Bitcoin_Milestone_2']['Answer 1'][index] is not np.nan:
                    questions.append(question.replace(u'\xa0', u' '))
                    answers.append(dfs['Bitcoin_Milestone_2']['Answer 1'][index].replace(u'\xa0', u' '))
        
        return questions, answers

    def model_infer(self, question, confL):
        answers = []
        real_answers = []
        scores = []

        paras = self.sementic_retriever.retrieve(question, top_k=20)
        paras.extend(self.lexical_retriever.retrieve(question, top_k=3))

        for para in paras:
            QA_input = {
                'question': question,
                'context': para
            }
            res = self.model(QA_input)
            answer = res['answer']
            score = res['score']
            real_answers.append(answer)
            if score < confL:
                answer = self.getlongAns(para, answer, self.nlpSeg)
            answers.append(answer)
            scores.append(score)
        return answers, real_answers, scores, paras

    def test_one_sample(self, question):
        predict, real_predict, scores, paras = self.model_infer(question, 1.0)
        sc_args = np.argsort(np.array(scores))
        limit_predict = []
        limit_real_predict = []
        limit_scores = []
        for it in range(self.LIMIT):
            limit_predict.append(predict[sc_args[-it-1]])
            limit_real_predict.append(real_predict[sc_args[-it-1]])
            limit_scores.append(scores[sc_args[-it-1]])

        indexes = self.sen_sim.similarity(question, limit_predict)

        limit_sim_predict = []
        limit_sim_real_predict = []
        limit_sim_scores = []
        for it in range(self.LIMIT_RR):
            limit_sim_predict.append(limit_predict[indexes[-it-1]])
            limit_sim_real_predict.append(limit_real_predict[indexes[-it-1]])
            limit_sim_scores.append(limit_scores[indexes[-it-1]])

        final_predict = []
        final_real_predict = []
        final_scores = []
        for it in range(len(limit_sim_scores)):
            if limit_sim_scores[it] > 0:
                final_predict.append(limit_sim_predict[it])
                final_real_predict.append(limit_sim_real_predict[it])
                final_scores.append(limit_sim_scores[it])
        
        if len(final_predict) == 0:
            final_predict.append(limit_sim_predict[0])
            final_real_predict.append(limit_sim_real_predict[0])
            final_scores.append(limit_sim_scores[0])
        return final_predict, final_real_predict, final_scores
        

    def evaluate(self, question_path, data_name):
        measure = QA_Measure()

        good_arr = []
        questions, answers = self.load_qa_from_excel(question_path, data_name)

        count = 0
        _sum = 0.0
        _len = []
        for i in range(len(questions)):
            question = questions[i]
            t0 = time.time()
            answer = answers[i]
            final_predict, final_real_predict, final_scores = self.test_one_sample(question)
            t1 = time.time()

            _len.append(len(final_predict))
            sc = measure.compareAnswer(final_predict, answer)
            
            _sum += sc
            if sc >= 0.9:
                count += 1
                good_arr.append(i)
                # for it in range(len(final_real_predict)):
                #   print("Real Predict: " + str(final_real_predict[it]))
                #   print("Predict: " + str(final_predict[it]))
                #   print("Score: " + str(final_scores[it]))
                #   print("---------")
            # else:
            print("{:<11} {}".format("Question: ", question))
            print("{:<11} {}".format("Answer: ", answer))
            # print('Paras:')
            # for para in paras:
            #   print(para.to_dict()['content'])
            for it in range(len(final_real_predict)):
                # print("Real Predict: " + str(final_real_predict[it])) 
                print("{:<11} {}".format("Predict " + str(it) + ": ", str(final_predict[it])))
                # print("Score: {:<10}" + str(final_scores[it]))
                    # print("---------")
                # print("===================================================")
            print("{:<11} {:.3f}".format("Time: ", (t1 - t0)))
            print("{:<11} {}".format("Acc: ", str(sc)))
            print("===================================================")
        print(count)
        print(len(questions))

        acc = _sum / len(questions) * 1.0
        print("Accuracy: " + str(acc))

        from collections import Counter
        print(Counter(_len))
        return good_arr