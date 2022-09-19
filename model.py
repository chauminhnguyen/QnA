import numpy as np
import time
<<<<<<< HEAD
from retriever import N_Gram_Retriever, DPR_Retriever, TF_IDF, BM25
from utils import QA_Measure, Document, Sentence_Similarity, Snapshot
import pandas as pd
import re
from text_processing import process_stops, reverse_process_stops

TEXT_COLOR_RED = '\x1b[0;31;40m'
TEXT_COLOR_GREEN = '\x1b[0;32;40m'
BACKGROUND_COLOR_RED = '\x1b[0;31;41m'


class Model:
    def __init__(self, command, doc_path, model_name, device, **kwargs):
        if command == 'test':
            self.state = Infer(doc_path, model_name, device)
        elif command == 'evaluate':
            if not 'question_path' in kwargs and 'data_name' in kwargs:
                raise ValueError("Missing question_path and data_name when using 'evaluate' command.")
            self.state = Evaluate(doc_path, model_name, device, kwargs['question_path'], kwargs['data_name'])
        elif command == 'debug':
            if not 'question_path' in kwargs and 'data_name' in kwargs:
                raise ValueError("Missing question_path and data_name when using 'debug' command.")
            self.state = Debug(doc_path, model_name, device, kwargs['question_path'], kwargs['data_name'])
        
        else:
            raise ValueError('Command should be "test", "evaluate" or "debug"')
        
    def run(self):
        self.state.run()


class State:
    def __init__(self, doc_path, model_name, device):
        self.model = QuestionAnswering(doc_path, model_name, device)
        

class Debug(State):
    def __init__(self, doc_path, model_name, device, question_path, data_name):
        super().__init__(doc_path, model_name, device)
        self.LIMIT = 6
        self.LIMIT_RR = 3
        self.question_path = question_path
        self.data_name = data_name

    def run(self):
        measure = QA_Measure()

        good_arr = []
        questions, answers = self.model.load_qa_from_excel(self.question_path, self.data_name)

        count = 0
        _sum = 0.0
        _len = []
        for i in range(len(questions)):
            question = questions[i]
            t0 = time.time()
            answer = answers[i]
            
            logging = Snapshot()
            logging.set_question("[INFO] Question: " + question)
            logging.set_answer("[INFO] Answer: " + answer)
            
            final_predict, final_real_predict, final_scores = self.model.test_one_sample(question, logging)
            
            t1 = time.time()

            _len.append(len(final_predict))
            sc = measure.compareAnswerByF1(final_predict, answer)
            
            if sc >= 0.8:
                count += 1
                good_arr.append(i)
            else:
                print(BACKGROUND_COLOR_RED + logging.get_question() + '\x1b[0m')
                print(BACKGROUND_COLOR_RED + logging.get_answer() + '\x1b[0m')

                print(TEXT_COLOR_RED + "[INFO] Model inference results:" + '\x1b[0m')
                it = 0
                for res_dict in logging.get_model_results():
                    for k, v in res_dict.items():
                        if it < self.LIMIT:
                            print(TEXT_COLOR_GREEN + "{:<15} {}".format(k + ' ' + str(it) + ':', v) + '\x1b[0m')
                        else:
                            print("{:<15} {}".format(k + ' ' + str(it) + ':', v))
                    it += 1
                
                print(TEXT_COLOR_RED + "[INFO] Retriever results:" + '\x1b[0m')
                it = 0
                for res_dict in logging.get_retriever_results():
                    for k, v in res_dict.items():
                        if it < self.LIMIT_RR:
                            print(TEXT_COLOR_GREEN + "{:<15} {}".format(k + ' ' + str(it) + ':', v) + '\x1b[0m')
                        else:
                            print("{:<15} {}".format(k + ' ' + str(it) + ':', v))
                    it += 1
                print("===================================================")
        
            _sum += sc
        print(count)
        print(len(questions))

        acc = _sum / len(questions) * 1.0
        print("Accuracy: " + str(acc))

        from collections import Counter
        print(Counter(_len))
        return good_arr
    

class Evaluate(State):
    def __init__(self, doc_path, model_name, device, question_path, data_name):
        self.question_path = question_path
        self.data_name = data_name
        super().__init__(doc_path, model_name, device)
    
    def run(self):
        measure = QA_Measure()

        good_arr = []
        questions, answers = self.model.load_qa_from_excel(self.question_path, self.data_name)

        count = 0
        _sum = 0.0
        _len = []
        for i in range(len(questions)):
            question = questions[i]
            t0 = time.time()
            answer = answers[i]
            final_predict, final_real_predict, final_scores = self.model.test_one_sample(question)
            t1 = time.time()

            _len.append(len(final_predict))
            sc = measure.compareAnswerByF1(final_predict, answer)
            
            if sc >= 0.8:
                count += 1
                good_arr.append(i)

            print("{:<11} {}".format("Question: ", question))
            print("{:<11} {}".format("Answer: ", answer))
            for it in range(len(final_real_predict)):
                print("{:<11} {}".format("Predict " + str(it) + ": ", str(final_predict[it])))
            
            print("{:<11} {:.3f}".format("Time: ", (t1 - t0)))
            print("{:<11} {}".format("Acc: ", str(sc)))
            print("===================================================")
        
            _sum += sc
        print(count)
        print(len(questions))

        acc = _sum / len(questions) * 1.0
        print("Accuracy: " + str(acc))

        from collections import Counter
        print(Counter(_len))
        return good_arr


class Infer(State):
    def __init__(self, doc_path, model_name, device):
        super().__init__(doc_path, model_name, device)
    
    def run(self):
        while True:
            while True:
                question = input("Question: ")
                if question[-1] != '?':
                    print("Please input a question!")
                else:
                    break

            final_predict, final_real_predict, final_scores = self.model.test_one_sample(question)
            for it in range(len(final_real_predict)):
                print("{:<11} {}".format("Predict " + str(it) + ": ", str(final_predict[it])))
            print("===================================================")
=======
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
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114


class QuestionAnswering:
    def __init__(self, doc_path, model_name, device):
        self.device = device
        self.doc_path = doc_path
        self.LIMIT = 6
<<<<<<< HEAD
        self.LIMIT_RR = 1
        self.load_model(model_name)
    
    def load_model(self, model_name):
=======
        self.LIMIT_RR = 3
        self.load_model(model_name, self.device)
    
    def load_model(self, model_name, device):
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
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
<<<<<<< HEAD
        if self.device == 'cpu':
            self.model = pipeline('question-answering', model=model_name, tokenizer=model_name)
        else:
            self.model = pipeline('question-answering', model=model_name, tokenizer=model_name, device=int(self.device))
        # Load Sentence Similarity model
        self.sen_sim = Sentence_Similarity(self.device)
        
    def getlongAns(self, para, start, end):
        para = process_stops(para)
        res = list(re.finditer("\.", para))
        start_arr = [0]
        end_arr = []
        for ele in res:
            if ele.start() < start:
                start_arr.append(ele.start())
            if ele.end() > end:
                end_arr.append(ele.end())
        if len(end_arr) == 0:
            end_arr.append(-1)
        para = para[start_arr[-1]:end_arr[0]].replace('.', '').strip()
        para = reverse_process_stops(para)
        return para

=======
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
    
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
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
<<<<<<< HEAD
        elif data_name == 'Sherlock':
            for index, question in enumerate( dfs['Sherlock_Milestone_2']['Question']):
                if dfs['Sherlock_Milestone_2']['Answer 1'][index] is not np.nan:
                    questions.append(question.replace(u'\xa0', u' '))
                    answers.append(dfs['Sherlock_Milestone_2']['Answer 1'][index].replace(u'\xa0', u' '))
=======
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
        
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
<<<<<<< HEAD
                para = process_stops(para)
                answer = self.getlongAns(para, res['start'], res['end'])
                
            answers.append(answer)
            scores.append(score)
        return answers, real_answers, scores, paras
    
    def test_one_sample(self, question, logging : Snapshot = None):

        # ========================== Model Infer ==========================
        predict, real_predict, scores, paras = self.model_infer(question, 1.0)
        sc_args = np.argsort(np.array(scores))
        
        # Logging
        if logging is not None:
            for i in range(len(predict)):
                _predict = predict[sc_args[-i-1]]
                _real_predict = real_predict[sc_args[-i-1]]
                _score = str(scores[sc_args[-i-1]])
                
                logging.append_model_infer_results(_predict, _real_predict, _score)
        
        limit_predict = []
        limit_real_predict = []
        limit_scores = []
        limit_paras = []
        for i in range(self.LIMIT):
            limit_predict.append(predict[sc_args[-i-1]])
            limit_real_predict.append(real_predict[sc_args[-i-1]])
            limit_scores.append(scores[sc_args[-i-1]])
            limit_paras.append(paras[sc_args[-i-1]])

        # ========================== Sentence Similarity ==========================
        sen_scores = self.sen_sim.similarity(question, limit_paras)
        sen_model_scores = [(sen_score + model_score)/2 for sen_score, model_score in zip(sen_scores, limit_scores)]
        indexes = np.argsort(np.array(sen_scores).flatten())

        # Logging
        if logging is not None:
            for i in range(self.LIMIT):
                _limit_predict = limit_predict[indexes[-i-1]],
                _limit_real_predict = limit_real_predict[indexes[-i-1]]
                _limit_scores = str(limit_scores[indexes[-i-1]])
                _limit_sen_scores = str(sen_scores[indexes[-i-1]])

                logging.append_retriever_results(_limit_predict, _limit_real_predict, _limit_scores, _limit_sen_scores)
=======
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
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114

        limit_sim_predict = []
        limit_sim_real_predict = []
        limit_sim_scores = []
        for it in range(self.LIMIT_RR):
            limit_sim_predict.append(limit_predict[indexes[-it-1]])
            limit_sim_real_predict.append(limit_real_predict[indexes[-it-1]])
            limit_sim_scores.append(limit_scores[indexes[-it-1]])
<<<<<<< HEAD
        
        final_predict = []
        final_real_predict = []
        final_scores = [] 
        for it in range(len(limit_sim_scores)):
            if limit_sim_scores[it] > 0:
                if limit_sim_predict[it] not in final_predict:
                    final_predict.append(limit_sim_predict[it])
                    final_real_predict.append(limit_sim_real_predict[it])
                    final_scores.append(limit_sim_scores[it])
=======

        final_predict = []
        final_real_predict = []
        final_scores = []
        for it in range(len(limit_sim_scores)):
            if limit_sim_scores[it] > 0:
                final_predict.append(limit_sim_predict[it])
                final_real_predict.append(limit_sim_real_predict[it])
                final_scores.append(limit_sim_scores[it])
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
        
        if len(final_predict) == 0:
            final_predict.append(limit_sim_predict[0])
            final_real_predict.append(limit_sim_real_predict[0])
            final_scores.append(limit_sim_scores[0])
<<<<<<< HEAD
        return final_predict, final_real_predict, final_scores
=======
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
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
