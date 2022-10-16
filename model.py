import numpy as np
import time
from retriever import N_Gram_Retriever, DPR_Retriever, TF_IDF, BM25
from utils import QA_Measure, Document, Sentence_Similarity, Snapshot, Exporter
import pandas as pd
import re
from text_processing import process_stops, reverse_process_stops
from spacy.lang.en import English
from transformers import pipeline
from collections import Counter

TEXT_COLOR_RED = '\x1b[0;31;40m'
TEXT_COLOR_GREEN = '\x1b[0;32;40m'
BACKGROUND_COLOR_RED = '\x1b[0;31;41m'


class Model:
    def __init__(self, cfg):
        command = cfg.command
        doc_path = cfg.data.path
        model_name = cfg.model.qa_name
        device = cfg.model.device
        question_path = cfg.question.path
        data_name = cfg.data.name
        export_path = cfg.report.path
        
        if command == 'test':
            self.state = Infer(cfg)
        elif command == 'evaluate':
            if question_path == None or data_name == None:
                raise ValueError("Missing question_path and data_name when using 'evaluate' command.")
            self.state = Evaluate(cfg)
        elif command == 'debug':
            if question_path == None or data_name == None:
                raise ValueError("Missing question_path and data_name when using 'debug' command.")
            self.state = Debug(cfg)
        
        else:
            raise ValueError('Command should be "test", "evaluate" or "debug"')
        
    def run(self):
        self.state.run()


class State:
    def __init__(self, cfg):
    # def __init__(self, doc_path, model_name, device):
        self.model = QuestionAnswering(cfg)
    
    def run(self):
        raise NotImplementedError
        

class Debug(State):
    def __init__(self, cfg):
        self.question_path = cfg.question.path
        self.data_name = cfg.data.name
        self.LIMIT = cfg.LIMIT
        self.LIMIT_RR = cfg.LIMIT_RR
        super().__init__(cfg)

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
            
            final_predict, _, _, _ = self.model.test_one_sample(question, logging)
            
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
                
                print(TEXT_COLOR_RED + "[INFO] Sentence Similarity results:" + '\x1b[0m')
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

        print(Counter(_len))
        return good_arr
    

class Evaluate(State):
    def __init__(self, cfg):
        self.question_path = cfg.question.path
        self.data_name = cfg.data.name
        self.export_path = cfg.report.path
        self.model_name = cfg.model.qa_name
        super().__init__(cfg)
    
    def run(self):
        measure = QA_Measure()

        good_arr = []
        bad_arr = []
        questions, answers = self.model.load_qa_from_excel(self.question_path, self.data_name)
        if self.export_path is not '':
            self.exporter = Exporter(questions, answers, self.data_name)

        count = 0
        _sum = 0.0
        _len = []

        for i in range(len(questions)):
            question = questions[i]
            t0 = time.time()
            answer = answers[i]
            final_predict, final_real_predict, final_scores, final_sen_scores = self.model.test_one_sample(question)
            t1 = time.time()

            _len.append(len(final_predict))
            sc = measure.compareAnswerByF1(final_predict, answer)
            sc_l = measure.list_compareAnswerByF1(final_predict, answer)

            if self.export_path is not '':
                self.exporter.update(final_predict, final_scores, sc_l)
            
            if sc >= 0.5:
                sc = 0.99999999
                count += 1
                good_arr.append(i)
            else:
                bad_arr.append(i)

            print("{:<15} {}".format("Question: ", question))
            print("{:<15} {}".format("Answer: ", answer))
            for it in range(len(final_real_predict)):
                print("{:<15} {}".format("Predict " + str(it) + ": ", str(final_predict[it])))
                print("{:<15} {}".format("Real Predict " + str(it) + ": ", str(final_real_predict[it])))
                print("{:<15} {}".format("Score " + str(it) + ": ", str(final_scores[it])))
                print("{:<15} {}".format("Sen Score " + str(it) + ": ", str(final_sen_scores[it])))
            
            print("{:<15} {:.3f}".format("Time: ", (t1 - t0)))
            print("{:<15} {}".format("Acc: ", str(sc)))
            print("===================================================")
        
            _sum += sc
        print(count)
        print(len(questions))

        acc = _sum / len(questions) * 1.0
        print("Accuracy: " + str(acc))

        print(Counter(_len))
        print('bad_arr', bad_arr)
        
        if self.export_path is not '':
            print("[INFO] Exporting to excel...")
            
            # len_result = {
            #     'n_question': len(questions),
            #     'n_ground_truth': len(answers),
            #     'n_predict': len(final_predict_l),
            #     'n_score': len(final_scores_l),
            #     'n_comp_score': len(comparison_scores_l)
            # }

            # {'n_question': 100, 'n_ground_truth': 100, 'n_predict': 300, 'n_score': 300}
            self.exporter.export(self.export_path)
            print("Done!")
        return good_arr


class Infer(State):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def run(self):
        while True:
            while True:
                question = input("Question: ")
                if question[-1] != '?':
                    print("Please input a question!")
                else:
                    break

            final_predict, final_real_predict, final_scores, final_sen_scores = self.model.test_one_sample(question)
            for it in range(len(final_predict)):
                print("{:<11} {}".format("Predict " + str(it) + ": ", str(final_predict[it])))
            print("===================================================")


class QuestionAnswering:
    def __init__(self, cfg):
        self.device = cfg.model.device
        self.doc_path = cfg.data.path
        self.model_name = cfg.model.qa_name
        self.sen_sim_name = cfg.model.sen_sim_name
        self.LIMIT = cfg.LIMIT
        self.LIMIT_RR = cfg.LIMIT_RR
        self.load_model()
    
    def load_model(self):
        # Load sentence segmentation model
        self.nlpSeg = English()  # just the language with no model
        self.nlpSeg.add_pipe("sentencizer")
        # Load data
        document_store = Document(self.doc_path)
        # Load Retriever model
        self.lexical_retriever = N_Gram_Retriever(document_store)
        self.sementic_retriever = DPR_Retriever(document_store, self.device)
        # Load QA model
        if self.device == 'cpu':
            self.model = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name)
        else:
            self.model = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name, device=int(self.device))
        # Load Sentence Similarity model
        self.sen_sim = Sentence_Similarity(self.sen_sim_name, self.device)
        
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
    
    def text_preprocess(self, text):
        text = text.lower()
        text = re.sub("\-|\'|\"|\+|\@|\#|\$|\%|\/|\)", "", text)
        return text

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
        elif data_name == 'Sherlock':
            for index, question in enumerate( dfs['Sherlock_Milestone_2']['Question']):
                if dfs['Sherlock_Milestone_2']['Answer 1'][index] is not np.nan:
                    questions.append(question.replace(u'\xa0', u' '))
                    answers.append(dfs['Sherlock_Milestone_2']['Answer 1'][index].replace(u'\xa0', u' '))
        
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
            # if len(answer) > 5 or answer.isalnum():
            if re.match('\w', answer) is not None:
                real_answers.append(answer)
                if score < confL:
                    para = process_stops(para)
                    answer = self.getlongAns(para, res['start'], res['end'])
            
                answers.append(answer)
                scores.append(score)
                
        return answers, real_answers, scores, paras
    
    def test_one_sample(self, question, logging : Snapshot = None):

        # ========================== Model Infer ==========================
        question = self.text_preprocess(question)
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
        paras_scores = self.sen_sim.similarity(question, limit_paras)
        sen_scores = self.sen_sim.similarity(question, limit_predict)
        sen_model_scores = [(sen_score + paras_score) for sen_score, paras_score in zip(sen_scores, paras_scores)]
        # sen_model_scores = [(sen_score + paras_score + model_score) for sen_score, paras_score, model_score in zip(sen_scores, paras_scores, limit_scores)]
        # sen_model_scores = [(sen_score + model_score) for sen_score, model_score in zip(sen_scores, limit_scores)]
        indexes = np.argsort(np.array(sen_model_scores).flatten())

        # Logging
        if logging is not None:
            for i in range(self.LIMIT):
                _limit_predict = limit_predict[indexes[-i-1]],
                _limit_real_predict = limit_real_predict[indexes[-i-1]]
                _limit_scores = str(limit_scores[indexes[-i-1]])
                _limit_sen_scores = str(sen_scores[indexes[-i-1]])

                logging.append_retriever_results(_limit_predict, _limit_real_predict, _limit_scores, _limit_sen_scores)

        limit_sim_predict = []
        limit_sim_real_predict = []
        limit_model_scores = []
        limit_sen_scores = []
        for it in range(self.LIMIT_RR):
            limit_sim_predict.append(limit_predict[indexes[-it-1]])
            limit_sim_real_predict.append(limit_real_predict[indexes[-it-1]])
            limit_model_scores.append(limit_scores[indexes[-it-1]])
            limit_sen_scores.append(sen_scores[indexes[-it-1]])
        
        final_predict = []
        final_real_predict = []
        final_model_scores = []
        final_sen_scores = []
        for it in range(len(limit_model_scores)):
            # final_predict.append(limit_sim_predict[it].lower())
            # final_real_predict.append(limit_sim_real_predict[it])
            # final_model_scores.append(limit_model_scores[it])
            # final_sen_scores.append(limit_sen_scores[it])

            if limit_model_scores[it] > 1e-4:
                if  limit_sim_predict[it].lower() in final_predict or \
                (limit_sen_scores[it] > 90. or limit_model_scores[it] > 0.9):
                    final_predict = [limit_sim_predict[it].lower()]
                    final_real_predict = [limit_sim_real_predict[it]]
                    final_model_scores = [limit_model_scores[it]]
                    final_sen_scores = [limit_sen_scores[it]]
                    break
                else:
                    final_predict.append(limit_sim_predict[it].lower())
                    final_real_predict.append(limit_sim_real_predict[it])
                    final_model_scores.append(limit_model_scores[it])
                    final_sen_scores.append(limit_sen_scores[it])

        if len(final_predict) == 0:
            final_predict.append(limit_sim_predict[0])
            final_real_predict.append(limit_sim_real_predict[0])
            final_model_scores.append(limit_model_scores[0])
            final_sen_scores.append(limit_sen_scores[0])
        return final_predict, final_real_predict, final_model_scores, final_sen_scores