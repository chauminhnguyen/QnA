import re
import pandas as pd
import numpy as np

import os
import re
import numpy as np

def splitContext(fileC):
  fileI = open(fileC, 'r', encoding='utf-8')
  context = fileI.read()

  return context

def get_passage(question_path, data_name):
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

def getlongAns(para, res, nlpSeg):
  doc = nlpSeg(para)
  for sent in doc.sents:
    if res in sent.text:
      return sent.text
  return res

def compareAnswer(answer, predict):
  '''
  Calculate the accuracy of the model. By couting the common words between the answer and the prediction. If the number of common words is greater than or equal to the length of the answer, the model is correct.
  answer (type: str): answer string
  predict (type: List): List of predict string
  '''

  max = 0.0
  score = 0.0
  answer = re.sub("(^\W*|\W*$)", "", answer)
  for ans in predict:
    ans = re.sub("(^\W*|\W*$)", "", ans)
    a1 = re.split("\W+", answer.lower())
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

def compareAnswerByF1(gt, predicts):
  '''
  Calculate the accuracy of the model. Using F1 score
  gt (type: str): answer string
  predicts (type: List): List of predict string
  '''
  from collections import Counter

  _max = 0
  gt = re.sub("(^\W*|\W*$)", "", gt)
  for predict in predicts:
    predict = re.sub("(^\W*|\W*$)", "", predict)
    gt_toks = re.split("\W+", gt.lower())
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

def QAsimilarity(question, answers, real_answer, scores, sen_trans, LIMIT):
  query_embedding = sen_trans.encode(question)
  sen_scores = []
  for p in answers:
      passage_embedding = sen_trans.encode(p)
      score = util.dot_score(query_embedding, passage_embedding)
      sen_scores.append(score)

  indexes = np.argsort(np.array(sen_scores).flatten())
  
  limitrr_predict = []
  limitrr_real_predict = []
  limitrr_scores = []
  for it in range(LIMIT):
    limitrr_predict.append(answers[indexes[-it-1]])
    limitrr_real_predict.append(real_answer[indexes[-it-1]])
    limitrr_scores.append(scores[indexes[-it-1]])
  return limitrr_predict, limitrr_real_predict, limitrr_scores

def QAContext(question, confL, retriever, nlpSeg, model):
  answers = []
  real_answers = []
  scores = []

  
  paras = [paras.to_dict()['content'] for paras in paras]

  paras.extend(tf_idf(question, tf_idf_paras, 3))
  for para in paras:
    QA_input = {
        'question': question,
        'context': para
    }
    res = model(QA_input)
    answer = res['answer']
    score = res['score']
    real_answers.append(answer)
    if score < confL:
      answer = getlongAns(para, answer, nlpSeg)
    answers.append(answer)
    scores.append(score)
  return answers, real_answers, scores, paras