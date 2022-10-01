from haystack.document_stores import FAISSDocumentStore
import re
import pandas as pd
import numpy as np

import time
from spacy.lang.en import English
from transformers import pipeline

nlpSeg = English()  # just the language with no model
sentencizer = nlpSeg.create_pipe("sentencizer")
nlpSeg.add_pipe(sentencizer)

document_store = FAISSDocumentStore(faiss_index_path="./sh/model.db", \
                                    faiss_config_path="./sh/model.json")

from haystack.nodes import DensePassageRetriever

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    # max_seq_len_query=64,
    max_seq_len_passage=128,
    batch_size=4,
    use_gpu=False,
    embed_title=True,
    use_fast_tokenizers=True,
)

from sentence_transformers import SentenceTransformer, util
sen_trans = SentenceTransformer('sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco')

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
          questions.append(question)
          answers.append(dfs['NY_Milestone_2']['Answer 1'][index])
  elif data_name == 'BitCoin':
      for index, question in enumerate( dfs['Bitcoin_Milestone_2']['Question']):
          if dfs['Bitcoin_Milestone_2']['Answer 1'][index] is not np.nan:
              questions.append(question)
              answers.append(dfs['Bitcoin_Milestone_2']['Answer 1'][index])
  
  return questions, answers

def getlongAns(para, res, nlpSeg):
  doc = nlpSeg(para)
  for sent in doc.sents:
    if res in sent.text:
      return sent.text
  return res

def compareAnswer(answer, predict):
  max = 0.0
  score = 0.0
  answer = re.sub("(^\W*|\W*$)", "", answer)
  for ans in predict:
    ans = re.sub("(^\W*|\W*$)", "", ans)
    a1 = re.split("\W+", answer.lower())
    a2 = re.split("\W+", ans.lower())
    count = 0
    if len(a1) > 0 and len(a2) > 0:
      for i in range(len(a2)):
        if a2[i] != "" and a2[i] in a1:
          count += 1
      lenM = len(a1)
      if lenM < len(a2):
        lenM = len(a2)
      score = 1.0 * count / lenM
      if score > max:
        max = score
  return max

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

def QAContext(question, confL, retriever, nlpSeg, model, TOP_K):
  answers = []
  real_answers = []
  scores = []

  paras = retriever.retrieve(question, top_k=15)
  for para in paras:
    QA_input = {
        'question': question,
        'context': para.to_dict()['content']
    }
    arr_res = model(QA_input, topk=TOP_K)
    if isinstance(arr_res, dict):
      answer = arr_res['answer']
      score = arr_res['score']
      real_answers.append(answer)
      if score < confL:
        answer = getlongAns(para.to_dict()['content'], answer, nlpSeg)
      answers.append(answer)
      scores.append(score)
    else:
      for res in arr_res:
        answer = res['answer']
        score = res['score']
        real_answers.append(answer)
        if score < confL:
          answer = getlongAns(para.to_dict()['content'], answer, nlpSeg)
        answers.append(answer)
        scores.append(score)
  return answers, real_answers, scores

def test(question, model_name):
    model = pipeline('question-answering', model=model_name, tokenizer=model_name)

    LIMIT = 6
    LIMIT_RR = 3
    TOP_K = 1
    
    predict, real_predict, scores = QAContext(question, 1.0, retriever, nlpSeg, model, TOP_K)
    
    # limitrr_predict = predict
    # limitrr_real_predict = real_predict
    # limitrr_scores = scores

    s_args = np.argsort(np.array(scores))
    limit_predict = []
    limit_real_predict = []
    limit_scores = []
    for it in range(LIMIT):
      limit_predict.append(predict[s_args[-it-1]])
      limit_real_predict.append(real_predict[s_args[-it-1]])
      limit_scores.append(scores[s_args[-it-1]])

    limitrr_predict, limitrr_real_predict, limitrr_scores = QAsimilarity(question, limit_predict, limit_real_predict, limit_scores, sen_trans, LIMIT_RR)
    final_predict = []
    final_real_predict = []
    final_scores = []
    for it in range(len(limitrr_scores)):
      if limitrr_scores[it] > 10e-4:
        final_predict.append(limitrr_predict[it])
        final_real_predict.append(limitrr_real_predict[it])
        final_scores.append(limitrr_scores[it])
    
    if len(final_predict) == 0:
      final_predict.append(limitrr_predict[0])
      final_real_predict.append(limitrr_real_predict[0])
      final_scores.append(limitrr_scores[0])

    return limitrr_predict, limitrr_real_predict, limitrr_scores