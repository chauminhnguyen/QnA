<<<<<<< HEAD
from utils import Document
import numpy as np
import re
from nltk.util import everygrams
from text_processing import process_stops, reverse_process_stops

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
=======
from stuff import Document
import numpy as np
import re
from nltk.util import everygrams
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

class IRetriever(object):
    def __init__(self, document : Document):
        '''
        Input:
            document (type: Document): document object
        '''
        self.document = document

    def load(self):
        # Load data and model
        raise Exception("NotImplementedException")

    def retrieve(self, query, top_k=20):
        '''
        Retrieve top_k paragraphs from the input document based on the query.
        Input:
            query (type: str): query string
            top_k (type: int): top k paragraphs to retrieve
        Output:
            retrieved_paraphs (type: list): list of retrieved paragraphs in document
        '''
        raise Exception("NotImplementedException")


class Lexical_Retriever(IRetriever):
    def __init__(self, document):
        '''
        Input:
            contents (type: list): list of paragraphs in document
        '''
        IRetriever.__init__(self, document)
<<<<<<< HEAD
        self.load()
    
    def load(self):
        self.paras = self.document.get_paras()
        # self.paras = [para.split() for para in self.paras]
        self.split_paras = []
        for para in self.paras:
            self.split_paras.append(self.text_preprocess(para))
        # build vocabulary
        self.vocab = set()
        for content in self.split_paras:
=======
        self.paras = self.document.get_paras()
        self.paras = [para.split() for para in self.paras]
        self.load()
    
    def load(self):
        # build vocabulary
        self.vocab = set()
        for content in self.paras:
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
            self.vocab.update(content)

    def sorter(self, item):
        id = list(item[1])
        return float(id[0])
    
<<<<<<< HEAD
    def text_preprocess(self, text):
        '''
        Output:
            prep_text (type: list): list of preprocessed text
        '''
        text = text.lower()
        text = re.sub("\,|\.|\-|\'|\"|\_|\+|\?|\!|\@|\#|\$|\%|\;|\[|\]|\/|\:|\)|\(", " ", text).split()
        text = [ps.stem(w.lower()) for w in text if not w.lower() in stop_words]
        return text
    
    def calc(self, qcontent):
        raise Exception("NotImplementedException")
=======
    def query_preprocess(self, query):
        '''
        Output:
            prep_query (type: list): list of preprocessed query
        '''
        query = re.sub("\,|\.|\-|\'|\"|\_|\+|\?|\!|\@|\#|\$|\%|\;|\[|\]|\/|\:|\)|\(", " ", query)
        query = query.lower().split()
        return query
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
        
    def get_top_k(self, prep_query, TF_IDF):      
        '''
        Output:
            ranked_paraphs (type: list): list of ranked paragraphs indices
        '''  
        arrs = []
        inverted_index = {}

        for i in range(len(prep_query)):
            arr = self.dic.get(prep_query[i])
            if arr == None:
                continue
            arr = (list(arr))
            arrs.append(arr)
        for i, querys in enumerate(arrs):
            q_arr = arrs[i]
            for ele in q_arr:
                x = ele[0]
                y = ele[1]
                l = TF_IDF[x, y]
                if inverted_index.get(y, None) is None:
                    inverted_index[y] = {l}
                else:
                    g = list(inverted_index.get(y))
                    f = float(l) + float(g[0])
                    inverted_index.pop(y)
                    inverted_index[y] = {f}
        arrs = np.array(list(inverted_index.items()))
        rank = sorted(arrs, key=self.sorter)
        return rank
    
<<<<<<< HEAD
    def retrieve(self, query, top_k):
        # raise Exception("NotImplementedException")
        prep_query = self.text_preprocess(query)
        score = self.calc(prep_query)
        rank = self.get_top_k(query, score)
=======
    def TF_weighting(self, qcontent):
        raise Exception("NotImplementedException")
    
    def retrieve(self, query, top_k):
        raise Exception("NotImplementedException")


class TF_IDF(Lexical_Retriever):
    def __init__(self, document):
        Lexical_Retriever.__init__(self, document)

    def TF_weighting(self, qcontent):
        self.dic = {}
        TF = np.zeros(((len(self.vocab)), len(self.paras)))
        for i, word in enumerate(self.vocab):
            if word in qcontent:
                for j, content in enumerate(self.paras):
                    TF[i, j] = content.count(word)
                    if self.dic.get(word, None) is None:
                        self.dic[word] = {(i, j)}
                    else:
                        self.dic[word].add((i, j))
            else:
                for j, content in enumerate(self.paras):
                    TF[i, j] = content.count(word)
        TF = TF / np.sum(TF, axis=0)
        
        return TF
    
    def retrieve(self, query, top_k):
        query = self.query_preprocess(query)
        TF = self.TF_weighting(query)
        DF = np.sum(TF != 0, axis=1)
        IDF =np.array([1 + np.log2(len(self.content)/DF)]).T
        TF_IDF = TF * IDF
        rank = self.get_top_k(query, TF_IDF)
        return_paras = []
        if len(rank) < top_k:
            top_k = len(rank)

        for i in range(1, top_k):
            return_paras.append(' '.join(self.paras[rank[-1*i][0]]))
        
        return return_paras

    
class BM25(Lexical_Retriever):
    def __init__(self, document):
        Lexical_Retriever.__init__(self, document)
    
    def TF_weighting(self, qcontent):
        vocab_dict = {}
        k = 1.2
        b = 0.75
        d_avg = sum([len(content) for content in self.paras])
        self.dic = {}
        TF = np.zeros(((len(self.vocab)), len(self.paras)))
        for i, word in enumerate(self.vocab):
            vocab_dict[word] = i
            if word in qcontent:
                for j, content in enumerate(self.paras):
                    TF[i, j] = content.count(word)
                    if self.dic.get(word, None) is None:
                        self.dic[word] = {(i, j)}
                    else:
                        self.dic[word].add((i, j))
            else:
                for j, content in enumerate(self.paras):
                    TF[i, j] = content.count(word)
        len_contents = np.array([len(content) for content in self.paras])
        TF = (TF * (k + 1)) / (np.sum(TF, axis=0) + k * (1-b+b*(len_contents/d_avg)))
        
        return TF, vocab_dict
    
    def retrieve(self, query, top_k):
        query = self.query_preprocess(query)
        TF, _ = self.TF_weighting(query)
        DF = np.sum(TF != 0, axis=1)
        IDF = np.array([1 + np.log2(len(self.content) + 0.5/DF + 0.5)]).T
        TF_IDF = TF * IDF
        rank = self.get_top_k(query, TF_IDF)
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
        return_paras = []
        if len(rank) < top_k:
            top_k = len(rank)

        for i in range(1, top_k):
            return_paras.append(' '.join(self.paras[rank[-1*i][0]]))
        
        return return_paras


<<<<<<< HEAD
class TF_IDF(Lexical_Retriever):
    def __init__(self, document):
        Lexical_Retriever.__init__(self, document)

    def get_vocab_dict(self):
        vocab_dict = {}
        for i, word in enumerate(self.vocab):
            vocab_dict[word] = i
        return vocab_dict
    
    def get_stem_vocab_dict(self):
        vocab_dict = {}
        for i, word in enumerate(self.vocab):
            vocab_dict[ps.stem(word)] = i
        return vocab_dict

    def calc(self, qcontent):
        self.dic = {}
        TF = np.zeros(((len(self.vocab)), len(self.split_paras)))
        for i, word in enumerate(self.vocab):
            if word in qcontent:
                for j, content in enumerate(self.split_paras):
                    TF[i, j] = content.count(word)
                    if self.dic.get(word, None) is None:
                        self.dic[word] = {(i, j)}
                    else:
                        self.dic[word].add((i, j))
            else:
                for j, content in enumerate(self.split_paras):
                    TF[i, j] = content.count(word)
        TF = TF / np.sum(TF, axis=0)
        
        DF = np.sum(TF != 0, axis=1)
        IDF = np.array([1 + np.log2(len(self.split_paras)/DF)]).T
        TF_IDF = TF * IDF
        return TF_IDF

    
class BM25(Lexical_Retriever):
    def __init__(self, document):
        Lexical_Retriever.__init__(self, document)

    def get_vocab_dict(self):
        vocab_dict = {}
        for i, word in enumerate(self.vocab):
            vocab_dict[word] = i
        return vocab_dict
    
    def get_stem_vocab_dict(self):
        vocab_dict = {}
        for i, word in enumerate(self.vocab):
            vocab_dict[ps.stem(word)] = i
        return vocab_dict
    
    def calc(self, qcontent):
        k = 1.2
        b = 0.75
        d_avg = sum([len(content) for content in self.split_paras])
        self.dic = {}
        TF = np.zeros(((len(self.vocab)), len(self.split_paras)))
        for i, word in enumerate(self.vocab):
            if word in qcontent:
                for j, content in enumerate(self.split_paras):
                    TF[i, j] = content.count(word)
                    if self.dic.get(word, None) is None:
                        self.dic[word] = {(i, j)}
                    else:
                        self.dic[word].add((i, j))
            else:
                for j, content in enumerate(self.split_paras):
                    TF[i, j] = content.count(word)
        len_contents = np.array([len(content) for content in self.split_paras])
        TF = (TF * (k + 1)) / (np.sum(TF, axis=0) + k * (1-b+b*(len_contents/d_avg)))
        
        # DF = np.sum(TF != 0, axis=1)
        # IDF =np.array([1 + np.log2(len(content)/DF)]).T

        DF = np.sum(TF != 0, axis=1)
        IDF = np.array([np.log2(1 + ((len(self.split_paras) - DF + 0.5)/(DF + 0.5)))]).T
        BM25 = TF * IDF
        
        return BM25


=======
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
class N_Gram_Retriever(IRetriever):
    def __init__(self, document):
        IRetriever.__init__(self, document)
        self.load()
    
    def load(self):
        self.paras = self.document.get_paras()
<<<<<<< HEAD
        self.split_paras = []
        for para in self.paras:
            self.split_paras.append(self.text_preprocess(para))
        
        self.vocab = set()
        for content in self.split_paras:
            self.vocab.update(content)
        
        self.IDF = self.IDF_calc()

    def get_vocab_dict(self):
        vocab_dict = {}
        for i, word in enumerate(self.vocab):
            vocab_dict[word] = i
        return vocab_dict
    
    def IDF_calc(self):
        self.vocab_dict = self.get_vocab_dict()
        TF = np.zeros(((len(self.vocab_dict)), len(self.paras)))
        for i, word in enumerate(self.vocab_dict):
            for j, content in enumerate(self.split_paras):
                TF[i, j] = content.count(word)

        # DF = np.sum(TF != 0, axis=1)
        DF = np.sum(TF, axis=1)
        IDF = np.array([1 + np.log2(len(self.paras)/DF)]).T
        return IDF
    
    def text_preprocess(self, text):
        text = text.lower()
        # text = process_stops(text)
        text = re.sub("\,|\.|\-|\'|\"|\+|\?|\!|\@|\#|\$|\%|\;|\[|\]|\/|\:|\)|\(", " ", text).split()
        text = [reverse_process_stops(ps.stem(w.lower())) for w in text if not w.lower() in stop_words]
        return text

    def retrieve(self, query, top_k):
        query = self.text_preprocess(query)
        
        res = []
        sc = np.zeros(len(self.split_paras))
        for it, para in enumerate(self.split_paras):
            gram_q = list(everygrams(query))
            gram_ps = list(everygrams(para))
            common = set(gram_q).intersection(set(gram_ps))
            
            if common:
                common_len = len(max(common, key=len))
                score = 0
                for ele in common:
                    for i, w in enumerate(ele):
                        if w in self.vocab_dict:
                            score += self.IDF[self.vocab_dict[w]] * (i + 1)
                sc[it] = score * common_len
=======
        self.split_paras = [para.split() for para in self.paras]
        
        self.bm25 = BM25(self.document)
        self.vocab = set()
        for content in self.split_paras:
            self.vocab.update(content)

    def query_preprocess(self, query):
        query = re.sub("\,|\.|\-|\'|\"|\_|\+|\?|\!|\@|\#|\$|\%|\;|\[|\]|\/|\:|\)|\(", " ", query).split()
        query = [w for w in query if not w.lower() in stop_words]
        return query

    def retrieve(self, query, top_k):
        query = self.query_preprocess(query)
        TF, vocab_dict = self.bm25.TF_weighting(query)

        DF = np.sum(TF != 0, axis=1)
        IDF = np.array([1 + np.log((len(self.split_paras) + 0.5)/(DF + 0.5))]).T
        TF_IDF = TF * IDF
        
        res = []
        sc = np.zeros(len(self.paras))
        for it, para in enumerate(self.paras):
            gram_q = list(everygrams(query)) 
            gram_ps = list(everygrams(para.lower().split()))
            common = set(gram_q).intersection(set(gram_ps))
            if common:
                common_len = len(max(common, key=len))
                # print(common, common_len)
                score = 0
                # print('===================')
                for ele in common:
                    score += TF_IDF[vocab_dict[ele[0]], it]
                    # print(ele[0])
                    # print(TF_IDF[vocab_dict[ele[0]], it])
                    # print(TF[vocab_dict[ele[0]], it])
                sc[it] = score * common_len
                # common_len[it] = len(max(common, key=len))
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
            
        args_sorted = np.argsort(sc)[::-1]
        
        for it in range(top_k):
            res.append(self.paras[args_sorted[it]])
        
        return res


class DPR_Retriever(IRetriever):
    def __init__(self, document, device):
        IRetriever.__init__(self, document)
        self.device = device
        self.load()

    
    def load(self):
        '''
        Load document store and return document
        Input:
            doc_path (type: str): path to .db file and .json file
        Output:
            doc (type: FaissDocumentStore): document store
        '''

        from haystack.nodes import DensePassageRetriever

<<<<<<< HEAD
        if self.device != 'cpu':
            self.device = 'cuda:' + str(self.device)

        self.retriever = DensePassageRetriever(
            document_store = self.document.get_document_store(),
            query_embedding_model = "./models/dpr-question_encoder-single-nq-base",
            passage_embedding_model = "./models/dpr-ctx_encoder-single-nq-base",
            # max_seq_len_query=64,
            max_seq_len_passage = 128,
            devices = [self.device],
=======
        self.retriever = DensePassageRetriever(
            document_store = self.document.get_document_store(),
            query_embedding_model = "facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model = "facebook/dpr-ctx_encoder-single-nq-base",
            # max_seq_len_query=64,
            max_seq_len_passage = 128,
            devices = ["cuda:" + str(self.device)],
>>>>>>> e76bba43cbb7f68efd0ac987bdd2ec9f36771114
            batch_size = 4,
            use_gpu = True,
            embed_title = True,
            use_fast_tokenizers = True,
        )
        self.document.update_embedding(self.retriever)
    
    def retrieve(self, query, top_k):
       paras = self.retriever.retrieve(query, top_k=top_k)
       return_paras = [para.to_dict()['content'] for para in paras]
       return return_paras