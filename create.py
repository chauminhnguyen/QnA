from haystack.utils import convert_files_to_docs
from haystack.document_stores import FAISSDocumentStore

document_store = FAISSDocumentStore(faiss_index_factory_str="HNSW")

doc_dir = './test'
dicts = convert_files_to_docs(dir_path=doc_dir, split_paragraphs=True)
document_store.write_documents(dicts)

from haystack.nodes import DensePassageRetriever

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    # max_seq_len_query=64,
    max_seq_len_passage=128,
    batch_size=4,
    use_gpu=True,
    embed_title=True,
    use_fast_tokenizers=True,
)

document_store.update_embeddings(retriever)
document_store.save('./sh/model.db')