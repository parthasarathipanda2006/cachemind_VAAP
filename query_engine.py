from core.cold_storage import ColdStorage
from core.embedding_model import embedding_model
from chatbot import chatbot
from cache.scrl_cache import SCRLCache
from core.embedding_model import embedding_model
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Initialize storages
cold_storage = ColdStorage(
    host="localhost",
    user="root",
    password="root",
    database="cold_storage"
)
hot_storage=SCRLCache(

    capacity= 100,
    discount_rate= None,
)
def find_context(results):
    context=[]
    for doc_id,_ in results:
        context.append(cold_storage.get_document(int(doc_id)))
    return context

while True:
    context=[]
    query = input("tell me your Query:\n")
    if query == "exit":
        break
    query_embedding = np.array(
        embedding_model.embed_query(query),
        dtype=np.float32
    )
    results=hot_storage.request(query_embedding)
    if len(results)>=10:
        logger.info("CACHE HIT")
        print(context)
        context=find_context(results)
        response=chatbot.invoke(
            {
                "context":"\n".join(context),
                "question":query
            }
        )
        print(response)
    else :
        #CACHE MISS
        cold_embedding=[]
        cold_results = cold_storage.search_bm25(query, top_k=10)
        if not cold_results:
            print("No results found in cold storage.")
            continue
        for result in cold_results[:10]:
            doc_embedding = np.array(
                embedding_model.embed_documents([result.text])[0],
                dtype=np.float32
            )
            hot_storage.insert(str(result.id),doc_embedding)
            cold_embedding.append(doc_embedding)

        hot_storage.update_retrieved(cold_embedding)
        results=hot_storage.request(query_embedding)
        context=find_context(results)
        if context==[]:
            logger.info("no relevant results found")
            continue
        elif len(context)>=10:
            print(context)
            logger.info("all relevant results found")
        elif len(context)<10:
            print(context)
            logger.info(f"only{len(context)} relevant results found")
        response=chatbot.invoke(
            {
                "context":"\n".join(context),
                "question":query
            }
        )
        print(response)
        
cold_storage.close()