import json


query_path = 'clothingadc_data/clothingadc_queries_2000.json'
# corpus_path = 'clothingadc_data/clothingadc_corpus_5000.json'
with open(query_path, 'r') as f:
    queries = json.load(f)

for corpus_type in ['5000','10000','20000','40000','60000','100000','500000','all']:
    corpus_path = f'clothingadc_data/clothingadc_corpus_{corpus_type}.json'
    
    with open(corpus_path, 'r') as f:
        corpus = json.load(f)
    
    for query in queries:
        path = query['img']
        assert path in corpus, f"Path {path} not found in corpus {corpus_path}"
    
    print(f"All queries in {query_path} are present in {corpus_path}")
    