from fastapi import FastAPI
import uvicorn
from typing import Union

import pandas as pd
import numpy as np 
import faiss

from sklearn.preprocessing import StandardScaler 
from catboost import CatBoostClassifier, Pool   

app = FastAPI()

all_base = None
valid_features = None 
valid_target = None 
scaler = None
search_index = None

@app.on_event('startup')
def start():
    global all_base
    global valid_features
    global valid_target
    global scaler 
    global search_index
    
    dict_base = {}
    col_list = ['6', '21', '25', '33', '44', '59', '65', '70']
    top_n = 100

    # Загрузка данных
    for i in range(72):
        dict_base[str(i)] = 'float32'
    valid_target = pd.read_csv('./static/validation_answer.csv', index_col=0)  
    all_base = pd.read_csv('./static/base.csv', index_col=0, dtype=dict_base)
    valid_features = pd.read_csv('./static/validation.csv', index_col=0, dtype=dict_base)       
    # Определение параметров алгоритма поиска
    # Количество кластеров
    n_cells = round(all_base.shape[0]**0.5)
    # Количество посещаемых кластеров
    n_probe = round(n_cells * 0.2)
    # Масштабирование данных
    scaler = StandardScaler()    
    scaler.fit(all_base)    
    base_sc = pd.DataFrame(scaler.transform(all_base),
                           columns=all_base.columns,
                           index=all_base.index)
    base_sc = base_sc.drop(col_list, axis=1) 
    # Размерность векторов
    dim = base_sc.shape[1] 
    # Создание индекса поиска
    quantiser = faiss.IndexFlatL2(dim)
    search_index = faiss.IndexIVFFlat(quantiser, dim, n_cells)
    search_index.train(np.ascontiguousarray(base_sc.values).astype('float32'))
    search_index.add(np.ascontiguousarray(base_sc.values).astype('float32'))
    search_index.nprobe = n_probe

@app.get('/')
def main() -> dict:
    return {'Send response to "http://localhost:5000/search" with params = list of items'}

@app.get('/search')
def match(items: Union[str, None] = None) -> dict:
    global all_base
    global valid_features
    global valid_target
    global scaler
    global search_index

    col_list = ['6', '21', '25', '33', '44', '59', '65', '70']
    top_n = 100
    try:
        query_items = items.split(',')
        sub_valid_query = valid_features.loc[query_items, :]
    except Exception as e:
        return {'status': 'Error', 'message': str(e)}
    # Масштабирование данных
    query_sc = pd.DataFrame(scaler.transform(sub_valid_query),
                            columns=valid_features.columns,
                            index=valid_features.loc[query_items, :].index)
    query_sc = query_sc.drop(col_list, axis=1)
    # Поиск топ 100 ближайших векторов
    _, valid_indices = search_index.search(np.ascontiguousarray(query_sc.values).astype('float32'),
                                    top_n) 
    i = sub_valid_query.shape[0]
    j = top_n  
    # Создание из топ 100 векторов ближайших векторов выборки 
    # для ранжирования с использованием модели машинного обучения
    rank_valid_indices = valid_indices.ravel()
    rank_valid_sub_base = all_base.iloc[rank_valid_indices, :].reset_index()
    all_valid = pd.concat([valid_features, valid_target], axis=1)
    rank_valid = all_valid.iloc[np.repeat(range(i), j), :].reset_index(drop=True)
    rank_valid = pd.concat([rank_valid, rank_valid_sub_base], axis=1, ignore_index=True)
    rank_valid = rank_valid.drop([72, 73], axis=1)
    valid_pool = Pool(data=rank_valid)
    # Загрузка обученной модели
    model = CatBoostClassifier()
    model.load_model('./static/all_trained_cb_class_model_ranker_100_Recall_0.7007874015748031_73.955')

    predicted = model.predict(valid_pool, prediction_type='Probability')
    # Формирование топ 5 рекомендаций на основе предсказания модели     
    predicted_array = predicted[:, 1].reshape(i, j)
    best_predicted_indices = (-predicted_array).argsort()[:, :5]
    best_predicted_neighbors = []
    for k, index in enumerate(best_predicted_indices):
        best_predicted_neighbors = np.concatenate((best_predicted_neighbors, valid_indices[k, index]), axis=0)
    best_predicted_neighbors = best_predicted_neighbors.reshape(i, 5)
    df_best_predicted_neighbors = pd.DataFrame(best_predicted_neighbors, index=sub_valid_query.index)
    df_best_predicted_neighbors = (df_best_predicted_neighbors
                                   .apply(lambda x: x.apply(lambda x: str(int(x)) + '-base')))
    
    return {'data': df_best_predicted_neighbors.reset_index(names='query_items').to_json(orient='index')}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)