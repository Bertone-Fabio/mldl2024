import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF

def get_validation_recalls(eval_dataset, db_desc, q_desc, k_values, save_dir ,print_results=False, faiss_gpu=False, dataset_name = "Name",num_queries_to_save=5):
        
        db_desc = db_desc.numpy()
        q_desc = q_desc.numpy()
        

        embed_size = db_desc.shape[1]
 
        faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        faiss_index.add(db_desc)

        # search for queries in the index
        _, predictions = faiss_index.search(q_desc, max(k_values))
        
        
        
        # start calculating recall_at_k
        ground_truth = eval_dataset.get_positives()
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print('\n') # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performance on {dataset_name}"))
        
        if num_queries_to_save > 0:
            save_predictions(predictions, eval_dataset, ground_truth, save_dir,num_queries_to_save)

        return d, predictions




def save_predictions(predictions, inference_dataset, ground_truth, save_dir,num_queries_to_save=5):
    os.makedirs(save_dir, exist_ok=True)
    # Limita il numero di query da salvare al minimo tra il numero richiesto e il numero totale di query
    num_queries_to_save = min(num_queries_to_save, len(predictions))
    q_idxs = np.random.choice(len(predictions), size = num_queries_to_save, replace = False)

    for q_idx in q_idxs:
        pred = predictions[q_idx]
        # Ottieni l'immagine della query e l'etichetta
        query_index = inference_dataset.num_db_images + q_idx  
        #_, query_label = inference_dataset[query_index] 
        query_image_path = inference_dataset.all_image_paths[query_index]
        query_image = Image.open(query_image_path)
        
        # Crea una cartella per la query corrente
        query_dir = os.path.join(save_dir, f"query_{q_idx}")
        os.makedirs(query_dir, exist_ok=True)
        
        # Salva l'immagine della query
        query_image_filename = f"query_{q_idx}.jpg"
        query_image_path = os.path.join(query_dir, query_image_filename)
        query_image.save(query_image_path)
        
        # Salva le immagini delle predizioni
        for i, pred_index in enumerate(pred):
            #_, pred_label = inference_dataset[pred_index] 
            pred_image_path = inference_dataset.all_image_paths[pred_index]
            pred_image = Image.open(pred_image_path)
            pred_type = "correct" if pred_index in ground_truth[q_idx] else "incorrect"
            pred_image_filename = f"pred_{i}_{pred_type}.jpg"
            pred_image_path = os.path.join(query_dir, pred_image_filename)
            pred_image.save(pred_image_path)
        
    return


