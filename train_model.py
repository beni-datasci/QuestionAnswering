import torch
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import json


#Training routine for BERT based NLP based on SQUAD2.0
#requieres pytorch with cuda support
#training data JSON can be found at: https://rajpurkar.github.io/SQuAD-explorer/
model_checkpoint = "bert-base-cased"
model_name = "bert-base-cased"
model_args = QuestionAnsweringArgs()
model_args.lazy_loading = True
model_args.train_batch_size = 8
model_args.evaluate_during_training = True
model_args.n_best_size = 3
model_args.num_train_epochs = 5
model_type = "bert"
train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"outputs/{model_type}",
    "best_model_dir": f"outputs/{model_type}/best_model",
    "evaluate_during_training": True,
    "max_seq_length": 64,
    "num_train_epochs": 5,
    "evaluate_during_training_steps": 1000,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "n_best_size": 3,
    # "use_early_stopping": True,
    # "early_stopping_metric": "mcc",
    # "n_gpu": 2,
    # "manual_seed": 4,
    "use_multiprocessing": False,
    "train_batch_size": 128,
    "eval_batch_size": 64,
    # "config": {
    #     "output_hidden_states": True
    # }
}
if __name__ == '__main__':
    model = QuestionAnsweringModel(
        model_type, model_name, args=train_args, use_cuda=True
    )
    
    #prepare data from json file 
    
    
    f = open("train-v2.0.json", encoding="utf-8")
    training_data = json.loads(f.read())["data"]
    f = open("dev-v2.0.json", encoding="utf-8")
    dev_data = json.loads(f.read())["data"]
    training_data_list = []
    #copy all training data in a list for the model to understand
    for x in training_data:
        training_data_list.extend(x["paragraphs"])
    #print(training_data_list)
    #only train first 1000 entries (time)
    model.train_model(training_data_list[0:1000], eval_data=dev_data[0]["paragraphs"])
