import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib
from transformers import DistilBertTokenizer, AutoTokenizer
from transformers import DistilBertForSequenceClassification, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
import wandb
#import warnings

#warnings.simplefilter('ignore')


matplotlib.use('Agg')

# data preparation
df = pd.read_csv('cleaned_synth_data.csv')
df.duplicated().sum()
df['entry'].str.len().plot.hist(bins=50)
df['labels'] = df['labels'].str.split(', ')
label_counts = [l for label in df['labels'] for l in label]
pd.Series(label_counts).value_counts()

# label encoder
multilabel = MultiLabelBinarizer()
labels = multilabel.fit_transform(df['labels']).astype('float32')
texts = df['entry'].tolist()

# build model
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.5, random_state=42)
checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=len(labels[0]), problem_type="multi_label_classification")

# encode data
class CaseNoteDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=250):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx])
        
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')
        # encoding = self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

train_dataset = CaseNoteDataset(train_texts, train_labels, tokenizer)
test_dataset = CaseNoteDataset(test_texts, test_labels, tokenizer)

# evaluation metrics
def multi_label_metrics(predictions, labels, threshold=0.3):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    
    
    f1 = f1_score(y_true, y_pred, average='macro')
    #print("F1: " + str(f1))
    #roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    hamming = hamming_loss(y_true, y_pred)
    #print("Hamming: " + str(hamming))

    metrics = {
        #"roc_auc": roc_auc,
        "hamming_loss": hamming,
        "f1": f1
    }
    
    return metrics

def compute_metrics(p:EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    
    return result

# training the model
# args = TrainingArguments(
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     output_dir = "./results",
#     num_train_epochs=5
# )
batch_size = 8
metric_name = "f1"
args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    #learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,

    #push_to_hub=True,
    report_to="wandb"
)

wandb.init(project="case-notes-classification", name="distilbert-base")

trainer = Trainer(model=model, 
                  args=args,
                  train_dataset=train_dataset, 
                  eval_dataset=test_dataset, 
                  compute_metrics=compute_metrics)

trainer.train()
trainer.evaluate()
trainer.save_model("distilbert-finetuned-case-notes-multi-label")
with open("multi-label-binarizer.bin", "wb") as f:
    pickle.dump(multilabel, f)
