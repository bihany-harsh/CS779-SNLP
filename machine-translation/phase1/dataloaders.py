import json
import pickle
from lang import Language
import torch
from torch.utils.data import Dataset, DataLoader

with open("/content/drive/MyDrive/neural_machine_translation/train_data1.json", "r") as file: # FOR GOOGLE COLAB (after mounting the gdrive)
  data = json.load(file)

# English-Hindi
eng_hi_source_sent_train = []
eng_hi_target_sent_train = []
eng_hi_id_train = []

for lang_pair, lang_data in data.items():
  if lang_pair == "English-Hindi":
    print(f"Language pair: {lang_pair}")
    for d_type, d_entry in lang_data.items():
      print(f"  Data type: {d_type}")
      for id, pair in d_entry.items():
        if d_type == "Train":
          eng_hi_source_sent_train.append(pair["source"])
          eng_hi_target_sent_train.append(pair["target"])
          eng_hi_id_train.append(id)

class TranslationDataset(Dataset):
  def __init__(self, source_lang, target_lang, source_sents, target_sents):
    self.source_lang = source_lang
    self.target_lang = target_lang
    # self.source_sents = torch.tensor(source_sents)
    # self.target_sents = torch.tensor(target_sents)
    self.source_sents = source_sents
    self.target_sents = target_sents


  def __len__(self):
    return len(self.source_sents)

  def __getitem__(self, idx):
    source_sent = self.source_sents[idx]
    target_sent = self.target_sents[idx]
    # source_idx_from_sent =  self.source_lang.idx_from_sentence(list(source_sent.numpy()))
    # target_idx_from_sent =  self.target_lang.idx_from_sentence(list(target_sent.numpy()))
    source_idx_from_sent = self.source_lang.idx_from_sentence(source_sent)
    target_idx_from_sent =  self.target_lang.idx_from_sentence(target_sent)

    return torch.tensor(source_idx_from_sent), torch.tensor(target_idx_from_sent)
  
# Languages
en_lang = Language(lang="en")
hi_lang = Language(lang="hi")

# Load the en_lang instance 
with open('/content/drive/MyDrive/neural_machine_translation/saves/language_instances/en_lang.pkl', 'rb') as f: # FOR GOOGLE COLAB (after mounting the gdrive)
    en_lang = pickle.load(f)

# Load the hi_lang instance
with open('/content/drive/MyDrive/neural_machine_translation/saves/language_instances/hi_lang.pkl', 'rb') as f: # FOR GOOGLE COLAB (after mounting the gdrive)
    hi_lang = pickle.load(f)

def collate_fn(batch):
  source_batch, target_batch = zip(*batch)
  sorted_indices = sorted(range(len(source_batch)), key=lambda x: source_batch[x].size(0), reverse=True)
  sorted_source_batch = [source_batch[i] for i in sorted_indices]
  sorted_target_batch = [target_batch[i] for i in sorted_indices]

  source_padded = torch.nn.utils.rnn.pad_sequence(sorted_source_batch, padding_value=1) # <EOS> as padding
  target_padded = torch.nn.utils.rnn.pad_sequence(sorted_target_batch, padding_value=1)

  source_lengths = torch.LongTensor([len(x) for x in sorted_source_batch])
  target_lengths = torch.LongTensor([len(x) for x in sorted_target_batch])

  return source_padded, source_lengths, target_padded, target_lengths

dataset = TranslationDataset(
    source_lang=en_lang,
    target_lang=hi_lang,
    source_sents=eng_hi_source_sent_train,
    target_sents=eng_hi_target_sent_train
)

BATCH_SIZE = 64

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# Get the first batch
src_batch, src_lengths, trg_batch, trg_lengths = next(iter(dataloader))

# Convert sequences back to sentences and print
for i in range(3):
    src_sent = dataset.source_lang.sentence_from_idx([idx.item() for idx in src_batch[:, i]])
    trg_sent = dataset.target_lang.sentence_from_idx([idx.item() for idx in trg_batch[:, i]])

    print(f"Source sentence {i+1}: {src_sent}")
    print(f"Target sentence {i+1}: {trg_sent}")
    print("------")
