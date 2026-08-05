import json

import torch
from tokenizers import Tokenizer, models
from ML.model import RnnModelForClassification, SkipGramEmbeddingModel
from torch.nn.utils.rnn import pad_sequence

@torch.no_grad()
def infer(sequence, tokenizer, model, device='cpu'):
    if isinstance(sequence, list):
        encoded_sequence = tokenizer.encode_batch(sequence)
        encoded_sequence = [torch.LongTensor(seq.ids) for seq in encoded_sequence]
        encoded_ids = pad_sequence(encoded_sequence,
                                batch_first=True,
                                padding_value=tokenizer.padding['pad_id'])
    else:
        encoded_sequence = tokenizer.encode(sequence)
        encoded_ids = torch.LongTensor(encoded_sequence.ids).unsqueeze(0) # [1, L]

    pred = model(encoded_ids.to(device)) # [B, C,] or # [B, L, E]
    if len(pred.shape) > 2:
        out = pred.sum(1) # [B, E]
    else:
        if pred.shape[0] == 1:
            pred = pred.squeeze(0)
        out = torch.softmax(pred,0).cpu().numpy()

    return out

def load_for_inference(model_path, tokenizer_file, labe_dict_path=None, embedding_dim=256, hidden_dim=512,
 num_layers=1, window_size=2, skip_gram=False):
    if labe_dict_path is not None:
        with open(labe_dict_path) as f:
            label_dict = json.load(f)
    else:
        label_dict = None

    tokenizer = Tokenizer(models.BPE())
    tokenizer = tokenizer.from_file(tokenizer_file)

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.padding['pad_id']
    if skip_gram:
        model = SkipGramEmbeddingModel(vocab_size, embedding_dim, pad_id, window_size)
    else:
        model = RnnModelForClassification(vocab_size, embedding_dim, pad_id, hidden_dim,
     num_layers, len(label_dict))
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    return model, tokenizer, label_dict
    
