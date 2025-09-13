import os
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from modeling_bart import BartForSequenceClassification

class SentimentClassifier(nn.Module):
    def __init__(self, args, tokenizer):
        super(SentimentClassifier, self).__init__()
        self.args = args

        # === TEXT-ONLY MODE: skip ResNet ===
        # keep a linear just to define embedding size
        self.image_dim = 2048  # resnet50.fc.out_features originally
        self.linear = nn.Linear(self.image_dim, 768)

        self.bart = BartForSequenceClassification.from_pretrained(
            "facebook/bart-base", num_labels=3
        )
        self.bart.resize_token_embeddings(len(tokenizer))

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels,
        image_pixels,
        extended_attention_mask=[],
    ):
        # ====== TEXT-ONLY MODE ======
        batch_size = input_ids.size(0)
        device = input_ids.device

        # create dummy image embeddings instead of ResNet features
        image_embedding = torch.zeros(batch_size, 768, device=device)

        if extended_attention_mask == []:
            outputs = self.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                image_embedding=image_embedding,
                extended_attention_mask=extended_attention_mask,
            )
        else:
            extended_attention_mask = extended_attention_mask.reshape(
                -1, 1, self.args.MAX_LEN, self.args.MAX_LEN
            )
            outputs = self.bart(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                image_embedding=image_embedding,
                extended_attention_mask=extended_attention_mask,
            )

        return outputs
