#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 09:50:28 2020

@author: liang
"""
import os
import json
from layoutlm import FunsdDataset, LayoutlmForTokenClassification,LayoutlmConfig
from tqdm import tqdm
from utils import Namespace
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np

import torch
from torch.utils.data import Dataset, SequentialSampler
from utils_fea import trans2examples, convert_examples_to_features, parsepdf4predict, write2txt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"                                                                             
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class pdfPredDataset(Dataset):
    def __init__(self, args, tokenizer, examples):

        features = convert_examples_to_features(
            examples,
            args.labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=False,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=args.pad_token_label_id,
        )

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
        )


class TablePredictHandler():
    def __init__(self, model_save_path):
        
        self.modelconfig = json.load(open(os.path.join(model_save_path, 'layoutLM_config.json'), 'r', encoding='utf-8'))
        self.config = Namespace(**self.modelconfig)
        # super(TablePredictHandler, self).__init__(self.config)
        self.model_save_path = model_save_path
        
        self.pretrainmodelpath = self.config.pretrainmodelpath
        self.loadmodel(model_save_path)
        
    def loadmodel(self,model_save_path):
        
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainmodelpath, do_lower_case=True)
        
        self.model_config = LayoutlmConfig.from_pretrained(os.path.join(model_save_path,'layoutLM_config.json'),
                                                            num_labels=self.config.num_labels,
                                                        )
        self.model = LayoutlmForTokenClassification.from_pretrained(model_save_path,config=self.model_config)
        
        
    def predict(self, pdf_file):
        
        #step1 parse pdf data for predict
        xlsxdf = parsepdf4predict(pdf_file)
        
        # write2txt(self.model_save_path, txtdata)
        pred_examples = trans2examples(xlsxdf)
        pred_dataset = pdfPredDataset(self.config, self.tokenizer, pred_examples)
        pred_dataloader = DataLoader(
                                        pred_dataset,
                                        batch_size=1,
                                    )
        
        print("***** Running evaluation %s *****", 'test')
        print("  Num examples = %d", len(pred_dataset))
        print("  Batch size = %d", 1)

        preds = None
        # device = torch.device('cuda')
        # self.model.to(device)
        # self.model.eval()
        for batch in tqdm(pred_dataloader, desc="Predict"):
            # print(batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0], #.to(args.device) .to('cuda:0')
                    "attention_mask": batch[1],
                    "token_type_ids" : batch[2],
                    "bbox": batch[4]
                }
                # dict_keys(['input_ids', 'attention_mask', 'labels', 'bbox', 'token_type_ids'])
                outputs = self.model(**inputs)
                logits = outputs[0]
                
            if self.config.decoder == 'crf':
                logits = logits.sequeeze(0)
    
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                
        if self.config.decoder == 'softmax':
            preds = np.argmax(preds, axis=2)
        elif self.config.decoder == 'crf':
            preds = preds.tolist()
            
        label_map = {i: label for i, label in enumerate(self.config.labels)}
    
        preds_list = [[] for _ in range(preds.shape[0])]
    
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                if preds[i, j] != self.config.pad_token_label_id:
                    preds_list[i].append(label_map[preds[i][j]])

        return xlsxdf, preds_list



if __name__ == '__main__':
    madel_save_path = "./model_funsd"
    data_dir = "./data"
    ta_extractor = TablePredictHandler(madel_save_path)
    xlsxdf, preds_list = ta_extractor.predict(data_dir)
    
    
    print(xlsxdf)
