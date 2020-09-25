#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:28:50 2020

@author: liang
"""
import os
from tqdm import tqdm
import pdfplumber
from tqdm import tqdm
from pdfminer.layout import LTChar, LTLine
import pandas as pd
from collections import Counter
import re

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes,
        actual_bboxes,
        file_name,
        page_size,
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f, open(
        box_file_path, encoding="utf-8"
    ) as fb, open(image_file_path, encoding="utf-8") as fi:
        words = []
        boxes = []
        actual_bboxes = []
        file_name = None
        page_size = None
        labels = []
        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                            boxes=boxes,
                            actual_bboxes=actual_bboxes,
                            file_name=file_name,
                            page_size=page_size,
                        )
                    )
                    guid_index += 1
                    words = []
                    boxes = []
                    actual_bboxes = []
                    file_name = None
                    page_size = None
                    labels = []
            else:
                splits = line.split("\t")
                bsplits = bline.split("\t")
                isplits = iline.split("\t")
                # print(isplits)
                # print(len(isplits))
                assert len(splits) == 2
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0]
                words.append(splits[0])
                if len(splits) >= 1:
                    labels.append("O")
                    box = bsplits[-1].replace("\n", "")
                    box = [int(b) for b in box.split()]
                    boxes.append(box)
                    actual_bbox = [int(b) for b in isplits[1].split()]
                    actual_bboxes.append(actual_bbox)
                    page_size = [int(i) for i in isplits[2].split()]
                    file_name = isplits[3].strip()
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )
    return examples



def trans2examples(excel_data, excel_name):
    
    def bbox_trans_func(box, width, length, num=1000):
        return [
            int(num * (box[0] / width)),
            int(num * (box[1] / length)),
            int(num * (box[2] / width)),
            int(num * (box[3] / length)),
        ]

    examples = []
    
    for sheet_id in excel_data.sheets:
        df = pd.read_excel(excel_data, sheet_name=str(sheet_id))
        words = df['token']
        boxes = []
        actual_bboxes = []
        labels = ["O"] * len(df['token'])
        
        file_name = excel_name + "_{}".format(str(sheet_id))
        page_size = [df['width'],df['height']]        
        
        for index,item in df.iterows():
            actual_bbox = [item['x0'],item['y0'],item['x1'],item['y1']]
            box = bbox_trans_func(actual_bbox,item['width'], item['height'])
            
            boxes.append(box)
            actual_bboxes.append(actual_bbox)  
            
        assert len(words) == len(boxes)
        assert len(words) == len(actual_bboxes)
        assert len(words) == len(labels)
        
        examples.append(
            InputExample(
                guid="%s-%d".format("pred", sheet_id+1),
                words=words,
                labels=labels,
                boxes=boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
            )
        )
    return examples

def convert_examples_to_features(
                                    examples,
                                    label_list,
                                    max_seq_length,
                                    tokenizer,
                                    cls_token_at_end=False,
                                    cls_token="[CLS]",
                                    cls_token_segment_id=1,
                                    sep_token="[SEP]",
                                    sep_token_extra=False,
                                    pad_on_left=False,
                                    pad_token=0,
                                    cls_token_box=[0, 0, 0, 0],
                                    sep_token_box=[1000, 1000, 1000, 1000],
                                    pad_token_box=[0, 0, 0, 0],
                                    pad_token_segment_id=0,
                                    pad_token_label_id=-1,
                                    sequence_a_segment_id=0,
                                    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size

        tokens = []
        token_boxes = []
        actual_bboxes = []
        label_ids = []
        for word, label, box, actual_bbox in zip(
            example.words, example.labels, example.boxes, example.actual_bboxes
        ):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend(
                [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1)
            )

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            actual_bboxes = actual_bboxes[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            actual_bboxes = [[0, 0, width, height]] + actual_bboxes
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            token_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length


        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
            )
        )
    return features

def bbox_string(box, width, length, num=1000):
    return (
        str(int(num * (box[0] / width)))
        + " "
        + str(int(num * (box[1] / length)))
        + " "
        + str(int(num * (box[2] / width)))
        + " "
        + str(int(num * (box[3] / length)))
    )

def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )

def within_bbox(bbox_bound, bbox_in):
    assert bbox_bound[0] <= bbox_bound[2]
    assert bbox_bound[1] <= bbox_bound[3]
    assert bbox_in[0] <= bbox_in[2]
    assert bbox_in[1] <= bbox_in[3]

    x_left = max(bbox_bound[0], bbox_in[0])
    y_top = max(bbox_bound[1], bbox_in[1])
    x_right = min(bbox_bound[2], bbox_in[2])
    y_bottom = min(bbox_bound[3], bbox_in[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_in_area = (bbox_in[2] - bbox_in[0]) * (bbox_in[3] - bbox_in[1])

    if bbox_in_area == 0:
        return False

    iou = intersection_area / float(bbox_in_area)

    return iou > 0.95

page_seg = "############"
def parsepdf4predict(pdf_file):
    # pdf_files = list(os.listdir(data_dir))#[:10]
    # pdf_files = [t for t in pdf_files if t.endswith('.pdf')]
        
    xlsx = pd.ExcelWriter(pdf_file.replace('.pdf','.xlsx'))
    
    pdf = pdfplumber.open(pdf_file)

    for page_id in tqdm(range(len(pdf.pages))):

        this_page = pdf.pages[page_id]
        token_array = []
        x0_array = []
        y0_array = []
        x1_array = []
        y1_array = []
        w_array = []
        h_array = []
    
        words = this_page.extract_words(x_tolerance=1.5)
    
        lines = []
        for obj in this_page.layout._objs:
            if not isinstance(obj, LTLine):
                continue
            lines.append(obj)
    
        for word in words:
            word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
            objs = []
            for obj in this_page.layout._objs:
                if not isinstance(obj, LTChar):
                    continue
                obj_bbox = (obj.bbox[0], float(this_page.height) - obj.bbox[3],
                            obj.bbox[2], float(this_page.height) - obj.bbox[1])
                if within_bbox(word_bbox, obj_bbox):
                    objs.append(obj)
            fontname = []
            for obj in objs:
                fontname.append(obj.fontname)
            if len(fontname) != 0:
                c = Counter(fontname)
                fontname, _ = c.most_common(1)[0]
            else:
                fontname = 'default'
    
            # format word_bbox
            width = int(this_page.width)
            height = int(this_page.height)
            x0 = word_bbox[0]
            y0 = word_bbox[1]
            x1 = word_bbox[2]
            y1 = word_bbox[3]
    
            word_text = re.sub(r"\s+", "", word['text'])
    
            token_array.append(word_text)
            x0_array.append(x0)
            y0_array.append(y0)
            x1_array.append(x1)
            y1_array.append(y1)
            w_array.append(width)
            h_array.append(height)
            
        union_array = list(zip(token_array,x0_array,y0_array,x1_array,y1_array,w_array,h_array))     
        pagedf = pd.DataFrame(union_array,columns=['token','x0','y0','x1','y1','width' , 'height'])
        
        pagedf.to_excel(xlsx, sheet_name=str(page_id), index=False)
            
           
    return xlsx
        
        
def write2txt(output_dir,txtdata):
    #############write2txt
    token_array,x0_array,y0_array,x1_array,y1_array,w_array,h_array = txtdata[0],txtdata[1],txtdata[2],txtdata[3],txtdata[4],txtdata[5],txtdata[6]
    data_name = 'test'
    with open(
        os.path.join(output_dir, data_name + ".txt"),
        "w",
        encoding="utf8",
    ) as fw, open(
        os.path.join(output_dir, data_name + "_box.txt"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(output_dir, data_name + "_image.txt"),
        "w",
        encoding="utf8",
    ) as fiw:
        
        for token,x0,y0,x1,y1,w,h in zip(token_array,x0_array,y0_array,x1_array,y1_array,w_array,h_array):
            if token == page_seg:
                fw.write('\n')
                fbw.write('\n')
                fiw.write('\n')
            else:
                fw.write(token + '\n')
                fbw.write(token + '\t'+ bbox_string([x0,y0,x1,y1], w, h) + '\n')
                fiw.write(token + '\t'+ actual_bbox_string([x0,y0,x1,y1], w, h)  + '\n')