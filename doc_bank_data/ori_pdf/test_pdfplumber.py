#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 19:08:56 2020

@author: liang
"""
import pdfplumber
import pandas as pd

pdf = pdfplumber.open("./430027-北科光大-2017年年度报告.pdf")
# for p0 in pdf.pages:
#     print(p0)
p0 = pdf.pages[7]
for table in p0.extract_tables(): 
#得到的table是嵌套list类型，转化成DataFrame更加方便查看和分析 
    # for line in table:
    df = pd.DataFrame(table[1:], columns=table[0]) 
    # print(line)
    df.to_excel("./tmp/{}.xlsx".format(p0), index=None)