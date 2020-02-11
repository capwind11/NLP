# -*- coding: UTF-8 -*-
from collections import Counter
import jieba
import os
import string
import re

source_raw = open('news-commentary-v13.zh-en.zh', 'r', encoding='utf-8')
target_raw = open('news-commentary-v13.zh-en.en', 'r', encoding='utf-8')
total = set()

def cut_word(file,outfile):
    content = file.readlines()
    f = open(outfile,'w',encoding='utf-8')
    for i in content[:10000]:
        words = list(jieba.cut(i.strip()))
        for j in range(words.count(' ')):
            words.remove(' ')
        f.write(' '.join(words)+'\n')
    f.close()

cut_word(target_raw,'target.txt')
cut_word(source_raw,'source.txt')
