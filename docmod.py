# -*- coding: utf-8 -*-
"""
Created on Thu May  4 05:15:53 2023

@author: Shree
"""
from docx import Document
document = Document("Otology (1).docx")
lines = document.paragraphs
for run in lines:
    print(run.text)