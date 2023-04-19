import json,os,re
import pandas as pd


def write_json(obj):
    '''
    写入/追加json文件
    :param obj:
    :return:
    '''

    #首先读取已有的json文件中的内容并将新传入的dict对象追加至list中
    item_list = {}
    with open('data/test_lm.json', 'r') as f:
        load_dict = json.load(f)
        keys = list(load_dict.keys())
        i = 0
        for idx, _ in enumerate(load_dict):
            table_id = keys[idx]
            entry = load_dict[table_id]
            for e in entry:
                e.append(obj[i])
                i = i + 1
            item_list[table_id] = entry

    #将追加的内容与原有内容写回（覆盖）原文件
    with open('data/test_lm2.json', 'w', encoding='utf-8') as f:
        json.dump(item_list, f, indent=2)


source_file='data/prediction.csv'

if os.path.isfile(source_file):
	# 处理CSV文件，将文件里的内容取出来，并组装成一个列表
    data = pd.read_csv(os.path.join(source_file), sep=',')
    obj = []
    for idx in range(len(data)):
        item = data.iloc[idx]
        text = item.GeneratedText.split('. ')
        text = text[0:2]
        decom = '. '.join(text)
        if not decom.endswith('.'):
            decom += '.'
        obj.append(decom)
    write_json(obj)
