import json
import csv

"""
需求：将json中的数据转换成tsv文件
"""
def csv_json():
    # 1.分别 读，创建文件
    json_fp = open("data/train.json", "r",encoding='utf-8')
    tsv_fp = open("data/type_data_train.tsv", "w",encoding='utf-8')

    # 2.提出表头和表的内容
    data_list = json.load(json_fp)
    sheet_title = {"sentence", "label"}
    sentence = []
    label = []
    for idx, _ in enumerate(data_list):
        entry = data_list[idx]
        sentence.append(entry["sent"])
        label.append(entry["action"])

    # 3.csv 写入器
    writer = csv.writer(tsv_fp, delimiter='\t', lineterminator='\n')

    # 4.写入表头
    writer.writerow(sheet_title)

    # 5.写入内容
    for num in range(len(data_list)):
        writer.writerow([sentence[num], label[num]])

    # 6.关闭两个文件
    json_fp.close()
    tsv_fp.close()


if __name__ == "__main__":
    csv_json()

