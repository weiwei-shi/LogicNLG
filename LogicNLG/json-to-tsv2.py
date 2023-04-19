import json
import csv

"""
需求：将json中的数据转换成tsv文件
"""
def csv_json():
    # 1.分别 读，创建文件
    json_fp = open("data/test_lm.json", "r",encoding='utf-8')
    # json_fp = open("data/test_lm_pos_neg.json", "r",encoding='utf-8')
    tsv_fp = open("data/decom_test.tsv", "w",encoding='utf-8')

    # 2.提出表头和表的内容
    data_list = json.load(json_fp)
    keys = list(data_list.keys())
    sheet_title = {"question"}
    sheet_data = []
    for idx, _ in enumerate(data_list):
        table_id = keys[idx]
        entry = data_list[table_id]
        for e in entry:
            sheet_data.append([e[0]])
            # sheet_data.append([e["pos"][0]])
            # sheet_data.append([e["neg"][0]])

    # 3.csv 写入器
    writer = csv.writer(tsv_fp, delimiter='\t', lineterminator='\n')

    # 4.写入表头
    writer.writerow(sheet_title)

    # 5.写入内容
    # print("{}\n".format(sheet_data))
    writer.writerows(sheet_data)

    # 6.关闭两个文件
    json_fp.close()
    tsv_fp.close()


if __name__ == "__main__":
    csv_json()
