'''
dumps：将python中的 字典 转换为 字符串
'''
import json

# templateP = {
#
#     1: {"Num":"P0301204024001", "label":'bolt', "CenterX":100, "CenterY":150, "Width":20, "Height":23},
#     2: {"Num":"P0301204024002", "label":'bolt', "CenterX":130, "CenterY":120, "Width":10, "Height":15},
#     3: {"Num": "P0301204024003", "label": 'oil', "CenterX": 253, "CenterY": 363, "Width": 40, "Height": 45},
# }
#
# print(templateP[2]['label'])

test_dict = {'bigberg': [7600, {1: [['iPhone', 6300], ['Bike', 800], ['shirt', 300]]}]}
# print(test_dict)
# print('---------------')
# print(test_dict['bigberg'])
# print('---------------')
# print(test_dict['bigberg'][1])
# print('---------------')
# print(test_dict['bigberg'][1][1])
# print(type(test_dict))
# dumps 将数据转换成字符串
json_str = json.dumps(test_dict)
print(json_str)
print(type(json_str))

'''
loads: 将 字符串 转换为 字典
'''
new_dict = json.loads(json_str)
print(new_dict)
print(type(new_dict))

'''
dump: 将数据写入json文件中
'''
with open("data/jason/record.json","w") as f:
    json.dump(new_dict,f)
    print("加载入文件完成...")


'''
load:把文件打开，并把字符串变换为数据类型
'''
with open("data/jason/record.json",'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict)
load_dict['smallberg'] = [8200,{1:[['Python',81],['shirt',300]]}]
print(load_dict)

with open("data/jason/record.json","w") as dump_f:
    json.dump(load_dict,dump_f)