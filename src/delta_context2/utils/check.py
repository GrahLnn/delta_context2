import json
import os

def is_done(item):
    file_path = 'data/core/record.json'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return False
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 验证done_list字段是否存在
    if 'done_list' not in data:
        raise KeyError("'done_list'字段在JSON文件中不存在")
    
    # 检查传入参数是否在done_list中
    return item in data['done_list']

def add_to_done_list(item):
    file_path = 'data/core/record.json'
    
    # 检查目录是否存在，不存在则创建
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 初始化数据结构
    data = {}
    
    # 如果文件存在，读取文件内容
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    
    # 验证done_list字段是否存在，不存在则初始化
    if 'done_list' not in data:
        data['done_list'] = []
    
    # 将新项目添加到done_list中
    if item not in data['done_list']:
        data['done_list'].append(item)
    
    # 写入数据到文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)