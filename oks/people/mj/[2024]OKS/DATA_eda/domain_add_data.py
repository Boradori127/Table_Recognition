import os
import json

def data_load(file_name):
    dir_path = "/home/oks/oks/data/intent_oks/korean/total_data"
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def add_domain(data):
    new_data = []
    for item in data:
        intent = item['intent']
        if "날짜_및_시간" in intent:    
            domain = "날짜_및_시간"
            topic = intent.split("_")[-1]
            new_intent = intent
        elif "택시" in intent:
            domain = "교통"
            new_intent = "교통_택시"
            topic = "택시"
        elif "기타_설정" in intent:
            domain = "오디오_볼륨"
            topic = "기타_설정"
            new_intent = intent
        elif "목록_생성_또는_추가" in intent:
            domain = "목록"
            topic = "생성_또는_추가"
            new_intent = intent
        elif "IoT" in intent:
            domain = "IoT"
            topic = intent.replace("IoT_", "")
            new_intent = intent
        elif "_" in intent:
            domain = intent.split("_")[0]
            new_intent = intent
            topic = "_".join(intent.split("_")[1:])
        else:
            domain = "민원"
            topic = intent
            new_intent = "민원_" + intent
            
        new_item = {
            'text': item['text'],
            'domain': domain,
            'topic' : topic,
            'intent': new_intent,
            'slot_out': item['slot_out']
        }
        new_data.append(new_item)
    return new_data

def save_data(data, file_name):
    dir_path = "/home/oks/oks/people/mj/[2024]OKS/DATA_eda/no_use/total_data"
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
def new_save_data_with_domain(data_list):
    for i in data_list:
        origin_data = data_load(i)
        new_data = add_domain(origin_data)
        save_data(new_data, i.split(".")[0]+"_with_domain_topic.json")
        print("save " + i.split(".")[0]+"_with_domain_topic.json")
        
data_list = ['kor_amazon_mixed_train.json', 'kor_amazon_mixed_val.json', 'kor_amazon_mixed_test.json']        
new_save_data_with_domain(data_list)


# data = data_load('kor_amazon_mixed_val_with_domain.json')
# print(data)