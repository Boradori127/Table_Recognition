import pandas as pd
import os

DATAPATH = "/SSL_NAS/OKS_2024/Ko_data"
DATANAME = ["A 음식점(15,726)_new.xlsx", "B 의류(15,826)_new.xlsx", 'C 학원(4,773)_new.xlsx']



'''
[df_columns]
Index(['SPEAKER', 'SENTENCE', 'DOMAINID', 'DOMAIN', 'CATEGORY', 'SPEAKERID',
       'SENTENCEID', 'MAIN', 'SUB', 'QA', 'QACNCT', 'MQ', 'SQ', 'UA', 'SA',
       '개체명', '용어사전', '지식베이스'],
      dtype='object')

[len(dataset)]
A : Total 15726, Question 8757
B : Total 15826, Question 8656
C : Total 4773, Question 2922

[Domain]
A(음식점) : 음식점
B(의복) : 의복의류점
C(학원) : 학원
'''

def count_unique_values(df):
    unique_values = {}
    for column in df.columns:
        unique_counts = df[column].value_counts().to_dict()
        unique_values[column] = unique_counts
    return unique_values


def count_values_containing_word(df, column_name, words):
    # 특정 컬럼의 값 중에서 특정 단어가 포함된 값의 개수를 셈
    for word in words:
        count = df[column_name].str.contains(word, na=False).sum()
        print(f"{word} row : {count}")
    return count


def filter_rows_without_word(df, column_name, words):
    # 특정 단어가 포함되지 않은 행들을 필터링
    for word in words:
        df = df[~df[column_name].str.contains(word, na=False)]
    return df


def dataset_EDA(filename):
    # QA에서 "Q"만 추출하기로
    # 
    df = pd.read_excel(filename)
    Question_df = df[df['QA'] == 'Q']
    domain_df = Question_df["DOMAIN"]
    intent_df = Question_df[['DOMAIN', 'MAIN']]

    unique_values_counts = count_unique_values(intent_df)

    words = ["문의", "요청", "요구", "질문", "선택"]
    # 결과 출력

    for column, value_counts in unique_values_counts.items():
        print(f"Column: {column}")
        for value, count in value_counts.items():
            print(f"    Value: {value}, Count: {count}")

    # count_values_containing_word(intent_df, "MAIN", words)
    # another_df = filter_rows_without_word(intent_df, "MAIN", words)
    # print(len(another_df))
    # print(another_df.head())


    # QA에서 "Q"만 추출하기로
    # domain + main -> intent
    # 지식 베이스에서 / 뒷 부분이 => slot으로 생각해서 eda 할거같애
    # 문의, 요청, 

    pass

def main():
    for data in DATANAME:
        filename = os.path.join(DATAPATH, data)
        print(filename)
        dataset_EDA(filename)

    pass

if __name__ == "__main__":
    main()