# -*- coding: utf-8 -*-
# coding: unicode_escape
import csv
import random
import pandas as pd
from data_collections.settings import *

def save_sample(file):
    df=pd.read_csv(file)
    df=df.sample(frac=1.0)
    cut_idx=int(round(0.001*df.shape[0]))
    df_sample=df.iloc[:cut_idx]
    print(df_sample)
    '''
    print(df_sample.shape)
    sample_file='../data/ALW/sample.csv'
    save=pd.DataFrame(df_sample)
    save.to_csv(sample_file)
    '''
def read_sample(file):
    n=sum(1 for line in open(file,encoding='utf-8'))-1
    s=1000
    skip=sorted(random.sample(range(1,n+1),n-s))
    df=pd.read_csv(file,skiprows=skip)
    sample_file = stack_overflow_sample+'/sample.csv'
    df.to_csv(sample_file)

def analyze(file):

    total_number = 0

    n_NoLongerNeeded = 0
    n_Obsolete = 0
    n_NotConstructiveOrOffTopic = 0
    n_TooChatty = 0
    n_RudeOrOffensive = 0
    n_Unwelcoming = 0
    n_Other = 0
    n_NA = 0
    length = []
    df_chunk = pd.read_csv(file, chunksize=10000)
    for df in df_chunk:
        df_chunk_number = len(df)
        total_number += df_chunk_number
        print('-------------read number {}-----------------------'.format(total_number))
        for i in range(df_chunk_number):
            text = df.iloc[i]['Text']
            text = text.strip().split()
            length.append(len(text))

            flag = df.iloc[i]['Flag']

            flag = str(flag).strip().split()
            flag = ''.join(flag)
            flag = flag.lower()
            if flag == 'nan': n_NA += 1
            elif flag == 'commentnolongerneeded': n_NoLongerNeeded += 1
            elif flag == 'commentobsolete': n_Obsolete += 1
            elif flag == 'commentnotconstructiveorofftopic': n_NotConstructiveOrOffTopic += 1
            elif flag == 'commenttoochatty': n_TooChatty += 1
            elif flag == 'commentrudeoroffensive': n_RudeOrOffensive += 1
            elif flag == 'commentunwelcoming':
                n_Unwelcoming += 1
            else:
                n_Other += 1
        print('max length : {}'.format(max(length)))
        print('min length : {}'.format(min(length)))
        print('average length : {}'.format(sum(length)/total_number))
        print('n_NA : {}'.format(n_NA))
        print('n_NoLongerNeeded : {}'.format(n_NoLongerNeeded))
        print('n_Obsolete : {}'.format(n_Obsolete))
        print('n_NotConstructiveOrOffTopic : {}'.format(n_NotConstructiveOrOffTopic))
        print('n_TooChatty : {}'.format(n_TooChatty))
        print('n_RudeOrOffensive : {}'.format(n_RudeOrOffensive))
        print('n_Unwelcoming : {}'.format(n_Unwelcoming))
        print('n_Other : {}'.format(n_Other))
        print('------------------------------------------------------------------------------')

    max_len = max(length)
    min_len = min(length)
    avg_len = sum(length) / total_number

    print('total number : {}'.format(total_number))
    print('max length : {}'.format(max_len))
    print('min length : {}'.format(min_len))
    print('average length : {}'.format(avg_len))

    print('No flag : {}'.format(n_NA))
    print('Comment No Longer Needed : {}'.format(n_NoLongerNeeded))
    print('Comment Obsolete : {}'.format(n_Obsolete))
    print('Comment Not Constructive Or Off Topic : {}'.format(n_NotConstructiveOrOffTopic))
    print('Comment Too Chatty : {}'.format(n_TooChatty))
    print('Comment Rude Or Offensive : {}'.format(n_RudeOrOffensive))
    print('Comment Unwelcoming : {}'.format(n_Unwelcoming))
    print('Comment Other : {}'.format(n_Other))

def new_sample(file):
    df_chunk = pd.read_csv(file, sep=',',chunksize=10000)

    f1=stack_overflow_sample+'/f1.csv'
    f2=stack_overflow_sample+'/f2.csv'
    total_number=0
    count=0
    for df in df_chunk:
        df_chunk_number = len(df)
        total_number += df_chunk_number
        print('read number : {}'.format(total_number))

        f1_list=[]
        f2_list=[]

        for i in range(len(df)):

            commentDate=df.iloc[i]['CommentDate']
            text=df.iloc[i]['Text']
            flag=df.iloc[i]['Flag']

            flag = str(flag).strip().split()
            flag = ''.join(flag)

            commentDate=str(commentDate)
            text=str(text)

            if flag=='CommentNotConstructiveOrOffTopic' or flag=='CommentTooChatty' or flag=='CommentRudeOrOffensive' or flag=='CommentUnwelcoming':
                f1_list.append([commentDate,text,flag])
            else:
                f2_list.append([commentDate,text,flag])
        f1_data=pd.DataFrame(data=f1_list,columns=['CommentDate','Text','Flag'])
        f2_data=pd.DataFrame(data=f2_list,columns=['CommentDate','Text','Flag'])
        f1_data.to_csv(f1,index=False,mode='a',encoding='utf-8')
        f2_data.to_csv(f2,index=False,mode='a',encoding='utf-8')


def get_data(file):
    n_NotConstructiveOrOffTopic = 0
    n_TooChatty = 0
    n_RudeOrOffensive = 0
    n_Unwelcoming = 0
    f1=open(stack_overflow_sample+'/f1.csv','a',encoding='utf-8')
    f2 = open(stack_overflow_sample+'/f2.csv', 'a', encoding='utf-8')

    with open(file,'r',encoding='utf-8') as csvfile:
        data=csv.reader(csvfile,delimiter=',')

        for index,row in enumerate(data):
            if index%50000==0:
                print("read item: {}".format(index+1))
            flag=row[2]
            flag = str(flag).strip().split()
            flag = ''.join(flag)
            if flag=='CommentNotConstructiveOrOffTopic' or flag=='CommentTooChatty' or flag=='CommentRudeOrOffensive' or flag=='CommentUnwelcoming':
                f1_writer=csv.writer(f1)
                f1_writer.writerow(row)
                if flag=='CommentNotConstructiveOrOffTopic':n_NotConstructiveOrOffTopic+=1
                elif flag=='CommentTooChatty':n_TooChatty+=1
                elif flag=='CommentRudeOrOffensive':n_RudeOrOffensive+=1
                elif flag=='CommentUnwelcoming':n_Unwelcoming+=1
            else:
                f2_writer=csv.writer(f2)
                f2_writer.writerow(row)
        print('n_NotConstructiveOrOffTopic : {}'.format(n_NotConstructiveOrOffTopic))
        print('n_TooChatty : {}'.format(n_TooChatty))
        print('n_RudeOrOffensive : {}'.format(n_RudeOrOffensive))
        print('n_Unwelcoming : {}'.format(n_Unwelcoming))

def get_f2_sample():
    #f1 = open('../data/ALW/f1.csv', 'a', encoding='utf-8')
    f2=stack_overflow_sample+'/f2.csv'
    f1=stack_overflow_sample+'/f1.csv'
    len_f1=sum(1 for line in open(f1, encoding='utf-8',errors='ignore')) - 1
    print(len_f1)
    n = sum(1 for line in open(f2, encoding='utf-8')) - 1
    s = 8*len_f1
    skip = sorted(random.sample(range(1, n + 1), n - s))
    df = pd.read_csv(f2, skiprows=skip)
    sample_file = stack_overflow_sample+'/f2_new.csv'
    df.to_csv(sample_file,index=False,header=False)







if __name__=='__main__':
    stack_comments_file=stack_overflow+'/stack_comments.csv'
#     save_sample(stack_comments_file)
#     read_sample(stack_comments_file)
#     analyze(stack_comments_file)
#     new_sample(stack_comments_file)
    #get_data(stack_comments_file)
    #get_f2_sample()


