import pandas as pd
import glob
import os
import argparse

'''Please run under ./preprocess/PHOENIX14T'''

def process_each_file(save_root,file_path):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    save_path_1 = os.path.join(save_root,file_path.split('.')[-3]+'.files')
    save_path_2 = os.path.join(save_root,file_path.split('.')[-3]+'.gloss')
    save_path_3 = os.path.join(save_root,file_path.split('.')[-3]+'.text')
    
    df = pd.read_csv(file_path,sep='|',skiprows=1,header=None)
    df.iloc[:,0].to_csv(save_path_1,index=False,header=False)
    df.iloc[:,-2].to_csv(save_path_2,index=False,header=False)
    df.iloc[:,-1].to_csv(save_path_3,index=False,header=False)
   
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser('preprocesss files')

    save_root = parser.add_argument('--save_root',default='../../Data/PHOENIX14T')
    
    args = parser.parse_args()
        
    files = glob.glob('*.csv')

    for file in files:
        process_each_file(args.save_root,file)
    
    print('finished!!!')
