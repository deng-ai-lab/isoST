import numpy as np
import pandas as pd

def get_dataset(root_path, slide_names):
    team = []
    slide_num = len(slide_names)

    print(f'===There are {slide_num} slides to train.===')
    for i in range(slide_num):
        u = pd.read_csv(f'{root_path}{slide_names[i]}.csv', index_col=0).values
        team.append(u)
    print("Data Loaded")
    return team
