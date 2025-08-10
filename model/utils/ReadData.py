# import torch.Tensor
# from src.utils.generate_bipartiteG import create_bipartite_graph
import torch


def get_sublist(a, b):
    index_0 = a.index(b[0])
    index_1 = a.index(b[1])

    # 确定起始和结束索引
    if index_0 < index_1:
        return a[index_0:index_1 + 1]
    else:
        return a[index_0:index_1 - 1:-1] + [b[1]]


def get_train_dataset_preprocess(data_lists, train_indexes, all_BiG21_list, all_BiG12_list):
    slide_num = len(train_indexes)

    # Filter training data
    u_lists = []
    for k in train_indexes:
        u_lists.append(data_lists[k])
    return u_lists


def get_dataset(root_path, slide_names, batch_num, device, mode):
    batch_data = []
    for b in range(batch_num):
        u_team = []
        slide_num = len(slide_names)
        print(f'===There are {slide_num} slides to train.===')
        for i in slide_names:
            u = torch.load(f'{root_path}/shuffled_{i}.pt').float().to(device)
            num1 = round(len(u) / batch_num)
            if b != batch_num - 1:
                u_ = u[b*num1:(b+1)*num1]
            else:
                u_ = u[b*num1:]
            u_team.append(u_)

        batch_data.append(u_team)
    print("Data Loaded")
    return batch_data


def get_group_dataset(root_path, grouped_slide_names, batch_num, device, mode):
    group_num = len(grouped_slide_names)
    batch_data = []
    for g in range(group_num):
        slide_names = grouped_slide_names[g]
        for b in range(batch_num):
            u_team = []
            slide_num = len(slide_names)
            print(f'===There are {slide_num} slides to train.===')
            for i in slide_names:
                u = torch.load(f'{root_path}/shuffled_{i}.pt').float().to(device)
                num1 = round(len(u) / batch_num)
                if b != batch_num - 1:
                    u_ = u[b*num1:(b+1)*num1]
                else:
                    u_ = u[b*num1:]
                u_team.append(u_)
            batch_data.append(u_team)
    print("Data Loaded")
    return batch_data