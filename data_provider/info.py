from data_provider.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, )
if __name__ == '__main__':
    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'custom': Dataset_Custom,
    }
    data = 'ETTh1'
    root_path = '../dataset/'
    data_path = 'ETTh1.csv'
    Data = data_dict[data]
    print(data_dict[data])
    timeenc = 0
    flag = 'train'
    shuffle_flag = False
    drop_last = False
    seq_len = 96
    label_len = 48
    pred_len = 96
    freq = 'h'
    features = 'M'
    target = 'OT'
    cycle = 24
    scale =False
    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq,
        cycle=cycle,
        scale=scale,
    )
    print(data_set)
    print(len(data_set))
    print(data_set[0][0])
    print(data_set[0][1])
    print(data_set[0][2])
    print(data_set[0][3])
    print(data_set[0][4])


