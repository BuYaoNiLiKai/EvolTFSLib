from data_provider.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, )
if __name__ == '__main__':
    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'custom': Dataset_Custom,
    }
    data = 'ETTm2'
    root_path = '../dataset/'
    data_path = 'ETTm2.csv'
    Data = data_dict[data]
    print(data_dict[data])
    timeenc = 0
    flag = 'train'
    shuffle_flag = False
    drop_last = False
    seq_len = 20
    label_len = 5
    pred_len = 10
    freq = 't'
    features = 'M'
    target = 'OT'
    cycle = 24
    scale =False
    fb=True
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
        fb=fb
    )
    print(data_set)
    print(len(data_set))
    for i in data_set[11000*3]:
        # print(i)
        print(i)
        print("___________________________")



