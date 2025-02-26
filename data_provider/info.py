from torch.utils.data import DataLoader

from data_provider.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, )
if __name__ == '__main__':
    data_dict = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'custom': Dataset_Custom,
    }
    data = 'ETTh2'
    root_path = '../dataset/'
    data_path = 'ETTh2.csv'
    Data = data_dict[data]
    print(data_dict[data])
    timeenc = 0
    flag = 'train'
    shuffle_flag = False
    drop_last = False
    seq_len = 96
    label_len = 48
    pred_len = 192
    freq = 'h'
    features = 'M'
    target = 'OT'
    cycle = 24
    scale =False
    station_type = 'fb'
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
        station_type=station_type,
    )
    batch_size = 32
    num_workers = 0

    print(len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    print(len(data_loader))
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,_) in enumerate(data_loader):
        print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
        break


