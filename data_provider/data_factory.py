
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    shuffle_flag = False if (flag == 'test') else True
    batch_size = args.batch_size
    drop_last = False
    
    data_set = Data(
        args = args,
        flag=flag
    )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    return data_set, data_loader
