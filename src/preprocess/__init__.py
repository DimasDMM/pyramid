import json

def sanity_check(dataset):
    n_bad = 0
    for i, item in enumerate(dataset):
        length = len(item['tokens'])
        for entity in item['entities']:
            if entity['span'][1] <= length:
                pass
            else:
                n_bad += 1
                break
    return n_bad

def split_dataset(dataset, train_ratio=0.81, dev_ratio=0.09):
    """
    By default, splits first 81%, subsequent 9%, and last 10% as train,
    dev and test set, respectively.
    """

    train_idx = [0, int(train_ratio * len(dataset))]
    dev_idx = [train_idx[1], train_idx[1] + int(dev_ratio * len(dataset))]
    test_idx = [dev_idx[1], len(dataset)]

    train_dataset = dataset[train_idx[0]:train_idx[1]]
    dev_dataset = dataset[dev_idx[0]:dev_idx[1]]
    test_dataset = dataset[test_idx[0]:test_idx[1]]

    return train_dataset, dev_dataset, test_dataset

def store_datasets(dataset_name, train_dataset, dev_dataset, test_dataset):
    train_file = './data/train.%s.json' % dataset_name
    valid_file = './data/valid.%s.json' % dataset_name
    test_file = './data/test.%s.json' % dataset_name

    with open(train_file, 'w') as fp:
        json.dump(train_dataset, fp)
    with open(valid_file, 'w') as fp:
        json.dump(dev_dataset, fp)
    with open(test_file, 'w') as fp:
        json.dump(test_dataset, fp)
