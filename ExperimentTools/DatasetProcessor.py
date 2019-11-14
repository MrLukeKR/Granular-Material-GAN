import random, math


def dataset_to_train_and_test(dataset, split_percent):
    shuffled = random.shuffle(dataset)
    train_files = list()
    test_files = list()

    for data in range(len(dataset)):
        if data / len(dataset) < split_percent:
            train_files += shuffled[data]
        else:
            test_files += shuffled[data]

    return train_files, test_files


def chunks(dataset, n):
    dataset_chunks = list()

    jump = math.floor(len(dataset) / n)

    for i in range(n):
        ind = i * jump
        dataset_chunks.append(dataset[ind:ind + jump])

    return dataset_chunks


def dataset_to_k_cross_fold(dataset, k):
    random.shuffle(dataset)
    groups = chunks(dataset, k)
    train_files = list()
    test_files = groups

    for group in groups:
        train_files.append(list(g for g in groups if g is not group))

    return train_files, test_files
