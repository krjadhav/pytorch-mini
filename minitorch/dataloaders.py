import numpy as np

class DataLoader:
    """
    This takes in data, target and shuffles them
    Using a generator to yield batches of data
    Shuffling is done with numpy random shuffle on the indexes
    Assuming the number of data points is equal to the number of target points
    Also assuming the number of data points don't go crazy large
    """
    def __init__(self, data, target, batch_size, shuffle=False):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(data)) # Creates a seperate list of indexes from 0 to len(data)

    def __iter__(self):
        data_size = len(self.data)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        for i in range(0, data_size, self.batch_size):
            batch_idxs = self.indexes[i: i + self.batch_size]
            yield self.data[batch_idxs], self.target[batch_idxs]

    def __len__(self):
        """
        Returns the number of batches in the dataset
        """
        return int(np.ceil(len(self.data) / self.batch_size))