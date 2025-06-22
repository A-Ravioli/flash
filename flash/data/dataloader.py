"""
Data loader implementation.
"""

import random
import math
class DataLoader:
    """
    Data loader class that provides batches of data.
    
    Attributes:
        dataset: Dataset to load from
        batch_size: Number of samples in each batch
        shuffle: Whether to shuffle the data
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Initialize a DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples in each batch
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        """
        Create an iterator for batches.
        
        Returns:
            Iterator that yields batches of data
        """
        # TODO: Create indices for all samples
        indices = list(range(len(self.dataset)))
        # TODO: Shuffle indices if needed
        if self.shuffle:
            random.shuffle(indices)
        # TODO: Return iterator that yields batches
        for i in range(0, len(indices), self.batch_size):
            yield self.dataset[indices[i:i+self.batch_size]]
        
    def __len__(self):
        """
        Get the number of batches in the data loader.
        
        Returns:
            Number of batches
        """
        # TODO: Calculate and return the number of batches 
        return math.ceil(len(self.dataset) / self.batch_size)