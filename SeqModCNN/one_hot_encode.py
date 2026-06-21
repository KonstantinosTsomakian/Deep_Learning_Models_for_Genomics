def one_hot_encode_sequence_list(sequence_list):
    import numpy as np
    import torch
    


    encoded_sequence_list = []
    unique_nucleotides = sorted(set("".join(sequence_list)))
    nucleotides_to_index = {char: idx for idx, char in enumerate(unique_nucleotides)}
    num_chars = len(nucleotides_to_index)
   

    for sequence in sequence_list:
        one_hot = np.zeros((len(sequence), num_chars), dtype=int)

        for i, char in enumerate(sequence):
            if char in nucleotides_to_index:
                one_hot[i, nucleotides_to_index[char]] = 1
            else:
                raise ValueError(f"Character '{char}' not found in char_to_index.")
        encoded_sequence_list.append(one_hot.T)
    
    return torch.tensor(encoded_sequence_list).float()