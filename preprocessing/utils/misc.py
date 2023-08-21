def remove_indices(input_list, indices_to_remove):
    new_list = []
    for index, item in enumerate(input_list):
        if index not in indices_to_remove:
            new_list.append(item)
    return new_list
