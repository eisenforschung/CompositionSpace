import numpy as np
import trimesh


def centroid_np(data_frame):
    length = len(data_frame["x"])
    sum_x = np.sum(data_frame["x"])
    sum_y = np.sum(data_frame["y"])
    sum_z = np.sum(data_frame["z"])
    return sum_x / length, sum_y / length, sum_z / length


def centroid_df(data_frame):
    length = len(data_frame["x"])
    sum_x = np.sum(data_frame["x"])
    sum_y = np.sum(data_frame["y"])
    sum_z = np.sum(data_frame["z"])
    return sum_x / length, sum_y / length, sum_z / length


def centroid_np_array(array):
    length = len(array[:, 0])
    sum_x = np.sum(array[:, 0])
    sum_y = np.sum(array[:, 1])
    sum_z = np.sum(array[:, 2])
    return sum_x / length, sum_y / length, sum_z / length


def correct_indices(array_wrong, reference):
    arr_w = array_wrong.flatten()
    for i in range(len(arr_w)):
        arr_w[i] = np.argwhere(reference == arr_w[i]).flatten()
    return arr_w.reshape(len(array_wrong), 3)


def magn_fac(d, x1, y1, z1, x2, y2, z2):
    A = (x2) ** 2
    B = (y2) ** 2
    C = (z2) ** 2
    t = np.sqrt((d**2) / (A + B + C))
    return t, -t


def cyl_endPoints(d, x1, x2, y1, y2, z1, z2):
    t1, t2 = magn_fac(d, x1, y1, z1, x2, y2, z2)

    U1 = x1 + (x2) * t1
    V1 = y1 + (y2) * t1
    W1 = z1 + (z2) * t1

    U2 = x1 + (x2) * t2
    V2 = y1 + (y2) * t2
    W2 = z1 + (z2) * t2

    return [U1, V1, W1, U2, V2, W2]


def plot_edit_mesh(nodes, simplices, simplices_edit, key):
    print("**Out mesh for plotting**")
    a = simplices[simplices_edit].flatten()
    ordered, index_or, inverse_or = np.unique(
        a,
        return_index=True,
        return_inverse=True,
    )
    ordered_corr = np.arange(len(ordered))
    sorted_id = a[np.sort(index_or)]
    sorted_id
    edited_a = np.zeros(len(a), dtype="int")
    for i in range(len(sorted_id)):
        for j in range(len(a)):
            if sorted_id[i] == a[j]:
                edited_a[j] = i

    nodes_selected = nodes[sorted_id]
    nodes_selected_edit_zeros = np.hstack(
        (nodes_selected, (np.zeros(len(nodes_selected)).reshape(-1, 1)))
    )
    mesh_tr = trimesh.Trimesh(
        vertices=nodes_selected_edit_zeros,
        faces=edited_a.reshape(simplices[simplices_edit].shape),
    )
    # mesh_tr.export(f'!!! CHANGEME !!!/mesh{key}.stl');
    return nodes_selected_edit_zeros, edited_a.reshape(simplices[simplices_edit].shape)


def plot_stl(file_path_prefix, nodes, simplices, key):
    mesh_tr = trimesh.Trimesh(vertices=nodes, faces=simplices)
    if file_path_prefix == "":
        mesh_tr.export(f"mesh{key}.stl")
    elif file_path_prefix.endswith("/"):
        mesh_tr.export(f"{file_path_prefix}mesh{key}.stl")
    else:
        mesh_tr.export(f"{file_path_prefix}/mesh{key}.stl")
