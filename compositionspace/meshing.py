import os
from functools import reduce

import flatdict as fd
import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from tqdm import tqdm

from compositionspace.geometryutils import (
    centroid_np,
    correct_indices,
    cyl_endPoints,
    plot_edit_mesh,
    plot_stl,
)


def meshes_normals(config_file_path, input_file_path, output_file_path):
    """TODO improve docstring."""
    config = {}
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as yml:
            config = fd.FlatDict(yaml.safe_load(yml), delimiter="/")

    print("This function needs inspection and a clean example how to use it")
    return

    # TODO this needs to be changed to load from the NeXus file directly
    Clusters = {}
    with h5py.File(input_file_path, "r") as hdfr:
        group1 = hdfr.get("1")
        No_clusters = list(group1.keys())  # list(list(group1.attrs.values())[1])
        for i in No_clusters:
            Clusters[i] = np.array(group1.get(i))

    cluster_combine_lst = []
    for i in Clusters.keys():
        print(i)
        image = Clusters[i][:, 0:3]

        df = pd.DataFrame(image, columns=("x", "y", "z"))
        df["ID"] = [int(i)] * len(image)
        cluster_combine_lst.append(df)
    # end of change

    # TODO this needs to be changed to load grid data from NeXus file
    # and other parameter from yaml config file like shown in other *.py files
    Node_list_plot = []
    no_elements = config["voxelization/edge_length"]
    dist_cut = config["meshing/dist_cut"]
    normal_end_length = config["meshing/normal_end_length"]
    nodes_edit_lst = []
    simp_plot_edit_lst = []

    keys = [1, 2, 3]  # 0,7

    with h5py.File(output_file_path, "w") as hdfw:
        G_normal_avg_vec = hdfw.create_group("normal_vec")
        G_nodes_edit_z = hdfw.create_group("nodes")
        G_simplices = hdfw.create_group("simplices")

        G_normal_ends = hdfw.create_group("normal_ends")
        for key in keys:
            print("key = ", key)
            ## load the Precipitate cantroids
            # gr = hdfr.get("{}".format(key))
            # vox_ratio_preci_col = list(list(gr.attrs.values())[3])
            # Preci_cent_col = list(list(gr.attrs.values())[2])
            # cent = np.array(gr.get("0"))
            # print("Centroids = ", len(cent))
            Df_cent = cluster_combine_lst[key]

            ## PCA on centroid values
            Data = Df_cent.drop(["ID"], axis=1)
            pca = PCA(n_components=3)
            fit_pca = pca.fit(Data.values)
            X_pca = pca.transform(Data.values)

            ## Define the boundaries of your PCAcentroid data
            # nodes = np.array([[max(X_pca[:, 0]),max(X_pca[:, 1])], [max(X_pca[:, 0]),min(X_pca[:, 1])],
            #         [min(X_pca[:, 0]),max(X_pca[:, 1])] , [min(X_pca[:, 0]),min(X_pca[:, 1])]])

            ## Set a 2D regular grid on the PCA centroids
            # no_elements = 50 # defined above

            nodes = []
            for i in np.arange(min(X_pca[:, 0]), max(X_pca[:, 0]), no_elements):
                for j in np.arange(min(X_pca[:, 1]), max(X_pca[:, 1]), no_elements):
                    nodes.append([i, j])
            nodes = np.asarray(nodes)

            print(nodes[0, 1] - nodes[1, 1])
            print((min(X_pca[:, 0]) - max(X_pca[:, 0])) / no_elements)
            print((min(X_pca[:, 1]) - max(X_pca[:, 1])) / no_elements)

            ##Triangulate the grid
            tri = Delaunay(nodes)

            ## Remove the empty grid triangles
            atoms_xy = np.delete(X_pca, 2, 1)
            P = atoms_xy
            check = []
            for i in tqdm(range(len(tri.simplices))):
                # print(i)
                vertices = nodes[tri.simplices[i]]
                A = vertices[0]
                B = vertices[1]
                C = vertices[2]

                AB = B - A
                BC = C - B
                CA = A - C
                AP = P - A
                BP = P - B
                CP = P - C

                a = np.cross(AB, AP)
                b = np.cross(BC, BP)
                c = np.cross(CA, CP)

                a1 = np.where(a < 0, -2, 1)
                # afinal= np.where(a1 >=0, 1, a1)

                b1 = np.where(b < 0, -2, 1)
                # bfinal= np.where(b1 >=0, 1, b1)

                c1 = np.where(c < 0, -2, 1)
                # cfinal= np.where(c1 >=0, 1, c1)

                product = a1 * b1 * c1

                if len(np.argwhere(product == 1)) != 0:
                    check.append(i)
                    # print("1")
                elif len(np.argwhere(product == -8)) != 0:
                    check.append(i)
                    # print("-8")

            # print(tri.simplices)
            # G_normal_2d.create_dataset("{}".format(int(key)), data = [nodes, tri.simplices, check])

            nodes = nodes
            simplices = tri.simplices
            simplices_edit = check

            plot2 = nodes[np.unique(simplices[simplices_edit].flatten())]
            nodes_edit = np.concatenate(
                (plot2, np.atleast_2d(np.zeros((1, len(plot2)))).T), axis=1
            )

            nodes_edit, simp_plot_edit = plot_edit_mesh(
                nodes, simplices, simplices_edit, key
            )
            plot2 = nodes_edit[:, 0:2]
            ## Define the depth at each Node:

            dist_cut = dist_cut

            for i in tqdm(range(len(nodes_edit))):
                center = plot2[i]
                shift_org_x = X_pca[:, 0] - plot2[i][0]
                shift_org_y = X_pca[:, 1] - plot2[i][1]

                a = np.argwhere(shift_org_x < dist_cut)
                b = np.argwhere(shift_org_y < dist_cut)
                c = np.argwhere(shift_org_x > -dist_cut)
                d = np.argwhere(shift_org_y > -dist_cut)

                k = reduce(np.intersect1d, (a, b, c, d))
                if len(k) == 0:
                    continue
                slice_pts = np.take(X_pca, k, axis=0)
                centroid = centroid_np(pd.DataFrame(slice_pts, columns=["x", "y", "z"]))
                nodes_edit[i] = centroid

            nodes_edit = nodes_edit
            ## Take PCA inverseof the edited nodes
            nodes_edit_Df = pd.DataFrame(data=nodes_edit, columns=["x", "y", "z"])
            nodes_inv = pd.DataFrame(
                data=pca.inverse_transform(nodes_edit_Df.values),
                columns=["x", "y", "z"],
            )
            nodes_edit = nodes_inv.values
            Node_list_plot.append(nodes_edit)
            print("key = **************", key)

            G_nodes_edit_z.create_dataset("{}".format(int(key)), data=nodes_edit)
            G_simplices.create_dataset("{}".format(int(key)), data=simp_plot_edit)
            plot_stl("", nodes_edit, simp_plot_edit, key)
            ##Find Normal at each Triangle:
            # def normal2Plane(simplice,nodes):
            sim_edit_node = np.unique(simplices[simplices_edit].flatten())

            normal_avg_lst = []

            for node in tqdm(range(len(nodes_edit))):
                patch = simplices[np.argwhere(simplices == sim_edit_node[node])[:, 0]]

                delete_it_lst = []
                for i in patch.flatten():
                    # print(i)

                    if i not in sim_edit_node:
                        # print("{}_NOT".format(i))
                        delet_it = np.argwhere(patch == i)[:, 0]
                        # patch_edit = np.delete(patch, delet_it,axis = 0)
                        delete_it_lst.append(delet_it.flatten().tolist())

                flat_list = [item for sublist in delete_it_lst for item in sublist]

                unique_flat_lst = np.unique(np.array(flat_list))

                # print("this     ",unique_flat_lst)
                if len(unique_flat_lst) == 0:
                    patch_edit = patch
                else:
                    patch_edit = np.delete(
                        patch, np.unique(np.array(flat_list)), axis=0
                    )
                patch_edit = correct_indices(patch_edit, sim_edit_node)
                # print(patch_edit)

                normals_local = []
                for i in range(len(patch_edit)):
                    vertices = nodes_edit[patch_edit[i]]
                    # print(vertices)

                    A = vertices[0]
                    B = vertices[1]
                    C = vertices[2]
                    AB = B - A
                    AC = C - A
                    normal1 = np.cross(AB, AC)
                    normal1_norm = normal1 / np.linalg.norm(normal1)

                    normals_local.append(normal1_norm.tolist())
                normal_average = np.average(np.array(normals_local), axis=0)
                normal_avg_lst.append(normal_average.tolist())
            normal_avg_arr = np.array(normal_avg_lst)

            nodes_inv["X_vec"] = normal_avg_arr[:, 0]
            nodes_inv["Y_vec"] = normal_avg_arr[:, 1]
            nodes_inv["Z_vec"] = normal_avg_arr[:, 2]
            nodes_inv.to_csv(
                "prec_{}_{}_Nodes_and_Normals_3D.csv".format(int(key), no_elements),
                index=False,
            )
            G_normal_avg_vec.create_dataset("{}".format(int(key)), data=normal_avg_arr)
            ## Define normal ends or the cylinder ends
            normal_magni = []
            for i in range(len(nodes_edit)):
                x1, y1, z1 = nodes_edit[i]
                x2, y2, z2 = normal_avg_lst[i]
                normal_magni.append(
                    cyl_endPoints(normal_end_length, x1, x2, y1, y2, z1, z2)
                )
            nodes_edit_lst.append(nodes_edit)
            simp_plot_edit_lst.append(simp_plot_edit)
            G_normal_ends.create_dataset(
                "{}".format(int(key)), data=np.array(normal_magni)
            )

            # with h5py.File('prec_{}_{}_Normal_ends_PCA_INV_3D.hdf5'.format(int(key), no_elements), 'w') as f:
            #    dset1 = f.create_dataset("normal_ends", data=np.array(normal_magni))
