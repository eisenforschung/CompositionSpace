"""Utility functions to visualize some of the generated content using HDF5/XDMF."""

import os

import h5py
import lxml.etree as et
import numpy as np


def link_to_hfive_data_item(
    xml_node_parent, hdf_location, dims=None, hdf_payload=None, item_name=None
):
    """Add child node to xml_node_parent pointing to heavy data inside hdf_location."""
    if not et.iselement(xml_node_parent):
        raise TypeError(f"Argument xml_node_parent is not an etree.SubElement!")
    itm = et.SubElement(xml_node_parent, "DataItem")
    if hdf_location != "":
        itm.text = f"{hdf_location}"
        itm.attrib["Format"] = "HDF"
    else:
        raise ValueError(f"Argument hdf_location is an empty string!")
    if dims is not None:
        if isinstance(dims, int):
            itm.attrib["Dimensions"] = f"{dims}"
        elif isinstance(dims, np.ndarray):
            itm.attrib["Dimensions"] = f"{' '.join(str(val) for val in dims)}"
        else:
            raise TypeError(f"Unexpected value for optional argument dims!")
    if hdf_payload is not None:
        if isinstance(hdf_payload, np.ndarray):
            if hdf_payload.dtype in (np.uint32, np.uint64):
                itm.attrib["NumberType"] = "UInt"
            if hdf_payload.dtype in (np.int32, np.int64):
                itm.attrib["NumberType"] = "Int"
            elif hdf_payload.dtype in (np.float32, np.float64):
                itm.attrib["NumberType"] = "Float"
            if hdf_payload.itemsize in [4, 8]:
                itm.attrib["Precision"] = f"{hdf_payload.itemsize}"
        else:
            raise TypeError(f"Argument hdf_payload is not a np.ndarray!")
    # itm.attrib["Format"] = "XML"
    if item_name is not None:
        itm.attrib["Name"] = f"{item_name}"


def generate_xdmf_for_visualizing_content(file_path: str, entry_id: int = 1):
    """Take NeXus/HDF5 file_path, inspect for plottable content, create XDMF for it."""
    print(f"Inspecting {file_path}...")
    if not os.path.isfile(file_path):
        raise IOError(f"Results file {'file_path'} does not exist!")
    with h5py.File(file_path, "r") as h5r:
        # firstly, we need to get metadata about the discretization grid to visualize
        trg = f"/entry{entry_id}/voxelization/cg_grid"
        req_fields = ["dimensionality", "extent", "origin", "cell_dimensions"]
        for field in req_fields:
            if f"{trg}/{field}" not in h5r:
                return
        grid_metadata = {}
        for field in req_fields:
            grid_metadata[field] = h5r[f"{trg}/{field}"][...]
            print(
                f"Found {field}, {grid_metadata[field].dtype}, {np.shape(grid_metadata[field])}, {grid_metadata[field]}"
            )

        # secondly, let's build the XML tree for the XDMF document
        root_node = et.Element("Xdmf")
        root_node.attrib["Version"] = "2.0"
        roi_node = et.SubElement(root_node, "Domain")
        grid_node = et.SubElement(roi_node, "Grid")
        grid_node.attrib["Name"] = f"entry{entry_id}/voxelization/cg_grid"
        grid_node.attrib["GridType"] = "Uniform"
        topo_node = et.SubElement(grid_node, "Topology")
        topo_node.attrib["TopologyType"] = "3DCoRectMesh"
        topo_node.attrib["NumberOfElements"] = (
            f"{' '.join(str(val) for val in grid_metadata['extent'][::-1])}"
        )
        geom_node = et.SubElement(grid_node, "Geometry")
        geom_node.attrib["GeometryType"] = "ORIGIN_DXDYDZ"
        link_to_hfive_data_item(
            geom_node,
            hdf_location=f"{file_path}:/entry{entry_id}/voxelization/cg_grid/origin",
            hdf_payload=grid_metadata["origin"],
            dims=3,
            item_name="Origin",
        )
        link_to_hfive_data_item(
            geom_node,
            hdf_location=f"{file_path}:/entry{entry_id}/voxelization/cg_grid/cell_dimensions",
            hdf_payload=grid_metadata["cell_dimensions"],
            dims=3,
            item_name="Spacing",
        )
        # add visualization for results of voxelization
        trg = f"/entry{entry_id}/voxelization"
        if f"{trg}/weight" in h5r:
            attr = et.SubElement(grid_node, "Attribute")
            attr.attrib["Name"] = "total, weight"
            attr.attrib["AttributeType"] = "Scalar"
            attr.attrib["Center"] = "Node"
            link_to_hfive_data_item(
                attr,
                f"{file_path}:{trg}/weight",
                hdf_payload=h5r[f"{trg}/weight"][:],
                dims=grid_metadata["extent"],
            )

            for grp in h5r[f"{trg}"].keys():
                if grp.startswith("element"):
                    if f"{trg}/{grp}/name" in h5r and f"{trg}/{grp}/weight" in h5r:
                        attr = et.SubElement(grid_node, "Attribute")
                        attr.attrib["Name"] = (
                            f"{grp}, {h5r[f"{trg}/{grp}/name"][()].decode("utf-8")}, weight"
                        )
                        attr.attrib["AttributeType"] = "Scalar"
                        attr.attrib["Center"] = "Node"
                        link_to_hfive_data_item(
                            attr,
                            f"{file_path}:{trg}/{grp}/weight",
                            hdf_payload=h5r[f"{trg}/{grp}/weight"][:],
                            dims=grid_metadata["extent"],
                        )

    # finally, write composed XML tree to disk
    with open(f"{file_path}.xdmf", "w") as fp:
        fp.write(
            et.tostring(
                et.ElementTree(root_node),
                xml_declaration=True,
                # encoding="utf-8",
                doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>',
                pretty_print=True,
            ).decode("utf-8")
        )
