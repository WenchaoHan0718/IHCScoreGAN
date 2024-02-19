import os
import numpy as np
import xml.etree.ElementTree as et

def mask_from_regions(annot_tree, scale_factor, shape):
    from PIL import Image, ImageDraw

    img = Image.new('L', shape, 0)
    for region_tree in annot_tree.iter('Region'):
        polygon = []
        for vertex in region_tree.iter('Vertex'):
            polygon.append((round(float(vertex.attrib['X']) * scale_factor), round(float(vertex.attrib['Y']) * scale_factor)))
        if len(polygon)<=2: continue
        ImageDraw.Draw(img).polygon(polygon, outline=255, fill=255)
    
    return img

def get_counts(annotation_tree):
    kvp = {'(3+) Nuclei':None,'(2+) Nuclei':None,'(1+) Nuclei':None,'(0+) Nuclei':None,'Total Nuclei':None}
    for x in annotation_tree.iter('AttributeHeader'):
        if x.attrib['Name'] in kvp.keys():
            kvp[x.attrib['Name']] = x.attrib['Id']
    kvp = {v: k for k, v in kvp.items()}

    counts = {v: 0 for v in kvp.values()}
    for attribute in annotation_tree.iter('Attribute'):
        # region_counts += (region_tree.attrib['InputRegionId'],)
        if attribute.attrib['Name'] in kvp.keys():
            counts[kvp[attribute.attrib['Name']]] += int(attribute.attrib['Value'])

    return counts

def xml_to_mask(annot_path, scale_factor, shape, counts_to_match):
    tree = et.parse(annot_path)
    root = tree.getroot()
    annotation_trees = root.findall('Annotation')
    if shape is None: shape = (max([int(float(x.attrib['X'])) for x in root.iter('Vertex')]), max([int(float(x.attrib['Y'])) for x in root.iter('Vertex')]))
    annot_trees = []
    annot_counts = []
    for annotation_tree in annotation_trees:
        if annotation_tree.find('InputAnnotationId') is not None:
            input_annotation_id = annotation_tree.find('InputAnnotationId').text
            annot_count = {'Id': input_annotation_id}
            annot_count.update(get_counts(annotation_tree))
            annot_counts.append(annot_count)
        else:
            annot_tree = {'Id': annotation_tree.attrib['Id'], 'Tree': annotation_tree}
            annot_trees.append(annot_tree)
    for annot_count in annot_counts:
        if annot_count['Total Nuclei']==0: 
            continue
        if all([annot_count[key]==counts_to_match[key] for key in counts_to_match.keys()]):
            matches = [annot_tree for annot_tree in annot_trees if annot_tree['Id']==annot_count['Id']]
            counted_annot_tree = matches[0] if len(matches)>0 else None
    if counted_annot_tree is None: return None
    mask = mask_from_regions(counted_annot_tree['Tree'], scale_factor, shape)
    return mask