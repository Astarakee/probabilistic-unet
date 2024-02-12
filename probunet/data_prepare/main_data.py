import os
import numpy as np
import SimpleITK as itk
from sitk_stuff import read_nifti
from json_pickle_stuff import write_json
from paths_dirs_stuff import path_contents, path_contents_pattern, create_path


def GetTimePointScan(subject_abs_path, pattern="time1"):
    filenames = path_contents_pattern(subject_abs_path, ".nii.gz")
    filenames = [x for x in filenames if pattern in x]
    return filenames


def GetMRSequence(scan_list):
    sequence = {}
    for item in scan_list:
        if "_t1.nii.gz" in item:
            sequence["t1"] = item
        elif "_t1ce.nii.gz" in item:
            sequence["t1ce"] = item
        elif "_t2.nii.gz" in item:
            sequence["t2"] = item
        elif "_flair.nii.gz" in item:
            sequence["flair"] = item
        elif "_seg.nii.gz" in item:
            sequence["seg"] = item
        else:
            pass
    return sequence


def normalize_array(seq_abs_path):
    img_array, _, img_size, _, _, _ = read_nifti(seq_abs_path)
    mean = img_array.mean()
    std = img_array.std()
    img_array -= mean
    img_array /= (max(std, 1e-8))
    return img_array


data_path = "/mnt/workspace/data/GBM/UCSF-ALPTDG/UCSF_POSTOP_GLIOMA_DATASET_FINAL_v1.0"
save_path = "/mnt/workspace/data/GBM/UCSF-ALPTDG/mehdi_prob_growth"
subjects = path_contents(data_path)
subjects = [x for x in subjects if '.xlsx' not in x]
n_subjects = len(subjects)


for ix, case in enumerate(subjects):
    print("working on subject {} out of {}".format(ix+1, n_subjects))
    case_abs = os.path.join(data_path, case)
    case_time1 = GetTimePointScan(case_abs, pattern="time1")
    case_time2 = GetTimePointScan(case_abs, pattern="time2")
    case_seq1 = GetMRSequence(case_time1)
    case_seq2 = GetMRSequence(case_time2)

    time1_data = []
    time2_data = []

    for k1,v1 in case_seq1.items():
        seq_abs_path = os.path.join(case_abs, v1)
        if k1 != "seg":
            vol_norm = normalize_array(seq_abs_path)
            time1_data.append(vol_norm)
    seg1_abs_path = os.path.join(case_abs,case_seq1["seg"])
    seg_array1, _, seg_size1, _, _, _ = read_nifti(seg1_abs_path)
    seg_array1 = seg_array1.astype("uint8")
    time1_data.append(seg_array1)

    for k2,v2 in case_seq2.items():
        seq_abs_path2 = os.path.join(case_abs, v2)
        if k2 != "seg":
            vol_norm2 = normalize_array(seq_abs_path2)
            time2_data.append(vol_norm2)
    seg2_abs_path = os.path.join(case_abs,case_seq2["seg"])
    seg_array2, _, seg_size2, _, _, _ = read_nifti(seg2_abs_path)
    seg_array2 = seg_array2.astype("uint8")
    time2_data.append(seg_array2)

    time1_data_np = np.array(time1_data)
    time2_data_np = np.array(time2_data)
    ch, dep, row, col = np.shape(time1_data_np)
    data_stack = np.zeros((3, ch, dep, row, col), np.float32)
    data_stack[0,::] = time1_data_np
    data_stack[1,::] = time2_data_np
    data_stack[2,::] = time2_data_np

    #data_itk = itk.GetImageFromArray(data_stack)
    #abs_path_write = os.path.join(save_path, "subject.nii.gz")
    #itk.WriteImage(data_itk, abs_path_write)
    subject_name = case+".npy"
    write_abs_path = os.path.join(save_path, subject_name)
    np.save(write_abs_path, data_stack)


identifiers = path_contents_pattern(save_path, ".npy")
multi_shape = {}
for item in identifiers:
    item = item.split('.npy')[0]
    multi_shape[item] = (3,5,155,240,240)
write_json(os.path.join(save_path, "multi_shapes.json"), multi_shape)