#!/usr/bin/env python3
"""
Notebook 06: PyRadiomics Feature Extraction for CSF3 Cluster
=============================================================

Standalone script converted from 06_pyradiomics_feature_extraction.ipynb
for execution on the University of Manchester CSF3 cluster.

This script extracts comprehensive radiomic features (GLCM, GLRLM, GLSZM, 
GLDM, shape, first-order) from T1ce MRI volumes using PyRadiomics.

Note: Uses the patched PyRadiomics fork (Kozmosa/pyradiomics-fix-configparser)
      to fix Python 3.10+ compatibility issues with deprecated ConfigParser.

Usage:
    1. Copy MU_Glioma_Post data to CSF3 scratch space
    2. Update DATA_DIR path below
    3. Submit via: qsub radiomics_joshua.sh

Author: Joshua Alliet
Date: February 2026
"""

import os
import sys
import glob
import time
import re
import logging
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR CSF3 SETUP
# =============================================================================

# Base directory containing MU_Glioma_Post data on CSF3
# Using ~ expands to your home directory; scratch is a symlink there
DATA_DIR = os.path.expanduser('~/scratch/MU_Glioma_Post')

# Directory containing patient imaging folders (PatientID_XXXX)
MU_GLIOMA_DIR = os.path.join(DATA_DIR, 'MU_Glioma_Post')

# Clinical data Excel file
CLINICAL_DATA_PATH = os.path.join(DATA_DIR, 'MU_Glioma_Post_ClinicalData_July2025.xlsx')

# Output directory for results
OUTPUT_DIR = os.path.join(DATA_DIR, 'pyradiomics_results')

# Batch save frequency (save intermediate results every N patients)
BATCH_SIZE = 20

# Random seed for reproducibility
RANDOM_STATE = 42

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, '06_extraction.log') if os.path.exists(OUTPUT_DIR) else '06_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress PyRadiomics verbose output
radiomics_logger = logging.getLogger('radiomics')
radiomics_logger.setLevel(logging.WARNING)

warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_first_timepoint_files(patient_dir_path):
    """
    For a given patient directory, find the first timepoint's T1ce volume
    and segmentation mask.

    Filename convention: PatientID_XXXX_Timepoint_N_brain_<modality>.nii.gz
    Segmentation mask:   PatientID_XXXX_Timepoint_N_tumorMask.nii.gz

    Returns:
        dict with keys 't1c', 'seg' mapped to file paths, or None if missing.
    """
    timepoint_dirs = sorted(
        [d for d in os.listdir(patient_dir_path)
         if os.path.isdir(os.path.join(patient_dir_path, d))],
        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
    )

    if not timepoint_dirs:
        return None

    first_tp_path = os.path.join(patient_dir_path, timepoint_dirs[0])
    nii_files = glob.glob(os.path.join(first_tp_path, '*.nii.gz'))

    file_map = {}
    for f in nii_files:
        basename = os.path.basename(f).lower()
        if basename.endswith('_brain_t1c.nii.gz'):
            file_map['t1c'] = f
        elif basename.endswith('_tumormask.nii.gz'):
            file_map['seg'] = f

    if 't1c' in file_map and 'seg' in file_map:
        file_map['timepoint'] = timepoint_dirs[0]
        return file_map

    return None


def extract_patient_id(dir_name):
    """Extract normalised patient ID from directory name."""
    match = re.match(r'(PatientID_\d{4})', str(dir_name))
    if match:
        return match.group(1)
    return str(dir_name)


def binarise_segmentation(seg_nifti_path, output_path):
    """
    Convert multi-label segmentation mask into binary whole-tumour mask.
    All non-zero labels become 1 (tumour); 0 remains 0 (background).
    """
    seg_img = sitk.ReadImage(seg_nifti_path)
    seg_array = sitk.GetArrayFromImage(seg_img)

    binary_array = (seg_array > 0).astype(np.uint8)

    binary_img = sitk.GetImageFromArray(binary_array)
    binary_img.CopyInformation(seg_img)

    sitk.WriteImage(binary_img, output_path)
    return output_path


def extract_features_for_patient(patient_id, file_map, extractor, temp_dir):
    """
    Extract radiomic features for a single patient using T1ce volume
    and binarised segmentation mask.

    Returns:
        Dict of feature_name: value, or None if extraction failed.
    """
    t1c_path = file_map['t1c']
    seg_path = file_map['seg']

    try:
        binary_seg_path = os.path.join(temp_dir, f'{patient_id}_seg_binary.nii.gz')
        binary_seg_path = binarise_segmentation(seg_path, binary_seg_path)

        check_img = sitk.ReadImage(binary_seg_path)
        check_array = sitk.GetArrayFromImage(check_img)
        tumour_voxels = np.sum(check_array > 0)

        if tumour_voxels == 0:
            logger.warning(f'{patient_id} has empty segmentation mask. Skipping.')
            return None

        result = extractor.execute(t1c_path, binary_seg_path)

        features = {}
        for key, value in result.items():
            if not key.startswith('diagnostics_'):
                try:
                    features[key] = float(value) if hasattr(value, 'item') else float(value)
                except (TypeError, ValueError):
                    features[key] = value

        features['patient_id'] = patient_id
        features['tumour_voxels'] = int(tumour_voxels)

        # Clean up temporary binary mask
        if os.path.exists(binary_seg_path):
            os.remove(binary_seg_path)

        return features

    except Exception as e:
        logger.error(f'ERROR extracting features for {patient_id}: {e}')
        return None


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def main():
    logger.info('=' * 70)
    logger.info('PYRADIOMICS FEATURE EXTRACTION - CSF3 CLUSTER')
    logger.info('=' * 70)

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    temp_dir = os.path.join(OUTPUT_DIR, 'temp_binary_masks')
    os.makedirs(temp_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Verify paths exist
    # -------------------------------------------------------------------------
    logger.info('\n1. Verifying data paths...')
    
    if not os.path.exists(MU_GLIOMA_DIR):
        logger.error(f'MU_GLIOMA_DIR not found: {MU_GLIOMA_DIR}')
        logger.error('Please update DATA_DIR in this script to point to your data location.')
        sys.exit(1)
    
    if not os.path.exists(CLINICAL_DATA_PATH):
        logger.error(f'Clinical data not found: {CLINICAL_DATA_PATH}')
        sys.exit(1)
    
    logger.info(f'  MU_GLIOMA_DIR: OK')
    logger.info(f'  CLINICAL_DATA_PATH: OK')

    # -------------------------------------------------------------------------
    # 2. Load clinical data for patient ID alignment
    # -------------------------------------------------------------------------
    logger.info('\n2. Loading clinical data...')
    
    clinical_raw = pd.read_excel(CLINICAL_DATA_PATH, sheet_name='MU Glioma Post')
    clinical_raw.columns = clinical_raw.columns.str.strip()

    patient_id_col = None
    for candidate in ['Patient ID', 'Patient_ID', 'patient_id', 'PatientID']:
        if candidate in clinical_raw.columns:
            patient_id_col = candidate
            break

    if patient_id_col is None:
        logger.error('Could not find Patient ID column in clinical data.')
        sys.exit(1)

    clinical_patient_ids = set(clinical_raw[patient_id_col].astype(str).values)
    logger.info(f'  Clinical patients: {len(clinical_patient_ids)}')

    # -------------------------------------------------------------------------
    # 3. Build patient file registry
    # -------------------------------------------------------------------------
    logger.info('\n3. Scanning imaging directories...')
    
    all_patient_dirs = sorted([
        d for d in os.listdir(MU_GLIOMA_DIR)
        if os.path.isdir(os.path.join(MU_GLIOMA_DIR, d)) and d.startswith('PatientID_')
    ])
    logger.info(f'  Patient directories found: {len(all_patient_dirs)}')

    patient_file_registry = {}
    missing_patients = []

    for patient_dir_name in all_patient_dirs:
        patient_path = os.path.join(MU_GLIOMA_DIR, patient_dir_name)
        files = find_first_timepoint_files(patient_path)

        if files is not None:
            patient_file_registry[patient_dir_name] = files
        else:
            missing_patients.append(patient_dir_name)

    logger.info(f'  Patients with T1ce + segmentation: {len(patient_file_registry)}')
    logger.info(f'  Patients missing required files: {len(missing_patients)}')

    # -------------------------------------------------------------------------
    # 4. Align with clinical data
    # -------------------------------------------------------------------------
    imaging_patient_ids = set(extract_patient_id(d) for d in patient_file_registry.keys())
    overlap = imaging_patient_ids & clinical_patient_ids

    matched_patients = [
        pid for pid in patient_file_registry.keys()
        if extract_patient_id(pid) in overlap
    ]

    logger.info(f'\n4. Patient alignment:')
    logger.info(f'  Imaging patients: {len(imaging_patient_ids)}')
    logger.info(f'  Clinical patients: {len(clinical_patient_ids)}')
    logger.info(f'  Matched (overlap): {len(overlap)}')
    logger.info(f'  Patients to process: {len(matched_patients)}')

    # -------------------------------------------------------------------------
    # 5. Configure PyRadiomics extractor
    # -------------------------------------------------------------------------
    logger.info('\n5. Configuring PyRadiomics extractor...')
    
    extractor_settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'normalizeScale': 1,
        'normalize': False,
        'removeOutliers': None,
        'label': 1
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_settings)
    extractor.enableAllFeatures()
    extractor.enableImageTypeByName('Original')

    logger.info(f'  Enabled feature classes: {list(extractor.enabledFeatures.keys())}')
    logger.info(f'  Bin width: {extractor_settings["binWidth"]}')

    # -------------------------------------------------------------------------
    # 6. Full cohort extraction
    # -------------------------------------------------------------------------
    logger.info('\n6. Starting full extraction...')
    logger.info(f'  Patients to process: {len(matched_patients)}')
    logger.info(f'  Intermediate saves every {BATCH_SIZE} patients')

    all_features_list = []
    failed_patients = []
    full_start = time.time()

    for i, patient_id in enumerate(matched_patients):
        patient_start = time.time()

        features = extract_features_for_patient(
            patient_id=patient_id,
            file_map=patient_file_registry[patient_id],
            extractor=extractor,
            temp_dir=temp_dir
        )

        elapsed = time.time() - patient_start

        if features is not None:
            all_features_list.append(features)
            logger.info(f'[{i+1:3d}/{len(matched_patients)}] {patient_id}: OK ({elapsed:.1f}s)')
        else:
            failed_patients.append(patient_id)
            logger.warning(f'[{i+1:3d}/{len(matched_patients)}] {patient_id}: FAILED ({elapsed:.1f}s)')

        # Intermediate save
        if (i + 1) % BATCH_SIZE == 0:
            interim_df = pd.DataFrame(all_features_list)
            interim_path = os.path.join(OUTPUT_DIR, '06_radiomic_features_interim.csv')
            interim_df.to_csv(interim_path, index=False)
            logger.info(f'  -> Interim save: {len(all_features_list)} patients extracted')

    full_elapsed = time.time() - full_start

    # -------------------------------------------------------------------------
    # 7. Save final results
    # -------------------------------------------------------------------------
    logger.info('\n7. Saving results...')

    full_features_df = pd.DataFrame(all_features_list)
    final_path = os.path.join(OUTPUT_DIR, '06_full_radiomic_features.csv')
    full_features_df.to_csv(final_path, index=False)

    logger.info(f'  Saved: {final_path}')
    logger.info(f'  Shape: {full_features_df.shape}')

    # Save failed patients list
    if failed_patients:
        failed_path = os.path.join(OUTPUT_DIR, '06_failed_patients.txt')
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_patients))
        logger.info(f'  Failed patients saved: {failed_path}')

    # -------------------------------------------------------------------------
    # 8. Summary
    # -------------------------------------------------------------------------
    logger.info('\n' + '=' * 70)
    logger.info('EXTRACTION COMPLETE')
    logger.info('=' * 70)
    logger.info(f'  Total time: {full_elapsed/60:.1f} minutes')
    logger.info(f'  Successful: {len(all_features_list)}')
    logger.info(f'  Failed: {len(failed_patients)}')
    logger.info(f'  Features per patient: {len(full_features_df.columns) - 2}')  # -2 for patient_id, tumour_voxels
    logger.info(f'  Output: {final_path}')

    # Clean up temp directory
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logger.info(f'  Cleaned up: {temp_dir}')

    logger.info('\nDone.')


if __name__ == '__main__':
    main()
