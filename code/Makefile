anatomical_prepoc:
	python fmri_utils/anat_preproc/anatomical_figures.py
	
func_preproc:
	pyhton fmri_utils/func_preproc/volume_realignment.py -f sub-10159_task-rest_bold.nii -n 1 -s 5 -p r_

registration:
	python fmri_utils/registration/quality_report.py
	python fmri_utils/registration/registration_report.py load 1 sub-10159 sub-10171 sub-10189 sub-10193 sub-10206 sub-10217 sub-10225

segmentation:
	python fmri_utils/segmentation/kmeans_figures.py
	python fmri_utils/segmentation/mrf_em_figures.py
