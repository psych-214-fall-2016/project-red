anatomical_prepoc:

func_preproc:

registration:
	python registration/quality_report.py
	python registration/registration_report.py load 1 sub-10159 sub-10171 sub-10189 sub-10193 sub-10206 sub-10217 sub-10225

segmentation:
	python segmentation/kmeans_figures.py
	python segmentation/mrf_ef_figures.py
