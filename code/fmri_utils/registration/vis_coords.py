import matplotlib.pyplot as plt
import csv
import nibabel as nib
import numpy as np
import numpy.linalg as npl

above_project_red= '/Users/Zuzanna/Documents/Berkeley/classes/PSYCH214_fMRI/project/'

coord_file = open('coordinate_info.csv','r')
coord_info = [line for line in csv.reader(coord_file)][1:]

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'


MNI_row = [i for i in range(len(coord_info)) if 'MNI' in coord_info[i]]
subject_rows = [i for i in range(len(coord_info)) if 'MNI' not in coord_info[i]]

def show_slices(coord_line, zoom):

    img = nib.load(above_project_red+coord_line[0])
    data = img.get_data()
    affine = npl.inv(img.affine)
    mat, vec = nib.affines.to_matvec(affine)

    fig, axes = plt.subplots(1,2)
    mid = str_to_list(coord_line[3])
    z = [mid[-1]]
    mid_coord = np.array(mid).dot(mat) + vec

    fig.suptitle(coord_line[1])

    axes[0].imshow(data[mid_coord[0],...].T)
    axes[0].scatter([mid_coord[1]],[mid_coord[2]], c = [0, 1, 0])

    if zoom:
        axes[0].set_xlim([80, 180])
        axes[0].set_ylim([50, 100])
    else:
        axes[0].set_xlim([0, data.shape[1]])
        axes[0].set_ylim([0, data.shape[2]])

    axes[0].set_xlabel('y-axis')
    axes[0].set_ylabel('z-axis')

    axes[1].imshow(data[...,mid_coord[-1]].T)
    axes[1].scatter([mid_coord[0]],[mid_coord[1]], c = [0, 1, 0])

    for landmark in range(4,9):
        pt_xy = coord_line[landmark]

        if pt_xy != 'None':
            pt_xyz = str_to_list(coord_line[landmark])+z

            coord = np.array(pt_xyz).dot(mat) + vec

            axes[1].scatter([coord[0]],[coord[1]], s = 4, c = [1,0,0])

    if zoom:
        axes[1].set_xlim([60, data.shape[0]])
        axes[1].set_ylim([100, data.shape[1]])
    else:
        axes[1].set_xlim([0, data.shape[0]])
        axes[1].set_ylim([0, data.shape[1]])

    axes[1].set_xlabel('x-axis')
    axes[1].set_ylabel('y-axis')

    return fig


def str_to_list(txt):
    clean = txt.replace('[','').replace(']','')
    return [float(i) for i in clean.split(', ')]

MNI_fig = show_slices(coord_info[MNI_row[0]],1)
MNI_fig.savefig('MNI_fig.png')
MNI_fig.show()

for s in range(len(subject_rows)):
    subj_fig = show_slices(coord_info[subject_rows[s]],1)
    subj_fig.savefig(coord_info[subject_rows[s]][1]+'.png')
    subj_fig.show()
