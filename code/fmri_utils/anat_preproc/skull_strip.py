from nipype.interfaces import fsl as fsl
# This function strips the skull from the brain, creating a brain only file as well as
# a brain mask file.

# This is the FSL Brain Extraction Tool (BET) and has several protocols that can be
# altered to optimize the results. Some of these protocols are described below.

# Much of my knowledge about this came from the Nipype beginners guide.

"""Always make sure to look at the result of the skull stripping step to determine
    if the function did a good job. This step has the most variability in the quality
    of the results. If the output file does not look sufficient, change the fractional
    intensity and/or the vertical_gradient until the result reaches the desired quality."""


def structural_skull_strip(reorient_file , out_file):
    skull_stripper = fsl.BET()
    skull_stripper.inputs.in_file = reorient_file
    skull_stripper.inputs.out_file = out_file
    # frac = fractional intensity threshold. Larger numbers equate to more of the
    # head being stripped. I chose this fractional intensity based on some research
    # and discussions with lab mates who use fsl
    skull_stripper.inputs.frac = 0.3
    # Proescu et al. 2012 describes how to optimize BET for people with Multiple
    # Sclerosis. They reccommend using the reduce_bias option. This option does some bias
    # correction and cuts off some of the residual neck voxels. Too much neck can cause
    # serious problems with BET because the center of mass can be severely altered.
    # This step adds a lot of time to the process but it benefits the image
    # significantly.
    skull_stripper.inputs.reduce_bias = True
    # Positive vertical_gradient (or Threshold gradient) values cause a larger brain
    # outline at the bottom and a smaller at the top. In other words, it sharpens up the
    # image to make less 'jagged' along the edges
    skull_stripper.inputs.vertical_gradient = .05
    skull_stripper.cmdline
    SS_results = skull_stripper.run()
    return SS_results
