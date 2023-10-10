"""
Corrects for possible error by deleting any drop with has more than 60% error
from the hermite line
"""

# def apply_resolution_correcti(self):
#     # Create a matrix to filter the wrong values that should be out
#     filter = full((32, 32), True)
#     for i, (d, _) in enumerate(CLASSES_DIAMETER):
#         vpred = V_D_Lhermitte_1988(d)
#         for j, (v, _) in enumerate(CLASSES_VELOCITY):
#             filter[i, j] = abs((vpred - v)) < 0.6 * v
#
#     # Apply the filter to every matrix
#     for item in self:
#         item.matrix *= filter

"""
In case the Data commes from the 3D stereo, it aplles the correction so as
to not have data from a resolution smaller then the parsivel.
"""

# def apply_resolution_correction(self):
#     # Zeros the first two diameters bins for the matrix
#     for item in self:
#         for i in range(32):
#             item.matrix[0, i] = 0.0
#             item.matrix[1, i] = 0.0
