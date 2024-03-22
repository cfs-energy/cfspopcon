from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import matplotlib.pyplot as plt

geqdsk = OMFITgeqdsk('../../example_cases/DIII-D/g193802.04490')
#plt.plot(geqdsk['AuxQuantities']['PSI_NORM'], geqdsk['fluxSurfaces']['geo']['delta'])
geqdsk.keys()