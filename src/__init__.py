# from __future__ import absolute_import
import _boost_math, boost, simulate_visibilities, Bulm, _Bulm, info, calib, arrayinfo, _omnical
# import ._omnical, ._boost_math, .boost, .simulate_visibilities, .Bulm, ._Bulm, .calib, .arrayinfo

"""init file for pyuvdata."""
# from uvbase import *
# from parameter import *
# from uvdata import *
# from utils import *
# from telescopes import *
# from uvfits import *
# from fhd import *
# from miriad import *
# from uvcal import *
# from calfits import *
# from uvbeam import *
from .uvbase import *
from .parameter import *
from .uvdata import *
from .utils import *
from .telescopes import *
from .uvfits import *
from .fhd import *
from .miriad import *
from .uvcal import *
from .calfits import *
from .uvbeam import *
import version

__version__ = version.version
try:
    # DATA_PATH = __path__[0] + '/../data'
    DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/data'
except:
    print('No DATA_PATH has been set.')
# from __future__ import absolute_import
# import ._omnical, ._Bulm, _boost_math

