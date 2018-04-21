"""
given an RA and a longitude on Earth,
calculate LST when that RA is at zenith.
"""
import ephem
import numpy as np
import datetime
import pytz
from astropy.time import Time
import astropy.constants as const
from astropy.time import Time
from astropy import coordinates as crd
from astropy import units as unt

def RA2LST(RA, lon, lat, jd):
    """
    RA : float
         right ascension (J2000) in degrees
    lon : float
          longitude East of observer in degrees

    lat : float
          latitude North of observer in degrees

    jd : float
          julian date to anchor conversion

    return LST_RA (LST in hours)
    """
    # get observer
    obs = ephem.Observer()
    obs.lon = lon * np.pi / 180.0
    obs.lat = lat * np.pi / 180.0
    obs.date = Time(jd, format='jd', scale='utc').datetime
    #obs.date = datetime.datetime(2000, 03, 20, 12, 0, 0, 0, pytz.UTC)

    # get RA at zenith of observer in degrees
    ra_now = obs.radec_of(0, np.pi/2)[0] * 180 / np.pi

    # get LST of observer
    LST_now = obs.sidereal_time() * 12.0 / np.pi 

    # get the LST of the RA via difference
    LST_RA = LST_now + (RA - ra_now) / 15.0
    return LST_RA
