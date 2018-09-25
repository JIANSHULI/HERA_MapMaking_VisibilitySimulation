import numpy as np

def brightness_temperature_calculator_21cm(ionization_ratio=0.5, densitiy_ratio=1., h=0.7, omiga_b=0.04, omiga_m=0.3, red_shift=6., temperature_spin=20., temperature_cmb=20, difference_over_cmb=False):
    if not difference_over_cmb:
        T_21 = 26.8 * ionization_ratio * densitiy_ratio * (h * omiga_b / 0.0327) * (omiga_m / 0.307)**(-0.5) * ((1 + red_shift) / 10.)**0.5
    else:
        T_21 = 26.8 * ionization_ratio * densitiy_ratio * (h * omiga_b / 0.0327) * (omiga_m / 0.307) ** (-0.5) * ((1 + red_shift) / 10.)**0.5 * ((temperature_spin - temperature_cmb) / temperature_spin)

    print('ionization_ratio={0}, density_ratio={1}, h={2}, omiga_b={3}, omiga_m={4}, red_shift={5}, temperture_spin={6}, temperature_cmb={7}, difference_over_cmb={8} \n'.format(ionization_ratio, densitiy_ratio, h, omiga_b, omiga_m, red_shift, temperature_spin, temperature_cmb,difference_over_cmb))
    print('T_21 = {0} mk'.format(T_21))
    
    return T_21 * 10.**(-3)

def temperature_to_jansky_calculator(k=1.3806504*10.**(-23), object_temperature=0.01, c=299792458., frequency=150, beam_angular_resolution=24.):
    jansky_per_beam = 2. * k * object_temperature * (c / (frequency*10.**6))**2 * (np.pi/180./60. * beam_angular_resolution)**2 * 10.**(26)
    
    print('object_temperature={0}, frequency={1}, beam_angular_resolution={2} \n'.format(object_temperature, frequency, beam_angular_resolution))
    print('jansky_per_beam = {0}'.format(jansky_per_beam))
    
    return jansky_per_beam

def effective_gain_calculator(k=1.3806504*10.**(-23), total_area=53878.314, effective_ratio=0.7):
    effective_gain = total_area*effective_ratio / (2 * k) * 10.**(-26)
    
    print('effective_area={0} m^2 \n'.format(total_area*effective_ratio))
    print('effective_gain = {0} K/jansky \n'.format(effective_gain))
    
    return effective_gain

def system_temperature_calculator(frequency=150, integration_time=10., frequency_channel_width=9.78*10.**4.):
    system_temperature = (100. + 120.*(frequency/150)**(-2.55)) / (frequency_channel_width * integration_time)**0.5
    
    print('frequency={0}, integration_time={1}, frequency_channel_width={2} \n'.format(frequency, integration_time, frequency_channel_width))
    print('system_temperature = {0} \n'.format(system_temperature))
    
    return system_temperature
    