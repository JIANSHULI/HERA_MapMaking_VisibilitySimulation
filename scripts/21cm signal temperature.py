import numpy as np


def brightness_temperature_calculator_21cm(neutral_ratio=0.5, densitiy_ratio=1., h=0.7, omiga_b=0.04, omiga_m=0.3, red_shift=6., temperature_spin=20., temperature_cmb=20, difference_over_cmb=False):
	if not difference_over_cmb:
		T_21 = 26.8 * neutral_ratio * densitiy_ratio * (h * omiga_b / 0.0327) * (omiga_m / 0.307) ** (-0.5) * ((1 + red_shift) / 10.) ** 0.5
	else:
		T_21 = 26.8 * neutral_ratio * densitiy_ratio * (h * omiga_b / 0.0327) * (omiga_m / 0.307) ** (-0.5) * ((1 + red_shift) / 10.) ** 0.5 * ((temperature_spin - temperature_cmb) / temperature_spin)
	
	print('neutral_ratio={0}, density_ratio={1}, h={2}, omiga_b={3}, omiga_m={4}, red_shift={5}, temperture_spin={6} mk, temperature_cmb={7} mk, difference_over_cmb={8} \n'.format(neutral_ratio, densitiy_ratio, h, omiga_b, omiga_m, red_shift, temperature_spin, temperature_cmb, difference_over_cmb))
	print('T_21 = {0} mk'.format(T_21))
	
	return T_21 * 10. ** (-3)


def temperature_to_jansky_calculator(k=1.3806504 * 10. ** (-23), object_temperature=0.01, c=299792458., frequency=150, bubble_angular_radius=8.):
	jansky_per_beam = 2. * k * object_temperature * (c / (frequency * 10. ** 6)) ** 2 * np.pi * (np.pi / 180. / 60. * bubble_angular_radius) ** 2 * 10. ** (26)
	
	print('object_temperature={0} K, frequency={1} MHz, bubble_angular_radius={2} arcmin \n'.format(object_temperature, frequency, bubble_angular_radius))
	print('jansky_per_beam = {0} uJy/beam'.format(jansky_per_beam * 10. ** 6))
	
	return jansky_per_beam


def effective_gain_calculator(k=1.3806504 * 10. ** (-23), total_area=53878.314, effective_ratio=0.7):
	effective_gain = total_area * effective_ratio / (2 * k) * 10. ** (-26)
	
	print('effective_area={0} m^2 \n'.format(total_area * effective_ratio))
	print('effective_gain = {0} K/jansky \n'.format(effective_gain))
	
	return effective_gain


def rms_temperature_calculator(frequency=150, integration_time=10., frequency_channel_width=9.78 * 10. ** 4.):
	system_temperature = (100. + 120. * (frequency / 150) ** (-2.55)) / (frequency_channel_width * integration_time) ** 0.5
	
	print('frequency={0} MHz, integration_time={1} s, frequency_channel_width={2} HZ \n'.format(frequency, integration_time, frequency_channel_width))
	print('system_temperature = {0} K \n'.format((100. + 120. * (frequency / 150) ** (-2.55))))
	print('rms_temperature = {0} K \n'.format(system_temperature))
	
	return system_temperature
	
	
print ('Run on Cluster')