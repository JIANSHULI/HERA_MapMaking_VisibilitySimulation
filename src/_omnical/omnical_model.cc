//Jeff 2012-07-19
#include <stdint.h>
#include <stdio.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <numeric>
#include <algorithm>
#include <functional>
#include <numeric>
#include "include/omnical_redcal.h"
#include "include/omnical_model.h"
#include <algorithm>
#define uint unsigned int
using namespace std;
const string FILENAME = "calibration_omni.cc";
const float SPEEDC = 299.792458;
const float PI = atan2(0, -1);
const bool DEBUG = false;
const float UBLPRECISION = pow(10, -3);
const float MIN_NONE_ZERO = pow(10, -10);
const float MAX_NONE_INF = pow(10, 10);
const float MAX_10_POW = 20; //single precision float max is 3.4*10^38, so I'm limiting the power to be 20.
const float MAX_POW_2 = pow(10, 10); //limiting the base of ^2 to be 10^10
const float X4_LONGITUDE = -69.987182;
const float X4_LATITUDE = 45.297728;
const float X4_ELEVATION = 171;
const float X4_TIMESHIFT = 28957;//seconds to add to raw data header time to get correct UTC
const float DEF_LONGITUDE = -69.987182;
const float DEF_LATITUDE = 45.297728;
const float DEF_ELEVATION = 171;
const int NUM_OBJECTS = 30;//Number of satellites we have in tracked_bodies_X4.tle;

///////////////////////////////////////
//command line and Python interaction//
///////////////////////////////////////

string ftostr(float f){
	ostringstream buffer;
	buffer << f;
	return buffer.str();
}

string itostr(int i, uint len){
	ostringstream buffer;
	buffer << abs(i);
	string raw = buffer.str();//unpadded int
	string output;
	if ( i >= 0) {
		output = "";
	} else {
		output = "-";
	}
	for (uint n = 0; n < len - raw.size(); n ++) output += "0";
	output += raw;
	return output;
};

vector<float> strtovf(string in){
	//cout << "DEBUG " << in << endl;
	stringstream stream(in);
	vector<float> output;
	float tmpf;
	while (stream.good()){
		stream >> tmpf;
		//cout << tmpf << endl;
		output.push_back(tmpf);
	}
	//output.pop_back(); //sometimes get a 0 in end..tricky...
	return output;
}

vector<float> tp2xyz (vector<float> thephi){
	vector<float> xyz(3,0);
	xyz[0] = sin(thephi[0])*cos(thephi[1]);
	xyz[1] = sin(thephi[0])*sin(thephi[1]);
	xyz[2] = cos(thephi[0]);
	return xyz;
}
vector<float> tp2xyz (float t, float p){
	vector<float> xyz(3,0);
	xyz[0] = sin(t)*cos(p);
	xyz[1] = sin(t)*sin(p);
	xyz[2] = cos(t);
	return xyz;
}

vector<float> xyz2tp (vector<float> xyz){
	string METHODNAME = "xyz2tp";
	vector<float> thephi(2,0);
	float r = sqrt(xyz[2]*xyz[2]+xyz[1]*xyz[1]+xyz[0]*xyz[0]);
	if( r == 0){
		cout << "#!!#" << FILENAME << "#!!#" << METHODNAME << ": !!FATAL error!! input x,y,z are all 0!" << endl;
		return thephi;
	}
	thephi[0] = acos(xyz[2] / r);

	thephi[1] = atan2(xyz[1],xyz[0]);
	return thephi;
}

vector<float> xyz2tp (float x, float y, float z){
	string METHODNAME = "xyz2tp";
	vector<float> thephi(2,0);
	float r = sqrt(x*x + y*y + z*z);
	if( r == 0){
		cout << "#!!#" << FILENAME << "#!!#" << METHODNAME << ": !!FATAL error!! input x,y,z are all 0!" << endl;
		return thephi;
	}
	thephi[0] = acos(z / r);

	thephi[1] = atan2(y,x);
	return thephi;
}

vector<float> tp2rd (vector<float> thephi){
	vector<float> rd(2,0);
	rd[0] = thephi[1];
	rd[1] = PI/2 - thephi[0];
	return rd;
}
vector<float> tp2rd (float t, float p){
	vector<float> rd(2,0);
	rd[0] = p;
	rd[1] = PI/2 - t;
	return rd;
}
vector<float> rd2tp (vector<float> rd){
	vector<float> tp(2,0);
	tp[1] = rd[0];
	tp[0] = PI/2 - rd[1];
	return tp;
}
vector<float> rd2tp (float r, float d){
	vector<float> tp(2,0);
	tp[1] = r;
	tp[0] = PI/2 - d;
	return tp;
}
vector<float> tp2aa (vector<float> thephi){//alt-az
	vector<float> aa(2,0);
	aa[0] = PI/2 - thephi[0];
	aa[1] = PI - thephi[1];
	return aa;
}
vector<float> tp2aa (float t, float p){//alt-az
	vector<float> aa(2,0);
	aa[0] = PI/2 - t;
	aa[1] = PI - p;
	return aa;
}
vector<float> aa2tp (vector<float> aa){//alt-az
	vector<float> tp(2,0);
	tp[1] = PI - aa[1];
	tp[0] = PI/2 - aa[0];
	return tp;
}
vector<float> aa2tp (float alt, float az){
	vector<float> tp(2,0);
	tp[1] = PI - az;
	tp[0] = PI/2 - alt;
	return tp;
}

void matrixDotV(vector<vector<float> > * A, vector<float> * b, vector<float> * x){
	int i, j;
	double sum;
	int n = min(A->size(),x->size());
	int m = min(A->at(0).size(),b->size());
	for(i = 0; i < n; i++){
		sum = 0.0;
		for(j = 0; j < m; j++){
			sum = sum + (A->at(i))[j] * (x->at(j));
		}
		(x->at(i)) = sum;
	}
	return;
}

//void iqDemod(vector<vector<vector<vector<vector<float> > > > > *data, vector<vector<vector<vector<vector<float> > > > > *data_out, int nIntegrations, int nFrequencies, int nAnt){
	//string METHODNAME = "iqDemod";
	//int nChannels = nAnt * 4; //a factor of 2 from consolidating x and y polarizations, and another factor of 2 from consolidating iq
	//int n_xxi = nAnt * (nAnt + 1)/2;

	//if ( data->size() != 1 or data_out->size() != 4 or (data->at(0)).size() != nIntegrations or (data_out->at(0)).size() != nIntegrations or (data->at(0))[0].size() != nFrequencies or (data_out->at(0))[0].size() != 2 * nFrequencies or (data->at(0))[0][0].size() != nChannels * ( nChannels + 1 ) / 2  or (data_out->at(0))[0][0].size() != nAnt * ( nAnt + 1 ) / 2 ){
		//cout << "#!!#" << FILENAME << "#!!#" << METHODNAME << ": FATAL I/O MISMATCH! The input array and IQ array are initialized at (p, t, f, bl) = (" << data->size() << ", " << (data->at(0)).size() << ", " <<  (data->at(0))[0].size()  << ", " <<  (data->at(0))[0][0].size() << ") and (" << data_out->size() << ", " << (data_out->at(0)).size() << ", " <<  (data_out->at(0))[0].size()  << ", " <<  (data_out->at(0))[0][0].size() << "), where as the parameters are specified as (t, f, ant) = (" << nIntegrations << ", "  << nFrequencies << ", " << nAnt << "). Exiting!!" << endl;
		//return;
	//}
	//vector<vector<float> > *freq_slice;
	//int prevk, k1i, k1q, prevk1i, prevk1q, k2xi, k2xq, k2yi, k2yq, prevk2xi, prevk2yi, bl;
	//float a1xx_re, a1xx_im, a2xx_re, a2xx_im, a3xx_re, a3xx_im, a1xy_re, a1xy_im, a2xy_re, a2xy_im, a3xy_re, a3xy_im, a1yx_re, a1yx_im, a2yx_re, a2yx_im, a3yx_re, a3yx_im, a1yy_re, a1yy_im, a2yy_re, a2yy_im, a3yy_re, a3yy_im;
	//int c2nchan1 = 2 * nChannels - 1; //frequently used constant
	//for (int t = 0; t < nIntegrations; t++){
		////cout << t << endl;
		//for (int f = 0; f < nFrequencies; f++){
			//freq_slice = &((data->at(0))[t][f]);
			////loop for xx and xy
			//for (int k1 = 0; k1 < nAnt; k1++){
				//prevk = (2 * nAnt - k1 - 1) * k1 / 2;
				//k1i = 2*k1;
				//k1q = k1i + 2 * nAnt;
				//prevk1i = (c2nchan1 - k1i)*k1i/2;
				//prevk1q = (c2nchan1 - k1q)*k1q/2;
				//for (int k2 = k1; k2 < nAnt; k2++){
					//k2xi = 2 * k2;
					//k2xq = k2xi + 2 * nAnt;
					//k2yi = k2xi + 1;
					//k2yq = k2xq + 1;
					//prevk2xi = (c2nchan1 - k2xi) * k2xi / 2;
					//prevk2yi = (c2nchan1-k2yi) * k2yi / 2;
					//// performing complex arithmetic: 0 index --> real
					//// 1 index --> imag
					//a1xx_re = freq_slice->at(prevk1i+k2xi)[0] + freq_slice->at(prevk1q+k2xq)[0];
					//a1xx_im = freq_slice->at(prevk1i+k2xi)[1] + freq_slice->at(prevk1q+k2xq)[1];
					//a2xx_re = freq_slice->at(prevk1i+k2xq)[0] - freq_slice->at(prevk2xi+k1q)[0];
					//a2xx_im = freq_slice->at(prevk1i+k2xq)[1] + freq_slice->at(prevk2xi+k1q)[1];
					//a3xx_re = -1 * a2xx_im;
					//a3xx_im = a2xx_re;
					//a1xy_re = freq_slice->at(prevk1i+k2yi)[0] + freq_slice->at(prevk1q+k2yq)[0];
					//a1xy_im = freq_slice->at(prevk1i+k2yi)[1] + freq_slice->at(prevk1q+k2yq)[1];
					//a2xy_re = freq_slice->at(prevk1i+k2yq)[0] - freq_slice->at(prevk2yi+k1q)[0];
					//a2xy_im = freq_slice->at(prevk1i+k2yq)[1] + freq_slice->at(prevk2yi+k1q)[1];
					//a3xy_re = -1 * a2xy_im;
					//a3xy_im = a2xy_re;

					////writing to output matrix
					//bl = prevk + k2;
					//if (f == 0){
						//(data_out->at(0))[t][2*nFrequencies-1][bl][0] = ( a1xx_re + a3xx_re);
						//(data_out->at(0))[t][2*nFrequencies-1][bl][1] = -1*( a1xx_im + a3xx_im);
						//(data_out->at(1))[t][2*nFrequencies-1][bl][0] = (a1xy_re + a3xy_re);
						//(data_out->at(1))[t][2*nFrequencies-1][bl][1] = -1*(a1xy_im + a3xy_im);
					//}

					//(data_out->at(0))[t][nFrequencies-1+f][bl][0] = ( a1xx_re + a3xx_re);
					//(data_out->at(0))[t][nFrequencies-1+f][bl][1] = -1*( a1xx_im + a3xx_im);
					//(data_out->at(0))[t][nFrequencies-1-f][bl][0] = a1xx_re - a3xx_re;
					//(data_out->at(0))[t][nFrequencies-1-f][bl][1] = a1xx_im - a3xx_im;
					//(data_out->at(1))[t][nFrequencies-1+f][bl][0] = (a1xy_re + a3xy_re);
					//(data_out->at(1))[t][nFrequencies-1+f][bl][1] = -1*(a1xy_im + a3xy_im);
					//(data_out->at(1))[t][nFrequencies-1-f][bl][0] = a1xy_re - a3xy_re;
					//(data_out->at(1))[t][nFrequencies-1-f][bl][1] = a1xy_im - a3xy_im;
				//}
			//}
				////loop for yy and yx
				////computational difference: k1i = 2*k1 (+ 1)
			//for (int k1=0; k1 < nAnt; k1++){
				//prevk = (2*nAnt-k1-1)*k1/2;
				//k1i = 2*k1 + 1;
				//k1q = k1i + 2 * nAnt;
				//prevk1i = (c2nchan1 - k1i)*k1i/2;
				//prevk1q = (c2nchan1 - k1q)*k1q/2;
				//for (int k2=k1; k2 < nAnt; k2++){
					//k2xi = 2*k2;
					//k2xq = k2xi + 2*nAnt;
					//k2yi = k2xi + 1;
					//k2yq = k2xq + 1;
					//prevk2xi = (c2nchan1-k2xi)*k2xi/2;
					//prevk2yi = (c2nchan1-k2yi)*k2yi/2;
					//// performing complex arithmetic: 0 index --> real
					//// 1 index --> imag
					//a1yx_re = freq_slice->at(prevk1i+k2xi)[0] + freq_slice->at(prevk1q+k2xq)[0];
					//a1yx_im = freq_slice->at(prevk1i+k2xi)[1] + freq_slice->at(prevk1q+k2xq)[1];
					//a2yx_re = freq_slice->at(prevk1i+k2xq)[0] - freq_slice->at(prevk2xi+k1q)[0];
					//a2yx_im = freq_slice->at(prevk1i+k2xq)[1] + freq_slice->at(prevk2xi+k1q)[1];
					//a3yx_re = -1 * a2yx_im;
					//a3yx_im = a2yx_re;
					//a1yy_re = freq_slice->at(prevk1i+k2yi)[0] + freq_slice->at(prevk1q+k2yq)[0];
					//a1yy_im = freq_slice->at(prevk1i+k2yi)[1] + freq_slice->at(prevk1q+k2yq)[1];
					//a2yy_re = freq_slice->at(prevk1i+k2yq)[0] - freq_slice->at(prevk2yi+k1q)[0];
					//a2yy_im = freq_slice->at(prevk1i+k2yq)[1] + freq_slice->at(prevk2yi+k1q)[1];
					//a3yy_re = -1 * a2yy_im;
					//a3yy_im = a2yy_re;

					////writing to output matrix
					//bl = prevk + k2;
					//if (f == 0){
						//(data_out->at(2))[t][2*nFrequencies-1][bl][0] = ( a1yx_re + a3yx_re);
						//(data_out->at(2))[t][2*nFrequencies-1][bl][1] = -1*( a1yx_im + a3yx_im);
						//(data_out->at(3))[t][2*nFrequencies-1][bl][0] = (a1yy_re + a3yy_re);
						//(data_out->at(3))[t][2*nFrequencies-1][bl][1] = -1*(a1yy_im + a3yy_im);
					//}
					//(data_out->at(2))[t][nFrequencies-1+f][bl][0] = (a1yx_re + a3yx_re);
					//(data_out->at(2))[t][nFrequencies-1+f][bl][1] = -1*(a1yx_im + a3yx_im);
					//(data_out->at(2))[t][nFrequencies-1-f][bl][0] = a1yx_re - a3yx_re;
					//(data_out->at(2))[t][nFrequencies-1-f][bl][1] = a1yx_im - a3yx_im;
					//(data_out->at(3))[t][nFrequencies-1+f][bl][0] = (a1yy_re + a3yy_re);
					//(data_out->at(3))[t][nFrequencies-1+f][bl][1] = -1*(a1yy_im + a3yy_im);
					//(data_out->at(3))[t][nFrequencies-1-f][bl][0] = a1yy_re - a3yy_re;
					//(data_out->at(3))[t][nFrequencies-1-f][bl][1] = a1yy_im - a3yy_im;
				//}
			//}
		//}
	//}
	//return;
//}

//void iqDemodLarge(vector<vector<vector<vector<float> > > > *data, vector<vector<vector<vector<float> > > > *data_out, int nIntegrations, int nFrequencies, int nAnt){
	//string METHODNAME = "iqDemodLarge";
	//int nChannels = nAnt * 4; //a factor of 2 from consolidating x and y polarizations, and another factor of 2 from consolidating iq
	//int n_xxi = nAnt * (nAnt + 1)/2;

	//if ( data->size() != 1 or data_out->size() != 4 or (data->at(0)).size() != nIntegrations or (data_out->at(0)).size() != nIntegrations or (data->at(0))[0].size() != nFrequencies or (data_out->at(0))[0].size() != 2 * nFrequencies or (data->at(0))[0][0].size() != nChannels * ( nChannels + 1 ) or (data_out->at(0))[0][0].size() != nAnt * ( nAnt + 1 ) ){
		//cout << "#!!#" << FILENAME << "#!!#" << METHODNAME << ": FATAL I/O MISMATCH! The input array and IQ array are initialized at (p, t, f, bl) = (" << data->size() << ", " << (data->at(0)).size() << ", " <<  (data->at(0))[0].size()  << ", " <<  (data->at(0))[0][0].size() << ") and (" << data_out->size() << ", " << (data_out->at(0)).size() << ", " <<  (data_out->at(0))[0].size()  << ", " <<  (data_out->at(0))[0][0].size() << "), where as the parameters are specified as (t, f_in, f_out, bl_in, bl_out) = (" << nIntegrations << ", "  << nFrequencies << ", "  << 2 * nFrequencies << ", " << nChannels * ( nChannels + 1 ) << ", " << nAnt * ( nAnt + 1 ) << "). Exiting!!" << endl;
		//return;
	//}
	//vector<float> *freq_slice;
	//int prevk, k1i, k1q, prevk1i, prevk1q, k2xi, k2xq, k2yi, k2yq, prevk2xi, prevk2yi, bl;
	//float a1xx_re, a1xx_im, a2xx_re, a2xx_im, a3xx_re, a3xx_im, a1xy_re, a1xy_im, a2xy_re, a2xy_im, a3xy_re, a3xy_im, a1yx_re, a1yx_im, a2yx_re, a2yx_im, a3yx_re, a3yx_im, a1yy_re, a1yy_im, a2yy_re, a2yy_im, a3yy_re, a3yy_im;
	//int c2nchan1 = 2 * nChannels - 1; //frequently used constant
	//for (int t = 0; t < nIntegrations; t++){
		////cout << t << endl;
		//for (int f = 0; f < nFrequencies; f++){
			//freq_slice = &((data->at(0))[t][f]);
			////loop for xx and xy
			//for (int k1 = 0; k1 < nAnt; k1++){
				//prevk = (2 * nAnt - k1 - 1) * k1 / 2;
				//k1i = 2*k1;
				//k1q = k1i + 2 * nAnt;
				//prevk1i = (c2nchan1 - k1i)*k1i/2;
				//prevk1q = (c2nchan1 - k1q)*k1q/2;
				//for (int k2 = k1; k2 < nAnt; k2++){
					//k2xi = 2 * k2;
					//k2xq = k2xi + 2 * nAnt;
					//k2yi = k2xi + 1;
					//k2yq = k2xq + 1;
					//prevk2xi = (c2nchan1 - k2xi) * k2xi / 2;
					//prevk2yi = (c2nchan1-k2yi) * k2yi / 2;
					//// performing complex arithmetic: 0 index --> real
					//// 1 index --> imag
					//a1xx_re = freq_slice->at(gc(prevk1i+k2xi, 0)) + freq_slice->at(gc(prevk1q+k2xq, 0));
					//a1xx_im = freq_slice->at(gc(prevk1i+k2xi, 1)) + freq_slice->at(gc(prevk1q+k2xq, 1));
					//a2xx_re = freq_slice->at(gc(prevk1i+k2xq, 0)) - freq_slice->at(gc(prevk2xi+k1q, 0));
					//a2xx_im = freq_slice->at(gc(prevk1i+k2xq, 1)) + freq_slice->at(gc(prevk2xi+k1q, 1));
					//a3xx_re = -1 * a2xx_im;
					//a3xx_im = a2xx_re;
					//a1xy_re = freq_slice->at(gc(prevk1i+k2yi, 0)) + freq_slice->at(gc(prevk1q+k2yq, 0));
					//a1xy_im = freq_slice->at(gc(prevk1i+k2yi, 1)) + freq_slice->at(gc(prevk1q+k2yq, 1));
					//a2xy_re = freq_slice->at(gc(prevk1i+k2yq, 0)) - freq_slice->at(gc(prevk2yi+k1q, 0));
					//a2xy_im = freq_slice->at(gc(prevk1i+k2yq, 1)) + freq_slice->at(gc(prevk2yi+k1q, 1));
					//a3xy_re = -1 * a2xy_im;
					//a3xy_im = a2xy_re;

					////writing to output matrix
					//bl = prevk + k2;
					//if (f == 0){
						//(data_out->at(0))[t][2*nFrequencies-1][gc(bl, 0)] = ( a1xx_re + a3xx_re);
						//(data_out->at(0))[t][2*nFrequencies-1][gc(bl, 1)] = -1*( a1xx_im + a3xx_im);
						//(data_out->at(1))[t][2*nFrequencies-1][gc(bl, 0)] = (a1xy_re + a3xy_re);
						//(data_out->at(1))[t][2*nFrequencies-1][gc(bl, 1)] = -1*(a1xy_im + a3xy_im);
					//}

					//(data_out->at(0))[t][nFrequencies-1+f][gc(bl, 0)] = ( a1xx_re + a3xx_re);
					//(data_out->at(0))[t][nFrequencies-1+f][gc(bl, 1)] = -1*( a1xx_im + a3xx_im);
					//(data_out->at(0))[t][nFrequencies-1-f][gc(bl, 0)] = a1xx_re - a3xx_re;
					//(data_out->at(0))[t][nFrequencies-1-f][gc(bl, 1)] = a1xx_im - a3xx_im;
					//(data_out->at(1))[t][nFrequencies-1+f][gc(bl, 0)] = (a1xy_re + a3xy_re);
					//(data_out->at(1))[t][nFrequencies-1+f][gc(bl, 1)] = -1*(a1xy_im + a3xy_im);
					//(data_out->at(1))[t][nFrequencies-1-f][gc(bl, 0)] = a1xy_re - a3xy_re;
					//(data_out->at(1))[t][nFrequencies-1-f][gc(bl, 1)] = a1xy_im - a3xy_im;
				//}
			//}
				////loop for yy and yx
				////computational difference: k1i = 2*k1 (+ 1)
			//for (int k1=0; k1 < nAnt; k1++){
				//prevk = (2*nAnt-k1-1)*k1/2;
				//k1i = 2*k1 + 1;
				//k1q = k1i + 2 * nAnt;
				//prevk1i = (c2nchan1 - k1i)*k1i/2;
				//prevk1q = (c2nchan1 - k1q)*k1q/2;
				//for (int k2=k1; k2 < nAnt; k2++){
					//k2xi = 2*k2;
					//k2xq = k2xi + 2*nAnt;
					//k2yi = k2xi + 1;
					//k2yq = k2xq + 1;
					//prevk2xi = (c2nchan1-k2xi)*k2xi/2;
					//prevk2yi = (c2nchan1-k2yi)*k2yi/2;
					//// performing complex arithmetic: 0 index --> real
					//// 1 index --> imag
					//a1yx_re = freq_slice->at(gc(prevk1i+k2xi, 0)) + freq_slice->at(gc(prevk1q+k2xq, 0));
					//a1yx_im = freq_slice->at(gc(prevk1i+k2xi, 1)) + freq_slice->at(gc(prevk1q+k2xq, 1));
					//a2yx_re = freq_slice->at(gc(prevk1i+k2xq, 0)) - freq_slice->at(gc(prevk2xi+k1q, 0));
					//a2yx_im = freq_slice->at(gc(prevk1i+k2xq, 1)) + freq_slice->at(gc(prevk2xi+k1q, 1));
					//a3yx_re = -1 * a2yx_im;
					//a3yx_im = a2yx_re;
					//a1yy_re = freq_slice->at(gc(prevk1i+k2yi, 0)) + freq_slice->at(gc(prevk1q+k2yq, 0));
					//a1yy_im = freq_slice->at(gc(prevk1i+k2yi, 1)) + freq_slice->at(gc(prevk1q+k2yq, 1));
					//a2yy_re = freq_slice->at(gc(prevk1i+k2yq, 0)) - freq_slice->at(gc(prevk2yi+k1q, 0));
					//a2yy_im = freq_slice->at(gc(prevk1i+k2yq, 1)) + freq_slice->at(gc(prevk2yi+k1q, 1));
					//a3yy_re = -1 * a2yy_im;
					//a3yy_im = a2yy_re;

					////writing to output matrix
					//bl = prevk + k2;
					//if (f == 0){
						//(data_out->at(2))[t][2*nFrequencies-1][gc(bl, 0)] = ( a1yx_re + a3yx_re);
						//(data_out->at(2))[t][2*nFrequencies-1][gc(bl, 1)] = -1*( a1yx_im + a3yx_im);
						//(data_out->at(3))[t][2*nFrequencies-1][gc(bl, 0)] = (a1yy_re + a3yy_re);
						//(data_out->at(3))[t][2*nFrequencies-1][gc(bl, 1)] = -1*(a1yy_im + a3yy_im);
					//}
					//(data_out->at(2))[t][nFrequencies-1+f][gc(bl, 0)] = (a1yx_re + a3yx_re);
					//(data_out->at(2))[t][nFrequencies-1+f][gc(bl, 1)] = -1*(a1yx_im + a3yx_im);
					//(data_out->at(2))[t][nFrequencies-1-f][gc(bl, 0)] = a1yx_re - a3yx_re;
					//(data_out->at(2))[t][nFrequencies-1-f][gc(bl, 1)] = a1yx_im - a3yx_im;
					//(data_out->at(3))[t][nFrequencies-1+f][gc(bl, 0)] = (a1yy_re + a3yy_re);
					//(data_out->at(3))[t][nFrequencies-1+f][gc(bl, 1)] = -1*(a1yy_im + a3yy_im);
					//(data_out->at(3))[t][nFrequencies-1-f][gc(bl, 0)] = a1yy_re - a3yy_re;
					//(data_out->at(3))[t][nFrequencies-1-f][gc(bl, 1)] = a1yy_im - a3yy_im;
				//}
			//}
		//}
	//}
	//return;
//}

int gc(int a, int b){
	return 2 * a + b;
}


int get1DBL(int i, int j, int nAntenna){//0 indexed
	int output;
	if (i <= j) {
		output = ( ( 2 * nAntenna - 1 - i ) * i / 2 + j );
	} else {
		output = ( ( 2 * nAntenna - 1 - j ) * j / 2 + i );
	}
	return output;
}


vector<int> get2DBL(int bl, int nAntenna){//bl counts cross corrs AND auto corrs
	if(bl < nAntenna){
		vector<int> v(2);
		v[0] = 0;
		v[1] = bl;
		return v;
	} else{
		vector<int> v;
		v = get2DBL(bl-nAntenna, nAntenna-1);
		v[0] = v[0] + 1;
		v[1] = v[1] + 1;
		return v;
	}
	vector<int> v(2, -1);
	return v;
}

vector<int> get2DBLCross(int bl, int nAntenna){//bl only counts cross corrs
	if(bl < nAntenna - 1){
		vector<int> v(2);
		v[0] = 0;
		v[1] = bl + 1;
		return v;
	} else{
		vector<int> v;
		v = get2DBLCross(bl - nAntenna + 1, nAntenna - 1);
		v[0] = v[0] + 1;
		v[1] = v[1] + 1;
		return v;
	}
	vector<int> v(2, -1);
	return v;
}

bool contains(vector<vector<float> > * UBL, vector<float> bl){//automatically checks for the opposite direction
	for (unsigned int i = 0; i < UBL->size(); i++){
		if ( ( fabs((&(UBL->at(i)))->at(0) - bl[0]) < UBLPRECISION && fabs((&(UBL->at(i)))->at(1) - bl[1]) < UBLPRECISION ) or ( fabs((&(UBL->at(i)))->at(0) + bl[0]) < UBLPRECISION && fabs((&(UBL->at(i)))->at(1) + bl[1]) < UBLPRECISION ) ){
			return true;
		}
	}
	return false;
}

int indexUBL(vector<vector<float> > * UBL, vector<float> bl){//give the 1-indexed index of a baseline inside the unique baseline list; the opposite direction will give -index
	for (unsigned int i = 0; i < UBL->size(); i++){
		if ( fabs((&(UBL->at(i)))->at(0) - bl[0]) < UBLPRECISION && fabs((&(UBL->at(i)))->at(1) - bl[1]) < UBLPRECISION ) {
			return 1+i;
		} else if ( fabs((&(UBL->at(i)))->at(0) + bl[0]) < UBLPRECISION && fabs((&(UBL->at(i)))->at(1) + bl[1]) < UBLPRECISION ){
			return -1-i;
		}
	}
	return 0;
}


bool contains_int(vector<int> * list, int j){
	for (uint i = 0; i < list->size(); i ++){
		if ( list->at(i) == j ){
			return true;
		}
	}
	return false;
}

void addPhase(vector<float> * x, float phi){
	float am = amp(x);
	float ph = phase(x->at(0), x->at(1));
	ph = phaseWrap(ph + phi);
	x->at(0) = am * cos(ph);
	x->at(1) = am * sin(ph);
	return;
}


float vectorDot(vector<float>* v1, vector<float>* v2){//v1.v2
	string METHODNAME = "vectorDot";
	if ( v1->size() != v2->size() ){
		cout << "#!!#" << FILENAME << "#!!#" << METHODNAME << ": FATAL INPUT MISMATCH! lengths of vectors are " << v1->size() << " and " << v2->size() << ". 0 returned!!!" << endl;
		return 0;
	}
	double sum = 0;
	for (unsigned int i = 0; i < v1->size(); i ++){
		sum += (v1->at(i) * v2->at(i));
	}
	return float(sum);
}


vector<float> matrixDotV(vector<vector<float> >* m, vector<float>* v){//m.v
	string METHODNAME = "maxtrixDotV";
	vector<float> u(m->size(), 0);
	if ( (m->at(0)).size() != v->size() ){
		cout << "#!!#" << FILENAME << "#!!#" << METHODNAME << ": FATAL INPUT MISMATCH! Dimensions of matrix and of vector are " << m->size() << "x" << (m->at(0)).size() << " and " << v->size() << ". 0 vector  returned!!!" << endl;
		return u;
	}
	for (unsigned int i = 0; i < m->size(); i ++){
		u[i] = vectorDot( &(m->at(i)), v);
	}
	return u;
}

vector<vector<float> > rotationMatrix(float x, float y, float z){//approximation for a rotation matrix rotating around x,y,z axis, {{1, -z, y}, {z, 1, -x}, {-y, x, 1}}
	vector<vector<float> > r(3, vector<float>(3,1));
	r[0][1] =-z;
	r[0][2] = y;
	r[1][0] = z;
	r[1][2] =-x;
	r[2][0] =-y;
	r[2][1] = x;
	return r;
}

vector<vector<float> > rotationMatrixZ(float z){/*approximation for a rotation matrix rotating around x,y,z axis, Ry[roty_] := {{Cos[roty], 0, Sin[roty]}, {0, 1, 0}, {-Sin[roty], 0,
    Cos[roty]}};
Rx[roty_] := {{1, 0, 0}, {0, Cos[roty], -Sin[roty]}, {0, Sin[roty],
    Cos[roty]}};
Rz[rotz_] := {{Cos[rotz], -Sin[rotz], 0}, {Sin[rotz], Cos[rotz],
    0}, {0, 0, 1}};*/
	vector<vector<float> > r(3, vector<float>(3, 0));
	r[0][0] = cos(z);
	r[0][1] =-sin(z);
	r[1][0] = sin(z);
	r[1][1] = cos(z);
	r[2][2] = 1;

	return r;
}


vector<float> getBL(int i, int j, vector<vector<float> > *antloc){
	vector<float> bl(2,0);
	bl[0] = (&(antloc->at(j)))->at(0) - (&(antloc->at(i)))->at(0);
	bl[1] = (&(antloc->at(j)))->at(1) - (&(antloc->at(i)))->at(1);
	return bl;
}

int countUBL(vector<vector<float> > *antloc ){
	vector<float> bl;
	vector<vector<float> > UBL;
	for (unsigned int i = 0; i < antloc->size() - 1; i++){
		for (unsigned int j = i + 1; j < antloc->size(); j++){
			bl = getBL(i, j, antloc);
			if (!contains(&UBL, bl)) {
				UBL.push_back(bl);
			}
		}
	}
	return UBL.size();
}

int lookupAnt(float x, float y, vector<vector<float> > antloc){
	for (unsigned int i = 0; i < antloc.size(); i++){
		if ( x == antloc[i][0] && y == antloc[i][1]){
			return i;
		}
	}
	return -1;
}

void computeUBL(vector<vector<float> > * antloc, vector<vector<float> > * listUBL){
	int numAntenna = antloc->size();
	vector<float> baseline(2);
	vector<vector<float> > UBLtmp;
	for (int i = 0; i < numAntenna; i++){
		for (int j = i + 1; j < numAntenna; j++){
			baseline = getBL(i, j, antloc);
			if ( ! (contains(&UBLtmp, baseline)) ) {
				UBLtmp.push_back(baseline);
			}
		}
	}
	for (unsigned int i = 0; i < UBLtmp.size(); i++){
		listUBL->at(i) = UBLtmp[i];
	}
	return;
}

vector<float> modelToMeasurement(vector<float> *modelCor, float ampcal, float phasecal){
	string METHODNAME = "modelToMeasurement";
	vector<float> measurement (2, 0.0);
	float modelAmp = sqrt( modelCor->at(0) * modelCor->at(0) + modelCor->at(1) * modelCor->at(1) );
	float modelPhase = phase( modelCor->at(0), modelCor->at(1) );
	float measurementAmp = modelAmp * pow( 10, min(MAX_10_POW,ampcal) );
	float measurementPhase = phaseWrap( modelPhase + phasecal );

	//cout <<  "#!#" << FILENAME << "#!#" << METHODNAME << ": " << "modelAmp: " << modelAmp << " modelPhase: " << modelPhase << " measurementAmp: " << measurementAmp << " measurementPhase: " << measurementPhase << " ampcal: " << ampcal << " phasecal: " << phasecal << endl;
	measurement[0] = measurementAmp * cos(measurementPhase);
	measurement[1] = measurementAmp * sin(measurementPhase);
	return measurement;
}

void computeUBLcor(vector<vector<float> >* calibratedData, vector<int> *UBLindex, vector<vector<float> > *UBLcor, vector<bool> *goodAnt){//average each group of calibrated redundant baselines to get the estimate for that ubl, only useful when directly applying a set of calpars instead of using logcal to fit for them.
}

vector<float> getModel(int i, int j, vector<vector<float> > *antloc, vector<vector<float> > *listUBL, vector<vector<float> > *UBLcor){
	vector<float> baseline = getBL(i, j, antloc);
	for (uint k = 0; k < listUBL->size(); k++){
		if ( baseline[0] == (&(listUBL->at(k)))->at(0) && baseline[1] == (&(listUBL->at(k)))->at(1) ){
			return UBLcor->at(k);
		} else	if ( baseline[0] == -(&(listUBL->at(k)))->at(0) && baseline[1] == -(&(listUBL->at(k)))->at(1) ){
			return conjugate( UBLcor->at(k) );
		}
	}
	return vector<float>(UBLcor->at(0).size(), 0);
}

vector<vector<float> > ReverseEngineer(vector<float> * ampcalpar, vector<float> * phasecalpar, vector<vector<float> > * UBLcor, vector<vector<float> > * antloc, vector<vector<float> > * listUBL){
	string METHODNAME = "ReverseEngineer";
	int numAntenna = antloc->size();
	int numCrosscor = numAntenna * ( numAntenna - 1 ) / 2;
	vector<float> complexdummy (2, 0.0);//Just an awkward way to initialize output
	vector<vector<float> > output (numCrosscor, complexdummy);

	int cnter = 0;
	for (int i = 0; i < numAntenna; i++){
		for (int j = i + 1; j < numAntenna; j++){
			vector<float> cor(2, 0.0);
			cor = getModel(i, j, antloc, listUBL, UBLcor);
			output[cnter] = modelToMeasurement( &cor, (ampcalpar->at(i) + ampcalpar->at(j)), (phasecalpar->at(j) - phasecalpar->at(i)) );
			cnter ++;
		}
	}
	return output;
}

void ReverseEngineer(vector<vector<float> >* output, vector<float> * calpar, int numAntenna, vector<int> * UBLindex){
	string METHODNAME = "ReverseEngineer";
	//int numCrosscor = numAntenna * ( numAntenna - 1 ) / 2;
	int cnter = 0;
	vector<float> cor(2, 0.0);
	for (int i = 0; i < numAntenna; i++){
		for (int j = i + 1; j < numAntenna; j++){
			int ubl = fabs(UBLindex->at(cnter)) - 1;
			if(UBLindex->at(cnter) > 0){
				cor[0] = calpar->at(3 + 2 * numAntenna + 2 * ubl);
				cor[1] = calpar->at(3 + 2 * numAntenna + 2 * ubl + 1);
			} else{
				cor[0] = calpar->at(3 + 2 * numAntenna + 2 * ubl);
				cor[1] = -calpar->at(3 + 2 * numAntenna + 2 * ubl + 1);
			}
			output->at(cnter) = modelToMeasurement( &cor, (calpar->at(3 + i) + calpar->at(3 + j)), (calpar->at(3 + numAntenna + j) - calpar->at(3 + numAntenna + i)) );
			cnter ++;
		}
	}
	return;
}

float chiSq(vector<vector<float> > * dataf, vector<vector<float> > * sdevf, vector<vector<float> > * antloc, vector<float> * ampcalpar, vector<float> * phasecalpar, vector<vector<float> > * UBLcor, int numAntenna, vector<vector<float> > * listUBL){
	string METHODNAME = "chiSq";
	uint numCrosscor = numAntenna * ( numAntenna - 1 ) / 2;
	uint numAutocor = numAntenna * ( numAntenna + 1 ) / 2;
	if ( dataf->size() != numAutocor) {
		cout << "#!#" << FILENAME << "#!#" << METHODNAME << ": !!!!FATAL ERROR!!!! Length of data is " << dataf->size() << ", not consistent with expected " << numAutocor << " from specified " << numAntenna << " antenna!" << endl;
		return 0.0;
	}
	if ( dataf->size() != sdevf->size()) {
		cout << "#!#" << FILENAME << "#!#" << METHODNAME << ": !!!!FATAL ERROR!!!! Length of data is " << dataf->size() << ", not consistent with length of standard deviation input " << sdevf->size() << "!" << endl;
		return 0.0;
	}
	float output = 0.0;
	vector<float> bl(2);

	vector<vector<float> > Ax(numCrosscor, bl);
	vector<vector<float> > y(numCrosscor, bl);
	vector<vector<float> > N(numCrosscor, bl);// Here N is technicaly n, noise in units of sdev, not the N covariant matrix

	int cnter = 0;
	for ( int i = 0; i < numAntenna; i++){
		for ( int j = i + 1; j < numAntenna; j++){
			int index = get1DBL(i, j, numAntenna);
			y[cnter] = dataf->at(index);
			N[cnter] = sdevf->at(index);
			cnter ++;
		}
	}

	Ax = ReverseEngineer(ampcalpar, phasecalpar, UBLcor, antloc, listUBL);

	for (uint i = 0; i < numCrosscor; i ++){
		output = output + square(( Ax[i][0] - y[i][0] ) / max(N[i][0], MIN_NONE_ZERO)) + square( ( Ax[i][1] - y[i][1] ) / max(N[i][1], MIN_NONE_ZERO));
	}

	return output;
}

bool fillChiSq(vector<vector<float> >* dataf, vector<vector<float> >* sdevf, vector<float>* calpar, int numAntenna, vector<int>* UBLindex, vector<bool>* goodAnt){
	string METHODNAME = "fillChiSq";
	uint numCrosscor = numAntenna * ( numAntenna - 1 ) / 2;
	uint numAutocor = numAntenna * ( numAntenna + 1 ) / 2;
	if ( dataf->size() != numAutocor) {
		cout << "#!#" << FILENAME << "#!#" << METHODNAME << ": !!!!FATAL ERROR!!!! Length of data is " << dataf->size() << ", not consistent with expected " << numAutocor << " from specified " << numAntenna << " antenna!" << endl;
		return false;
	}
	if ( dataf->size() != sdevf->size()) {
		cout << "#!#" << FILENAME << "#!#" << METHODNAME << ": !!!!FATAL ERROR!!!! Length of data is " << dataf->size() << ", not consistent with length of standard deviation input " << sdevf->size() << "!" << endl;
		return false;
	}
	float output = 0.0;
	vector<float> bl(2);

	vector<vector<float> > Ax(numCrosscor, bl);
	ReverseEngineer(&Ax, calpar, numAntenna, UBLindex);

	for (uint i = 0; i < numCrosscor; i ++){
		vector<int> a = get2DBLCross(i, numAntenna);
		if(goodAnt->at(a[0]) and goodAnt->at(a[1])){
			int blauto = get1DBL(a[0], a[1], numAntenna);
			//cout << a[0] << " " << a[1] << " (" << Ax[i][0] << "," << Ax[i][1] << ")" << "(" << dataf->at(blauto)[0] << "," << dataf->at(blauto)[1] << ") " << square(( Ax[i][0] - dataf->at(blauto)[0] ) / max(sdevf->at(blauto)[0], MIN_NONE_ZERO)) + square( ( Ax[i][1] - dataf->at(blauto)[1] ) / max(sdevf->at(blauto)[1], MIN_NONE_ZERO)) << endl;
			output = output + square(( Ax[i][0] - dataf->at(blauto)[0] ) / max(sdevf->at(blauto)[0], MIN_NONE_ZERO)) + square( ( Ax[i][1] - dataf->at(blauto)[1] ) / max(sdevf->at(blauto)[1], MIN_NONE_ZERO));
		}
	}
	calpar->at(0) = output;
	return true;
}

///////////////MAJOR STUFF///////////////////
/////////////////////////////////////////////

void substractComplexPhase(vector<float> *a, vector<float> *b, float angle){
	float amptmp = amp(a);
	float phasetmp = phase(a->at(0), a->at(1));
	phasetmp = phaseWrap( phasetmp - angle );
	b->at(0) = amptmp * cos(phasetmp);
	b->at(1) = amptmp * sin(phasetmp);
	return;
}


//Logcal functions

///////////////REDUNDANT BASELINE CALIBRATION STUFF///////////////////
/////////////////////////////////////////////


/******************************************************/
/******************************************************/

vector<float> minimizecomplex(vector<vector<float> >* a, vector<vector<float> >* b){
	vector<float> sum1(2, 0);
	for (uint i =0; i < a->size(); i++){
		sum1[0] += a->at(i)[0] * b->at(i)[0] + a->at(i)[1] * b->at(i)[1];
		sum1[1] += a->at(i)[1] * b->at(i)[0] - a->at(i)[0] * b->at(i)[1];
	}
	float sum2 = pow(norm(b), 2);
	sum1[0] = sum1[0] / sum2;
	sum1[1] = sum1[1] / sum2;
	return sum1;
}

void runAverage1d(vector<float> *in, vector<float> *out, uint w){//compute running average with running length 2w+1. The first and last w elements are averaged with less elements.
	string METHODNAME = "runAverage1d";
	if(in->size() != out->size()){
		printf("#!!#%s#!!#%s: FATAL ERROR: input and output arrays have different dimensions: %lu vs %lu. ABORT!\n", FILENAME.c_str(), METHODNAME.c_str(), in->size(), out->size());
		return;
	}
	uint l = in->size();
	double sum = 0;
	uint n = 0;//number of items in sum
	for (uint i = 0; i < min(w, l); i++){
		sum += in->at(i);
		n++;
	}
	for (uint i = 0; i < out->size(); i++){
		if(i + w < l){
			sum += in->at(i + w);
			n++;
		}
		if(i - w -1 >= 0){
			sum += -(in->at(i - w - 1));
			n += -1;
		}
		out->at(i) = float(sum / n);
	}
}
void runAverage(vector<vector<vector<vector<float> > > > *in, int dimension, int w){//compute running average along dimension with running length 2w+1. The first and last w elements are averaged with less elements. Input array is modified!
	string METHODNAME = "runAverage";
	//if(in->size() != out->size() or in->at(0).size() != out->at(0).size()){
		//printf("#!!#%s#!!#%s: FATAL ERROR: input and output arrays have different dimensions: (%i, %i) vs (%i, %i). ABORT!\n", FILENAME, METHODNAME.c_str(), in->size(), in->at(0).size(), out->size(), out->at(0).size());
		//return;
	//}
	vector<float> dummy, dummy2;
	switch(dimension){
		case 0:
			dummy = vector<float>(in->size(), 0);
			break;
		case 1:
			dummy = vector<float>(in->at(0).size(), 0);
			break;
		case 2:
			dummy = vector<float>(in->at(0)[0].size(), 0);
			break;
		case 3:
			dummy = vector<float>(in->at(0)[0][0].size(), 0);
			break;
		default:
			printf("#!!#%s#!!#%s: FATAL ERROR: input array does not contain dimension %i. ABORT!\n", FILENAME.c_str(), METHODNAME.c_str(), dimension);
			return;
			break;
	}
	dummy2 = dummy;


	for (unsigned int t = 0; t < in->size(); t++){
		for (unsigned int f = 0; f < in->at(0).size(); f++){
			for (unsigned int b = 0; b < in->at(0)[0].size(); b++){
				for (unsigned int ri = 0; ri < in->at(0)[0][0].size(); ri++){
					switch(dimension){
						case 0:
							for (unsigned int i = 0; i < dummy.size(); i ++){
								dummy[i] = in->at(i)[f][b][ri];
							}
								runAverage1d(&dummy, &dummy2, w);
							for (unsigned int i = 0; i < dummy.size(); i ++){
								in->at(i)[f][b][ri] = dummy2[i];
							}
							break;
						case 1:
							for (unsigned int i = 0; i < dummy.size(); i ++){
								dummy[i] = in->at(t)[i][b][ri];
							}
								runAverage1d(&dummy, &dummy2, w);
							for (unsigned int i = 0; i < dummy.size(); i ++){
								in->at(t)[i][b][ri] = dummy2[i];
							}
							break;
						case 2:
							for (unsigned int i = 0; i < dummy.size(); i ++){
								dummy[i] = in->at(t)[f][i][ri];
							}
								runAverage1d(&dummy, &dummy2, w);
							for (unsigned int i = 0; i < dummy.size(); i ++){
								in->at(t)[f][i][ri] = dummy2[i];
							}
							break;
						case 3:
							for (unsigned int i = 0; i < dummy.size(); i ++){
								dummy[i] = in->at(t)[f][b][i];
							}
								runAverage1d(&dummy, &dummy2, w);
							for (unsigned int i = 0; i < dummy.size(); i ++){
								in->at(t)[f][b][i] = dummy2[i];
							}
							break;
					}
					if (dimension == 3) break;
				}
				if (dimension == 2) break;
			}
			if (dimension == 1) break;
		}
		if (dimension == 0) break;
	}

}

