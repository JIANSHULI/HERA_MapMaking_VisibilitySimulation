#ifndef OMNICAL_REDCAL_H
#define OMNICAL_REDCAL_H

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
#include <algorithm>
#include <functional>
#include <numeric>
#define uint unsigned int
using namespace std;

struct redundantinfo{
	int nAntenna;//number of good antennas among all (64) antennas, same as the length of subsetant
	//int nUBL;//number of unique baselines
	//int nBaseline;
	//int nCross;
	//vector<int> subsetant;//the index of good antennas in all (64) antennas, currently not used
	//vector<vector<float> > antloc;//3d antloc for each good antenna. strongly recommend using idealized antloc rather than literal ones
	//vector<int> subsetbl;//the index of good baselines (auto included) in all baselines
	//vector<vector<float> > ubl;//unique baseline vectors
	vector<int> bltoubl;//cross bl number to ubl index
	//vector<int> reversed;//cross only bl if reversed -1, otherwise 1
	//vector<int> reversedauto;//auto included bl if reversed -1, otherwise 1, currently not used
	//vector<int> autoindex;//index of auto bls among good bls, currently not used
	//vector<int> crossindex;//index of cross bls among good bls
	vector<vector<int> > bl2d;//from 1d bl to a pair of antenna numbers, (0,0), (0,1) (0,2) etc
	//vector<vector<int> > totalVisibilityId;//totalVisibilityId
	vector<int> ublcount;//for each ubl, the number of good cross bls corresponding to it
    // ublindex looks like this:
    // 0: [(i0a,j0a,k0a), (i0b,j0b,k0b), ... ]
    // 1: [(i1a,j1a,k1a), (i1b,j1b,k1b), ... ]
    // and so on, with i,j = ant indices, and k = index of this bl in bl2dmatrix?
    // the front index is the unique baseline index
	vector<vector<int> > ublindex;//for each ubl, the vector<int> contains the index of a baseline in bl2d
	vector<vector<int> > bl1dmatrix;//a symmetric matrix where col/row numbers are antenna indices and entries are 1d baseline index not counting auto corr
	vector<vector<float> > degenM;//degenM.(phase calibrations as a vector of nAnt) will generate a vector of length (nAnt + nUBL) which, when added to the existing calibration parameters' phases, will remove the linear phase field
	//vector<vector<int> > A;//A matrix for logcal amplitude
	//vector<vector<int> > B;//B matrix for logcal phase
	vector<vector<int> > Atsparse;//At matrix for logcal amplitude, in sparse form, all entries are one
	vector<vector<vector<int> > > Btsparse;//Bt matrix for logcal phase, entries can be -1 or one so there's one more dimension than Atsparse
	vector<vector<float> > AtAi;//(AtA)^-1
	vector<vector<float> > BtBi;//(BtB)^-1
	//vector<vector<float> > AtAiAt;//(AtA)^-1At
	//vector<vector<float> > BtBiBt;//(BtB)^-1Bt
	//vector<vector<float> > PA;//A(AtA)^-1At
	//vector<vector<float> > PB;//B(BtB)^-1Bt
	//vector<vector<float> > ImPA;//I-PA
	//vector<vector<float> > ImPB;//I-PB

};

struct calmemmodule{//temporary memory modules for logcaladditive and lincal
	vector<float> amp1;
	vector<float> amp2;
	vector<float> amp3;
	vector<float> pha1;
	vector<float> pha2;
	vector<float> pha3;
	vector<float> x1;
	vector<float> x2;
	vector<float> x3;
	vector<float> x4;
	vector<vector<float> > g0;
	vector<vector<float> > g1;
	vector<vector<float> > g2;
	vector<vector<float> > g3;
	vector<vector<float> > adata1;
	vector<vector<float> > adata2;
	vector<vector<float> > adata3;
	vector<vector<float> > cdata1;
	vector<vector<float> > cdata2;
	vector<vector<float> > cdata3;
	vector<vector<float> > ubl1;//nubl by 2
	vector<vector<float> > ubl2;
	vector<vector<float> > ubl3;
	vector<vector<float> > ubl0;
	vector<vector<float> > ublgrp1;//regrouped baseline according to ubl, second dimension of unequal length of ubl count
	vector<vector<float> > ublgrp2;
	vector<vector<vector<float> > > ubl2dgrp1;//regrouped baseline according to ubl, second dimension of unequal length of ubl count
	vector<vector<vector<float> > > ubl2dgrp2;
};

//bool readredundantinfo(string filepath, redundantinfo* info);
void initcalmodule(calmemmodule* module, redundantinfo* info);

////void iqDemod(vector<vector<vector<vector<vector<float> > > > > *data, vector<vector<vector<vector<vector<float> > > > > *data_out, int nIntegrations, int nFrequencies, int nAnt);

////void iqDemodLarge(vector<vector<vector<vector<float> > > > *data, vector<vector<vector<vector<float> > > > *data_out, int nIntegrations, int nFrequencies, int nAnt);//For large set of data (more than ~200 time slices), it takes more than 24G of memory to allocate a large enough 5D vector. So we consolidate the last two dimensions (baseline and re/im) together

/*void iqDemodBinary(string data, string data_out, int nIntegrations, int nFrequencies, int nAnt);//Used for large odfs for which there's not enough memory to take in all the data, so we have to read and iq and write without saving too much into memory*/

float square(float x);

float amp(vector<float> * x);

float amp(float x, float y);

float phase(float re, float im);

float phase(vector<float> * c);

float norm(vector<vector<float> > * v);

vector<float> conjugate (vector<float> x);

///////////////PHASE CALIBRATE STUFF///////////////////
/////////////////////////////////////////////

float phaseWrap (float x, float offset = -atan2(0,-1));//Wrap phase to be on (offset, offset+2pi]



///////////////CHI SQUARE STUFF///////////////////
/////////////////////////////////////////////


///////////////POINT SOURCE STUFF///////////////////
/////////////////////////////////////////////
float median (vector<float> list); //Not using pointer because the process involves sorting which will modify the list, which may be bad

float medianAngle (vector<float> *list); //Not using pointer because the process involves sorting which will modify the list, which may be bad

float mean (vector<float> *v, int start = -1, int end = -1);// take mean from start to end indices of vector v. 0 indexed

vector<float> stdev(vector<float> *v);//returns {mean, sample standard deviation}. Created by Hrant and modified by Jeff

float meanAngle (vector<float> *list);

vector<float> stdevAngle(vector<float> *v);

bool invert(vector<vector<int> > * AtNinvAori, vector<vector<double> > * AtNinvAinv );
bool invert(vector<vector<float> > * AtNinvAori, vector<vector<double> > * AtNinvAinv );
///////////////REDUNDANT BASELINE CALIBRATION STUFF///////////////////
/////////////////////////////////////////////

void vecmatmul(vector<vector<double> > * Afitting, vector<float> * v, vector<float> * ampfit);
void vecmatmul(vector<vector<float> > * Afitting, vector<float> * v, vector<float> * ampfit);
void logcaladd(vector<vector<float> >* data, vector<vector<float> >* additivein, redundantinfo* info, vector<float>* calpar, vector<vector<float> >* additiveout, int computeUBLFit, int compute_calpar, calmemmodule* module);
void lincal(vector<vector<float> >* data, vector<vector<float> >* additivein, redundantinfo* info, vector<float>* calpar, vector<vector<float> >* additiveout, int computeUBLFit, calmemmodule* module, float convergethresh, int maxiter, float stepsize);//if command is 1, compute the ubl estimates given data and calpars, rather than read ubl estimates from input; additive term will only be updated if lincal can achieve a lower chi^2
void gaincal(vector<vector<float> >* data, vector<vector<float> >* additivein, redundantinfo* info, vector<float>* calpar, vector<vector<float> >* additiveout, calmemmodule* module, float convergethresh, int maxiter, float stepsize);
//void loadGoodVisibilities(vector<vector<vector<vector<float> > > > * rawdata, vector<vector<vector<vector<float> > > >* receiver, redundantinfo* info, int xy);
void removeDegen(vector<float> *calpar, redundantinfo * info, calmemmodule* module);//forces the calibration parameters to have average 1 amp, and no shifting the image in phase. Note: 1) If you have not absolute calibrated the data, there's no point in running this, because this can only ensure that the calpars don't screw up already absolute calibrated data. 2) the antloc and ubl stored in redundant info must be computed from idealized antloc, otherwise the error in antloc from perfect redundancy will roll into this result, in an unknown fashion.
#endif
