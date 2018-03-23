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
#include <algorithm>
#define uint unsigned int
using namespace std;
const string FILENAME = "omnical_redcal.cc";
const float PI = atan2(0, -1);
const float UBLPRECISION = pow(10, -3);
const float MIN_NONE_ZERO = pow(10, -10);
const float MAX_NONE_INF = pow(10, 10);
const float MAX_POW_2 = pow(10, 10); //limiting the base of ^2 to be 10^10

void initcalmodule(calmemmodule* module, redundantinfo* info){
	int nant = info->nAntenna;
	int nbl = info->bl2d.size();
    int nubl = info->ublindex.size();
	int ncross = nbl;
	(module->amp1).resize(ncross);
	(module->amp2).resize(ncross);
	(module->amp3).resize(ncross);
	(module->pha1).resize(ncross);
	(module->pha2).resize(ncross);
	(module->pha3).resize(ncross);
	(module->x1).resize(nubl + nant);
	(module->x2).resize(nubl + nant);
	(module->x3).resize(nubl + nant);
	(module->x4).resize(nubl + nant);

	(module->g1).resize(nant);
	for (int i = 0; i < nant; i++){
		(module->g1)[i].resize(2);
	}
	(module->g0) = (module->g1);
	(module->g2) = (module->g1);
	(module->g3) = (module->g1);

	(module->adata1).resize(nbl);
	for (int i = 0; i < nbl; i++){
		(module->adata1)[i].resize(2);
	}
	(module->adata2) = (module->adata1);
	(module->adata3) = (module->adata1);

	(module->cdata1).resize(ncross);
	for (int i = 0; i < ncross; i++){
		(module->cdata1)[i].resize(2);
	}
	(module->cdata2) = (module->cdata1);
	(module->cdata3) = (module->cdata1);


	(module->ubl1).resize(nubl);
	for (int i = 0; i < nubl; i++){
		(module->ubl1)[i].resize(2);
	}
	(module->ubl0) = (module->ubl1);
	(module->ubl2) = (module->ubl1);
	(module->ubl3) = (module->ubl1);

	(module->ublgrp1).resize(nubl);
	for (int i = 0; i < nubl; i++){
		(module->ublgrp1)[i].resize(info->ublcount[i]);
	}
	(module->ublgrp2) = (module->ublgrp1);

	(module->ubl2dgrp1).resize(nubl);
	for (int i = 0; i < nubl; i++){
		(module->ubl2dgrp1)[i].resize(info->ublcount[i]);
		for (int j = 0; j < info->ublcount[i]; j ++){
			(module->ubl2dgrp1)[i][j].resize(2);
		}
	}
	(module->ubl2dgrp2) = (module->ubl2dgrp1);
	return;
}

float square(float x){
	return pow( max(min(x, MAX_POW_2), -MAX_POW_2), 2);
}


float amp(vector<float> * x){
	return sqrt( square(x->at(0))  + square(x->at(1)) );
}

float amp(float x, float y){
	return sqrt( square(x)  + square(y) );
}

float phase(float re, float im){
	/*if (re == 0 and im == 0){
		return 0;
	}*/
	return atan2(im, re);
}



float norm(vector<vector<float> > * v){
	float res = 0;
	for (unsigned int i = 0; i < v->size(); i++){
		for (unsigned int j = 0; j < v->at(i).size(); j++){
			res += pow(v->at(i)[j], 2);
		}
	}
	return pow(res, 0.5);
}

float phase(vector<float> * c){
	return atan2(c->at(1), c->at(0));
};

vector<float> conjugate (vector<float> x){
	vector<float> y = x;
	y[1] = -x[1];
	return y;
}

float phaseWrap (float x, float offset/*default -pi*/){
	while ( x <= offset ){
		x += 2 * PI;
	}
	while ( x > offset + 2 * PI ){
		x -= 2 * PI;
	}
	return x;
}

float median (vector<float> list){
	int l = list.size();
	if (l == 0) return 0;
	sort(list.begin(), list.end());
	int index = floor( l / 2 );
	if(l % 2 == 1) return list[index];
	else return (list[index] + list[index - 1]) / 2;
}

float medianAngle (vector<float> *list){
	string METHODNAME = "medianAngle";
	//cout << "#!#" << FILENAME << "#!#" << METHODNAME << " DBG ";
	vector<float> xList(list->size());
	vector<float> yList(list->size());
	for (unsigned int i = 0; i < list->size(); i++){
		//cout << list[i] << " ";
		xList[i] = cos(list->at(i));
		yList[i] = sin(list->at(i));
	}
	//cout << " median is " << atan2(median(yList), median(xList)) << endl;
	return atan2(median(yList), median(xList));
}

float mean (vector<float> *v, int start, int end){// take mean from start to end indices of vector v. 0 indexed
	string METHODNAME = "mean";
	int size = (int) (v->size());//size is originally unsigned

	if (size <= 0){
		cout << "#!#" << FILENAME << "#!#" << METHODNAME << " !!WARNING!! mean of an empty array requested!!";
		return 0;
	}


	if (end > size - 1 or start > size - 1){
		cout << "#!#" << FILENAME << "#!#" << METHODNAME << " !!WARNING!! start/end index requested at "<< start << "/" << end << " is out of array length of " << v->size() << "!!";
	}
	int a,b;
	if (start < 0 or start > size - 1) a = 0; else a = start;
	if (end < 0 or end > size - 1) b = size - 1; else b = end;
	float sum = accumulate(v->begin() + a, v->begin() + b, 0.0);
	//cout <<  start << " " << end << " " << a << " " << b << " " << sum << " " << endl;
	//cout << sum << endl;
	return sum / (b - a + 1);
}

vector<float> stdev(vector<float> *v){//returns {mean, sample standard deviation}. Created by Hrant and modified by Jeff
	string METHODNAME = "stdev";
	vector <float> result(2,0);
	int n = v-> size();
	if ( n <= 1){
		cout << "#!#" << FILENAME << "#!#" << METHODNAME << " !!WARNING!! standard deviation of an empty or unit array requested!!";
		return result;
	}
	float m = mean(v);
	//cout << m << endl;
	result[0] = m;
	float var = 0;
	for (int i = 0; i < n; i ++){
		var += square( v->at(i) - m );
	}
	/*standard deviation*/result[1] = sqrt( var / (n - 1) );
	return result;
}

float meanAngle (vector<float> *list){
    string METHODNAME = "meanAngle";
    //cout << "#!#" << FILENAME << "#!#" << METHODNAME << " DBG ";
    vector<float> xList(list->size());
    vector<float> yList(list->size());
    for (unsigned int i = 0; i < list->size(); i++){
        //cout << list[i] << " ";
        xList[i] = cos(list->at(i));
        yList[i] = sin(list->at(i));
    }
    //cout << " median is " << atan2(median(yList), median(xList)) << endl;
    return atan2(mean(&yList), mean(&xList));
}

vector<float> stdevAngle(vector<float> *v){
    string METHODNAME = "stdevAngle";
    vector <float> result(2,0);
    int n = v-> size();
    if ( n <= 1){
        cout << "#!#" << FILENAME << "#!#" << METHODNAME << " !!WARNING!! standard deviation of an empty or unit array requested!!";
        return result;
    }
    float m = meanAngle(v);
    //cout << m << endl;
    result[0] = m;
    float var = 0;
    for (int i = 0; i < n; i ++){
        var += square( phaseWrap(v->at(i) - m) );
    }
    /*standard deviation*/result[1] = sqrt( var / n );
    return result;
}

///////////////MAJOR STUFF///////////////////
/////////////////////////////////////////////

//Logcal functions

//XXX linear algebra
bool invert(vector<vector<int> > * AtNinvAori, vector<vector<double> > * AtNinvAinv ){//Gauss–Jordan elimination
	string METHODNAME = "invert";
	//WARNING: PASSING OF ARGUMENT VALUES NOT TESTED !!!!!!!!!!!!

	/* 	subroutine invert(np,n,A,B)
	! B = A^{-1}, where A is an arbitraty non-singular matrix.
	! In the process, A gets destroyed (A=I on exit).
	! Note that A need NOT be symmetric.
	! This routine is designed for small matrices where you dont care about
	! speed or numerical stability.
	! Written & tested by Max Tegmark 000506
	*/

	uint i=0, j=0, k=0, n = AtNinvAori->size();//todo check size
	vector<vector<float> > AtNinvAqaz(AtNinvAori->size(), vector<float>(AtNinvAori->at(0).size(),0));
	vector<vector<float> > *AtNinvA = &AtNinvAqaz;
	for( i = 0; i < AtNinvAori->size(); i++){
		for( j = 0; j < AtNinvAori->at(0).size(); j++){
			AtNinvA->at(i)[j] = AtNinvAori->at(i)[j];
		}
	}
	double r;
	//clock_t t1,t2;
	//t1=clock();


	for( i = 0; i < n; i++){// set matrix to one
		for (j = 0; j < n; j++){
			(AtNinvAinv->at(i))[j]=0.0;
		}
		(AtNinvAinv->at(i))[i]=1.0;
	}
	//t2=clock();
	//if(TIME) cout << ((float)t2-(float)t1) / CLOCKS_PER_SEC << "sec ";
	for ( i = 0; i < n; i++){ // find inverse by making A(i,i)=1

		if(isnan(AtNinvA->at(i)[i])){
			printf("%s: FATAL ERROR: input matrix AtNinvA of size %i by %i has NaN on diagonal! ABORT!\n", METHODNAME.c_str(), n, n);
			return false;
		}

		if(fabs((AtNinvA->at(i))[i]) < MIN_NONE_ZERO){
			if (i == n - 1){
				return false;
			}
			for(j = i+1; j < n; j++){//find a row to add to row i to make its diagonal non zero
				if(fabs((AtNinvA->at(j))[i]) >= MIN_NONE_ZERO and !isnan((AtNinvA->at(j))[i])){
					for(k = 0; k < n; k++){
						(AtNinvA->at(i))[k]=(AtNinvA->at(i))[k] + (AtNinvA->at(j))[k];
						(AtNinvAinv->at(i))[k]= (AtNinvAinv->at(i))[k] + (AtNinvAinv->at(j))[k];
					}
					break;
				}
				if (j == n - 1){
					return false;
				}
			}
		}
		r = (AtNinvA->at(i))[i];
		for (j = 0; j < n; j++){
			(AtNinvA->at(i))[j]=(AtNinvA->at(i))[j]/r;
			(AtNinvAinv->at(i))[j]= (AtNinvAinv->at(i))[j]/r;
			//if(j == i and (AtNinvAinv->at(i))[j]<0) {
				//printf("%s: FATAL ERROR: inverse matrix AtNinvAinv of size %i by %i has negative number on diagonal! ABORT!\n", METHODNAME.c_str(), n, n);
				//return false;
			//}
		}
		// Zero remaining elements A(*,i)
		for (k = 0; k < n; k++){
			if( k != i ){
				r = (AtNinvA->at(k))[i];
				for(j = 0; j < n; j++){
					(AtNinvA->at(k))[j] = (AtNinvA->at(k))[j] - r*(AtNinvA->at(i))[j];
					(AtNinvAinv->at(k))[j] = (AtNinvAinv->at(k))[j] - r*(AtNinvAinv->at(i))[j];
				}
			}
		}
	}
	//t2=clock();
	//if(TIME) cout << ((float)t2-(float)t1) / CLOCKS_PER_SEC << "sec ";
	return true;
}

//XXX linear algebra
bool invert(vector<vector<float> > * AtNinvAori, vector<vector<double> > * AtNinvAinv ){//Gauss–Jordan elimination
	string METHODNAME = "invert";
	//WARNING: PASSING OF ARGUMENT VALUES NOT TESTED !!!!!!!!!!!!

	/* 	subroutine invert(np,n,A,B)
	! B = A^{-1}, where A is an arbitraty non-singular matrix.
	! In the process, A gets destroyed (A=I on exit).
	! Note that A need NOT be symmetric.
	! This routine is designed for small matrices where you dont care about
	! speed or numerical stability.
	! Written & tested by Max Tegmark 000506
	*/

	uint i=0, j=0, k=0, n = AtNinvAori->size();//todo check size
	vector<vector<float> > AtNinvAqaz(AtNinvAori->size(), vector<float>(AtNinvAori->at(0).size(),0));
	vector<vector<float> > *AtNinvA = &AtNinvAqaz;
	for( i = 0; i < AtNinvAori->size(); i++){
		for( j = 0; j < AtNinvAori->at(0).size(); j++){
			AtNinvA->at(i)[j] = AtNinvAori->at(i)[j];
		}
	}
	double r;
	//clock_t t1,t2;
	//t1=clock();


	for( i = 0; i < n; i++){// set matrix to one
		for (j = 0; j < n; j++){
			(AtNinvAinv->at(i))[j]=0.0;
		}
		(AtNinvAinv->at(i))[i]=1.0;
	}
	//t2=clock();
	//if(TIME) cout << ((float)t2-(float)t1) / CLOCKS_PER_SEC << "sec ";
	for ( i = 0; i < n; i++){ // find inverse by making A(i,i)=1

		if(isnan(AtNinvA->at(i)[i])){
			printf("%s: FATAL ERROR: input matrix AtNinvA of size %i by %i has NaN on diagonal! ABORT!\n", METHODNAME.c_str(), n, n);
			return false;
		}

		if(fabs((AtNinvA->at(i))[i]) < MIN_NONE_ZERO){
			if (i == n - 1){
				return false;
			}
			for(j = i+1; j < n; j++){//find a row to add to row i to make its diagonal non zero
				if(fabs((AtNinvA->at(j))[i]) >= MIN_NONE_ZERO and !isnan((AtNinvA->at(j))[i])){
					for(k = 0; k < n; k++){
						(AtNinvA->at(i))[k]=(AtNinvA->at(i))[k] + (AtNinvA->at(j))[k];
						(AtNinvAinv->at(i))[k]= (AtNinvAinv->at(i))[k] + (AtNinvAinv->at(j))[k];
					}
					break;
				}
				if (j == n - 1){
					return false;
				}
			}
		}
		r = (AtNinvA->at(i))[i];
		for (j = 0; j < n; j++){
			(AtNinvA->at(i))[j]=(AtNinvA->at(i))[j]/r;
			(AtNinvAinv->at(i))[j]= (AtNinvAinv->at(i))[j]/r;
			//if(j == i and (AtNinvAinv->at(i))[j]<0) {
				//printf("%s: FATAL ERROR: inverse matrix AtNinvAinv of size %i by %i has negative number on diagonal! ABORT!\n", METHODNAME.c_str(), n, n);
				//return false;
			//}
		}
		// Zero remaining elements A(*,i)
		for (k = 0; k < n; k++){
			if( k != i ){
				r = (AtNinvA->at(k))[i];
				for(j = 0; j < n; j++){
					(AtNinvA->at(k))[j] = (AtNinvA->at(k))[j] - r*(AtNinvA->at(i))[j];
					(AtNinvAinv->at(k))[j] = (AtNinvAinv->at(k))[j] - r*(AtNinvAinv->at(i))[j];
				}
			}
		}
	}
	//t2=clock();
	//if(TIME) cout << ((float)t2-(float)t1) / CLOCKS_PER_SEC << "sec ";
	return true;
}



///////////////REDUNDANT BASELINE CALIBRATION STUFF///////////////////
/////////////////////////////////////////////


/******************************************************/
/******************************************************/
//XXX linear algebra
void vecmatmul(vector<vector<double> > * Afitting, vector<float> * v, vector<float> * ampfit){
	int i, j;
	double sum;
	int n = Afitting->size();//todo size check
	int m = v->size();
	for(i=0; i < n; i++){
		sum = 0.0;
		for(j = 0; j < m; j++){
			sum = sum + (Afitting->at(i))[j] * (v->at(j));
		}
		(ampfit->at(i)) = sum;
	}
	return;
}

//XXX linear algebra
void vecmatmul(vector<vector<float> > * Afitting, vector<float> * v, vector<float> * ampfit){
	int i, j;
	double sum;
	int n = Afitting->size();//todo size check
	int m = v->size();
	for(i=0; i < n; i++){
		sum = 0.0;
		for(j = 0; j < m; j++){
			sum = sum + (Afitting->at(i))[j] * (v->at(j));
		}
		(ampfit->at(i)) = sum;
	}
	return;
}

//XXX linear algebra
void vecmatmul(vector<vector<int> > * A, vector<float> * v, vector<float> * yfit){
	int i, j;
	double sum;
	int n = A->size();//todo size check
	int m = v->size();
	for(i=0; i < n; i++){
		sum = 0.0;
		for(j = 0; j < m; j++){
			sum = sum + (A->at(i))[j] * (v->at(j));
		}
		(yfit->at(i)) = sum;
	}
	return;
}


/******************************************************/
/******************************************************/
vector<float> minimizecomplex(vector<vector<float> >* a, vector<vector<float> >* b){//A*c = B where A and B complex vecs, c complex number, solve for c
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

void logcaladd(vector<vector<float> >* data, vector<vector<float> >* additivein, redundantinfo* info, vector<float>* calpar, vector<vector<float> >* additiveout, int computeUBLFit, int compute_calpar, calmemmodule* module){//if computeUBLFit is 1, compute the ubl estimates given data and calpars, rather than read ubl estimates from input
	int nubl = info->ublindex.size();
    int ai, aj; // antenna indices
	////initialize data and g0 ubl0
	for (unsigned int b = 0; b < (module->cdata1).size(); b++){
		module->cdata1[b][0] = data->at(b)[0] - additivein->at(b)[0];
		module->cdata1[b][1] = data->at(b)[1] - additivein->at(b)[1];
	}
	float amptmp;
	unsigned int cbl;
	for (int a = 0; a < info->nAntenna; a++){
		amptmp = pow(10, calpar->at(3 + a));
		module->g0[a][0] = amptmp * cos(calpar->at(3 + info->nAntenna + a));
		module->g0[a][1] = amptmp * sin(calpar->at(3 + info->nAntenna + a));
	}
	if (computeUBLFit != 1){
		for (int u = 0; u < nubl; u++){
			module->ubl0[u][0] = calpar->at(3 + 2 * info->nAntenna + 2 * u);
			module->ubl0[u][1] = calpar->at(3 + 2 * info->nAntenna + 2 * u + 1);
		}
	} else{//if computeUBLFit is 1, compute the ubl estimates given data and calpars, rather than read ubl estimates from input
		for (int u = 0; u < nubl; u++){
			for (unsigned int i = 0; i < module->ubl2dgrp1[u].size(); i++){
				cbl = info->ublindex[u][i];
                ai = info->bl2d[cbl][0]; aj = info->bl2d[cbl][1];
				module->ubl2dgrp1[u][i][0] = module->cdata1[cbl][0];
				module->ubl2dgrp1[u][i][1] = module->cdata1[cbl][1];
				module->ubl2dgrp2[u][i][0] = module->g0[ai][0] * module->g0[aj][0] + module->g0[ai][1] * module->g0[aj][1];
				module->ubl2dgrp2[u][i][1] = (module->g0[ai][0] * module->g0[aj][1] - module->g0[ai][1] * module->g0[aj][0]);
			}

			module->ubl0[u] = minimizecomplex(&(module->ubl2dgrp1[u]), &(module->ubl2dgrp2[u]));
		}
	}



	int nant = info->nAntenna;
	int ncross = info->bl2d.size();
	////read in amp and args
	for (int b = 0; b < ncross; b++){
		ai = info->bl2d[b][0];
		aj = info->bl2d[b][1];
		if ((data->at(b)[0] - additivein->at(b)[0] == 0) and (data->at(b)[1] - additivein->at(b)[1] == 0)){//got 0, quit
			for(int i = 3; i < 3 + 2 * nant + 2 * nubl; i++){
				calpar->at(i) = 0;
			}
			calpar->at(1) = INFINITY;
			return;
		}

		module->amp1[b] = log10(amp(data->at(b)[0] - additivein->at(b)[0], data->at(b)[1] - additivein->at(b)[1])) - calpar->at(3 + ai) - calpar->at(3 + aj);
		//module->pha1[b] = phase(data->at(b)[0] - additivein->at(b)[0], data->at(b)[1] - additivein->at(b)[1]) * info->reversed[b];
		module->pha1[b] = phaseWrap(phase(data->at(b)[0] - additivein->at(b)[0], data->at(b)[1] - additivein->at(b)[1]) + calpar->at(3 + nant + ai) - calpar->at(3 + nant + aj));
	}

	////rewrap args//TODO: use module->ubl0
	for(int i = 0; i < nubl; i ++){
		for (uint j = 0; j < (module->ublgrp1)[i].size(); j ++){
			(module->ublgrp1)[i][j] = module->pha1[info->ublindex[i][j]];
		}
	}

	for (int i = 0; i < nubl; i++){
		(module->ubl1)[i][1] = medianAngle(&((module->ublgrp1)[i]));
	}

	for (int b = 0; b < ncross; b++) {
		module->pha1[b] = phaseWrap(module->pha1[b], (module->ubl1)[info->bltoubl[b]][1] - PI);
	}

	fill(module->x3.begin(), module->x3.end(), 0);////At.y
	for (unsigned int i = 0; i < info->Atsparse.size(); i++){
		for (unsigned int j = 0; j < info->Atsparse[i].size(); j++){
			module->x3[i] += module->amp1[info->Atsparse[i][j]];
		}
	}
	fill(module->x4.begin(), module->x4.end(), 0);////Bt.y
	for (unsigned int i = 0; i < info->Btsparse.size(); i++){
		for (unsigned int j = 0; j < info->Btsparse[i].size(); j++){
			module->x4[i] += module->pha1[info->Btsparse[i][j][0]] * info->Btsparse[i][j][1];
		}
	}
	vecmatmul(&(info->AtAi), &(module->x3), &(module->x1));
	vecmatmul(&(info->BtBi), &(module->x4), &(module->x2));
	//vecmatmul(&(info->AtAiAt), &(module->amp1), &(module->x1));////This is actually slower than seperate multiplications
	//vecmatmul(&(info->BtBiBt), &(module->pha1), &(module->x2));


	for(int b = 0; b < ncross; b++) {
		ai = info->bl2d[b][0];
		aj = info->bl2d[b][1];
		float amp = pow(10, module->x1[nant + info->bltoubl[b]] + module->x1[ai] + module->x1[aj] + calpar->at(3 + ai) + calpar->at(3 + aj));
		float phase =  module->x2[nant + info->bltoubl[b]] - module->x2[ai] + module->x2[aj] - calpar->at(3 + nant + ai) + calpar->at(3 + nant + aj);
		additiveout->at(b)[0] = data->at(b)[0] - amp * cos(phase);
		additiveout->at(b)[1] = data->at(b)[1] - amp * sin(phase);
	}
	if(compute_calpar == 0){////compute additive term only
		calpar->at(1) = pow(norm(additiveout), 2);
		//cout << norm(additiveout) << endl;
		return;
	} else if(compute_calpar == 1){////compute full set of calpars
		for(int a = 0; a < nant; a++){
			calpar->at(3 + a) += module->x1[a];
			calpar->at(3 + nant + a) += module->x2[a];
		}
		for(int u = 0; u < nubl; u++){
			calpar->at(3 + 2 * nant + 2 * u) = pow(10, module->x1[nant + u]) * cos(module->x2[nant + u]);
			calpar->at(3 + 2 * nant + 2 * u + 1) = pow(10, module->x1[nant + u]) * sin(module->x2[nant + u]);
		}
		calpar->at(1) = pow(norm(additiveout), 2);
	}
	return;
}



void lincal(vector<vector<float> >* data, vector<vector<float> >* additivein, redundantinfo* info, vector<float>* calpar, vector<vector<float> >* additiveout, int computeUBLFit, calmemmodule* module, float convergethresh, int maxiter, float stepsize){
	int nubl = info->ublindex.size();
    int ai, aj; // antenna indices
	////initialize data and g0 ubl0
	for (unsigned int b = 0; b < (module->cdata1).size(); b++){
		module->cdata1[b][0] = data->at(b)[0] - additivein->at(b)[0];
		module->cdata1[b][1] = data->at(b)[1] - additivein->at(b)[1];
	}
	float amptmp;
	unsigned int cbl;
	float stepsize2 = 1 - stepsize;
	for (int a = 0; a < info->nAntenna; a++){
		amptmp = pow(10, calpar->at(3 + a));
		module->g0[a][0] = amptmp * cos(calpar->at(3 + info->nAntenna + a));
		module->g0[a][1] = amptmp * sin(calpar->at(3 + info->nAntenna + a));
	}
	if (computeUBLFit != 1){
		for (int u = 0; u < nubl; u++){
			module->ubl0[u][0] = calpar->at(3 + 2 * info->nAntenna + 2 * u);
			module->ubl0[u][1] = calpar->at(3 + 2 * info->nAntenna + 2 * u + 1);
		}
	} else{//if computeUBLFit is 1, compute the ubl estimates given data and calpars, rather than read ubl estimates from input
		for (int u = 0; u < nubl; u++){
			for (unsigned int i = 0; i < module->ubl2dgrp1[u].size(); i++){
				cbl = info->ublindex[u][i];
                ai = info->bl2d[cbl][0]; aj = info->bl2d[cbl][1];
				module->ubl2dgrp1[u][i][0] = module->cdata1[cbl][0];
				module->ubl2dgrp1[u][i][1] = module->cdata1[cbl][1];
				module->ubl2dgrp2[u][i][0] = module->g0[ai][0] * module->g0[aj][0] + module->g0[ai][1] * module->g0[aj][1];
				module->ubl2dgrp2[u][i][1] = (module->g0[ai][0] * module->g0[aj][1] - module->g0[ai][1] * module->g0[aj][0]);
			}

			module->ubl0[u] = minimizecomplex(&(module->ubl2dgrp1[u]), &(module->ubl2dgrp2[u]));
		}
	}

	float gre, gim, starting_chisq, chisq, chisq2, delta;
	int a1, a2; // antenna indices
	chisq = 0;
	for (unsigned int b = 0; b < (module->cdata2).size(); b++){
		a1 = info->bl2d[b][0];
		a2 = info->bl2d[b][1];
		gre = module->g0[a1][0] * module->g0[a2][0] + module->g0[a1][1] * module->g0[a2][1];
		gim = module->g0[a1][0] * module->g0[a2][1] - module->g0[a1][1] * module->g0[a2][0];
		//module->cdata2[b][0] = gre * module->ubl0[info->bltoubl[b]][0] - gim * module->ubl0[info->bltoubl[b]][1] * info->reversed[b];
		module->cdata2[b][0] = gre * module->ubl0[info->bltoubl[b]][0] - gim * module->ubl0[info->bltoubl[b]][1];// * info->reversed[b];
		//module->cdata2[b][1] = gre * module->ubl0[info->bltoubl[b]][1] * info->reversed[b] + gim * module->ubl0[info->bltoubl[b]][0];
		module->cdata2[b][1] = gre * module->ubl0[info->bltoubl[b]][1] + gim * module->ubl0[info->bltoubl[b]][0];
		delta = (pow(module->cdata2[b][0] - module->cdata1[b][0], 2) + pow(module->cdata2[b][1] - module->cdata1[b][1], 2));
		chisq += delta;
        // XXX have a starting_chisq_ant?
		//if (delta != 0){
			//cout << delta << " " << module->cdata2[b][0]-1 << " " << module->cdata2[b][1] << " " << module->ubl0[info->bltoubl[b]][0]-1 << " " << module->ubl0[info->bltoubl[b]][1] * info->reversed[b] << " " <<  a1 << " " <<  a2 << " " <<  b << " " << info->reversed[b] << endl;
		//}
		//cout << gre << " " << gim << " " << module->ubl0[info->bltoubl[b]][0] << " " << module->ubl0[info->bltoubl[b]][1] * info->reversed[b] << " " <<  a1 << " " <<  a2 << " " <<  b << " " << info->reversed[b] << endl;
	}
	starting_chisq = chisq;
	//cout << "lincal DBG v " << module->cdata1[DBGbl][0] << " " <<  module->cdata1[DBGbl][1] << endl<<flush;
	//cout << "lincal DBG c0 g0 g0 " << module->ubl0[info->nUBL - 1][0] << " " <<  module->ubl0[info->nUBL -1][1]  << " " << module->g0[DBGg1][0] << " " <<  module->g0[DBGg1][1]  << " " << module->g0[DBGg2][0] << " " <<  module->g0[DBGg2][1] << endl<<flush;
	//cout << "lincal DBG c0g0g0 "  << module->cdata2[DBGbl][0] << " " << module->cdata2[DBGbl][1] << endl<<flush;

	////start iterations
	int iter = 0;
	float componentchange = 100;
	while(iter < maxiter and componentchange > convergethresh){
		iter++;
		//cout << "iteration #" << iter << endl; cout.flush();
		////calpar g

		for (unsigned int a3 = 0; a3 < module->g3.size(); a3++){////g3 will be containing the final dg, g1, g2 will contain a and b as in the cost function LAMBDA = ||a + b*g||^2
			for (unsigned int a = 0; a < module->g3.size(); a++){
				cbl = info->bl1dmatrix[a3][a];
                // cbl is unsigned, so gauranteed not < 0
				if (cbl > module->cdata1.size() or info->ublcount[info->bltoubl[cbl]] < 2){//badbl or ubl has only 1 bl
					module->g1[a] = vector<float>(2,0);
					module->g2[a] = vector<float>(2,0);
				}else if(info->bl2d[cbl][1] == a3){
					module->g1[a] = module->cdata1[cbl];
					//module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * info->reversed[cbl]);
					module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1]);
					//module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * info->reversed[cbl] - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
					module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
				}else{
					module->g1[a][0] = module->cdata1[cbl][0];
					module->g1[a][1] = -module->cdata1[cbl][1];////vij needs to be conjugated
					//module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]));////Mi-j needs to be conjugated
					module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * (-1));////Mi-j needs to be conjugated
					//module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]) - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
					module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * (-1) - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
				}
			}
			//(module->g1)[a3] = vector<float>(2,0);
			//(module->g2)[a3] = (module->g1)[a3];
			//for (unsigned int a = a3 + 1; a < module->g3.size(); a++){
				//cbl = info->bl1dmatrix[a3][a];
				//if (cbl < 0 or cbl > module->cdata1.size() or info->ublcount[info->bltoubl[cbl]] < 2){//badbl or ubl has only 1 bl
					//module->g1[a] = vector<float>(2,0);
					//module->g2[a] = vector<float>(2,0);
				//}else{
					//module->g1[a][0] = module->cdata1[cbl][0];
					//module->g1[a][1] = -module->cdata1[cbl][1];////vij needs to be conjugated
					//module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]));////Mi-j needs to be conjugated
					//module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]) - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
				//}
			//}
			module->g3[a3] = minimizecomplex(&(module->g1), &(module->g2));
		}

		////ubl M
		for (int u = 0; u < nubl; u++){
			for (unsigned int i = 0; i < module->ubl2dgrp1[u].size(); i++){
				cbl = info->ublindex[u][i];
                ai = info->bl2d[cbl][0]; aj = info->bl2d[cbl][1];
				module->ubl2dgrp1[u][i][0] = module->cdata1[cbl][0];
				module->ubl2dgrp1[u][i][1] = module->cdata1[cbl][1] ;
				module->ubl2dgrp2[u][i][0] = module->g0[ai][0] * module->g0[aj][0] + module->g0[ai][1] * module->g0[aj][1];
				module->ubl2dgrp2[u][i][1] = (module->g0[ai][0] * module->g0[aj][1] - module->g0[ai][1] * module->g0[aj][0]);
			}

			module->ubl3[u] = minimizecomplex(&(module->ubl2dgrp1[u]), &(module->ubl2dgrp2[u]));
		}


		////Update g and ubl, do not update single-bl bls since they are not reversible. Will reverse this step later is chisq increased
		//float fraction;
		for (unsigned int a = 0; a < module->g3.size(); a++){
			module->g0[a][0] = stepsize2 * module->g0[a][0] + stepsize * module->g3[a][0];
			module->g0[a][1] = stepsize2 * module->g0[a][1] + stepsize * module->g3[a][1];

		}
		for (unsigned int u = 0; u < module->ubl3.size(); u++){
			if ((info->ublcount)[u] > 1){
				module->ubl0[u][0] = stepsize2 * module->ubl0[u][0] + stepsize * module->ubl3[u][0];
				module->ubl0[u][1] = stepsize2 * module->ubl0[u][1] + stepsize * module->ubl3[u][1];
			}
		}

		//compute chisq and decide convergence
		chisq2 = 0;
		for (unsigned int b = 0; b < (module->cdata2).size(); b++){
			if ((info->ublcount)[info->bltoubl[b]] > 1){//automatically use 0 for single-bl ubls, their actaul values are not updated yet
				a1 = info->bl2d[b][0];
				a2 = info->bl2d[b][1];
				gre = module->g0[a1][0] * module->g0[a2][0] + module->g0[a1][1] * module->g0[a2][1];
				gim = module->g0[a1][0] * module->g0[a2][1] - module->g0[a1][1] * module->g0[a2][0];
				module->cdata2[b][0] = gre * module->ubl0[info->bltoubl[b]][0] - gim * module->ubl0[info->bltoubl[b]][1];
				module->cdata2[b][1] = gre * module->ubl0[info->bltoubl[b]][1] + gim * module->ubl0[info->bltoubl[b]][0];
				chisq2 += (pow(module->cdata2[b][0] - module->cdata1[b][0], 2) + pow(module->cdata2[b][1] - module->cdata1[b][1], 2));
			}
		}
		componentchange = (chisq - chisq2) / chisq;

        if (componentchange > 0){//if improved, keep g0 and ubl0 updates, and update single-bl ubls and chisq
			chisq = chisq2;
			for (unsigned int u = 0; u < module->ubl3.size(); u++){
			//make sure there's no error on unique baselines with only 1 baseline
				for (unsigned int i = 0; i < module->ubl2dgrp1[u].size(); i++){
					cbl = info->ublindex[u][i];
                    ai = info->bl2d[cbl][0]; aj = info->bl2d[cbl][1];
					module->ubl2dgrp1[u][i][0] = module->cdata1[cbl][0];
					module->ubl2dgrp1[u][i][1] = module->cdata1[cbl][1];
					module->ubl2dgrp2[u][i][0] = module->g0[ai][0] * module->g0[aj][0] + module->g0[ai][1] * module->g0[aj][1];
					module->ubl2dgrp2[u][i][1] = (module->g0[ai][0] * module->g0[aj][1] - module->g0[ai][1] * module->g0[aj][0]);
				}
				module->ubl3[u] = minimizecomplex(&(module->ubl2dgrp1[u]), &(module->ubl2dgrp2[u]));
				module->ubl0[u][0] = module->ubl3[u][0];
				module->ubl0[u][1] = module->ubl3[u][1];
			}
		} else {//reverse g0 and ubl0 changes
			iter--;
			for (unsigned int a = 0; a < module->g3.size(); a++){
				module->g0[a][0] = (module->g0[a][0] - stepsize * module->g3[a][0]) / stepsize2;
				module->g0[a][1] = (module->g0[a][1] - stepsize * module->g3[a][1]) / stepsize2;

			}
			for (unsigned int u = 0; u < module->ubl3.size(); u++){
				if ((info->ublcount)[u] > 1){
					module->ubl0[u][0] = (module->ubl0[u][0] - stepsize * module->ubl3[u][0]) / stepsize2;
					module->ubl0[u][1] = (module->ubl0[u][1] - stepsize * module->ubl3[u][1]) / stepsize2;
				}
			}
		}
	}


	////update calpar and additive term
	if(componentchange > 0 or iter > 1){
		for (unsigned int a = 0; a < module->g0.size(); a++){
			calpar->at(3 + a) = log10(amp(&(module->g0[a])));
			calpar->at(3 + info->nAntenna + a) = phase(&(module->g0[a]));
		}
		int tmp = 3 + 2 * info->nAntenna;
		for (unsigned int u = 0; u < module->ubl0.size(); u++){
			calpar->at(tmp + 2 * u) = module->ubl0[u][0];
			calpar->at(tmp + 2 * u + 1) = module->ubl0[u][1];
		}

		calpar->at(0) += iter;
		calpar->at(2) = chisq;
		for (unsigned int b = 0; b < (module->cdata2).size(); b++){
			additiveout->at(b)[0] = module->cdata1[b][0] - module->cdata2[b][0];
			additiveout->at(b)[1] = module->cdata1[b][1] - module->cdata2[b][1];
		}
        // Create a chisq for each antenna. Right now, this is done at every time and
        // frequency, since that's how lincal is called, but that can be summed later
        // over all times and frequencies to get a chisq for each antenna.
        int chisq_ant = 3 + 2*(info -> nAntenna + nubl);
        for (int b = 0; b < (module -> cdata2).size(); b++){
            delta  = pow(module->cdata2[b][0] - module->cdata1[b][0], 2);
            delta += pow(module->cdata2[b][1] - module->cdata1[b][1], 2);
            a1 = info->bl2d[b][0];
            a2 = info->bl2d[b][1];
            calpar -> at(chisq_ant + a1) += delta;
            calpar -> at(chisq_ant + a2) += delta;
        }
	
	}else{////if chisq didnt decrease, keep everything untouched
		calpar->at(0) += 0;
		calpar->at(2) = starting_chisq;
        // XXX do we need to put a dummy value in for chisq per ant?
	}

	//cout << "lincal DBG v "  << module->cdata1[DBGbl][0] << " " << module->cdata1[DBGbl][1] << endl<<flush;
	//cout << "lincal DBG c0g0g0 "  << module->cdata2[DBGbl][0] << " " << module->cdata2[DBGbl][1] << endl<<flush;
	return;
}


void gaincal(vector<vector<float> >* data, vector<vector<float> >* additivein, redundantinfo* info, vector<float>* calpar, vector<vector<float> >* additiveout, calmemmodule* module, float convergethresh, int maxiter, float stepsize){
    int nubl = info->ublindex.size();

	////initialize data and g0 ubl0
	for (unsigned int b = 0; b < (module->cdata1).size(); b++){
		module->cdata1[b][0] = data->at(b)[0] - additivein->at(b)[0];
		module->cdata1[b][1] = data->at(b)[1] - additivein->at(b)[1];
	}
	float amptmp;
	unsigned int cbl;
	float stepsize2 = 1 - stepsize;
	for (int a = 0; a < info->nAntenna; a++){
		amptmp = pow(10, calpar->at(3 + a));
		module->g0[a][0] = amptmp * cos(calpar->at(3 + info->nAntenna + a));
		module->g0[a][1] = amptmp * sin(calpar->at(3 + info->nAntenna + a));
	}

	for (int u = 0; u < nubl; u++){
		module->ubl0[u][0] = 1;
		module->ubl0[u][1] = 0;
	}


	float gre, gim, starting_chisq, chisq, chisq2, delta;
	int a1, a2;
	chisq = 0;
	for (unsigned int b = 0; b < (module->cdata2).size(); b++){
		a1 = info->bl2d[b][0];
		a2 = info->bl2d[b][1];
		gre = module->g0[a1][0] * module->g0[a2][0] + module->g0[a1][1] * module->g0[a2][1];
		gim = module->g0[a1][0] * module->g0[a2][1] - module->g0[a1][1] * module->g0[a2][0];
		//module->cdata2[b][0] = gre * module->ubl0[info->bltoubl[b]][0] - gim * module->ubl0[info->bltoubl[b]][1] * info->reversed[b];
		module->cdata2[b][0] = gre * module->ubl0[info->bltoubl[b]][0] - gim * module->ubl0[info->bltoubl[b]][1];// * info->reversed[b];
		//module->cdata2[b][1] = gre * module->ubl0[info->bltoubl[b]][1] * info->reversed[b] + gim * module->ubl0[info->bltoubl[b]][0];
		module->cdata2[b][1] = gre * module->ubl0[info->bltoubl[b]][1] + gim * module->ubl0[info->bltoubl[b]][0];
		delta = (pow(module->cdata2[b][0] - module->cdata1[b][0], 2) + pow(module->cdata2[b][1] - module->cdata1[b][1], 2));
		chisq += delta;
		//if (delta != 0){
			//cout << delta << " " << module->cdata2[b][0]-1 << " " << module->cdata2[b][1] << " " << module->ubl0[info->bltoubl[b]][0]-1 << " " << module->ubl0[info->bltoubl[b]][1] * info->reversed[b] << " " <<  a1 << " " <<  a2 << " " <<  b << " " << info->reversed[b] << endl;
		//}
		//cout << gre << " " << gim << " " << module->ubl0[info->bltoubl[b]][0] << " " << module->ubl0[info->bltoubl[b]][1] * info->reversed[b] << " " <<  a1 << " " <<  a2 << " " <<  b << " " << info->reversed[b] << endl;
	}
	starting_chisq = chisq;
	//cout << "lincal DBG v " << module->cdata1[DBGbl][0] << " " <<  module->cdata1[DBGbl][1] << endl<<flush;
	//cout << "lincal DBG c0 g0 g0 " << module->ubl0[info->nUBL - 1][0] << " " <<  module->ubl0[info->nUBL -1][1]  << " " << module->g0[DBGg1][0] << " " <<  module->g0[DBGg1][1]  << " " << module->g0[DBGg2][0] << " " <<  module->g0[DBGg2][1] << endl<<flush;
	//cout << "lincal DBG c0g0g0 "  << module->cdata2[DBGbl][0] << " " << module->cdata2[DBGbl][1] << endl<<flush;

	////start iterations
	int iter = 0;
	float componentchange = 100;
	while(iter < maxiter and componentchange > convergethresh){
		iter++;
		//cout << "iteration #" << iter << endl; cout.flush();
		////calpar g

		for (unsigned int a3 = 0; a3 < module->g3.size(); a3++){////g3 will be containing the final dg, g1, g2 will contain a and b as in the cost function LAMBDA = ||a + b*g||^2
			for (unsigned int a = 0; a < module->g3.size(); a++){
				cbl = info->bl1dmatrix[a3][a];
                // cbl is unsigned and so gauranteed >= 0
				if (cbl > module->cdata1.size() or info->ublcount[info->bltoubl[cbl]] < 2){//badbl or ubl has only 1 bl
					module->g1[a] = vector<float>(2,0);
					module->g2[a] = vector<float>(2,0);
				}else if(info->bl2d[cbl][1] == a3){
					module->g1[a] = module->cdata1[cbl];
					//module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * info->reversed[cbl]);
					module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1]);
					//module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * info->reversed[cbl] - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
					module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
				}else{
					module->g1[a][0] = module->cdata1[cbl][0];
					module->g1[a][1] = -module->cdata1[cbl][1];////vij needs to be conjugated
					//module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]));////Mi-j needs to be conjugated
					module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * (-1));////Mi-j needs to be conjugated
					//module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]) - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
					module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * (-1) - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
				}
			}
			//(module->g1)[a3] = vector<float>(2,0);
			//(module->g2)[a3] = (module->g1)[a3];
			//for (unsigned int a = a3 + 1; a < module->g3.size(); a++){
				//cbl = info->bl1dmatrix[a3][a];
				//if (cbl < 0 or cbl > module->cdata1.size() or info->ublcount[info->bltoubl[cbl]] < 2){//badbl or ubl has only 1 bl
					//module->g1[a] = vector<float>(2,0);
					//module->g2[a] = vector<float>(2,0);
				//}else{
					//module->g1[a][0] = module->cdata1[cbl][0];
					//module->g1[a][1] = -module->cdata1[cbl][1];////vij needs to be conjugated
					//module->g2[a][0] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][0] + module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]));////Mi-j needs to be conjugated
					//module->g2[a][1] = (module->g0[a][0] * module->ubl0[info->bltoubl[cbl]][1] * (-info->reversed[cbl]) - module->g0[a][1] * module->ubl0[info->bltoubl[cbl]][0]);
				//}
			//}
			module->g3[a3] = minimizecomplex(&(module->g1), &(module->g2));
		}


		////Update g and ubl, do not update single-bl bls since they are not reversible. Will reverse this step later is chisq increased
		//float fraction;
		for (unsigned int a = 0; a < module->g3.size(); a++){
			module->g0[a][0] = stepsize2 * module->g0[a][0] + stepsize * module->g3[a][0];
			module->g0[a][1] = stepsize2 * module->g0[a][1] + stepsize * module->g3[a][1];

		}

		//compute chisq and decide convergence
		chisq2 = 0;
		for (unsigned int b = 0; b < (module->cdata2).size(); b++){
			if ((info->ublcount)[info->bltoubl[b]] > 1){//automatically use 0 for single-bl ubls, their actaul values are not updated yet
				a1 = info->bl2d[b][0];
				a2 = info->bl2d[b][1];
				gre = module->g0[a1][0] * module->g0[a2][0] + module->g0[a1][1] * module->g0[a2][1];
				gim = module->g0[a1][0] * module->g0[a2][1] - module->g0[a1][1] * module->g0[a2][0];
				//module->cdata2[b][0] = gre * module->ubl0[info->bltoubl[b]][0] - gim * module->ubl0[info->bltoubl[b]][1] * info->reversed[b];
				module->cdata2[b][0] = gre * module->ubl0[info->bltoubl[b]][0] - gim * module->ubl0[info->bltoubl[b]][1];
				//module->cdata2[b][1] = gre * module->ubl0[info->bltoubl[b]][1] * info->reversed[b] + gim * module->ubl0[info->bltoubl[b]][0];
				module->cdata2[b][1] = gre * module->ubl0[info->bltoubl[b]][1] + gim * module->ubl0[info->bltoubl[b]][0];
				chisq2 += (pow(module->cdata2[b][0] - module->cdata1[b][0], 2) + pow(module->cdata2[b][1] - module->cdata1[b][1], 2));
				//cout << gre << " " << gim << " " << module->ubl0[info->bltoubl[b]][0] << " " << module->ubl0[info->bltoubl[b]][1] * info->reversed[b] << " " <<  a1 << " " <<  a2 << " " <<  b << " " << info->reversed[b] << endl;
			}
		}
		componentchange = (chisq - chisq2) / chisq;

		if (componentchange > 0){//if improved, keep g0 and ubl0 updates, and update single-bl ubls and chisq
			chisq = chisq2;
		} else {//reverse g0 and ubl0 changes
			iter--;
			for (unsigned int a = 0; a < module->g3.size(); a++){
				module->g0[a][0] = (module->g0[a][0] - stepsize * module->g3[a][0]) / stepsize2;
				module->g0[a][1] = (module->g0[a][1] - stepsize * module->g3[a][1]) / stepsize2;
			}
		}
	}


	////update calpar and additive term
	if(componentchange > 0 or iter > 1){
		for (unsigned int a = 0; a < module->g0.size(); a++){
			calpar->at(3 + a) = log10(amp(&(module->g0[a])));
			calpar->at(3 + info->nAntenna + a) = phase(&(module->g0[a]));
		}

		calpar->at(0) += iter;
		calpar->at(2) = chisq;
		for (unsigned int b = 0; b < (module->cdata2).size(); b++){
			additiveout->at(b)[0] = module->cdata1[b][0] - module->cdata2[b][0];
			additiveout->at(b)[1] = module->cdata1[b][1] - module->cdata2[b][1];
		}
	}else{////if chisq didnt decrease, keep everything untouched
		calpar->at(0) += 0;
		calpar->at(2) = starting_chisq;
	}
	//cout << "lincal DBG v "  << module->cdata1[DBGbl][0] << " " << module->cdata1[DBGbl][1] << endl<<flush;
	//cout << "lincal DBG c0g0g0 "  << module->cdata2[DBGbl][0] << " " << module->cdata2[DBGbl][1] << endl<<flush;
	return;
}

//void loadGoodVisibilities(vector<vector<vector<vector<float> > > > * rawdata, vector<vector<vector<vector<float> > > >* receiver, redundantinfo* info, int xy){////0 for xx 3 for yy
//	for (unsigned int t = 0; t < receiver->size(); t++){
//		for (unsigned int f = 0; f < receiver->at(0).size(); f++){
//			for (unsigned int bl = 0; bl < receiver->at(0)[0].size(); bl++){
//				receiver->at(t)[f][bl][0] = rawdata->at(xy)[t][f][2 * info->subsetbl[bl]];
//				receiver->at(t)[f][bl][1] = rawdata->at(xy)[t][f][2 * info->subsetbl[bl] + 1];
//			}
//		}
//	}
//	return;
//}


void removeDegen(vector<float> *calpar, redundantinfo * info, calmemmodule* module){//forces the calibration parameters to have average 1 amp, and no shifting the image in phase. Note: 1) If you have not absolute calibrated the data, there's no point in running this, because this can only ensure that the calpars don't screw up already absolute calibrated data. 2) the antloc and ubl stored in redundant info must be computed from idealized antloc, otherwise the error in antloc from perfect redundancy will roll into this result, in an unknown fashion.
	////load data
    int nubl = info->ublindex.size();
	vector<float> pha1(info->nAntenna, 0);
	for (int a = 0 ; a < info->nAntenna; a ++){
		pha1[a] = calpar->at(3 + info->nAntenna + a);
	}
	for (int u = 0 ; u < nubl; u ++){
		module->ubl1[u][0] = amp(calpar->at(3 + 2 * info->nAntenna + 2 * u), calpar->at(3 + 2 * info->nAntenna + 2 * u + 1));
		module->ubl1[u][1] = phase(calpar->at(3 + 2 * info->nAntenna + 2 * u), calpar->at(3 + 2 * info->nAntenna + 2 * u + 1));
	}

	////compute amp delta
	float ampfactor = 0;//average |g|, divide ant calpar by this, multiply ublfit by square this
	for (int a = 0 ; a < info->nAntenna; a ++){
		ampfactor += pow(10, calpar->at(3 + a));
	}
	ampfactor = ampfactor / info->nAntenna;
	//cout << ampfactor << endl;

	////compute phase delta
	vecmatmul(&(info->degenM), &(pha1), &(module->x1));//x1: add ant calpar and ubl fit by this

	////correct ant calpar
	for (int a = 0 ; a < info->nAntenna; a ++){
		calpar->at(3 + a) = calpar->at(3 + a) - log10(ampfactor);
		calpar->at(3 + info->nAntenna + a) = phaseWrap(calpar->at(3 + info->nAntenna + a) + module->x1[a]);
	}

	////correct ublfit
	for (int u = 0 ; u < nubl; u ++){
		module->ubl2[u][0] = module->ubl1[u][0] * ampfactor * ampfactor;
		module->ubl2[u][1] = module->ubl1[u][1] + module->x1[info->nAntenna + u];
		calpar->at(3 + 2 * info->nAntenna + 2 * u) = module->ubl2[u][0] * cos(module->ubl2[u][1]);
		calpar->at(3 + 2 * info->nAntenna + 2 * u + 1) = module->ubl2[u][0] * sin(module->ubl2[u][1]);
	}
	return;
}

