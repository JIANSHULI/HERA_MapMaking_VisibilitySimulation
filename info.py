import _omnical as _O
import numpy as np, numpy.linalg as la
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import scipy.sparse as sps

#  ___        _              _          _   ___       __
# | _ \___ __| |_  _ _ _  __| |__ _ _ _| |_|_ _|_ _  / _|___
# |   / -_) _` | || | ' \/ _` / _` | ' \  _|| || ' \|  _/ _ \
# |_|_\___\__,_|\_,_|_||_\__,_\__,_|_||_\__|___|_||_|_| \___/

class RedundantInfo(_O.RedundantInfo):
    '''Metadata used by _omnical.redcal which is passed into C++ routines.  Fields are as follows:
    'nAntenna', number of usable ants (not total number)
    'nBaseline', number of bls, matches first dim of bltoubl/bl2d, now python only
    'subsetant', (nAntenna,) antenna numbers used; index i corresponds to antenna number ai
    'antloc', (nAntenna,3) float, antpos from which degeneracies (unsolvable cal params) are determined
    'bltoubl', (nBaseline,) for each bl in bl2d, the index of corresponding unique bl in ubl/ublcount/ublindex
    'bl2d', (nBaseline,2) the i,j indices of ants in subsetant for each bl
    'ublcount', (nUBL,) number of bls contributing to each ubl
    'ublindex', (nBaseline,) bl2d index for each bl contributing to each ubl
    'bl1dmatrix', (nAntenna,nAntenna) for each i,j antenna pair, the bl2d index of that bl
    'degenM', (nAntenna+nUBL,nAntenna) matrix projecting out degenerate cal params
    'At', (ncross,nAntenna+nUBL), sparse, matrix containing amp cal equations
    'Bt', (ncross,nAntenna+nUBL), sparse, matrix containing phs cal equations
    'AtAi', precomputed matrix [At A]^-1 used to weight measurements
    'BtBi', precomputed matrix [Bt B]^-1 used to weight measurements
    ------------------------------------------------------------------------
    'nUBL', number of unique bls; XXX legacy only
    'subsetbl', (nBaseline,) for each bl in bl2d, the index in totalVisibilityId; XXX legacy only
    'ubl', (nUBL,3) float, sep vector for each unique baseline; XXX unused?
    'reversed', for each bl in crossindex, -1 if flipped wrt corresponding ubl, otherwise 1; XXX legacy only
    'crossindex', indices in bl2d in totVisibilityId; XXX legacy only
    'totalVisibilityId', (nBaselines, 2) i,j for every bl; defines data order into omnical; XXX legacy only'''
    def __init__(self, filename=None):
        _O.RedundantInfo.__init__(self)
        if filename: self.from_npz(filename)
    def _get_AtBt(self, key):
        '''for convenience of multiplication in update()'''
        assert(key in ['At','Bt'])
        tmp = _O.RedundantInfo.__getattribute__(self, key+'sparse')
        matrix = np.zeros((self.nAntenna + len(self.ublcount), len(self.bl2d)))
        for i in tmp: matrix[i[0],i[1]] = i[2]
        return sps.csr_matrix(matrix)
    def _set_AtBt(self, key, val):
        assert(key in ['At','Bt'])
        nonzeros = np.array(val.nonzero()).transpose()
        self.__setattr__(key+'sparse', np.array([[i,j,val[i,j]] for i,j in nonzeros], dtype=np.int32))
    def __getattribute__(self, key):
        if key in ['At','Bt']: return self._get_AtBt(key)
        else: return _O.RedundantInfo.__getattribute__(self, key)
    def __setattr__(self, key, val):
        if key in ['At','Bt']: return self._set_AtBt(key, val)
        else: return _O.RedundantInfo.__setattr__(self, key, val)
    def __getitem__(self,k): return self.__getattribute__(k)
    def __setitem__(self,k,val): return self.__setattr__(k,val)
    def bl_order(self):
        '''Return (i,j) baseline tuples in the order that they should appear in data.  Antenna indicies
        are in real-world order (as opposed to the internal ordering used in subsetant).'''
        return [(self.subsetant[i],self.subsetant[j]) for (i,j) in self.bl2d]
    def to_npz(self, filename):
        '''Write enough to a numpy npz file to enable RedundantInfo to be reconstructed.'''
        reds = self.get_reds()
        antpos = self.get_antpos()
        # XXX how expensive is constructing At,Bt, AtAi, BtBi?  Do we need to store them?
        np.savez(filename, reds=reds, antpos=antpos)
    def from_npz(self, filename):
        '''Initialize RedundantInfo from a numpy npz file written by 'to_npz'.'''
        npz = np.load(filename)
        # XXX how expensive is constructing At,Bt, AtAi, BtBi?  Do we need to store them?
        self.init_from_reds(npz['reds'], npz['antpos'])
    def update(self):
        '''Initialize other arrays from fundamental arrays'''
        #The sparse matrices are treated a little differently because they are not rectangular
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            if self.AtAi.size == 0:
                self.AtAi = la.pinv(self.At.dot(self.At.T).todense(),rcond=1e-6).astype(np.float32)#(AtA)^-1
                self.BtBi = la.pinv(self.Bt.dot(self.Bt.T).todense(),rcond=1e-6).astype(np.float32)#(BtB)^-1
    def ant_index(self, i):
        try: return self._ant2ind[i]
        except(AttributeError):
            self._ant2ind = {}
            for x,ant in enumerate(self.subsetant): self._ant2ind[ant] = x
            return self._ant2ind[i]
    def init_same(self,reds):
        ants = {}
        for ubl_gp in reds:
            for (i,j) in ubl_gp:
                ants[i] = ants.get(i,0) + 1
                ants[j] = ants.get(j,0) + 1
        self.subsetant = np.array(ants.keys(), dtype=np.int32)
        bl2d = np.array([(self.ant_index(i),self.ant_index(j),u) for u,ubl_gp in enumerate(reds) for i,j in ubl_gp], dtype=np.int32)
        self.bl2d = bl2d[:,:2]
        self.bltoubl = bl2d[:,2]
        self.blperant = np.array([ants[a] for a in sorted(ants.keys())], dtype=int)
        
    def init_from_reds(self, reds, antpos):
        '''Initialize RedundantInfo from a list where each entry is a group of redundant baselines.
        Each baseline is a (i,j) tuple, where i,j are antenna indices.  To ensure baselines are
        oriented to be redundant, it may be necessary to have i > j.  If this is the case, then
        when calibrating visibilities listed as j,i data will have to be conjugated.'''
        reds = [[(int(i),int(j)) for i,j in gp] for gp in reds]
        self.init_same(reds)
        self.nAntenna = self.subsetant.size
        nUBL = len(reds)
        self.nBaseline = self.bl2d.shape[0]
        self.ublcount = np.array([len(ubl_gp) for ubl_gp in reds], dtype=np.int32)
        self.ublindex = np.arange(self.nBaseline, dtype=np.int32)
        bl1dmatrix = (2**31-1) * np.ones((self.nAntenna,self.nAntenna),dtype=np.int32)
        for n,(i,j) in enumerate(self.bl2d): bl1dmatrix[i,j], bl1dmatrix[j,i] = n,n
        self.bl1dmatrix = bl1dmatrix
        #A: A matrix for logcal amplitude
        A,B = np.zeros((self.nBaseline,self.nAntenna+nUBL)), np.zeros((self.nBaseline,self.nAntenna+nUBL))
        for n,(i,j) in enumerate(self.bl2d):
            A[n,i], A[n,j], A[n,self.nAntenna+self.bltoubl[n]] = 1,1,1
            B[n,i], B[n,j], B[n,self.nAntenna+self.bltoubl[n]] = -1,1,1
        self.At, self.Bt = sps.csr_matrix(A).T, sps.csr_matrix(B).T
        # XXX nothing up to this point requires antloc; in principle, degenM can be deduced
        # from reds alone, removing need for antpos.  So that'd be nice, someday
        self.antloc = antpos.take(self.subsetant, axis=0).astype(np.float32)
        self.ubl = np.array([np.mean([antpos[j]-antpos[i] for i,j in ublgp],axis=0) for ublgp in reds], dtype=np.float32)
        # XXX why are 1,0 appended to positions/ubls?
        a = np.array([np.append(ai,1) for ai in self.antloc], dtype=np.float32)
        d = np.array([np.append(ubli,0) for ubli in self.ubl], dtype=np.float32)
        m1 = -a.dot(la.pinv(a.T.dot(a))).dot(a.T)
        m2 = d.dot(la.pinv(a.T.dot(a))).dot(a.T)
        self.degenM = np.append(m1,m2,axis=0)
        self.update()
    def get_reds(self):
        '''After initialization, return redundancies in the same format used in init_from_reds.  Requires that
        ublcount, ublindex, subsetant, and bl2d be set.'''
        reds = []
        x = 0
        for y in self.ublcount:
            reds.append([(self.subsetant[self.bl2d[k,0]],self.subsetant[self.bl2d[k,1]]) for k in self.ublindex[x:x+y]])
            x += y
        return reds
    def get_antpos(self):
        '''After initialization, return antenna positions in the format used by init_from_reds.'''
        antpos = np.zeros((self.subsetant.max()+1,3),dtype=np.float)
        for i,ant in enumerate(self.subsetant): antpos[ant] = self.antloc[i]
        return antpos
    def get_xy_AB(self):
        '''XXX need to define where/how this function is used.
        return xyA, xyB, yxA, yxB for logcal cross polarizations'''
        na = self.nAntenna
        nu = len(self.ublcount)
        A = self.At.T.todense()
        B = self.Bt.T.todense()
        bl2dcross = self.bl2d
        #print na, nu, B.shape wesdcxaz
        xyA = np.zeros((len(self.bl2d), 2*na+nu), dtype='int8')
        yxA = np.zeros_like(xyA)
        xyB = np.zeros_like(xyA)
        yxB = np.zeros_like(xyA)
        xyA[:, 2*na:] = A[:, na:]
        xyB[:, 2*na:] = B[:, na:]
        for i in range(len(xyA)):
            xyA[i, bl2dcross[i,0]] = A[i, bl2dcross[i,0]]
            xyA[i, na + bl2dcross[i,1]] = A[i, bl2dcross[i,1]]
            xyB[i, bl2dcross[i,0]] = B[i, bl2dcross[i,0]]
            xyB[i, na + bl2dcross[i,1]] = B[i, bl2dcross[i,1]]
        yxA[:, :na] = xyA[:, na:2*na]
        yxA[:, na:2*na] = xyA[:, :na]
        yxA[:, 2*na:] = xyA[:, 2*na:]
        yxB[:, :na] = xyB[:, na:2*na]
        yxB[:, na:2*na] = xyB[:, :na]
        yxB[:, 2*na:] = xyB[:, 2*na:]
        return xyA, xyB, yxA, yxB

##########
import os

KEYS = [
    'nAntenna', # number of usable ants (not total number)
    'nUBL', # number of unique bls, matches first dim of ubl/ublcount/ublindex
    'nBaseline', # number of bls, matches first dim of bltoubl/bl2d, now python only
    'subsetant', # (nAntenna,) antenna numbers used; index i corresponds to antenna number ai, now python only
    'antloc', # (nAntenna,3) float,  idealized antpos from which redundancy is determined XXX not sure of lin/log cal need this.  if not, could restrict this to ArrayInfo and remove from RedundantInfo
    'subsetbl', # (nBaseline,) for each bl in bl2d, the index in totalVisibilityId; now python only
    'ubl', # (nUBL,3) float, sep vector for each unique baseline i think not necessary for lin/log cal, now python only
    'bltoubl', # (nBaseline,) for each bl in bl2d, the index of corresponding unique bl in ubl/ublcount/ublindex
    'reversed', # for each entry in crossindex, 1 if baseline is flipped wrt corresponding ubl, otherwise -1
    'reversedauto', # XXX to read old files
    'autoindex', # XXX to read old files
    'crossindex', # indices in bl2d of crosses XXX if we mandate no autos, then crossindex not necessary
    'bl2d', # (nBaseline,2) the i,j indices of ants in subsetant for each bl
    'ublcount', # (nUBL,) number of bls contributing to each ubl XXX can determine this from ublindex
    'ublindex', # (nUBL, ublcount[i], 3) ant1,ant2,blindex for each bl contributing to each ubl
    'bl1dmatrix', # (nAntenna,nAntenna) for each i,j antenna pair, the index of where that bl appears in crossindex
    'degenM', # (nAntenna+nUBL,nAntenna)
    'At', # (ncross,nAntenna+nUBL), sparse
    'Bt', # (ncross,nAntenna+nUBL), sparse
    'AtAi', # precomputed matrix
    'BtBi', # precomputed matrix
    'totalVisibilityId', # (all_baselines, 2) ai,aj antenna numbers for every possible bl; defines order of data to be loaded into omnical solver XXX if totalVisibilityId only holds good data, then subsetbl becomes pointless
]

float_infokeys = ['degenM','AtAi','BtBi']#,'AtAiAt','BtBiBt','PA','PB','ImPA','ImPB']

MARKER = 9999999

class RedundantInfoLegacy(RedundantInfo):
    def __init__(self, filename=None, verbose=False, preview_only=False, txtmode=False, threshold=128):
        RedundantInfo.__init__(self, filename=None)
        self.threshold = threshold
        if filename:
            if txtmode: self.fromfile_txt(filename, verbose=verbose)
            else: self.fromfile(filename, verbose=verbose, preview_only=preview_only)
        self.totalVisibilityId = np.zeros_like(self.bl2d) # XXX placeholder for now
    def make_dd(self, data):
        '''Legacy interface to create the data dict used by 'order_data' from the array of all
        visibilities the was formerly used by omnical.  Makes use of info.subsetbl and info._reversed,
        which are only preserved for this legacy interface.''' # XXX get rid of this function someday
        dd = {}
        for bl,ind,rev in zip(self.bl_order(),self.subsetbl,self._reversed):
            if rev == -1: bl = bl[::-1]
            dd[bl] = data[...,ind]
        return dd
    def fromfile_txt(self, filename, verbose=False):
        '''Initialize from (txt) file.  This is a legacy interface; writing to txt files is no longer supported.'''
        if verbose: print 'Reading redundant info from %s' % filename
        d = np.array([np.array(map(float, line.split())) for line in open(filename)])
        self.from_array(d, verbose=verbose)
        self.update()
        if verbose: print "done. nAntenna,nUBL,nBaseline = %i,%i,%i" % (len(self.subsetant),len(self.ublcount),self.nBaseline)
    def tofile(self, filename, overwrite=False, verbose=False):
        '''XXX DOCSTRING'''
        assert(not os.path.exists(filename) or overwrite)
        if verbose: print 'Writing info to', filename
        d = self.to_array(verbose=verbose)
        f = open(filename,'wb')
        d.tofile(f)
        f.close()
        if verbose: print "Info file successfully written to", filename
    def fromfile(self, filename, verbose=False, preview_only=False): # XXX what is preview?
        '''Initialize from (binary) file.'''
        if verbose: print 'Reading redundant info from %s' % filename
        datachunk = np.fromfile(filename)
        markerindex = np.where(datachunk == MARKER)[0]
        # XXX uneven array
        d = np.array([np.array(datachunk[markerindex[i]+1:markerindex[i+1]]) for i in range(len(markerindex)-1)])
        self.from_array(d, verbose=verbose, preview_only=preview_only) # XXX do i need to store preview return case?
        self.update()
        if verbose: print "done. nAntenna,nUBL,nBaseline = %i,%i,%i" % (len(self.subsetant),len(self.ublcount),self.nBaseline)
    def to_array(self, verbose=False):
        # XXX from_array and to_array do not match, need to change that, but this affects fromfile & fromfile_txt
        d = KEYS
        if self.nAntenna <= self.threshold: d = d[:-3] + d[-1:]
        self.crossindex = np.arange(len(self.bl2d)) # XXX legacy for file format.
        self.reversedauto = np.ones(self.nAntenna) # XXX legacy for file format
        self.autoindex = np.arange(self.nAntenna) # XXX legacy for file format
        self.reversed = np.ones(self.nBaseline) # XXX legacy for file format
        self.nUBL = len(self.ublcount) # XXX legacy for file format
        self.subsetbl = np.arange(self.nBaseline, dtype=np.int32) # XXX legacy for file format
        def fmt(k):
            if k in ['At','Bt']:
                sk = self[k].T
                if self.nAntenna > self.threshold:
                    row,col = sk.nonzero()
                    return np.vstack([row,col,sk[row,col]]).T
                else: return sk.todense()
            elif k in ['ublindex']: # XXX legacy for file format
                d = np.zeros((self.ublindex.size,3), dtype=np.int)
                d[:,2] = self.ublindex
                return d
            else: return self[k]
        d = [fmt(k) for k in d]
        d = [[MARKER]]+[k for i in zip(d,[[MARKER]]*len(d)) for k in i]
        return np.concatenate([np.asarray(k).flatten() for k in d])
    def from_array(self, d, verbose=False, preview_only=False):
        '''Initialize fields from data contained in 2D float array used to store data to file.'''
        # XXX from_array and to_array do not match, need to change that, but this affects fromfile & fromfile_txt
        # XXX maybe not all of these variables should be exposed (i.e. some could be C only)
        # XXX at the least, should validate array dimensions in C wrapper
        self.nAntenna = int(d[0][0]) # XXX did we mean d[0,0]?
        #self.nUBL = int(d[1][0]) # XXX
        nUBL = int(d[1][0]) # XXX
        self.nBaseline = int(d[2][0]) # XXX
        self.subsetant = d[3].astype(np.int32) # index of good antennas
        self.antloc = d[4].reshape((self.nAntenna,3)).astype(np.float32)
        self.subsetbl = d[5].astype(np.int32) # index of good bls (+autos) within all bls
        self.ubl = d[6].reshape((nUBL,3)).astype(np.float32) # unique bl vectors
        self.bltoubl = d[7].astype(np.int32) # cross bl number to ubl index
        reverse = d[8].astype(np.int32) # cross only bl if reversed -1, else 1
        #self.reversed = np.ones_like(reverse)
        self._reversed = reverse
        self.reversedauto = d[9].astype(np.int32) # XXX check comment: index of good autos within all bls
        self.autoindex = d[10].astype(np.int32) # index of auto bls among good bls
        #self.crossindex = d[11].astype(np.int32) # index of cross bls among good bls
        crossindex = d[11].astype(np.int32) # index of cross bls among good bls
        #self.crossindex = np.arange(len(crossindex), dtype=np.int32)
        #ncross = len(self.crossindex)
        # XXX maybe add this as a function
        if preview_only: return #ncross - self.nUBL - self.nAntenna + 2 # XXX return value here, normally not returning anything
        #self.bl2d = d[12].reshape(self.nBaseline,2).astype(np.int32) # 1d bl index to (i,j) antenna pair
        bl2d = d[12].reshape(-1,2).astype(np.int32) # 1d bl index to (i,j) antenna pair
        bl2dc = bl2d[crossindex]
        if True: # XXX I worry from_array, particularly this block, has not been effectively tested since removing reversed
            bl2dc0 = np.where(reverse == 1, bl2dc[:,0], bl2dc[:,1])
            bl2dc1 = np.where(reverse == 1, bl2dc[:,1], bl2dc[:,0])
            bl2dc[:,0],bl2dc[:,1] = bl2dc0,bl2dc1
        self.bl2d = bl2dc
        self.ublcount = d[13].astype(np.int32) # for each ubl, number of corresponding good cross bls
        #self.ublindex = d[14].reshape(ncross,3).astype(np.int32) # for each ubl, the vector<int> contains (i,j,ant1,ant2,crossbl)
        ublindex = d[14].reshape(-1,3).astype(np.int32) # for each ubl, the vector<int> contains (i,j,crossbl)
        newind = np.arange(self.nBaseline)[crossindex] = np.arange(len(crossindex))
        ublindex[:,2] = newind[ublindex[:,2]]
        self.ublindex = ublindex[:,2] # XXX could clean this up
        self.bl1dmatrix = d[15].reshape((self.nAntenna,self.nAntenna)).astype(np.int32) #a symmetric matrix where col/row numbers are antenna indices and entries are 1d baseline index not counting auto corr
        self.degenM = d[16].reshape((self.nAntenna+nUBL,self.nAntenna)).astype(np.float32)
        if self.nAntenna > self.threshold:
            #sparse_entries = d[16].reshape((len(d[16])/3,3))
            sparse_entries = d[17].reshape((-1,3))
            row,column,value = sparse_entries[:,0],sparse_entries[:,1],sparse_entries[:,2]
            self.At = sps.csr_matrix((value,(row,column)), shape=(len(self.bl2d),self.nAntenna+nUBL)).T
            #sparse_entries = d[17].reshape((len(d[17])/3,3))
            sparse_entries = d[18].reshape((-1,3))
            row,column,value = sparse_entries[:,0],sparse_entries[:,1],sparse_entries[:,2]
            self.Bt = sps.csr_matrix((value,(row,column)), shape=(len(self.bl2d),self.nAntenna+nUBL)).T
            self.AtAi = d[19].reshape((self.nAntenna + nUBL,self.nAntenna + nUBL)).astype(np.float32)
            self.BtBi = d[20].reshape((self.nAntenna + nUBL,self.nAntenna + nUBL)).astype(np.float32)
            self.totalVisibilityId = d[21].reshape(-1,2).astype(np.int32)
        else:
            # XXX why astype(int) here, but not above?
            self.At = sps.csr_matrix(d[17].reshape((-1,self.nAntenna+nUBL)).astype(np.int32)).T # A matrix for logcal amplitude
            #self.Bt = sps.csr_matrix(d[18].reshape((-1,self.nAntenna+nUBL)).astype(np.int32)).T # B matrix for logcal phase
            try: self.totalVisibilityId = d[19].reshape(-1,2).astype(np.int32)
            except(IndexError): # old files were saved w/o this field
                pass
        # XXX overwriting Bt because it depends on reversed
        crosspair = [p for p in self.bl2d]
        B = np.zeros((len(crosspair),self.nAntenna+nUBL))
        for i,cp in enumerate(crosspair): B[i,cp[0]], B[i,cp[1]], B[i,self.nAntenna+self.bltoubl[i]] = -1,1,1
        # XXX setting Bt twice segfaults
        self.Bt = sps.csr_matrix(B).T
    def compare(self, info, verbose=False, tol=1e-5): # XXX after trimming things, this func is almost meaningless
        '''compare with another RedundantInfo, output True if they are the same and False if they are different'''
        try:
            floatkeys = float_infokeys#['antloc','ubl','AtAi','BtBi','AtAiAt','BtBiBt','PA','PB','ImPA','ImPB']
            #intkeys = ['nAntenna','bltoubl','reversed','bl2d','ublcount','bl1dmatrix']
            intkeys = ['nAntenna','bltoubl','bl2d','ublcount','bl1dmatrix']
            #infomatrices=['At','Bt']
            infomatrices=['At']
            specialkeys = ['ublindex']
            allkeys = floatkeys + intkeys + infomatrices + specialkeys#['antloc','ubl','nAntenna','nUBL','nBaseline','subsetant','subsetbl','bltoubl','reversed','reversedauto','autoindex','crossindex','bl2d','ublcount','bl1dmatrix','AtAi','BtBi','AtAiAt','BtBiBt','PA','PB','ImPA','ImPB','A','B','At','Bt']
            diff = []
            for key in floatkeys:
                if verbose: print key, 'checking'
                #ok = np.allclose(self[key], info[key], tol)
                ok = round(la.norm(np.array(self[key])-np.array(info[key]))/tol) == 0
                if verbose: print key, ok
                if not ok: return False
                #try: diff.append(round(la.norm(np.array(self[key])-np.array(info[key]))/tol)==0)
                #except: diff.append(False)
            for key in intkeys:
                if verbose: print key, 'checking'
                ok = la.norm(np.array(self[key])-np.array(info[key])) == 0
                if verbose: print key, ok
                if not ok: return False
                #try: diff.append(la.norm(np.array(self[key])-np.array(info[key]))==0)
                #except: diff.append(False)
            for key in infomatrices:
                if verbose: print key, 'checking'
                # XXX made a switch here. ok?
                #ok = la.norm(np.array(self[key])-np.array(info[key])) == 0
                #ok = la.norm((self[key]-info[key]).todense()) == 0
                ok = np.all(self[key].todense() == info[key].todense())
                if verbose: print key, ok
                if not ok: return False
                #try: diff.append(la.norm((self[key]-info[key]).todense())==0)
                #except: diff.append(False)
            diff.append(True)

            bool = True
            for i in diff: bool &= i
            #print the first key found different (this will only trigger when the two info's have the same shape, so probably not very useful)
            if verbose and not bool:
                for i in range(len(diff)):
                    if not diff[i]: print allkeys[i]
            return bool
        except(ValueError):
            print "info doesn't have the same shape"
            return False
