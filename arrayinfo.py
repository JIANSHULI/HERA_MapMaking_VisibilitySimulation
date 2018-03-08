import numpy as np
import itertools
from info import RedundantInfo

def filter_reds(reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None):
    '''Filter redundancies to include/exclude the specified bls, antennas, and unique bl groups.'''
    if ubls or ex_ubls:
        bl2gp = {}
        for i,gp in enumerate(reds):
            for bl in gp: bl2gp[bl] = bl2gp[bl[::-1]] = bl2gp.get(bl,[]) + [i]
        if ubls: ubls = reduce(lambda x,y: x+y, [bl2gp[bl] for bl in ubls if bl2gp.has_key(bl)])
        else: ubls = range(len(reds))
        if ex_ubls: ex_ubls = reduce(lambda x,y: x+y, [bl2gp[bl] for bl in ex_ubls if bl2gp.has_key(bl)])
        else: ex_ubls = []
        reds = [gp for i,gp in enumerate(reds) if i in ubls and i not in ex_ubls]
    if bls is None: bls = [bl for gp in reds for bl in gp]
    if ex_bls: bls = [(i,j) for i,j in bls if (i,j) not in ex_bls and (j,i) not in ex_bls]
    if ants: bls = [(i,j) for i,j in bls if i in ants and j in ants]
    if ex_ants: bls = [(i,j) for i,j in bls if i not in ex_ants and j not in ex_ants]
    bld = {}
    for bl in bls: bld[bl] = bld[bl[::-1]] = None
    reds = [[bl for bl in gp if bld.has_key(bl)] for gp in reds]
    return [gp for gp in reds if len(gp) > 1]

def compute_reds(antpos, tol=0.1):
    '''Return redundancies on the basis of antenna positions.  As in RedundantInfo.init_from_reds, each
    list element consists of a list of (i,j) antenna indices whose separation vectors (pos[j]-pos[i])
    fall within the specified tolerance of each other.  'antpos' is a (nant,3) array of antenna positions.'''
    bls = [(i,j) for i in xrange(antpos.shape[0]) for j in xrange(i+1,antpos.shape[0])]
    # Coarsely sort bls using absolute grid (i.e. not relative separations); some groups may have several uids
    def sep(i,j): return antpos[j] - antpos[i]
    def uid(s): return tuple(map(int,np.around(s/tol)))
    ublgp,ubl_v = {},{}
    for bl in bls:
        s = sep(*bl); u = uid(s)
        ubl_v[u] = ubl_v.get(u,0) + s
        ublgp[u] = ublgp.get(u,[]) + [bl]
    for u in ubl_v: ubl_v[u] /= len(ublgp[u])
    # Now combine neighbors and Hermitian neighbors if within tol
    def neighbors(u):
        for du in itertools.product((-1,0,1),(-1,0,1),(-1,0,1)): yield (u[0]+du[0],u[1]+du[1],u[2]+du[2])
    for u in ubl_v.keys(): # Using 'keys' here allows dicts to be modified, but results in missing keys
        if not ubl_v.has_key(u): continue # bail if this has been popped already
        for nu in neighbors(u):
            if not ubl_v.has_key(nu): continue # bail if nonexistant
            if u == nu: continue
            if np.linalg.norm(ubl_v[u] - ubl_v[nu]) < tol:
                ubl_v[u] = ubl_v[u] * len(ublgp[u]) + ubl_v.pop(nu) * len(ublgp[nu])
                ublgp[u] += ublgp.pop(nu)
                ubl_v[u] /= len(ublgp[u]) # final step in weighted avg of ubl vectors
        for nu in neighbors((-u[0],-u[1],-u[2])): # Find Hermitian neighbors
            if not ubl_v.has_key(nu): continue # bail if nonexistant
            if np.linalg.norm(ubl_v[u] + ubl_v[nu]) < tol: # note sign reversal
                ubl_v[u] = ubl_v[u] * len(ublgp[u]) - ubl_v.pop(nu) * len(ublgp[nu]) # note sign reversal
                ublgp[u] += [(j,i) for i,j in ublgp.pop(nu)]
                ubl_v[u] /= len(ublgp[u]) # final step in weighted avg of ubl vectors
    return [v for v in ublgp.values() if len(v) > 1] # no such thing as redundancy of one
    
def compute_reds_total(antpos, tol=0.1):
    '''Return redundancies on the basis of antenna positions.  As in RedundantInfo.init_from_reds, each
    list element consists of a list of (i,j) antenna indices whose separation vectors (pos[j]-pos[i])
    fall within the specified tolerance of each other.  'antpos' is a (nant,3) array of antenna positions.'''
    bls = [(i,j) for i in xrange(antpos.shape[0]) for j in xrange(i+1,antpos.shape[0])]
    # Coarsely sort bls using absolute grid (i.e. not relative separations); some groups may have several uids
    def sep(i,j): return antpos[j] - antpos[i]
    def uid(s): return tuple(map(int,np.around(s/tol)))
    ublgp,ubl_v = {},{}
    for bl in bls:
        s = sep(*bl); u = uid(s)
        ubl_v[u] = ubl_v.get(u,0) + s
        ublgp[u] = ublgp.get(u,[]) + [bl]
    for u in ubl_v: ubl_v[u] /= len(ublgp[u])
    # Now combine neighbors and Hermitian neighbors if within tol
    def neighbors(u):
        for du in itertools.product((-1,0,1),(-1,0,1),(-1,0,1)): yield (u[0]+du[0],u[1]+du[1],u[2]+du[2])
    for u in ubl_v.keys(): # Using 'keys' here allows dicts to be modified, but results in missing keys
        if not ubl_v.has_key(u): continue # bail if this has been popped already
        for nu in neighbors(u):
            if not ubl_v.has_key(nu): continue # bail if nonexistant
            if u == nu: continue
            if np.linalg.norm(ubl_v[u] - ubl_v[nu]) < tol:
                ubl_v[u] = ubl_v[u] * len(ublgp[u]) + ubl_v.pop(nu) * len(ublgp[nu])
                ublgp[u] += ublgp.pop(nu)
                ubl_v[u] /= len(ublgp[u]) # final step in weighted avg of ubl vectors
        for nu in neighbors((-u[0],-u[1],-u[2])): # Find Hermitian neighbors
            if not ubl_v.has_key(nu): continue # bail if nonexistant
            if np.linalg.norm(ubl_v[u] + ubl_v[nu]) < tol: # note sign reversal
                ubl_v[u] = ubl_v[u] * len(ublgp[u]) - ubl_v.pop(nu) * len(ublgp[nu]) # note sign reversal
                ublgp[u] += [(j,i) for i,j in ublgp.pop(nu)]
                ubl_v[u] /= len(ublgp[u]) # final step in weighted avg of ubl vectors
    return [v for v in ublgp.values() if len(v) >= 1] # no such thing as redundancy of one

class ArrayInfo:
    '''Store information about an antenna array needed for computing redundancy and indexing matrices.'''
    # XXX i think we're moving toward getting rid of this class, because it is all derivative
    def __init__(self, nTotalAnt, badAntenna=[], badUBLpair=[]):
        self.nTotalAnt = nTotalAnt
        self.antennaLocation = np.zeros((nTotalAnt, 3))
        # XXX don't like next 2 lines.  need to avoid guessing.
        side = int(nTotalAnt**.5)
        for a in range(nTotalAnt): self.antennaLocation[a] = np.array([a/side, a%side, 0])
        self.badAntenna = badAntenna
        self.badUBLpair = badUBLpair
        #PAPER miriad convention by default
        self.totalVisibilityId = np.concatenate([[[i,j] for i in range(j+1)] for j in range(nTotalAnt)])
    def filter_reds(self, reds, bls=None, ex_bls=None, ants=None, ex_ants=None, ubls=None, ex_ubls=None):
        '''Filter redundancies to include/exclude the specified bls, antennas, and unique bl groups.'''
        return filter_reds(reds, bls=bls, ex_bls=ex_bls, ants=ants, ex_ants=ex_ants, ubls=ubls, ex_ubls=ex_ubls)
    def compute_reds(self, tol=0.1):
        '''Return redundancies on the basis of antenna positions.  As in RedundantInfo.init_from_reds, each
        list element consists of a list of (i,j) antenna indices whose separation vectors (pos[j]-pos[i])
        fall within the specified tolerance of each other.'''
        return compute_reds(self.antennaLocation, tol=tol)
    def compute_redundantinfo(self, tol=1e-6):
        '''Use provided antenna locations (in arrayinfoPath) to derive redundancy equations'''
        reds = self.compute_reds(tol=tol)
        reds = self.filter_reds(reds, bls=self.totalVisibilityId.keys(), 
                ex_ants=list(self.badAntenna), ex_ubls=[tuple(p) for p in self.badUBLpair])
        info = RedundantInfo()
        info.init_from_reds(reds, self.antennaLocation)
        return info

import scipy.sparse as sps
import numpy.linalg as la

class ArrayInfoLegacy(ArrayInfo):
    '''Legacy interface/mechanism for ArrayInfo.  Deprecated.'''
    def __init__(self, nTotalAnt, badAntenna=[], badUBLpair=[], tol=1e-6):
        ArrayInfo.__init__(self, nTotalAnt, badAntenna=badAntenna, badUBLpair=badUBLpair)
        self.antennaLocationTolerance = tol
        self.nTotalBaselineAuto = (nTotalAnt + 1) * nTotalAnt / 2
        self.nTotalBaselineCross = (nTotalAnt - 1) * nTotalAnt / 2
        self.totalVisibilityUBL = None
        self._gen_totalVisibilityId_dic()
    def _gen_totalVisibilityId_dic(self):
        self.totalVisibilityId_dic = {}
        for i, (a1,a2) in enumerate(self.totalVisibilityId): self.totalVisibilityId_dic[(a1,a2)] = i
    def compute_redundantinfo(self, arrayinfoPath=None, tol=1e-6):
        if arrayinfoPath is not None: self.read_arrayinfo(arrayinfoPath)
        ArrayInfo.compute_redundantinfo(self, tol=tol)
    def read_arrayinfo(self, arrayinfopath, verbose=False):
        '''array info is the minimum set of information to uniquely describe a 
        redundant array, and is needed to compute redundant info. It includes, 
        in each line, bad antenna indices, bad unique bl indices, tolerance 
        of error when checking redundancy, antenna locations, and visibility's 
        antenna pairing conventions. Unlike redundant info which is a self-contained 
        dictionary, items in array info each have their own fields in the instance.'''
        if verbose: print "Reading", arrayinfopath
        with open(arrayinfopath) as f: rawinfo = [[float(x) for x in line.split()] for line in f]
        self.badAntenna = np.array(rawinfo[0], dtype=np.int)
        if self.badAntenna[0] < 0: self.badAntenna = np.zeros(0) # XXX special significance for < 0?
        rawpair = np.array(rawinfo[1], dtype=np.int)
        if rawpair.shape[0] == 0 or rawpair.shape[0] % 2 != 0 or rawpair.min() < 0: # XXX shouldn't accept bad states
            self.badUBLpair = np.array([])
        else: self.badUBLpair = np.reshape(rawpair,(len(rawpair)/2,2))
        self.antennaLocationTolerance = rawinfo[2][0]
        for a in range(len(self.antennaLocation)):
            assert(len(rawinfo[a+3]) == 3)
            self.antennaLocation[a] = np.array(rawinfo[a+3])
        bl = 0
        vis_id = []
        max_bl_cnt = self.nTotalAnt * (self.nTotalAnt + 1) / 2
        maxline = len(rawinfo)
        while len(rawinfo[bl + 3 + len(self.antennaLocation)]) == 2: # XXX don't like while loop
            assert(bl < max_bl_cnt)
            vis_id.append(np.array(rawinfo[bl + 3 + len(self.antennaLocation)], dtype=np.int))
            bl += 1
            if bl + 3 + len(self.antennaLocation) >= maxline: break
        self.totalVisibilityId = np.array(vis_id, dtype=np.int)
        self._gen_totalVisibilityId_dic()
    def get_baseline(self,bl): # XXX unused except for legacy _compute_redundantinfo
        '''inverse function of totalVisibilityId, calculate the bl index from 
        the antenna pair. It allows flipping of a1 and a2, will return same result'''
        bl = tuple(bl)
        try: return self.totalVisibilityId_dic[bl]
        except(KeyError): pass
        try: return self.totalVisibilityId_dic[bl[::-1]]
        except(KeyError): return None
    def compute_UBL(self,tolerance = 0.1): # XXX unused except legacy _compute_redundantinfo
        '''XXX DOCSTRING'''
        if tolerance == 0:
            tolerance = np.min(np.linalg.norm(np.array(self.antennaLocation) - self.antennaLocation[0], axis=1)) / 1e6
        ubl = {}
        for bl, (a1,a2) in enumerate(self.totalVisibilityId):
            if a1 != a2 and a1 not in self.badAntenna and a2 not in self.badAntenna:
                loc_tuple = tuple(np.round((self.antennaLocation[a2] - self.antennaLocation[a1]) / float(tolerance)) * tolerance)
                neg_loc_tuple = tuple(np.round((self.antennaLocation[a1] - self.antennaLocation[a2]) / float(tolerance)) * tolerance)
                if loc_tuple in ubl: ubl[loc_tuple].add(bl + 1)
                elif neg_loc_tuple in ubl: ubl[neg_loc_tuple].add(- bl - 1)
                else:
                    if loc_tuple[0] >= 0: ubl[loc_tuple] = set([bl + 1])
                    else: ubl[neg_loc_tuple] = set([-bl - 1])
        #calculate actual average of the gridded bls vectors to get an accurate representation of the ubl vector
        ubl_vec = np.zeros((len(ubl), 3))
        self.totalVisibilityUBL = {}
        ublcount = np.zeros(len(ubl))
        for u, grid_ubl_vec in enumerate(ubl):
            for bl in ubl[grid_ubl_vec]:
                assert bl != 0
                a1, a2 = self.totalVisibilityId[abs(bl) - 1]
                if bl > 0: ubl_vec[u] = ubl_vec[u] + self.antennaLocation[a2] - self.antennaLocation[a1]
                else: ubl_vec[u] = ubl_vec[u] + self.antennaLocation[a1] - self.antennaLocation[a2]
                self.totalVisibilityUBL[(a1, a2)] = u
            ublcount[u] = len(ubl[grid_ubl_vec])
            ubl_vec[u] = ubl_vec[u] / ublcount[u]
        reorder = (ubl_vec[:,1]*1e9 + ubl_vec[:,0]).argsort()
        rereorder = reorder.argsort()
        for key in self.totalVisibilityUBL:
            self.totalVisibilityUBL[key] = rereorder[self.totalVisibilityUBL[key]]
        ubl_vec = ubl_vec[reorder]
        #now I need to deal with the fact that no matter how coarse my grid is, it's possible for a single group of ubl to fall into two adjacent grids. So I'm going to check if any of the final ubl vectors are seperated by less than tolerance. If so, merge them
        ublmap = {}
        for u1 in range(len(ubl_vec)):
            for u2 in range(u1):
                if la.norm(ubl_vec[u2] - ubl_vec[u1]) < tolerance or la.norm(ubl_vec[u2] + ubl_vec[u1]) < tolerance:
                    ublmap[u1] = u2
                    ubl_vec[u2] = (ubl_vec[u1] * ublcount[u1] + ubl_vec[u2] * ublcount[u2]) / (ublcount[u1] + ublcount[u2])
                    break
            ublmap[u1] = u1
        merged_ubl_vec = []
        for u in range(len(ubl_vec)):
            if ublmap[u] == u:
                merged_ubl_vec.append(ubl_vec[u])
                ublmap[u] = len(merged_ubl_vec) - 1
            else: ublmap[u] = ublmap[ublmap[u]]
        merged_ubl_vec = np.array(merged_ubl_vec)
        for key in self.totalVisibilityUBL:
            self.totalVisibilityUBL[key] = ublmap[self.totalVisibilityUBL[key]]
        return ubl_vec
    def compute_redundantinfo(self, arrayinfoPath=None, tol=1e-6): # XXX remove this legacy interface?
        '''Legacy version of compute_redundantinfo if you need subsetbls for data ordering.'''
        self.antennaLocationTolerance = tol
        if arrayinfoPath is not None: self.read_arrayinfo(arrayinfoPath)
        info = RedundantInfo()
        # exclude bad antennas
        info['subsetant'] = subsetant = np.array([i for i in xrange(self.antennaLocation.shape[0]) 
                if i not in self.badAntenna], dtype=np.int32)
        info['nAntenna'] = nAntenna = len(subsetant) # XXX maybe have C api automatically infer this
        info['antloc'] = antloc = np.array([self.antennaLocation[i] for i in subsetant], dtype=np.float32)
        ublall = self.compute_UBL(tol)
        #delete the bad ubl's
        badUBL = {}
        def dis(a1,a2): return np.linalg.norm(a1-a2)
        for a1,a2 in self.badUBLpair:
            bl = self.antennaLocation[a1] - self.antennaLocation[a2]
            for i,ubl in enumerate(ublall):
                if dis(bl,ubl) < tol or dis(bl,-ubl) < tol: badUBL[i] = None
        ubl2goodubl = {}
        def f(i,u):
            ubl2goodubl[i] = len(ubl2goodubl)
            return u
        info['ubl'] = ubl = np.array([f(i,u) for i,u in enumerate(ublall) if not badUBL.has_key(i)], dtype=np.float32)
        for k in badUBL: ubl2goodubl[k] = -1
        nUBL = ubl.shape[0] # XXX maybe have C api automatically infer this
        badubl = [ublall[i] for i in badUBL]
        #find nBaseline (include auto bls) and subsetbl
        #bl2d:  from 1d bl index to a pair of antenna numbers
        bl2d = [] # XXX cleaner way to do this?
        for i,ai in enumerate(antloc):
            for j,aj in enumerate(antloc[:i+1]):
                blij = ai - aj
                flag = False
                for bl in badubl:
                    if dis(blij,bl) < tol or dis(blij,-bl) < tol:
                        flag = True
                        break
                if not flag: bl2d.append((i,j))
        # exclude pairs that are not in totalVisibilityId
        tmp = []
        for p in bl2d:
            bl = (subsetant[p[0]],subsetant[p[1]])
            if self.totalVisibilityId_dic.has_key(bl): tmp.append(p)
            elif self.totalVisibilityId_dic.has_key(bl[::-1]): tmp.append(p[::-1])
        bl2d = np.array(tmp, dtype=np.int32)
        crossindex = np.array([i for i,p in enumerate(bl2d) if p[0] != p[1]], dtype=np.int32)
        nBaseline = len(bl2d)
        bl2d = bl2d[crossindex] # make bl2d only hold crosscorrelations
        info['nBaseline'] = len(bl2d) # XXX maybe have C api infer this
        # from a pair of good antenna index to bl index
        info['subsetbl'] = np.array([self.get_baseline([subsetant[bl[0]],subsetant[bl[1]]]) 
                for bl in bl2d], dtype=np.int32)
        #bltoubl: cross bl number to ubl index
        def findublindex(p1,p2):
            a1,a2 = subsetant[p1],subsetant[p2]
            if (a1,a2) in self.totalVisibilityUBL: return ubl2goodubl[self.totalVisibilityUBL[(a1,a2)]]
        info['bltoubl'] = bltoubl = np.array([findublindex(*p) for p in bl2d if p[0] != p[1]], dtype=np.int32)
        #reversed:   cross only bl if reversed -1, otherwise 1
        crosspair = [p for p in bl2d if p[0] != p[1]]
        reverse = []
        for k,cpk in enumerate(crosspair):
            bl = antloc[cpk[0]] - antloc[cpk[1]]
            if dis(bl,ubl[bltoubl[k]]) < tol: reverse.append(-1)
            elif dis(bl,-ubl[bltoubl[k]]) < tol: reverse.append(1)
            else : raise ValueError('bltoubl[%d] points to wrong ubl index' % (k))
        reverse = np.array(reverse, dtype=np.int32)
        info._reversed = reverse # XXX store this to remember what we did
        bl2d0 = np.where(reverse == 1, bl2d[:,0], bl2d[:,1])
        bl2d1 = np.where(reverse == 1, bl2d[:,1], bl2d[:,0])
        bl2d[:,0],bl2d[:,1] = bl2d0,bl2d1
        crosspair = [p for p in bl2d if p[0] != p[1]] # recompute crosspair for reversed indices
        info.bl2d = bl2d
        #ublcount:  for each ubl, the number of good cross bls corresponding to it
        cnt = {}
        for bl in bltoubl: cnt[bl] = cnt.get(bl,0) + 1
        info['ublcount'] = np.array([cnt[i] for i in range(nUBL)], dtype=np.int32)
        #ublindex:  //for each ubl, the set of corresponding indices of baselines in bl2d
        cnt = {}
        for i,(a1,a2) in enumerate(crosspair): cnt[bltoubl[i]] = cnt.get(bltoubl[i],[]) + [[a1,a2,i]]
        ublindex = np.concatenate([np.array(cnt[i],dtype=np.int32) for i in range(nUBL)])
        newind = np.arange(nBaseline)[crossindex] = np.arange(crossindex.size, dtype=np.int32)
        info.ublindex = newind[ublindex[:,2]]
        #bl1dmatrix: a symmetric matrix where col/row numbers index ants and entries are bl index (no auto corr)
        bl1dmatrix = (2**31-1) * np.ones((nAntenna,nAntenna),dtype=np.int32) # XXX don't like 2**31-1.  whence this number?
        for i,cp in enumerate(crosspair): bl1dmatrix[cp[1],cp[0]], bl1dmatrix[cp[0],cp[1]] = i,i
        info['bl1dmatrix'] = bl1dmatrix
        #degenM:
        a = np.array([np.append(ai,1) for ai in antloc], dtype=np.float32)
        d = np.array([np.append(ubli,0) for ubli in ubl], dtype=np.float32)
        m1 = -a.dot(la.pinv(a.T.dot(a))).dot(a.T)
        m2 = d.dot(la.pinv(a.T.dot(a))).dot(a.T)
        info['degenM'] = np.append(m1,m2,axis=0)
        #A: A matrix for logcal amplitude
        A = np.zeros((len(crosspair),nAntenna+nUBL))
        for i,cp in enumerate(crosspair): A[i,cp[0]], A[i,cp[1]], A[i,nAntenna+bltoubl[i]] = 1,1,1
        info['At'] = sps.csr_matrix(A).T
        #B: B matrix for logcal phase
        B = np.zeros((len(crosspair),nAntenna+nUBL))
        #for i,cp in enumerate(crosspair): B[i,cp[0]], B[i,cp[1]], B[i,nAntenna+bltoubl[i]] = -reverse[i],reverse[i],1
        for i,cp in enumerate(crosspair): B[i,cp[0]], B[i,cp[1]], B[i,nAntenna+bltoubl[i]] = -1,1,1
        info['Bt'] = sps.csr_matrix(B).T
        info.update()
        return info
