print("Jianshu Li")

4+2
4+7

function mandel(z, maxiter)
    c = z
    for n in 1:maxiter
        if abs(z) > 2
            return n - 1
        end
        z = z^2 + c
    end
    return maxiter
end

mandel(1,100)

length(1:1)

list1 = 1:10
list2 = list1[2:5]

print("$list2")

import Pkg

# ENV["PYTHON"] = "/Users/JianshuLi/anaconda3/envs/Cosmology-Python27/bin/python"
ENV["PYTHON"] = "/nfs/blender/data/jshu_li/anaconda3/envs/Cosmology_python27/bin/python"
Pkg.add("PyCall")
Pkg.build("PyCall")
using PyCall
@pyimport numpy as np
@pyimport sys
print(sys.executable)

@pyimport HERA_MapMaking_VisibilitySimulation as mmvs
print(mmvs.DATA_PATH)

@pyimport matplotlib.pyplot as plt


array1 = np.array([i for i in np.arange(5)])
length(array1)
print(array1)


try
    print("Jianshu")
catch
    print("Li")
end
