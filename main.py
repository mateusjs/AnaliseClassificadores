import classificadores
import numpy as np

knn = []
dst = []
nb = []
svm = []
mlp = []

for _ in range(10):
    knn.append(classificadores.knn())
    dst.append(classificadores.dst())
    nb.append(classificadores.nb())
    svm.append(classificadores.svm())
    mlp.append(classificadores.mlp())
    classificadores.update_data()

with open("medias.csv", "w") as fp:

    fp.write("Knn, Dst, Nb, Svm, Mlp\n")
    for index in range(10):
        fp.write("%f, %f, %f, %f, %f\n" % (knn[index], dst[index], nb[index], svm[index], mlp[index]))

    fp.write("%f, %f, %f, %f, %f\n" % (np.mean(knn), np.mean(dst), np.mean(nb), np.mean(svm), np.mean(mlp)))

