import csv
import collections
import numpy as np
import time
import copy
import math
from joblib import Parallel, delayed

Geometry = collections.namedtuple("Geometry", "vertices, normals, faces, edges, adjacency")
Vertex = collections.namedtuple("Vertex",
                                "index,position,normal,neighbouringFaceIndices,neighbouringVerticesIndices,rotationAxis,theta")
Face = collections.namedtuple("Face",
                              "index,centroid,vertices,verticesIndices,faceNormal,area,edgeIndices,neighbouringFaceIndices,guidedNormal,rotationAxis,theta")
Edge = collections.namedtuple("Edge", "index,vertices,verticesIndices,length,facesIndices,edgeNormal")


def loadObj(filename):
    vertices = []
    normals = []
    faces = []
    edges = []
    adjacency = []
    with open(filename, newline='') as f:
        flines = f.readlines()
        # Read vertices
        indexCounter = 0;
        for row in flines:
            if row[0] == 'v' and row[1] == ' ':
                line = row.rstrip()
                line = line[2:len(line)]
                coords = line.split()
                coords = list(map(float, coords))
                v = Vertex(
                    index=indexCounter,
                    position=np.asarray([coords[0], coords[1], coords[2]]),
                    normal=np.asarray([0.0, 0.0, 0.0]),
                    neighbouringFaceIndices=[],
                    neighbouringVerticesIndices=[],
                    theta=0.0,
                    rotationAxis=np.asarray([0.0, 0.0, 0.0])
                )
                indexCounter += 1;
                vertices.append(v)
        # Read Faces
        indexCounter = 0;
        for row in flines:
            if row[0] == 'f':
                line = row.rstrip()
                line = line[2:len(line)]
                lineparts = line.strip().split()
                faceline = [];
                for fi in lineparts:
                    fi = fi.split('/')
                    faceline.append(int(fi[0]) - 1)
                f = Face(
                    index=indexCounter,
                    verticesIndices=[int(faceline[0]), int(faceline[1]), int(faceline[2])],
                    vertices=[],
                    centroid=np.asarray([0.0, 0.0, 0.0]),
                    faceNormal=np.asarray([0.0, 0.0, 0.0]),
                    edgeIndices=[],
                    area=0.0,
                    neighbouringFaceIndices=[],
                    guidedNormal=np.asarray([0.0, 0.0, 0.0]),
                    theta=0.0,
                    rotationAxis=np.asarray([0.0, 1.0, 0.0])
                )
                indexCounter += 1;
                faces.append(f)

        ## Connectivity
        # Which faces are neighbouring to each vertex
        for idx_f, f in enumerate(faces):
            for idx_v, v in enumerate(f.verticesIndices):
                vertices[v].neighbouringFaceIndices.append(f.index)

        # Which vertices are neighbouring to each vertex
        for idx_f, f in enumerate(faces):
            v0 = f.verticesIndices[0]
            v1 = f.verticesIndices[1]
            v2 = f.verticesIndices[2]
            if v1 not in vertices[v0].neighbouringVerticesIndices:
                vertices[v0].neighbouringVerticesIndices.append(v1)
            if v2 not in vertices[v0].neighbouringVerticesIndices:
                vertices[v0].neighbouringVerticesIndices.append(v2)
            if v0 not in vertices[v1].neighbouringVerticesIndices:
                vertices[v1].neighbouringVerticesIndices.append(v0)
            if v2 not in vertices[v1].neighbouringVerticesIndices:
                vertices[v1].neighbouringVerticesIndices.append(v2)
            if v0 not in vertices[v2].neighbouringVerticesIndices:
                vertices[v2].neighbouringVerticesIndices.append(v0)
            if v1 not in vertices[v2].neighbouringVerticesIndices:
                vertices[v2].neighbouringVerticesIndices.append(v1)

        # Process edges which edges belong to each face and which faces belong to each edge
        edges_ = []
        for idx_f, f in enumerate(faces):
            edge_ = [f.verticesIndices[0], f.verticesIndices[1]]
            if (edge_ not in edges_) and (edge_.reverse() not in edges_):
                edges_.append(edge_)
            edge_ = [f.verticesIndices[1], f.verticesIndices[2]]
            if (edge_ not in edges_) and (edge_.reverse() not in edges_):
                edges_.append(edge_)
            edge_ = [f.verticesIndices[2], f.verticesIndices[0]]
            if (edge_ not in edges_) and (edge_.reverse() not in edges_):
                edges_.append(edge_)

        for ind_e, e_ in enumerate(edges_):
            facesIndices = []
            for idx_f, f in enumerate(faces):
                if (e_[0] in f.verticesIndices) and (e_[1] in f.verticesIndices):
                    facesIndices.append(f.index)
            E = Edge(
                index=ind_e,
                vertices=[],
                verticesIndices=[e_[0], e_[1]],
                length=0.0,
                facesIndices=facesIndices,
                edgeNormal=np.asarray([1.0,0.0,0.0])
            )
            edges.append(E)
            for fIndex in facesIndices:
                faces[fIndex].edgeIndices.append(ind_e)

        # Process edges which edges belong to each face and which faces belong to each edge
        for idx_f, f in enumerate(faces):
            for idx_e, e in enumerate(f.edgeIndices):
                for idx_fe, fe in enumerate(edges[e].facesIndices):
                    if fe != f.index:
                        f.neighbouringFaceIndices.append(fe)

        # ## Positioning & orientation
        # # Get vertex objects for each face
        # for idx_f, f in enumerate(faces):
        #     v = [vertices[fvi] for fvi in f.verticesIndices]
        #     faces[idx_f] = faces[idx_f]._replace(vertices=v)
        # # Compute centroids
        # for idx_f, f in enumerate(faces):
        #     vPos = [vertices[i].position for i in f.verticesIndices]
        #     vPos = np.asarray(vPos)
        #     centroid = np.mean(vPos, axis=0)
        #     faces[idx_f] = faces[idx_f]._replace(centroid=centroid)
        # # Process cetroids and normals per face
        # for idx_f, f in enumerate(faces):
        #     bc = f.vertices[1].position - f.vertices[2].position
        #     ba = f.vertices[0].position - f.vertices[2].position
        #     normal = np.cross(bc, ba)
        #     faceArea = 0.5 * np.linalg.norm(normal)
        #     faces[idx_f] = faces[idx_f]._replace(area=faceArea)
        #     normalizedNormal = normal / np.linalg.norm(normal)
        #     f.faceNormal[0] = normalizedNormal[0]
        #     f.faceNormal[1] = normalizedNormal[1]
        #     f.faceNormal[2] = normalizedNormal[2]
        # # Process normals per vertex
        # for idx_v, v in enumerate(vertices):
        #     normal = np.asarray([0.0, 0.0, 0.0])
        #     for idx_f, f in enumerate(v.neighbouringFaceIndices):
        #         normal += faces[f].faceNormal
        #     normal = normal / len(v.neighbouringFaceIndices)
        #     normal = normal / np.linalg.norm(normal)
        #     v.normal[0] = normal[0]
        #     v.normal[1] = normal[1]
        #     v.normal[2] = normal[2]
        # # Process edge attributes
        # for idx_e, e in enumerate(edges):
        #     edges[idx_e] = edges[idx_e]._replace(vertices=[])
        #     edges[idx_e].vertices.append(vertices[e.verticesIndices[0]])
        #     edges[idx_e].vertices.append(vertices[e.verticesIndices[1]])
        #     length = np.linalg.norm(vertices[e.verticesIndices[0]].position - vertices[e.verticesIndices[1]].position)
        #     edges[idx_e] = edges[idx_e]._replace(length=length)

    return Geometry(
        vertices=vertices,
        normals=normals,
        faces=faces,
        edges=edges,
        adjacency=[]
    )


def addNoise(Geom, noiseLevel):
    avg_len = 0.0
    for idx_e, e in enumerate(Geom.edges):
        avg_len += e.length
    avg_len = avg_len / len(Geom.edges)
    stddev = avg_len * noiseLevel
    g = np.random.normal(0, stddev, len(Geom.vertices))
    #stddevList=[avg_len * np.random.uniform(low=0.01, high=0.3) for i in range(0,len(Geom.vertices))]
    #muList = [0.0 for i in range(0, len(Geom.vertices))]
    #g = np.random.normal(muList, stddevList, len(Geom.vertices))
    for idx_v, v in enumerate(Geom.vertices):
        Geom.vertices[idx_v].position[0] = Geom.vertices[idx_v].position[0] + Geom.vertices[idx_v].normal[0] * g[idx_v]
        Geom.vertices[idx_v].position[1] = Geom.vertices[idx_v].position[1] + Geom.vertices[idx_v].normal[1] * g[idx_v]
        Geom.vertices[idx_v].position[2] = Geom.vertices[idx_v].position[2] + Geom.vertices[idx_v].normal[2] * g[idx_v]


def exportObj(Geom, filename):
    V = []
    F = []
    for v in Geom.vertices:
        V.append(v.position)
    for f in Geom.faces:
        F.append(np.asarray(f.verticesIndices) + np.ones(np.shape(f.verticesIndices)))
    V = np.asarray(V)
    # V = np.round(V, 6)
    F = np.asarray(F)

    with open(filename, 'w') as writeFile:
        for j in range(0, np.size(V, axis=0)):
            line = "v " + str(V[j, 0]) + " " + str(V[j, 1]) + " " + str(V[j, 2])
            writeFile.write(line)
            writeFile.write('\n')
        for j in range(0, np.size(F, axis=0)):
            line = "f " + str(int(F[j, 0])) + " " + str(int(F[j, 1])) + " " + str(int(F[j, 2]))
            writeFile.write(line)
            writeFile.write('\n')
    print('Obj model ' + filename + ' exported')


def neighboursByFace(Geom, faceIndex, numOfNeighbours):
    patchFaces = []
    patchFaces.append(faceIndex)
    # for i in patchFaces:
    #     for j in Geom.faces[i].neighbouringFaceIndices:
    #         if len(patchFaces) < numOfNeighbours:
    #             for k in Geom.faces[j].neighbouringFaceIndices:
    #                 if k not in patchFaces:
    #                     patchFaces.append(k)
    for i in patchFaces:
        for j in Geom.faces[i].neighbouringFaceIndices:
            if len(patchFaces) < numOfNeighbours:
                if j not in patchFaces:
                    patchFaces.append(j)
    patchVertices = []
    for idx, i in enumerate(patchFaces):
        for j in Geom.faces[i].verticesIndices:
            if j not in patchVertices:
                patchVertices.append(j)
    W = np.zeros((len(patchVertices), len(patchVertices)))
    U = W
    Uf = U

    # for idx, i in enumerate(patchFaces):
    #     for j in Geom.faces[i].edgeIndices:
    #         a = Geom.edges[j].verticesIndices[0]
    #         a_ind = patchVertices.index(a)
    #         b = Geom.edges[j].verticesIndices[1]
    #         b_ind = patchVertices.index(b)
    #         W[a_ind, b_ind] = 1
    #         W[b_ind, a_ind] = 1
    # D=np.sum(W,axis=1)
    # L=D-W
    # S,U=np.linalg.eig(L)
    # I = np.argsort(S)
    # Uf=U[:,I]
    # Uf=U

    return W, Uf, patchVertices, patchFaces


def neighboursByVertex(Geom, VertexIndex, numOfNeighbours):
    patchVertices = []
    patchVertices.append(VertexIndex)
    for i in patchVertices:
        for j in Geom.vertices[i].neighbouringVerticesIndices:
            if j not in patchVertices:
                if len(patchVertices) < numOfNeighbours:
                    patchVertices.append(j)
    W = np.zeros((len(patchVertices), len(patchVertices)))
    for i in patchVertices:
        for j in Geom.vertices[i].neighbouringVerticesIndices:
            if j in patchVertices:
                a_ind = patchVertices.index(i)
                b_ind = patchVertices.index(j)
                W[a_ind, b_ind] = 1
                W[b_ind, a_ind] = 1

    D = np.sum(W, axis=1)
    L = D - W
    S, U = np.linalg.eig(L)
    I = np.argsort(S)
    Uf = U[:, I]

    return W, Uf, patchVertices


def updateGeometryConnectivity(Geom):
    print('Updating geometry connectivity')


def updateGeometryAttibutes(Geom, useGuided=False, numOfFacesForGuided=10):
    print('Updating geometry attributes')
    ## Positioning & orientation

    print('Get vertex objects for each face')
    for idx_f, f in enumerate(Geom.faces):
        v = [Geom.vertices[fvi] for fvi in Geom.faces[idx_f].verticesIndices]
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(vertices=v)
    print('Compute centroids')
    for idx_f, f in enumerate(Geom.faces):
        vPos = [Geom.vertices[i].position for i in Geom.faces[idx_f].verticesIndices]
        vPos = np.asarray(vPos)
        centroid = np.mean(vPos, axis=0)
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(centroid=centroid)
    print('Process cetroids and normals per face')
    for idx_f, f in enumerate(Geom.faces):
        bc = Geom.faces[idx_f].vertices[1].position - Geom.faces[idx_f].vertices[2].position
        ba = Geom.faces[idx_f].vertices[0].position - Geom.faces[idx_f].vertices[2].position
        normal = np.cross(bc, ba)
        faceArea = 0.5 * np.linalg.norm(normal)
        Geom.faces[idx_f] = Geom.faces[idx_f]._replace(area=faceArea)
        normalizedNormal = normal / np.linalg.norm(normal)
        Geom.faces[idx_f].faceNormal[0] = normalizedNormal[0]
        Geom.faces[idx_f].faceNormal[1] = normalizedNormal[1]
        Geom.faces[idx_f].faceNormal[2] = normalizedNormal[2]
    print('Process normals per vertex')
    for idx_v, v in enumerate(Geom.vertices):
        normal = np.asarray([0.0, 0.0, 0.0])
        for idx_f, f in enumerate(Geom.vertices[idx_v].neighbouringFaceIndices):
            normal += Geom.faces[f].faceNormal
        normal = normal / len(Geom.vertices[idx_v].neighbouringFaceIndices)
        normal = normal / np.linalg.norm(normal)
        Geom.vertices[idx_v].normal[0] = normal[0]
        Geom.vertices[idx_v].normal[1] = normal[1]
        Geom.vertices[idx_v].normal[2] = normal[2]
    print('Process edge attributes')
    for idx_e, e in enumerate(Geom.edges):
        Geom.edges[idx_e] = Geom.edges[idx_e]._replace(vertices=[])
        Geom.edges[idx_e].vertices.append(Geom.vertices[e.verticesIndices[0]])
        Geom.edges[idx_e].vertices.append(Geom.vertices[e.verticesIndices[1]])
        length = np.linalg.norm(
            Geom.vertices[e.verticesIndices[0]].position - Geom.vertices[e.verticesIndices[1]].position)
        Geom.edges[idx_e] = Geom.edges[idx_e]._replace(length=length)

    if useGuided:
        print('Process guided')
        numOfFaces_ = numOfFacesForGuided
        patches = []
        for i in range(0, len(Geom.faces)):
            # print('Start searching patces for face '+str(i)+' '+ str(time.time() - t))
            w, u, pv, p = neighboursByFace(Geom, i, numOfFaces_)
            patches.append(p)
            # print('Searching patces for face complete '+str(i)+ ' '+str(time.time() - t))
        for idx_f, f in enumerate(Geom.faces):
            if idx_f != f.index:
                print('Maybe prob', f.index)
            selectedPatches = []
            for p in patches:
                if f.index in p:
                    selectedPatches.append(p)
            patchFactors = []
            for p in selectedPatches:
                patchFaces = [Geom.faces[i] for i in p]
                patchNormals = [pF.faceNormal for pF in patchFaces]
                normalsDiffWithinPatch = [np.linalg.norm(patchNormals[0] - p, 2) for p in patchNormals]
                maxDiff = max(normalsDiffWithinPatch)
                patchNormals = np.asarray(patchNormals)
                M = np.matmul(np.transpose(patchNormals), patchNormals)
                w, v = np.linalg.eig(M)
                eignorm = np.linalg.norm(np.diag(v))
                patchFactor = eignorm * maxDiff
                patchFactors.append(patchFactor)
            minIndex = np.argmin(np.asarray(patchFactors))
            p = selectedPatches[minIndex]
            patchFaces = [Geom.faces[i] for i in p]
            weightedNormalFactors = [pF.area * pF.faceNormal for pF in patchFaces]
            weightedNormalFactors = np.asarray(weightedNormalFactors)
            weightedNormal = np.mean(weightedNormalFactors, axis=0)
            weightedNormal = weightedNormal / np.linalg.norm(weightedNormal)
            Geom.faces[f.index] = Geom.faces[f.index]._replace(guidedNormal=weightedNormal)


def updateVerticesWithNormals(Geom, N, nIter):
    for i in range(0, nIter):
        for idx_v, v in enumerate(Geom.vertices):
            updatedNormal = np.asarray([0.0, 0.0, 0.0])
            for vf in Geom.vertices[idx_v].neighbouringFaceIndices:
                # Geom.faces[vf]
                n = N[vf, :]
                # print(n)
                updateFactor = np.dot(n, (Geom.vertices[Geom.faces[vf].verticesIndices[0]].position - Geom.vertices[
                    idx_v].position)) * n + \
                               np.dot(n, (Geom.vertices[Geom.faces[vf].verticesIndices[1]].position - Geom.vertices[
                                   idx_v].position)) * n + \
                               np.dot(n, (Geom.vertices[Geom.faces[vf].verticesIndices[2]].position - Geom.vertices[
                                   idx_v].position)) * n
                # print(updateFactor)
                updatedNormal = updateFactor + updatedNormal
            updatedNormal = updatedNormal / (3 * len(v.neighbouringFaceIndices))
            Geom.vertices[idx_v] = Geom.vertices[idx_v]._replace(position=Geom.vertices[idx_v].position + updatedNormal)



def updateVerticesWithNormalsBilateral(Geom, N, nIter):
    for i in range(0, nIter):
        for idx_v, v in enumerate(Geom.vertices):
            updatedNormal = np.asarray([0.0, 0.0, 0.0])
            for vf in Geom.vertices[idx_v].neighbouringFaceIndices:
                # Geom.faces[vf]
                n = N[vf, :]
                # print(n)
                updateFactor = np.dot(n, (Geom.vertices[Geom.faces[vf].verticesIndices[0]].position - Geom.vertices[
                    idx_v].position)) * n + \
                               np.dot(n, (Geom.vertices[Geom.faces[vf].verticesIndices[1]].position - Geom.vertices[
                                   idx_v].position)) * n + \
                               np.dot(n, (Geom.vertices[Geom.faces[vf].verticesIndices[2]].position - Geom.vertices[
                                   idx_v].position)) * n
                # print(updateFactor)
                updatedNormal = updateFactor + updatedNormal
            updatedNormal = updatedNormal / (3 * len(v.neighbouringFaceIndices))
            Geom.vertices[idx_v] = Geom.vertices[idx_v]._replace(position=Geom.vertices[idx_v].position + updatedNormal)




def asCartesian(rthetaphi):
    # takes list rthetaphi (single coord)
    r = rthetaphi[0]
    theta = rthetaphi[1] * np.pi / 180  # to radian
    phi = rthetaphi[2] * np.pi / 180
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]


def asSpherical(xyz):
    # takes list xyz (single coord)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r) * 180 / np.pi  # to degrees
    phi = np.arctan2(y, x) * 180 / np.pi
    if phi < 0:
        phi = phi + 360
    if theta < 0:
        theta = theta + 360
    return [r, theta, phi]


def mat2Sph(M):
    for i in range(0, np.size(M, axis=1)):
        xyz = M[:, i]
        r, theta, phi = asSpherical(xyz)
        M[0, i] = r
        M[1, i] = theta
        M[2, i] = phi
    return M


def mat2Cartesian(M):
    for i in range(0, np.size(M, axis=1)):
        rthetaphi = M[:, i]
        x, y, z = asSpherical(rthetaphi)
        M[0, i] = x
        M[1, i] = y
        M[2, i] = z
    return M


# def rotate(a, axis, theta):
#     if theta != 0:
#         arot = a * math.cos(theta) + np.cross(axis, a) * math.sin(theta) + axis * np.dot(axis, a) * (
#                 1 - math.cos(theta))
#     else:
#         arot = a
#     return arot

def rotate(a, axis, theta):
    if theta != 0:
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        a = a * costheta + np.cross(axis, a) * sintheta + axis * np.dot(axis, a) * (1 - costheta)
    return a

def computeRotation(vec, target):
    vec = vec / np.linalg.norm(vec)
    target = target / np.linalg.norm(target)
    theta = math.acos(np.dot(vec, target) / (np.linalg.norm(vec) * np.linalg.norm(target)))
    axis = np.cross(vec, target)
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
    return axis, theta

##############################################################################
################################ Test ########################################
##############################################################################
#
# if False:
#     t = time.time()
#     print('Initialize, read model', time.time() - t)
#     # mModel = loadObj('E:\\_Groundwork\\Datasets\\cad_models_set_1\\block.obj')
#     mModel = loadObj('E:\\_Groundwork\\Datasets\\cad_models_set_1\\block_s.obj')
#     updateGeometryAttibutes(mModel)
#     numOfFaces = 10
#     print('Read model complete', time.time() - t)
#
#
#     NormalsOriginalFinal = np.empty(shape=[0, 8])
#     NormalsNoisyFinal=np.empty(shape=[0, 8])
#
#     patches = []
#     for i in range(0, len(mModel.faces)):
#         # print('Start searching patces for face '+str(i)+' '+ str(time.time() - t))
#         p = neighboursByFace(mModel, i, numOfFaces)
#         patches.append(p)
#         # print('Searching patces for face complete '+str(i)+ ' '+str(time.time() - t))
#     NormalsOriginal = np.empty(shape=[0, 8])
#     for p in patches:
#         patchFaces = [mModel.faces[i] for i in p]
#         normalsPatchFaces = []
#         for pF in patchFaces:
#             normalsPatchFaces.append(pF.faceNormal)
#         normalsPatchFaces = np.asarray(normalsPatchFaces)
#         normalsPatchFaces = np.transpose(normalsPatchFaces)
#         NormalsOriginal = np.concatenate((NormalsOriginal, normalsPatchFaces[:, 0:8]), axis=0)
#     # exportObj(mModel, 'E:\\_Groundwork\\_Python\\CVAE\\mModel.obj')
#     print('Complete', time.time() - t)
#
#
#
#
#     ###############################################################################
#     ###############################################################################
#     ###############################################################################
#     for repeat in range(0,50):
#         mModelToProcess = copy.deepcopy(mModel)
#         addNoise(mModelToProcess, 0.2)
#         updateGeometryAttibutes(mModelToProcess)
#         NormalsNoisy = np.empty(shape=[0, 8])
#         for p in patches:
#             patchFaces = [mModelToProcess.faces[i] for i in p]
#             normalsPatchFaces = []
#             for pF in patchFaces:
#                 normalsPatchFaces.append(pF.faceNormal)
#             normalsPatchFaces = np.asarray(normalsPatchFaces)
#             normalsPatchFaces = np.transpose(normalsPatchFaces)
#             NormalsNoisy = np.concatenate((NormalsNoisy, normalsPatchFaces[:, 0:8]), axis=0)
#         # exportObj(mModelToProcess, 'E:\\_Groundwork\\_Python\\CVAE\\mModelN1.obj')
#         print('Complete', time.time() - t)
#
#
#         NormalsOriginalFinal = np.concatenate((NormalsOriginalFinal, NormalsOriginal[:, 0:8]), axis=0)
#         NormalsNoisyFinal = np.concatenate((NormalsNoisyFinal, NormalsNoisy[:, 0:8]), axis=0)
#
#     print('Process complete')
#     ###############################################################################
#     ###############################################################################
#     ###############################################################################
#     # mModelToProcess = copy.deepcopy(mModel)
#     # addNoise(mModelToProcess, 0.2)
#     # updateGeometryAttibutes(mModelToProcess)
#     # NormalsNoisy = np.empty(shape=[0, 8])
#     # for p in patches:
#     #     patchFaces = [mModelToProcess.faces[i] for i in p]
#     #     normalsPatchFaces = []
#     #     for pF in patchFaces:
#     #         normalsPatchFaces.append(pF.faceNormal)
#     #     normalsPatchFaces = np.asarray(normalsPatchFaces)
#     #     normalsPatchFaces = np.transpose(normalsPatchFaces)
#     #     NormalsNoisy = np.concatenate((NormalsNoisy, normalsPatchFaces[:, 0:8]), axis=0)
#     # # exportObj(mModelToProcess, 'E:\\_Groundwork\\_Python\\CVAE\\mModelN2.obj')
#     # print('Complete', time.time() - t)
#     #
#     # NormalsOriginalFinal = np.concatenate((NormalsOriginalFinal, NormalsOriginal[:, 0:8]), axis=0)
#     # NormalsNoisyFinal = np.concatenate((NormalsNoisyFinal, NormalsNoisy[:, 0:8]), axis=0)
#     ###############################################################################
#     ###############################################################################
#     ###############################################################################
