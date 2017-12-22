import numpy as np
import cv2
from random import randint
from random import uniform
from timeit import default_timer as timer
import sys
import os
import time
import math
import gdal
from get_roundabouts import get_geo_corners
from get_roundabouts import get_roundabouts
from get_roundabouts import geo_to_img_coords

i = 0;
j = 1;
k = 2;
x = 0;
y = 1;

rMax = 300;
rMin = 10;

imgWidth = 0;
imgHeight = 0;

edges = [];

initialization_parameters = [];

offset = 50;

class Colony(object):
    def __init__(self,initialization_parameters, colonysize, limit, iterations, alfa, edges):
        self.circles = [Circle(edges,initialization_parameters) for _ in range(colonysize)];
        self.limit = limit;
        self.alfa = alfa;
        self.iterations = iterations;
        self.imageEdges = edges;
        self.detectedCircles = [];
        self.actualCircles = [];
        self.calculateProb();
        self.max_threshold = 0.9;
        self.min_threshold = 0.52;

    def getCounters(self):
        counters = [];
        for i in range(len(self.circles)):
            counters.append(self.circles[i].counter);
        return counters;

    def multiModalOptimizationSearch(self):

        print 'Entering multiModalOptimizationSearch...';
        mf = self.obtainMatchingFitnessDetectedCircles();
        print 'Detected circles: ' + str(len(self.detectedCircles));
        print 'Matching Fitness detected circles: ' + str(mf);
        maxIndex = np.argmax(mf);
        print 'maxIndex: ' + str(maxIndex);

        self.actualCircles.append(self.detectedCircles[maxIndex]);

        Esth = self.alfa * np.sqrt(np.power(imgWidth - 1, 2) + np.power(imgHeight - 1, 2) + np.power(rMax - rMin, 2));

        for index in range(len(self.detectedCircles)):
            Esdi = np.zeros(len(self.actualCircles));
            if len(self.actualCircles) == 1:
                Esdi = np.sqrt(np.power(self.detectedCircles[index].x - self.actualCircles[0].x, 2) + np.power(
                    self.detectedCircles[index].y - self.actualCircles[0].y, 2) + np.power(
                    self.detectedCircles[index].r - self.actualCircles[0].r, 2));
            else:
                for index2 in range(0, len(self.actualCircles) - 1):
                    # print 'Index2 = ' + str(index2);
                    Esdi[index2] = np.sqrt(np.power(self.detectedCircles[index].x - self.actualCircles[index2].x, 2) + np.power(
                        self.detectedCircles[index].y - self.actualCircles[index2].y, 2) + np.power(
                        self.detectedCircles[index].r - self.actualCircles[index2].r, 2));

            if (Esdi > Esth).all():
                print '         Actual circle found!';
                self.actualCircles.append(self.detectedCircles[index]);

        print 'Done.';

    def obtainMatchingFitnessDetectedCircles(self):
        mf = [0] * len(self.detectedCircles);
        for index in range(len(self.detectedCircles)):
            mf[index] = self.detectedCircles[index].obtainMatchingFitness();
        return mf;

    def calculateProb(self):
        mf = self.obtainMatchingFitnessCircles();
        mfsum = np.sum(mf);
        for index in range(len(self.circles)):
            if mfsum == 0:
                self.circles[index].probability = 0;
            else:
                self.circles[index].probability = mf[index] / mfsum;

    def obtainMatchingFitnessCircles(self):
        mf = [0] * len(self.circles);
        for index in range(len(self.circles)):
            mf[index] = self.circles[index].obtainMatchingFitness();
        return mf;

    def objectiveFunction(self,circle):
        # Function that has to be minimized
        N = float(len(circle.edges));
        matches = set(self.imageEdges) & set(circle.edges);

        if N == 0.0:
            div = 0.0;
        else:
            div = float(len(matches)) / N;

        return 1.0 - div;

    def generateNewCircleCandidates(self):
        # Function that generates new circle candidates depending on the probability assigned to each one of the circles.
        # When a new circle candidate is generated, its matching fitness is compared to the one that should be replaced.
        # Only if the new circle has a higher matching fitness, then the old circle is replaced by the new one.
        print 'Generating new circle candidates...';
        self.calculateProb();
        for index in range(len(self.circles)):
            randomfloat = uniform(0, 4.0 / len(self.circles));
            if self.circles[index].probability > randomfloat:
                mf = self.obtainMatchingFitnessCircles();
                newCandidate = self.modifyCircleCandidate(index);

                mf_new = newCandidate.obtainMatchingFitness();
                mf_old = mf[index];

                if mf_new > mf_old:
                    print ' New circle candidate added!';
                    self.circles[index] = newCandidate;
                else:
                    self.circles[index].counter += 1;

    def checkMatchingFitnessesAboveThreshold(self):
        mf = self.obtainMatchingFitnessCircles();
        for index in range(len(mf)):
            if mf[index] >= self.max_threshold:
                print 'One posible circle candidate detected!';
                self.detectedCircles.append(self.circles[index]);
                self.circles[index] = Circle(self.imageEdges,initialization_parameters);

    def modifyCircleCandidate(self,index):
        # Function that modifies a randomly chosen parameter of the circle specified by candidateIndexToModify, using the
        # equation proposed in chapter 3.2.4
        newCandidate = Circle(self.imageEdges,initialization_parameters);
        newCandidate.x = self.circles[index].x;
        newCandidate.y = self.circles[index].y;
        newCandidate.r = self.circles[index].r;

        parameterToModify1 = randint(0, 2);
        auxiliaryIndex1 = randint(0, len(self.circles) - 1);
        while auxiliaryIndex1 == index:
            auxiliaryIndex1 = randint(0, len(self.circles) - 1);

        parameterToModify2 = randint(0, 2);
        auxiliaryIndex2 = randint(0, len(self.circles) - 1);
        while auxiliaryIndex2 == index:
            auxiliaryIndex2 = randint(0, len(self.circles) - 1);

        if parameterToModify1 == 0:
            offset = uniform(-1, 1) * (newCandidate.x - self.circles[auxiliaryIndex1].x);
            newCandidate.x = newCandidate.x + offset;
        elif parameterToModify1 == 1:
            offset = uniform(-1, 1) * (newCandidate.y - self.circles[auxiliaryIndex1].y);
            newCandidate.y = newCandidate.y + offset;
        else:
            offset = uniform(-1, 1) * (newCandidate.r - self.circles[auxiliaryIndex1].r);
            newCandidate.r = newCandidate.r + offset;

        #if parameterToModify2 == 0:
        #    offset = uniform(-1, 1) * (newCandidate.x - self.circles[auxiliaryIndex2].x);
        #    newCandidate.x = newCandidate.x + offset;
        #elif parameterToModify2 == 1:
        #    offset = uniform(-1, 1) * (newCandidate.y - self.circles[auxiliaryIndex2].y);
        #    newCandidate.y = newCandidate.y + offset;
        #else:
        #    offset = uniform(-1, 1) * (newCandidate.r - self.circles[auxiliaryIndex2].r);
        #    newCandidate.r = newCandidate.r + offset;

        return newCandidate;

    def checkCounters(self):
        for index in range(len(self.circles)):
            if self.circles[index].counter >= limit:
                print 'One posible circle candidate detected!';
                self.detectedCircles.append(self.circles[index]);
                self.circles[index] = Circle(self.imageEdges,initialization_parameters);

class Circle(object):
    def __init__(self,imageEdges,initialization_parameters):
        index = randint(0,len(initialization_parameters)-1);
        self.x = initialization_parameters[index][0]+randint(-5,5);
        self.y = initialization_parameters[index][1]+randint(-5,5);
        self.r = initialization_parameters[index][2]+randint(-5,5);
        self.edges = [];
        self.imageEdges = imageEdges;
        self.counter = 0;
        self.probability = 0;

    def obtainMatchingFitness(self):
        self.MCA();
        J = self.objectiveFunction();
        if J >= 0:
            mf = 1 / (1 + J);
        else:
            mf = 1 + np.abs(J);
        return mf;

    def objectiveFunction(self):
        # Function that has to be minimized
        N = float(len(self.imageEdges));
        matches = set(self.edges)&set(self.imageEdges);

        if N == 0.0:
            div = 0.0;
        else:
            div = float(len(matches)) / N;

        return 1.0 - div;

    def MCA(self):
        # Function that implements the Midpoint Circle Algorithm

        x0 = int(round(self.x));
        y0 = int(round(self.y));
        r = int(round(self.r));
        x = 0;
        y = r;
        d = 3 - 2 * r;
        self.edges = [];
        while (x <= y):
            if ((x0 + x) < imgWidth) and ((x0 + x) > 0) and ((y0 + y) < imgHeight) and ((y0 + y) > 0):
                self.edges.append((x0+x,y0+y));
            if ((x0 - x) > 0) and ((x0 - x) < imgWidth) and ((y0 + y) < imgHeight) and ((y0 + y) > 0):
                self.edges.append((x0 - x, y0 + y));
            if ((y0 - y) > 0) and ((y0 - y) < imgHeight) and ((x0 + x) < imgWidth) and ((x0 + x) > 0):
                self.edges.append((x0 + x, y0 - y));
            if ((x0 - x) > 0) and ((x0 - x) < imgWidth) and ((y0 - y) > 0) and ((y0 - y) < imgHeight):
                self.edges.append((x0 - x, y0 - y));
            if ((x0 + y) < imgWidth) and ((x0 + y) > 0) and ((y0 + x) < imgHeight) and ((y0 + x) > 0):
                self.edges.append((x0 + y, y0 + x));
            if ((x0 - y) > 0) and ((x0 - y) < imgWidth) and ((y0 + x) < imgHeight) and ((y0 + x) > 0):
                self.edges.append((x0 - y, y0 + x));
            if ((y0 - x) > 0) and ((y0 - x) < imgHeight) and ((x0 + y) < imgWidth) and ((x0 + y) > 0):
                self.edges.append((x0 + y, y0 - x));
            if ((x0 - y) > 0) and ((x0 - y) < imgWidth) and ((y0 - x) > 0) and ((y0 - x) < imgHeight):
                self.edges.append((x0 - y, y0 - x));

            if (d < 0):
                d = d + 4 * x + 6
            else:
                d = d + 4 * (x - y) + 10
                y = y - 1
            x = x + 1


def transformEdgeVectorIndexesToShapeParameters(P):
    A = [[0.0,0.0],[0.0,0.0]];
    B = [[0.0, 0.0], [0.0, 0.0]];

    if (P[i][x]*(P[j][y]-P[k][y]) + P[j][x]*(P[k][y]-P[i][y]) + P[k][x]*(P[i][y]-P[j][y])) != 0:
        A[0][0] = np.power(P[j][x],2)+np.power(P[j][y],2)-(np.power(P[i][x],2)+np.power(P[i][y],2));
        A[0][1] = 2.0*(P[j][y]-P[i][y]);
        A[1][0] = np.power(P[k][x], 2) + np.power(P[k][y], 2) - (np.power(P[i][x], 2) + np.power(P[i][y], 2));
        A[1][1] = 2.0 * (P[k][y] - P[i][y]);

        B[0][0] = 2.0 * (P[j][x] - P[i][x]);
        B[0][1] = np.power(P[j][x], 2) + np.power(P[j][y], 2) - (np.power(P[i][x], 2) + np.power(P[i][y], 2));
        B[1][0] = 2.0 * (P[k][x] - P[i][x]);
        B[1][1] = np.power(P[k][x], 2) + np.power(P[k][y], 2) - (np.power(P[i][x], 2) + np.power(P[i][y], 2));

        x0 = np.linalg.det(A)/(4.0*((P[j][x]-P[i][x])*(P[k][y]-P[i][y])-(P[k][x]-P[i][x])*(P[j][y]-P[i][y])));
        y0 = np.linalg.det(B) / (4.0 * ((P[j][x] - P[i][x]) * (P[k][y] - P[i][y]) - (P[k][x] - P[i][x]) * (P[j][y] - P[i][y])));

        r = np.sqrt(float(np.power(x0-P[i][x],2)+np.power(y0-P[i][y],2)));

    else:
        return None;

    return [float(x0),float(y0),float(r)];

def get_edges_roundabouts(imagePath,edges_img):
    geo_corners = get_geo_corners(imagePath);
    osmconvert_rect = "%f,%f,%f,%f" % geo_corners
    os.system("./osmconvert32 " + "muc_umgebung.osm" + " -b=" + osmconvert_rect + " > sub_section.osm")
    roundabouts = get_roundabouts("sub_section.osm")

    imgHeight, imgWidth = edges_img.shape;
    print 'imgHeight: ' + str(imgHeight);
    print 'imgWidth: ' + str(imgWidth);

    edges_temp = [];
    edges_ret = [];
    for r in roundabouts:
        l_l = geo_to_img_coords(imagePath, r[1][0], r[1][1])
        u_r = geo_to_img_coords(imagePath, r[1][2], r[1][3])
        x = int(round(l_l[0] + (u_r[0] - l_l[0]) / 2));
        y = int(round(u_r[1] + (l_l[1] - u_r[1]) / 2));
        r = int(round(u_r[0] - x));
        initialization_parameters.append((x, y, r));

        crop_edges_img = edges_img[u_r[1] - offset:l_l[1] + offset, l_l[0] - offset:u_r[0] + offset]
        jj, ii = np.where(crop_edges_img == 255);

        ii = np.rint(ii);
        jj = np.rint(jj);

        ii = ii + int(round(l_l[0] - offset));
        jj = jj + int(round(u_r[1] - offset));

        ii = ii.astype(int);
        jj = jj.astype(int);

        edges_temp.append(zip(ii, jj));

        print "%f,%f,%f,%f" % (l_l[0], l_l[1], u_r[0], u_r[1])
        #cv2.rectangle(img, (int(round(l_l[0])), int(round(l_l[1]))), (int(round(u_r[0])), int(round(u_r[1]))),
                      #(0, 0, 255), thickness=2)

    for l in edges_temp:
        for e in l:
            edges_ret.append(e);

    return edges_ret;

if __name__ == "__main__":

    print 'Starting program.........';
    start_general = timer();
    if len(sys.argv) == 6:

        imagePath = sys.argv[1];
        imageCompleteName = os.path.basename(imagePath);
        imageName = os.path.splitext(imageCompleteName)[0];
        imageExtension = os.path.splitext(imageCompleteName)[1];
        imageFolder = os.path.dirname(imagePath);

        resultFolder = imageFolder + '/Results_for_' + imageName + '_' + time.strftime("%d%m%Y") + '_' + time.strftime("%H%M%S");

        print 'Image Extension = ' + str(imageExtension);

        print 'Loading .tif image!';
        ds = gdal.Open(imagePath);
        img = np.array(ds.ReadAsArray());
        img = img.astype(np.uint8);


        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB);
        edges_img = cv2.Canny(img, 50, 100);

        edges = get_edges_roundabouts(imagePath,edges_img);

        # Step 2: Required parameters for the ABC algorithm
        colonySize = int(sys.argv[2]);
        limit = int(sys.argv[3]);
        maxCycles = int(sys.argv[4]);
        alfa = float(sys.argv[5]);

        resultFolder = resultFolder + '_' + str(colonySize) + '_' + str(limit) + '_' + str(maxCycles) + '_' + str(alfa);
        os.makedirs(resultFolder);

        # Step 3: initialize circle candidates
        circleCandidates = Colony(initialization_parameters,colonySize, limit, maxCycles, alfa, edges);

        for index in range(1, circleCandidates.iterations):

            print '--------------------------------';
            print 'Iteration: ' + str(index) + ' of ' + str(circleCandidates.iterations);
            start_time = timer();


            # Step 6 and 7: modify the circle candidates and update the counter array A and the probabilities array
            circleCandidates.generateNewCircleCandidates();

            #print circleCandidates.obtainMatchingFitnessCircles();

            # Step 10: check if some circleCandidate counter has reached the amount specified with the variable limit.
            # If it has, then it is saved as a possible solution and a new circle candidate is generated in his place.
            circleCandidates.checkCounters();

            # Check if the matching fitness of any circle is above the threshold. If it is, then this circle is saved as
            # a detected circle.
            circleCandidates.checkMatchingFitnessesAboveThreshold();

            print circleCandidates.getCounters();

            end_time = timer();

            print 'Iteration: ' + str(index) + ' of ' + str(circleCandidates.iterations) + ' - ' + str(end_time - start_time);
            print '--------------------------------';

        print 'End.';


        # Final step: the multimodal optimization search is applied as described in chapter 3.4
        circleCandidates.multiModalOptimizationSearch();

        # The original image is saved inside the created folder:
        cv2.imwrite(resultFolder + '/Original' + imageExtension,img);

        edges_img = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB);

        for index in range(len(circleCandidates.actualCircles)):
            cv2.circle(img, (int(round(circleCandidates.actualCircles[index].x)), int(round(circleCandidates.actualCircles[index].y))),
                       int(round(circleCandidates.actualCircles[index].r)), (0,0,255),thickness=2);
            cv2.circle(edges_img, (
            int(round(circleCandidates.actualCircles[index].x)), int(round(circleCandidates.actualCircles[index].y))),
                       int(round(circleCandidates.actualCircles[index].r)), (0, 0, 255), thickness=2);

        cv2.imwrite(resultFolder + '/Circles_found' + imageExtension, img);
        cv2.imwrite(resultFolder + '/Edges' + imageExtension, edges_img);
        end_general = timer();

        print 'Total elapsed time: ' + str(end_general - start_general) + ' [s]';
    else:

        print 'main.py <imagePath> <colonySize> <limit> <maxCycles> <alfa>';