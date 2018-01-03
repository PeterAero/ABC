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

rMax = 0;
rMin = 5;

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
        self.max_threshold = 0.8;
        self.min_threshold = 0.65;
        self.mutationProb = 0.05;


    def getCounters(self):
        counters = [];
        for gc_index in range(len(self.circles)):
            counters.append(self.circles[gc_index].counter);
        return counters;


    def multiModalOptimizationSearch(self):

        print 'Entering multiModalOptimizationSearch...';
        mf = self.obtainMatchingFitnessDetectedCircles();
        print 'Detected circles: ' + str(len(self.detectedCircles));

        self.detectedCircles = [x for _, x in sorted(zip(mf,self.detectedCircles),reverse=True)];

        mf = self.obtainMatchingFitnessDetectedCircles();
        J = (1/mf[0])-1;
        print mf;
        print 'max_mf: ' + str(mf[0]) + ' - edges: ' + str(len(self.detectedCircles[0].edges));
        print 'J: ' + str(J) + ' - matches: ' + str((1-J)*len(self.detectedCircles[0].edges));

        Esth = self.alfa * np.sqrt(np.power(imgWidth - 1, 2) + np.power(imgHeight - 1, 2) + np.power(rMax - rMin, 2));

        self.actualCircles.append(self.detectedCircles[0]);
        print '         Actual circle found!';

        for mmos_index in range(1,len(self.detectedCircles)):
            Esdi = np.zeros(len(self.actualCircles));
            if len(self.actualCircles) == 1:
                Esdi = np.sqrt(np.power(self.detectedCircles[mmos_index].x - self.actualCircles[0].x, 2) + np.power(
                    self.detectedCircles[mmos_index].y - self.actualCircles[0].y, 2) + np.power(
                    self.detectedCircles[mmos_index].r - self.actualCircles[0].r, 2));
            else:
                for mmos_index2 in range(0, len(self.actualCircles) - 1):
                    Esdi[mmos_index2] = np.sqrt(np.power(self.detectedCircles[mmos_index].x - self.actualCircles[mmos_index2].x, 2) + np.power(
                        self.detectedCircles[mmos_index].y - self.actualCircles[mmos_index2].y, 2) + np.power(
                        self.detectedCircles[mmos_index].r - self.actualCircles[mmos_index2].r, 2));

            if (Esdi > Esth).all():
                print '         Actual circle found!';
                self.actualCircles.append(self.detectedCircles[mmos_index]);

        print 'Done.';


    def obtainMatchingFitnessDetectedCircles(self):
        mf = [0] * len(self.detectedCircles);
        for omfdc_index in range(len(self.detectedCircles)):
            mf[omfdc_index] = self.detectedCircles[omfdc_index].obtainMatchingFitness();
        return mf;


    def calculateProb(self):
        mf = self.obtainMatchingFitnessCircles();
        mfsum = np.sum(mf);
        for cp_index in range(len(self.circles)):
            if mfsum == 0:
                self.circles[cp_index].probability = 0;
            else:
                self.circles[cp_index].probability = mf[cp_index] / mfsum;


    def obtainMatchingFitnessCircles(self):
        mf = [0] * len(self.circles);
        for omfc_index in range(len(self.circles)):
            mf[omfc_index] = self.circles[omfc_index].obtainMatchingFitness();
        return mf;


    def generateNewCircleCandidates(self):
        # Function that generates new circle candidates depending on the probability assigned to each one of the circles.
        # When a new circle candidate is generated, its matching fitness is compared to the one that should be replaced.
        # Only if the new circle has a higher matching fitness, then the old circle is replaced by the new one.
        print 'Generating new circle candidates...';
        self.calculateProb();
        for gncc_index in range(len(self.circles)):
            randomfloat = uniform(0, 4.0 / len(self.circles));
            if self.circles[gncc_index].probability > randomfloat:
                mf = self.obtainMatchingFitnessCircles();

                candidate = self.modifyCircleCandidate(gncc_index);

                mf_new = candidate.obtainMatchingFitness();
                mf_old = mf[gncc_index];

                if mf_new > mf_old:
                    print ' New circle candidate added!';
                    self.circles[gncc_index] = candidate;
                else:
                    self.circles[gncc_index].counter += 1;


    def checkMatchingFitnessesAboveThreshold(self):
        mf = self.obtainMatchingFitnessCircles();
        for cmfat_index in range(len(mf)):
            if mf[cmfat_index] >= self.max_threshold:
                print 'One posible circle candidate detected!';
                self.detectedCircles.append(self.circles[cmfat_index]);
                print 'Counter: ' + str(self.circles[cmfat_index].counter);
                self.circles[cmfat_index] = Circle(self.imageEdges,initialization_parameters);


    def modifyCircleCandidate(self,mcc_index):
        # Function that modifies a randomly chosen parameter of the circle specified by candidateIndexToModify, using the
        # equation proposed in chapter 3.2.4
        newCandidate = Circle(self.imageEdges, initialization_parameters);

        newCandidate.x = self.circles[mcc_index].x;
        newCandidate.y = self.circles[mcc_index].y;
        newCandidate.r = self.circles[mcc_index].r;

        parameterToModify1 = randint(0, 2);
        auxiliaryIndex1 = randint(0, len(self.circles) - 1);
        while auxiliaryIndex1 == mcc_index:
            auxiliaryIndex1 = randint(0, len(self.circles) - 1);

        if parameterToModify1 == 0:
            offset = uniform(-1, 1) * (newCandidate.x - self.circles[auxiliaryIndex1].x);
            newCandidate.x = newCandidate.x + offset;
        elif parameterToModify1 == 1:
            offset = uniform(-1, 1) * (newCandidate.y - self.circles[auxiliaryIndex1].y);
            newCandidate.y = newCandidate.y + offset;
        else:
            offset = uniform(-1, 1) * (newCandidate.r - self.circles[auxiliaryIndex1].r);
            newCandidate.r = newCandidate.r + offset;

        randomMutation = uniform(0.0, 1.0);
        if randomMutation < self.mutationProb:
            parameterToModify1 = randint(0, 2);
            auxiliaryIndex1 = randint(0, len(self.circles) - 1);
            while auxiliaryIndex1 == mcc_index:
                auxiliaryIndex1 = randint(0, len(self.circles) - 1);

            if parameterToModify1 == 0:
                offset = uniform(-1, 1) * (newCandidate.x - self.circles[auxiliaryIndex1].x);
                newCandidate.x = newCandidate.x + offset;
            elif parameterToModify1 == 1:
                offset = uniform(-1, 1) * (newCandidate.y - self.circles[auxiliaryIndex1].y);
                newCandidate.y = newCandidate.y + offset;
            else:
                offset = uniform(-1, 1) * (newCandidate.r - self.circles[auxiliaryIndex1].r);
                newCandidate.r = newCandidate.r + offset;

        return newCandidate;


    def checkCounters(self):
        mf = self.obtainMatchingFitnessCircles();
        for cc_index in range(len(self.circles)):
            if self.circles[cc_index].counter >= limit:
                if mf[cc_index] >= self.min_threshold:
                    print 'One posible circle candidate detected!';
                    self.detectedCircles.append(self.circles[cc_index]);
                    self.circles[cc_index] = Circle(self.imageEdges,initialization_parameters);
                else:
                    self.circles[cc_index].counter = 0;


    def checkRadius(self):
        for cr_index in range(len(self.circles)):
            if self.circles[cr_index].r <= rMin:
                self.circles[cr_index] = Circle(self.imageEdges,initialization_parameters);
            elif self.circles[cr_index].r >= rMax:
                self.circles[cr_index] = Circle(self.imageEdges, initialization_parameters);


class Circle(object):
    def __init__(self,imageEdges,initialization_parameters):
        ic_index = randint(0,len(initialization_parameters)-1);
        xi = initialization_parameters[ic_index][0];
        yi = initialization_parameters[ic_index][1];
        ri = initialization_parameters[ic_index][2];
        self.x = xi+randint(-5,5);
        self.y = yi+randint(-5,5);
        self.r = ri+randint(-10,10);
        self.edges = [];
        self.imageEdges = imageEdges;
        self.counter = 0;
        self.probability = 0;


    def obtainMatchingFitness(self):
        self.MCA();
        J = self.objectiveFunction();

        if J >= 0:
            mf = 1.0 / (1.0 + J);
        else:
            mf = 1.0 + np.abs(J);
        return mf;


    def objectiveFunction(self):
        # Function that has to be minimized
        N = float(len(self.edges));
        matches = set(self.edges) & set(self.imageEdges);

        if N == 0.0:
            div = 0.0;
        else:
            div = float(len(matches)) / N;

        return 1.0 - div;


    def MCA(self):
        # Function that implements the Midpoint Circle Algorithm

        x0 = int(round(self.x));
        y0 = int(round(self.y));
        r = int(round(abs(self.r)));

        self.edges = [];

        x = 0;
        y = r;
        d = 3 - 2 * r;
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


def get_edges_roundabouts(imagePath,edges_img):
    geo_corners = get_geo_corners(imagePath);
    osmconvert_rect = "%f,%f,%f,%f" % geo_corners
    os.system("./osmconvert32 " + "muc_umgebung.osm" + " -b=" + osmconvert_rect + " > sub_section.osm")
    roundabouts = get_roundabouts("sub_section.osm")

    edges_temp = [];
    edges_ret = [];

    rad_max = 0;
    rad_min = 0;

    for r in roundabouts:
        l_l = geo_to_img_coords(imagePath, r[1][0], r[1][1])
        u_r = geo_to_img_coords(imagePath, r[1][2], r[1][3])
        x = int(round(l_l[0] + (u_r[0] - l_l[0]) / 2));
        y = int(round(u_r[1] + (l_l[1] - u_r[1]) / 2));
        rad = int(round(u_r[0] - x));
        initialization_parameters.append((x, y, rad));

        if rad_max < rad:
            rad_max = rad;
        if rad_min > rad:
            rad_min = rad;
        if rad_min == 0:
            rad_min = rad;

        crop_edges_img = edges_img[u_r[1] - offset:l_l[1] + offset, l_l[0] - offset:u_r[0] + offset]
        jj, ii = np.where(crop_edges_img == 255);

        ii = np.rint(ii);
        jj = np.rint(jj);

        ii = ii + int(round(l_l[0] - offset));
        jj = jj + int(round(u_r[1] - offset));

        ii = ii.astype(int);
        jj = jj.astype(int);

        edges_temp.append(zip(ii, jj));

    for l in edges_temp:
        for e in l:
            edges_ret.append(e);

    rad_min = int(round(0.3*rad_min));
    rad_max = int(round(1.3*rad_max));

    return edges_ret,rad_max,rad_min;


if __name__ == "__main__":

    print 'Starting program.........';
    start_general = timer();
    if len(sys.argv) == 6:

        # Step 2: Required parameters for the ABC algorithm
        imagePath = sys.argv[1];
        colonySize = int(sys.argv[2]);
        limit = int(sys.argv[3]);
        maxCycles = int(sys.argv[4]);
        alfa = float(sys.argv[5]);

        imageCompleteName = os.path.basename(imagePath);
        imageName = os.path.splitext(imageCompleteName)[0];
        imageExtension = os.path.splitext(imageCompleteName)[1];
        imageFolder = os.path.dirname(imagePath);

        resultFolder = imageFolder + '/Results_for_' + imageName + '_' + time.strftime("%d%m%Y") + '_' + time.strftime("%H%M%S");
        resultFolder = resultFolder + '_' + str(colonySize) + '_' + str(limit) + '_' + str(maxCycles) + '_' + str(alfa);
        os.makedirs(resultFolder);

        print 'Loading .tif image!';
        ds = gdal.Open(imagePath);
        img = np.array(ds.ReadAsArray());
        img = img.astype(np.uint8);

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB);
        edges_img = cv2.Canny(img, 50, 100);

        imgHeight, imgWidth = edges_img.shape;
        print 'imgHeight: ' + str(imgHeight);
        print 'imgWidth: ' + str(imgWidth);

        edges,rMax,rMin = get_edges_roundabouts(imagePath,edges_img);

        # Step 3: initialize circle candidates
        circleCandidates = Colony(initialization_parameters,colonySize, limit, maxCycles, alfa, edges);

        for index in range(1, circleCandidates.iterations):
            print '--------------------------------';
            print 'Iteration: ' + str(index) + ' of ' + str(circleCandidates.iterations);

            start_time = timer();

            # Step 6 and 7: modify the circle candidates and update the counter array A and the probabilities array
            circleCandidates.generateNewCircleCandidates();

            print 'max mf: ' + str(np.amax(circleCandidates.obtainMatchingFitnessCircles()));

            # Step 10: check if some circleCandidate counter has reached the amount specified with the variable limit.
            # If it has, then it is saved as a possible solution and a new circle candidate is generated in his place.
            circleCandidates.checkCounters();

            # Check if the matching fitness of any circle is above the threshold. If it is, then this circle is saved as
            # a detected circle.
            circleCandidates.checkMatchingFitnessesAboveThreshold();

            circleCandidates.checkRadius();

            end_time = timer();

            print 'Iteration: ' + str(index) + ' of ' + str(circleCandidates.iterations) + ' - ' + str(end_time - start_time);
            print '--------------------------------';

        print 'End';

        # Final step: the multimodal optimization search is applied as described in chapter 3.4
        circleCandidates.multiModalOptimizationSearch();

        # The original image is saved inside the created folder:
        cv2.imwrite(resultFolder + '/Original' + imageExtension,img);

        edges_img = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2RGB);

        print 'Circle   X   Y   R';

        for index in range(len(circleCandidates.actualCircles)):

            x = int(round(circleCandidates.actualCircles[index].x));
            y = int(round(circleCandidates.actualCircles[index].y));
            r = abs(int(round(circleCandidates.actualCircles[index].r)));

            print str(index) + '    ' + str(x) + '    ' + str(y) + '    ' + str(r);

            cv2.circle(img, (x, y), r, (0,0,255),thickness=2);
            cv2.circle(edges_img, (x, y), r, (0, 0, 255), thickness=1);

        cv2.imwrite(resultFolder + '/Circles_found' + imageExtension, img);
        cv2.imwrite(resultFolder + '/Edges' + imageExtension, edges_img);
        end_general = timer();

        print 'Total elapsed time: ' + str(end_general - start_general) + ' [s]';
    else:

        print 'main.py <imagePath> <colonySize> <limit> <maxCycles> <alfa>';