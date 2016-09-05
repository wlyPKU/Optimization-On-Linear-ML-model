package parallelADMM;

import Utils.*;
import math.DenseVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
public class SVM extends model.SVM {

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int featureDimension;
    private static int threadNum;

    private static List<LabeledData> labeledData;
    private static ADMMState model;

    private ADMMState[] localADMMState;
    private List<List<LabeledData>> localTrainCorpus = new ArrayList<List<LabeledData>>();

    //Parameter:
    private int lbfgsNumIteration = 10;
    private int lbfgsHistory = 10;
    private double rel_par = 1.0;
    private double rho = 1e-4;
    private double maxRho = 5;
    private double x_hat[];

    private class executeRunnable implements Runnable {
        int threadID;
        int iteNum;

        private executeRunnable(int threadID, int iteNum) {
            this.threadID = threadID;
            this.iteNum = iteNum;
        }

        public void run() {
            //Update x;
            parallelLBFGS.train(localADMMState[threadID], lbfgsNumIteration, lbfgsHistory,
                    rho, iteNum, localTrainCorpus.get(threadID), "SVM", model.z);
            model.x.plusDense(localADMMState[threadID].x);
        }
    }

    private void updateZ(){
        for(int id = 0; id < featureDimension; id++){
            x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.z.values[id];
            //z=Soft_threshold(lambda/rho,x+u);
            model.z.values[id] = (rho / (1.0 / lambda + threadNum * rho)) * (x_hat[id] + model.u.values[id]);
        }
    }
    private void updateU(){
        Arrays.fill(model.u.values, 0);
        for(int j = 0; j < threadNum; j++){
            for(int fID = 0; fID < featureDimension; fID++){
                localADMMState[j].u.values[fID] += (x_hat[fID] - model.z.values[fID]);
                model.u.values[fID] += localADMMState[j].u.values[fID];
            }
        }
        model.u.allDividedBy(threadNum);
    }

    private void updateX(int iteNumber){
        Arrays.fill(model.x.values, 0);
        ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
        for (int threadID = 0; threadID < threadNum; threadID++) {
            threadPool.execute(new executeRunnable(threadID, iteNumber));
        }
        threadPool.shutdown();
        while (!threadPool.isTerminated()) {
            try {
                    threadPool.awaitTermination(1, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                System.out.println("Waiting.");
                    e.printStackTrace();
                }
        }
        model.x.allDividedBy(threadNum);
    }
    private void testAndSummary(List<LabeledData>trainCorpus, List<LabeledData>testCorpus){
        long startTest = System.currentTimeMillis();
        double loss = SVMLoss(trainCorpus, model.x, model.z, lambda);
        double trainAuc = auc(trainCorpus, model.x);
        double testAuc = auc(testCorpus, model.x);
        double trainAccuracy = accuracy(trainCorpus, model.x);
        double testAccuracy = accuracy(testCorpus, model.x);
        long testTime = System.currentTimeMillis() - startTest;
        System.out.println("loss=" + loss + " trainAuc=" + trainAuc + " testAuc=" + testAuc
                + " trainAccuracy=" + trainAccuracy + " testAccuracy=" + testAccuracy
                + " testTime=" + testTime);
    }
    private void trainCore() {
        Collections.shuffle(labeledData);

        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        DenseVector oldModel = new DenseVector(featureDimension);
        localADMMState = new ADMMState[threadNum];
        for (int threadID = 0; threadID < threadNum; threadID++) {
            localADMMState[threadID] = new ADMMState(featureDimension);
            int from = trainCorpus.size() * threadID / threadNum;
            int to = trainCorpus.size() * (threadID + 1) / threadNum;
            List<LabeledData> localData = trainCorpus.subList(from, to);
            localTrainCorpus.add(localData);
        }

        x_hat = new double[model.featureNum];
        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            updateX(i);
            //Update z
            updateZ();
            //Update u
            updateU();
            //rho = Math.min(maxRho, rho * 1.1);

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime=" + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus);

            if(converge(oldModel, model.x)){
                //break;
            }
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDimension);
        }
    }
    private static void train() {
        SVM svmADMM = new SVM();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new ADMMState(featureDimension);
        long start = System.currentTimeMillis();
        svmADMM.trainCore();
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelADMM.SVM threadNum featureDimension train_path lambda trainRatio");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        if(argv.length >= 5){
            trainRatio = Double.parseDouble(argv[4]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        long startLoad = System.currentTimeMillis();
        labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train();
    }
}
