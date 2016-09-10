package parallelADMM;

import Utils.*;
import math.DenseVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

//TODO: To be checked ...

/**
 * Created by 王羚宇 on 2016/7/24.
 */
//According to https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
public class LinearRegressionLBFGS extends model.LinearRegression{

    private static double trainRatio = 0.5;
    private static int featureDimension;
    private static List<LabeledData> labeledData;
    private static ADMMState model;
    private ADMMState[] localADMMState;
    private static int threadNum;

    private double x_hat[];
    private List<List<LabeledData>> localTrainCorpus = new ArrayList<List<LabeledData>>();

    private double rho = 1e-4;
    private double maxRho = 5;
    private double rel_par = 1.0;
    private int lbfgsNumIteration = 10;
    private int lbfgsHistory = 10;

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
                    rho, iteNum, localTrainCorpus.get(threadID), "LinearRegressionModelParallel", model.z);
            model.x.plusDense(localADMMState[threadID].x);
        }
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
    private void updateZ(){
        for(int id = 0; id < featureDimension; id++){
            x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.z.values[id];
            //z=Soft_threshold(lambda/rho,x+u);
            model.z.values[id] = x_hat[id] + model.u.values[id];
        }
    }

    private void updateU(){
        Arrays.fill(model.u.values, 0);
        for(int j = 0; j < threadNum; j++){
            for(int fID = 0; fID < featureDimension; fID++){
                localADMMState[j].u.values[fID] += (localADMMState[j].x.values[fID] - model.z.values[fID]);
                model.u.values[fID] += localADMMState[j].u.values[fID];
            }
        }
        model.u.allDividedBy(threadNum);
    }

    private void trainCore() {
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData>trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        x_hat = new double[model.featureNum];
        DenseVector oldModel = new DenseVector(featureDimension);

        localADMMState = new ADMMState[threadNum];
        for (int threadID = 0; threadID < threadNum; threadID++) {
            localADMMState[threadID] = new ADMMState(featureDimension);
            int from = trainCorpus.size() * threadID / threadNum;
            int to = trainCorpus.size() * (threadID + 1) / threadNum;
            List<LabeledData> localData = trainCorpus.subList(from, to);
            localTrainCorpus.add(localData);
        }
        long totalBegin = System.currentTimeMillis();

        for (int i = 0; i < 100; i ++) {
            long startTrain = System.currentTimeMillis();
            updateX(i);
            //Update z
            updateZ();
            updateU();
            //rho = Math.min(rho * 1.1, maxRho);

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime=" + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model.x);

            //rho = Math.min(rho * 1.1, maxRho);
            if(converge(oldModel, model.x)){
                //break;
            }
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDimension);
            Arrays.fill(model.x.values, 0);
            System.out.println("Totaltime=" + (System.currentTimeMillis() - totalBegin) );

        }
    }

    private static void train() {
        LinearRegressionLBFGS lrADMM = new LinearRegressionLBFGS();
        model = new ADMMState(featureDimension);
        long start = System.currentTimeMillis();
        lrADMM.trainCore();
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LinearRegressionLBFGS threadNum featureDimension train_path [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        if(argv.length >=4){
            trainRatio = Double.parseDouble(argv[3]);
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
