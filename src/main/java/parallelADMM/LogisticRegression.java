package parallelADMM;

import Utils.*;
import math.DenseVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

//TODO: To be checked...
//According to the angel ADMM logistic regression

/**
 * Created by 王羚宇 on 2016/7/24.
 */
//https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
//https://web.stanford.edu/~boyd/papers/pdf/admm_slides.pdf
//https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
//https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
//http://www.simonlucey.com/lasso-using-admm/
//http://users.ece.gatech.edu/~justin/CVXOPT-Spring-2015/resources/14-notes-admm.pdf

//https://web.stanford.edu/~boyd/papers/admm/logreg-l1/logreg.html
public class LogisticRegression extends model.LogisticRegression{
    private static long start;

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int featureDimension;
    private static List<LabeledData> labeledData;
    private static ADMMState model;
    private static int threadNum;
    private double rho = 1e-4;
    //private double maxRho = 5;
    private int lbfgsNumIteration = 10;
    private int lbfgsHistory = 10;
    private double rel_par = 1.0;
    private ADMMState[] localADMMState;
    private List<List<LabeledData>> localTrainCorpus = new ArrayList<List<LabeledData>>();
    private double x_hat[];
    private static DenseVector oldModelZ;


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
                    rho, iteNum, localTrainCorpus.get(threadID), "logisticRegression", model.z);
        }
    }

    private double calculateRho(double rho){
        //https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf PG23
        double miu = 10;
        double pi_incr = 2, pi_decr = 2;
        double r = 0;
        for(int i = 0; i < featureDimension; i++){
            r += (model.x.values[i] - model.z.values[i]) * (model.x.values[i] - model.z.values[i]);
        }
        r = Math.sqrt(r);
        double s = 0;
        for(int i = 0; i < featureDimension; i++){
            s += (oldModelZ.values[i] - model.z.values[i]) * (oldModelZ.values[i] - model.z.values[i]);
        }
        s = Math.sqrt(s) * rho;
        if(r > miu * s){
            for(int fID = 0; fID < featureDimension; fID++){
                model.u.values[fID] /= pi_incr;
                for(int j = 0; j < threadNum; j++){
                    localADMMState[j].u.values[fID] /= pi_incr;
                }
            }
            return pi_incr * rho;
        }else if(s > miu * r){
            for(int fID = 0; fID < featureDimension; fID++){
                model.u.values[fID] *= pi_incr;
                for(int j = 0; j < threadNum; j++){
                    localADMMState[j].u.values[fID] *= pi_incr;
                }
            }
            return rho / pi_decr;
        }
        return rho;
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
        for(int threadID = 0; threadID < threadNum; threadID++) {
            model.x.plusDense(localADMMState[threadID].x);
        }
        model.x.allDividedBy(threadNum);
    }

    private void updateZ(){
        System.arraycopy(model.z.values, 0, oldModelZ.values, 0, featureDimension);
        for(int id = 0; id < featureDimension; id++){
            x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.z.values[id];
            //z=Soft_threshold(lambda/rho,x+u);
            model.z.values[id] = Utils.soft_threshold(lambda / rho * threadNum, x_hat[id] + model.u.values[id]);
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
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        DenseVector oldModel = new DenseVector(featureDimension);
        x_hat = new double[model.featureNum];
        oldModelZ = new DenseVector(featureDimension);

        localADMMState = new ADMMState[threadNum];
        for (int threadID = 0; threadID < threadNum; threadID++) {
            localADMMState[threadID] = new ADMMState(featureDimension);
            int from = trainCorpus.size() * threadID / threadNum;
            int to = trainCorpus.size() * (threadID + 1) / threadNum;
            List<LabeledData> localData = trainCorpus.subList(from, to);
            localTrainCorpus.add(localData);
        }
        long totalBegin = System.currentTimeMillis();
        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            long startTrain = System.currentTimeMillis();
            //Update x;
            updateX(i);
            //Update z
            updateZ();
            updateU();
            //rho = Math.min(maxRho, rho * 1.1);
            rho = calculateRho(rho);

            System.out.println("Iteration " + i);
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");
            totalIterationTime += trainTime;
            System.out.println("totalIterationTime " + totalIterationTime);

            testAndSummary(trainCorpus, testCorpus, model.x, lambda);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );
            if(modelType == 1) {
                if (totalIterationTime > maxTimeLimit) {
                    break;
                }
            }else if(modelType == 0){
                if(i > maxIteration){
                    break;
                }
            }else if (modelType == 2){
                if(converge(oldModel, model.x)){
                    break;
                }
            }
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDimension);
        }
    }


    private static void train() {
        LogisticRegression lrADMM = new LogisticRegression();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new ADMMState(featureDimension);
        start = System.currentTimeMillis();
        lrADMM.trainCore();
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelADMM.LogisticRegression threadNum FeatureDim train_path lambda trainRatio");
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);

        for(int i = 0; i < argv.length - 1; i++){
            if(argv[i].equals("Model")){
                //0: maxIteration  1: maxTime 2: earlyStop
                modelType = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("TimeLimit")){
                maxTimeLimit = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("StopDelta")){
                stopDelta = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("MaxIteration")){
                maxIteration = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("TrainRatio")){
                trainRatio = Double.parseDouble(argv[i+1]);
                if(trainRatio >= 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }
            }
        }
        System.out.println("ThreadNum " + threadNum);
        System.out.println("StopDelta " + stopDelta);
        System.out.println("FeatureDimension " + featureDimension);
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("ModelType " + modelType);
        System.out.println("Iteration Limit " + maxIteration);

        long startLoad = System.currentTimeMillis();
        labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        train();
    }
}
