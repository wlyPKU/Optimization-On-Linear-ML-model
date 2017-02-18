package parallelADMM;

import Utils.ADMMState;
import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

//TODO: To be checked ...

/**
 * Created by 王羚宇 on 2016/7/24.
 */
//According to https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
//https://web.stanford.edu/~boyd/papers/admm/

public class LinearRegressionSGDTest extends model.LinearRegression{
    private static int threadNum;
    private static double trainRatio = 0.5;
    private static int featureDimension;

    private static DenseVector oldModelZ;
    private static List<LabeledData> labeledData;
    private static List<LabeledData> reservedCorpus;

    private static ADMMState model;
    private ADMMState[] localADMMState;

    private double x_hat[];
    private List<List<LabeledData>> localTrainCorpus = new ArrayList<List<LabeledData>>();

    private static double rho = 0.1;
    private int lbfgsNumIteration = 10;
    private int lbfgsHistory = 10;
    double rel_par = 1.0;

    private static int sgdIterations = 3;
    private static double lr = 0.05;

    private static double ABSTOL = 1e-4;
    private static double RELTOL = 1e-3;

    private double calculateRho(double rho){
        //https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf PG23
        double miu = 10;
        double pi_incr = 2, pi_decr = 2;
        double R_Norm = 0;
        double S_Norm = 0;
        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < featureDimension; j++) {
                R_Norm += (localADMMState[i].x.values[j] - model.z.values[j])
                        * (localADMMState[i].x.values[j] - model.z.values[j]);
                S_Norm += (model.z.values[j] - oldModelZ.values[j]) * rho
                        * (model.z.values[j] - oldModelZ.values[j]) * rho;
            }
        }
        R_Norm = Math.sqrt(R_Norm);
        S_Norm = Math.sqrt(S_Norm);
        if(R_Norm > miu * S_Norm){
            for(int fID = 0; fID < featureDimension; fID++){
                model.u.values[fID] /= pi_incr;
                for(int j = 0; j < threadNum; j++){
                    localADMMState[j].u.values[fID] /= pi_incr;
                }
            }
            return pi_incr * rho;
        }else if(S_Norm > miu * R_Norm){
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

    private class executeRunnable implements Runnable {
        int threadID;
        int iteNum;

        private executeRunnable(int threadID, int iteNum) {
            this.threadID = threadID;
            this.iteNum = iteNum;
        }

        public void run() {
            //Update x;
            //double modelPenalty = - 2 * lr * lambda;
            for(int ite = 0; ite < sgdIterations; ite++) {
                for (LabeledData tuple : localTrainCorpus.get(threadID)) {
                    //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf Pg 3.
                    /* model pennalty */
                    //model.value[i] -= model.value[i] * 2 * lr * lambda / N;
                    for (int i = 0; i < tuple.data.indices.length; i++) {
                        int idx = tuple.data.indices[i];
                        localADMMState[threadID].x.values[idx] += -lr * rho /  (trainRatio * labeledData.size()) *
                                (-model.z.values[idx] + localADMMState[threadID].u.values[idx]);
                    }
                    double dotProd = localADMMState[threadID].x.dot(tuple.data);
                    localADMMState[threadID].x.plusGradient(tuple.data, - lr * (dotProd - tuple.label));
                }
            }
        }
    }

    private class updateUThread implements Runnable {
        int threadID;
        private updateUThread(int threadID){
            this.threadID = threadID;
        }
        public void run() {
            for(int fID = 0; fID < featureDimension; fID++){
                localADMMState[threadID].u.values[fID] += (localADMMState[threadID].x.values[fID] - model.z.values[fID]);
                model.u.values[fID] += localADMMState[threadID].u.values[fID];
            }
        }
    }

    private void updateX(int iteNumber, boolean verbose){
        long startTrain = System.currentTimeMillis();
        Arrays.fill(model.x.values, 0);
        ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
        for (int threadID = 0; threadID < threadNum; threadID++) {
            threadPool.execute(new executeRunnable(threadID, iteNumber));
        }
        threadPool.shutdown();
        while (!threadPool.isTerminated()) {
            try {
                while (!threadPool.awaitTermination(1, TimeUnit.MILLISECONDS)) {
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        for(int i = 0; i < threadNum; i++){
            model.x.plusDense(localADMMState[i].x);
        }
        model.x.allDividedBy(threadNum);
        if(verbose) {
            System.out.println("[Information]Update X costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");
        }
    }
    private void updateZ(boolean verbose){
        long startTrain = System.currentTimeMillis();
        System.arraycopy(model.z.values, 0, oldModelZ.values, 0, featureDimension);
        for(int id = 0; id < featureDimension; id++){
            x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.z.values[id];
            //z=Soft_threshold(lambda/rho,x+u);
            model.z.values[id] = (x_hat[id] + model.u.values[id]);
        }
        if(verbose) {
            System.out.println("[Information]Update Z costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");
        }
    }

    private void updateU(boolean verbose){
        long startTrain = System.currentTimeMillis();
        Arrays.fill(model.u.values, 0);
        ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
        for (int threadID = 0; threadID < threadNum; threadID++) {
            threadPool.execute(new updateUThread(threadID));
        }
        threadPool.shutdown();
        while (!threadPool.isTerminated()) {
            try {
                while (!threadPool.awaitTermination(1, TimeUnit.MILLISECONDS)) {
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        model.u.allDividedBy(threadNum);
        if(verbose) {
            System.out.println("[Information]Update U costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");
        }
    }

    private boolean judgeConverge(boolean verbose){
        double R_Norm = 0;
        double S_Norm = 0;
        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < featureDimension; j++) {
                R_Norm += (localADMMState[i].x.values[j] - model.z.values[j])
                        * (localADMMState[i].x.values[j] - model.z.values[j]);
                S_Norm += (model.z.values[j] - oldModelZ.values[j]) * rho
                        * (model.z.values[j] - oldModelZ.values[j]) * rho;
            }
        }
        R_Norm = Math.sqrt(R_Norm);
        S_Norm = Math.sqrt(S_Norm);
        double tmpNormX = 0, tmpNormZ = 0, tmpNormU = 0;
        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < featureDimension; j++) {
                tmpNormX += localADMMState[i].x.values[j] * localADMMState[i].x.values[j];
                tmpNormZ += model.z.values[j] * model.z.values[j];
                tmpNormU += localADMMState[i].u.values[j] * localADMMState[i].u.values[j];
            }
        }
        tmpNormX = Math.sqrt(tmpNormX);
        tmpNormZ = Math.sqrt(tmpNormZ);
        tmpNormU = Math.sqrt(tmpNormU);
        double EPS_PRI = Math.sqrt(threadNum) * ABSTOL +RELTOL * Math.max(tmpNormX, tmpNormZ);
        double EPS_DUAL = Math.sqrt(threadNum) * ABSTOL + RELTOL * rho * tmpNormU;
        if(verbose) {
            System.out.println("[Information]AbsoluteErrorDelta " + (EPS_PRI - R_Norm));
            System.out.println("[Information]RelativeErrorDelta " + (EPS_DUAL - S_Norm));
        }
        return R_Norm < EPS_PRI && S_Norm < EPS_DUAL;
    }

    private double trainCore(boolean verbose) {
        double startCompute = System.currentTimeMillis();
        Collections.shuffle(labeledData);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData>trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        x_hat = new double[model.featureNum];
        DenseVector oldModel = new DenseVector(featureDimension);

        localADMMState = new ADMMState[threadNum];
        for (int threadID = 0; threadID < threadNum; threadID++) {
            localADMMState[threadID] = new ADMMState(featureDimension);
        }
        long totalBegin = System.currentTimeMillis();

        oldModelZ = new DenseVector(featureDimension);
        if(verbose) {
            System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");
        }

        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            if(verbose) {
                System.out.println("[Information]Iteration " + i + " ---------------");
            }
            //Collections.shuffle(trainCorpus);
            localTrainCorpus = new ArrayList<List<LabeledData>>();
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = trainCorpus.size() * threadID / threadNum;
                int to = trainCorpus.size() * (threadID + 1) / threadNum;
                List<LabeledData> localData = trainCorpus.subList(from, to);
                localTrainCorpus.add(localData);
            }
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model.x, verbose);
            long startTrain = System.currentTimeMillis();
            //Update x
            updateX(i, verbose);
            //Update z
            updateZ(verbose);
            //Update u
            updateU(verbose);
            if(!rhoFixed){
                rho = calculateRho(rho);
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            if(verbose) {
                System.out.println("[Information]Current rho is " + rho);
                System.out.println("[Information]trainTime " + trainTime);
                totalIterationTime += trainTime;
                System.out.println("[Information]totalTrainTime " + totalIterationTime);
                System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin));
                System.out.println("[Information]HeapUsed " + ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed()
                        / 1024 / 1024 + "M");
                System.out.println("[Information]MemoryUsed " + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                        / 1024 / 1024 + "M");
            }
            if(modelType == 1) {
                if (totalIterationTime > maxTimeLimit) {
                    break;
                }
            }else if(modelType == 0) {
                if (i > maxIteration) {
                    break;
                }
            }
            if(converge(oldModel, model.x, trainCorpus, verbose)) {
                if (modelType == 2)
                    break;
            }
            judgeConverge(verbose);
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDimension);
            if(diverge){
                if(verbose) {
                    System.out.println("[Warning]Diverge happens!");
                }
                break;
            }
        }
        return test(trainCorpus, model.x);
    }

    private static double train(boolean verbose) {
        LinearRegressionSGDTest lrADMM = new LinearRegressionSGDTest();
        model = new ADMMState(featureDimension);
        long start = System.currentTimeMillis();
        double trainLoss = lrADMM.trainCore(verbose);
        long cost = System.currentTimeMillis() - start;
        if (verbose) {
            System.out.println("Training cost " + cost + " ms totally.");
        }
        return trainLoss;
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: ADMM.LinearRegression threadNum featureDimension train_path [trainRatio]");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        for(int i = 0; i < argv.length - 1; i++){
            if(argv[i].equals("Model")){
                //0: maxIteration  1: maxTime 2: earlyStop
                modelType = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("TimeLimit")){
                maxTimeLimit = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("StopDelta")){
                ABSTOL = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("MaxIteration")){
                maxIteration = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("RhoInitial")){
                rho = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("RhoFixed")){
                rhoFixed = Boolean.parseBoolean(argv[i + 1]);
            }
            if(argv[i].equals("TrainRatio")){
                trainRatio = Double.parseDouble(argv[i+1]);
                if(trainRatio > 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }
            }
            if(argv[i].equals("LR")){
                lr = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("SGDIterations")){
                sgdIterations = Integer.parseInt(argv[i + 1]);
            }
            if(argv[i].equals("DoNormalize")){
                doNormalize = Boolean.parseBoolean(argv[i + 1]);
            }
        }
        System.out.println("[Parameter]ThreadNum " + threadNum);
        System.out.println("[Parameter]StopDelta " + stopDelta);
        System.out.println("[Parameter]FeatureDimension " + featureDimension);
        System.out.println("[Parameter]File Path " + path);
        System.out.println("[Parameter]TrainRatio " + trainRatio);
        System.out.println("[Parameter]TimeLimit " + maxTimeLimit);
        System.out.println("[Parameter]ModelType " + modelType);
        System.out.println("[Parameter]Iteration Limit " + maxIteration);
        System.out.println("[Parameter]Rho Fixed " + rhoFixed);
        System.out.println("[Parameter]Rho " + rho);
        System.out.println("[Parameter]SGDIterations " + sgdIterations);
        System.out.println("[Parameter]LR " + lr);
        System.out.println("[Parameter]DoNormalize " + doNormalize);
        System.out.println(" ------------------------------------");

        long startLoad = System.currentTimeMillis();
        reservedCorpus = Utils.loadLibSVM(path, featureDimension);
        if(doNormalize){
            reservedCorpus = Utils.normalizeData(reservedCorpus, featureDimension);
        }
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");

        /* choose a good learning rate */
        labeledData = reservedCorpus.subList(0, Math.min(reservedCorpus.size(), Math.min(reservedCorpus.size() / 10, 10000)));
        Collections.shuffle(labeledData);
        double learningRates[] = {1, 0.33, 0.1, 0.033, 0.01, 0.0033, 0.001, 0.00033, 0.0001, 0.00033, 0.00001};
        double lowestObjectValue = 1e300;
        int minLearningRateIndex = 0;
        for(int i = 0; i < learningRates.length; i++){
            lr = learningRates[i];
            double currentObjectValue = train(false);
            System.out.println("[Learning rate test]Learning rate " + learningRates[i] + " objective value "
                    + currentObjectValue + " on " + labeledData.size() + " samples.");
            if(lowestObjectValue >  currentObjectValue){
                minLearningRateIndex = i;
                lowestObjectValue = currentObjectValue;
            }
        }
        lr = learningRates[minLearningRateIndex];
        labeledData = reservedCorpus;
        System.out.println("[Parameter]LearningRate " + lr);

        train(true);
    }
}
