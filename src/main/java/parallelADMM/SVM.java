package parallelADMM;

import Utils.*;
import math.DenseVector;

import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.lang.management.ManagementFactory;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//https://web.stanford.edu/~boyd/papers/admm/

public class SVM extends model.SVM {
    private static double lambda;
    private static int threadNum;
    private static double trainRatio = 0.5;
    private static int featureDimension;

    private static DenseVector oldModelZ;
    private static List<LabeledData> labeledData;
    private static ADMMState model;
    private ADMMState[] localADMMState;

    private double x_hat[];
    private List<List<LabeledData>> localTrainCorpus = new ArrayList<List<LabeledData>>();
    private static double rho = 0.01;
    private int lbfgsNumIteration = 10;
    private int lbfgsHistory = 10;
    private double rel_par = 1.0;

    private static double ABSTOL = 1e-4;
    private static double RELTOL = 1e-3;

    private class executeRunnable implements Runnable {
        int threadID;
        int iteNum;

        private executeRunnable(int threadID, int iteNum) {
            this.threadID = threadID;
            this.iteNum = iteNum;
        }

        public void run() {
            //Update x;
            parallelLBFGS.train(localADMMState[threadID], lbfgsNumIteration, lbfgsHistory,threadNum,
                    rho, iteNum, localTrainCorpus.get(threadID), "SVM", model.z);
        }
    }

    private void updateX(int iteNumber){
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
        for(int threadID = 0; threadID < threadNum; threadID++) {
            model.x.plusDense(localADMMState[threadID].x);
        }
        model.x.allDividedBy(threadNum);
        System.out.println("[Information]Update X costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");
    }
    private void updateZ(){
        long startTrain = System.currentTimeMillis();
        System.arraycopy(model.z.values, 0, oldModelZ.values, 0, featureDimension);
        for(int id = 0; id < featureDimension; id++){
            x_hat[id] = rel_par * model.x.values[id] + (1 - rel_par) * model.u.values[id];
            //z=Soft_threshold(lambda/rho,x+u);
            model.z.values[id] = (rho / (1.0 / lambda + threadNum * rho)) * (x_hat[id] + model.u.values[id]);
        }
        System.out.println("Update Z costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");
    }
    private void updateU(){
        long startTrain = System.currentTimeMillis();
        Arrays.fill(model.u.values, 0);
        for (int threadID = 0; threadID < threadNum; threadID++) {
            for(int fID = 0; fID < featureDimension; fID++){
                localADMMState[threadID].u.values[fID] += (localADMMState[threadID].x.values[fID] - model.z.values[fID]);
                model.u.values[fID] += localADMMState[threadID].u.values[fID];
            }
        }
        model.u.allDividedBy(threadNum);
        System.out.println("[Information]Update U costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");

    }

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

    private void trainCore() {
        double startCompute = System.currentTimeMillis();
        Collections.shuffle(labeledData);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        x_hat = new double[model.featureNum];
        DenseVector oldModel = new DenseVector(featureDimension);

        localADMMState = new ADMMState[threadNum];
        localTrainCorpus = new ArrayList<List<LabeledData>>();

        for (int threadID = 0; threadID < threadNum; threadID++) {
            localADMMState[threadID] = new ADMMState(featureDimension);
            int from = trainCorpus.size() * threadID / threadNum;
            int to = trainCorpus.size() * (threadID + 1) / threadNum;
            List<LabeledData> localData = trainCorpus.subList(from, to);
            localTrainCorpus.add(localData);
        }
        long totalBegin = System.currentTimeMillis();

        oldModelZ = new DenseVector(featureDimension);
        System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");

        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model.x, lambda);
            //Collections.shuffle(trainCorpus);
            localTrainCorpus = new ArrayList<List<LabeledData>>();
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = trainCorpus.size() * threadID / threadNum;
                int to = trainCorpus.size() * (threadID + 1) / threadNum;
                List<LabeledData> localData = trainCorpus.subList(from, to);
                localTrainCorpus.add(localData);
            }
            long startTrain = System.currentTimeMillis();
            //Update x;
            updateX(i);
            //Update z
            updateZ();
            //Update u
            updateU();


            if(!rhoFixed){
                rho = calculateRho(rho);
            }
            System.out.println("[Information]Current rho is " + rho);
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("[Information]trainTime " + trainTime);
            totalIterationTime += trainTime;
            System.out.println("[Information]totalTrainTime " + totalIterationTime);
            System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin));
            System.out.println("[Information]HeapUsed " + ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed()
                    / 1024 / 1024 + "M");
            System.out.println("[Information]MemoryUsed " + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                    / 1024 / 1024 + "M");
            if(modelType == 1) {
                if (totalIterationTime > maxTimeLimit) {
                    break;
                }
            }else if(modelType == 0){
                if(i > maxIteration){
                    break;
                }
            }
            if(converge(oldModel, model.x, trainCorpus, lambda)) {
                if (modelType == 2)
                    break;
            }
            judgeConverge();
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDimension);
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
        }
    }
    private static void train() {
        SVM svmADMM = new SVM();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new ADMMState(featureDimension);
        long start = System.currentTimeMillis();
        svmADMM.trainCore();
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }


    private boolean judgeConverge(){
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
        System.out.println("AbsoluteErrorDelta " + (EPS_PRI - R_Norm) + " RelativeErrorDelta " + (EPS_DUAL - S_Norm));
        return R_Norm < EPS_PRI && S_Norm < EPS_DUAL;
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelADMM.SVM threadNum featureDimension train_path lambda trainRatio");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
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
                if(trainRatio >= 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }
            }
        }
        System.out.println("[Parameter]ThreadNum " + threadNum);
        System.out.println("[Parameter]StopDelta " + stopDelta);
        System.out.println("[Parameter]FeatureDimension " + featureDimension);
        System.out.println("[Parameter]File Path " + path);
        System.out.println("[Parameter]Lambda " + lambda);
        System.out.println("[Parameter]TrainRatio " + trainRatio);
        System.out.println("[Parameter]TimeLimit " + maxTimeLimit);
        System.out.println("[Parameter]ModelType " + modelType);
        System.out.println("[Parameter]Iteration Limit " + maxIteration);
        System.out.println("[Parameter]Rho Fixed " + rhoFixed);
        System.out.println("[Parameter]Rho " + rho);
        System.out.println("------------------------------------");

        long startLoad = System.currentTimeMillis();
        labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        train();
    }
}
