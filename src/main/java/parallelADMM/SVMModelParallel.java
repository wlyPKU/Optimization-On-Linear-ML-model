package parallelADMM;


/**
 * Created by WLY on 2016/9/4.
 */

import Utils.ADMMFeatureState;
import Utils.LabeledData;
import Utils.Utils;
import Utils.parallelLBFGSFeature;
import math.DenseVector;
import math.SparseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

//According to https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
//https://web.stanford.edu/~boyd/papers/admm/

/**
 * Created by 王羚宇 on 2016/7/24.
 */
//Reference:
//https://github.com/niangaotuantuan/LASSO-Regression/blob/8338930ca6017927efcb362c17a37a68a160290f/LASSO_ADMM.m
//https://web.stanford.edu/~boyd/papers/pdf/admm_slides.pdf
//https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
//https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
//http://www.simonlucey.com/lasso-using-admm/
//http://users.ece.gatech.edu/~justin/CVXOPT-Spring-2015/resources/14-notes-admm.pdf
public class SVMModelParallel extends model.SVM {
    private static double lambda;
    private static int threadNum;
    private static double trainRatio = 0.5;
    private static int featureDimension;

    private static DenseVector oldModelZ;
    private static List<LabeledData> labeledData;
    private List<List<LabeledData>> localLabeledData = new ArrayList<List<LabeledData>>();
    private static ADMMFeatureState model;
    private ADMMFeatureState[] localADMMState;

    private static double rho = 2;
    private int lbfgsNumIteration = 10;
    private int lbfgsHistory = 10;

    static double ABSTOL = 1e-3;
    static double RELTOL = 1e-3;

    private double calculateRho(double rho){
        //https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf PG20
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

    private class executeRunnable implements Runnable
    {
        int threadID;
        int iteNum;
        private executeRunnable(int threadID, int iteNum){
            this.threadID = threadID;
            this.iteNum = iteNum;
        }
        public void run() {
            //Update x;
            parallelLBFGSFeature.train(localADMMState[threadID], lbfgsNumIteration, lbfgsHistory, threadNum,
                    rho, lambda, iteNum, localLabeledData.get(threadID), "SVM", model.z);
            computeAx(localADMMState[threadID], localLabeledData.get(threadID));
        }
        double computeAXi(DenseVector x, LabeledData data){
            double result = 0;
            SparseVector s = data.data;
            for(int i = 0; i < s.indices.length; i++){
                result += (s.values == null? 1:s.values[i]) * x.values[s.indices[i]];
            }
            return result;
        }
        void computeAx(ADMMFeatureState state, List<LabeledData> labeledData){
            for(int i = 0; i < state.sampleDimension; i++){
                state.AX[i] = computeAXi(state.x, labeledData.get(i));
            }
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
        System.out.println("[Information]Update X costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");
    }
    private void updateZ(){
        long startTrain = System.currentTimeMillis();
        System.arraycopy(model.z.values, 0, oldModelZ.values, 0, model.z.dim);
        for(int id = 0; id < model.z.dim; id++){
            double label = labeledData.get(id).label;
            if(label == 1){
                if(model.AX[id] + model.u.values[id] > 1.0 / threadNum){
                    model.z.values[id] = model.AX[id] + model.u.values[id];
                }else if(model.AX[id] + model.u.values[id] < 1.0 / threadNum - threadNum / rho){
                    model.z.values[id] = model.AX[id] + model.u.values[id] + threadNum / rho;
                }else{
                    model.z.values[id] = 1.0 / threadNum ;
                }
            }else if(label == -1){
                if(model.AX[id] + model.u.values[id] < - 1.0 / threadNum){
                    model.z.values[id] = model.AX[id] + model.u.values[id];
                }else if(model.AX[id] + model.u.values[id] > -1.0 / threadNum + threadNum / rho){
                    model.z.values[id] = model.AX[id] + model.u.values[id] - threadNum / rho;
                }else{
                    model.z.values[id] = -1.0 / threadNum ;
                }
            }

        }
        System.out.println("[Information]Update Z costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");
    }

    private void updateU(){
        long startTrain = System.currentTimeMillis();
        for (int id = 0; id < model.u.dim; id++) {
            model.u.values[id] += model.AX[id] - model.z.values[id];
        }
        for(int i = 0; i < threadNum; i++){
            System.arraycopy(model.u.values, 0, localADMMState[i].u.values, 0, model.u.dim);
        }
        System.out.println("[Information]Update U costs " + String.valueOf(System.currentTimeMillis() - startTrain) + " ms");

    }
    @SuppressWarnings("unused")
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
        System.out.println("[Information]AbsoluteErrorDelta " + (EPS_PRI - R_Norm));
        System.out.println("[Information]RelativeErrorDelta " + (EPS_DUAL - S_Norm));
        return R_Norm < EPS_PRI && S_Norm < EPS_DUAL;
    }

    private List<LabeledData> processLocalCorpus(int begin, int end){
        List<LabeledData> list = new ArrayList<LabeledData>();
        for(int id = 0; id < (int)(trainRatio * labeledData.size()); id++){
            LabeledData l = labeledData.get(id);
            SparseVector data = l.data;
            double value = l.label;
            List<Integer> featureIndex = new ArrayList<Integer>();
            List<Double> featureValue = new ArrayList<Double>();
            for(int i = 0; i < data.indices.length; i++){
                if(data.indices[i] >= end){
                    break;
                }
                if(data.indices[i] < begin){
                    continue;
                }
                if(data.values == null){
                    featureIndex.add(data.indices[i] - begin);
                }else{
                    featureIndex.add(data.indices[i] - begin);
                    featureValue.add(data.values[i]);
                }
            }
            int index[] = new int[featureIndex.size()];
            if(data.values == null){
                for(int i = 0; i < index.length; i++){
                    index[i] = featureIndex.get(i);
                }
                SparseVector newVector = new SparseVector(index.length, index);
                LabeledData insertData = new LabeledData(newVector, value);
                list.add(insertData);
            }else{
                double values[] = new double[featureIndex.size()];
                for(int i = 0; i < index.length; i++){
                    index[i] = featureIndex.get(i);
                    values[i] = featureValue.get(i);
                }
                SparseVector newVector = new SparseVector(index.length, index, values);
                LabeledData insertData = new LabeledData(newVector, value);
                list.add(insertData);
            }
        }
        return list;
    }

    private void trainCore() {
        Collections.shuffle(labeledData);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData>trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        DenseVector oldModel = new DenseVector(featureDimension);
        model = new ADMMFeatureState(featureDimension, trainCorpus.size(), 0);

        localADMMState = new ADMMFeatureState[threadNum];
        for (int threadID = 0; threadID < threadNum; threadID++) {
            int beginOffset = featureDimension * threadID / threadNum;
            int dimension = featureDimension / threadNum;
            localADMMState[threadID] = new ADMMFeatureState(dimension, trainCorpus.size(), beginOffset);
            List<LabeledData> tmp = processLocalCorpus(beginOffset, beginOffset + dimension);
            localLabeledData.add(tmp);
        }
        long totalBegin = System.currentTimeMillis();

        oldModelZ = new DenseVector(trainCorpus.size());

        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model.x, lambda);
            long startTrain = System.currentTimeMillis();
            //Update z
            updateZ();
            //Update u
            updateU();
            //Update x
            updateX(i);
            for(int id = 0; id < threadNum; id++){
                System.arraycopy(localADMMState[id].x.values, 0, model.x.values, localADMMState[id].beginOffset,
                        localADMMState[id].x.values.length);
            }
            //mergeAx
            Arrays.fill(model.AX, 0);
            for(int j = 0; j < trainCorpus.size(); j++){
                for(int id = 0; id < threadNum; id++){
                    model.AX[j] += localADMMState[id].AX[j];
                }
                model.AX[j] /= threadNum;
            }
            for(int id = 0; id < threadNum; id++){
                System.arraycopy(model.AX, 0, localADMMState[id].globalAX, 0, model.AX.length);
            }

            //rho = Math.min(rho * 1.1, maxRho);
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
            //judgeConverge();
            System.arraycopy(model.x.values, 0, oldModel.values, 0, featureDimension);
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
        }
    }

    private static void train() {
        SVMModelParallel svmLBFGS = new SVMModelParallel();
        long start = System.currentTimeMillis();
        svmLBFGS.trainCore();
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }
    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelADMM.SVM threadNum featureDimension train_path lambda trainRatio");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        trainRatio = 0.5;
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
