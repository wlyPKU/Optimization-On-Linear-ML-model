package parallelCD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;
import math.SparseMap;
import math.SparseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by 王羚宇 on 2016/7/26.
 */
//Ref: http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Pg. 7,14,18

//  model       每个线程共享
//  predictValue    每个线程共享
//  可能会发生冲突
public class LogisticRegressionArrayExpTable extends model.LogisticRegression{

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;

    private static SparseVector[] features;

    private static DenseVector modelOfU;
    private static DenseVector modelOfV;
    private static DenseVector model;
    private static double predictValue[];

    private static double featureSquare[];

    private static LabeledData[] trainCorpus;

    private static double[][] expTable = new double[5][100];

    private class updateWPositive implements Runnable
    {
        int from, to;
        private updateWPositive(int from, int to){
            this.from = from;
            this.to = to;
        }
        public double exp(double val) {
            double result = 1.0;
            int bitNum = 0;
            for(int i = 0; i < 3; i++){
                bitNum = (int)val % 100;
                if(bitNum >= 0) {
                    result *= expTable[i][bitNum];
                }else{
                    result /= expTable[i][-bitNum];
                }
                val*= 10;
            }
            return result;
        }
        public double EXP(double x){
            if(x > 10){
                return 0;
            }else if(x < -10){
                return 1;
            }else{
                return 1.0 / (1 + exp(x));
            }
        }
        public void run() {
            double C = 1000000;
            if(lambda != 0){
                C = 1.0 / lambda;
            }
            int[] indices;
            int idx;
            double firstOrderL, oldValue, xj;
            double[] values;
            for(int fIdx = from; fIdx < to; fIdx++){
                indices = features[fIdx].indices;
                values = features[fIdx].values;
                if(featureSquare[fIdx] != 0) {
                    //First Order L:
                    firstOrderL = 0;
                    oldValue = modelOfV.values[fIdx];
                    for (int i = 0; i < indices.length; i++) {
                        idx = indices[i];
                        xj = values[i];
                        LabeledData l = trainCorpus[idx];
                        //double tao = EXP(-l.label * predictValue[idx]);
                        //double tao = 1.0 / (1.0 + Math.exp(-l.label * predictValue[idx]));
                        double tao = EXP(-l.label * predictValue[idx]);
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                    firstOrderL *= C;
                    //误差的来源,0.25.
                    double Uj = 0.25 * C * featureSquare[fIdx];
                    double updateValue = (1 + firstOrderL) / Uj;
                    if (updateValue > modelOfU.values[fIdx]) {
                        modelOfU.values[fIdx] = 0;
                    } else {
                        modelOfU.values[fIdx] -= updateValue;
                    }
                    if(Math.abs(modelOfU.values[fIdx] - oldValue) > 1e-5)
                    {
                        //Update predictValue
                        for (int i = 0; i < indices.length; i++) {
                            idx = indices[i];
                            predictValue[idx] += values[i] * (modelOfU.values[fIdx] - oldValue);
                        }
                    }
                }
            }
        }
    }

    private class updateWNegative implements Runnable
    {
        int from, to;
        private updateWNegative(int from, int to){
            this.from = from;
            this.to = to;
        }
        public double exp(double val) {
            double result = 1.0;
            int bitNum = 0;
            for(int i = 0; i < 3; i++){
                bitNum = ((int)val) % 100;
                if(bitNum >= 0) {
                    result *= expTable[i][bitNum];
                }else{
                    result /= expTable[i][-bitNum];
                }
                val*= 100;
            }
            return result;
        }
        double EXP(double x){
            if(x > 10){
                return 0;
            }else if(x < -10){
                return 1;
            }else{
                return 1 / (1 + exp(x));
            }
        }
        public void run() {
            double C = 1000000;
            if(lambda != 0){
                C = 1.0 / lambda;
            }
            int[] indices;
            int idx;
            double firstOrderL, oldValue, xj;
            double[] values;
            for(int fIdx = from; fIdx < to; fIdx++){
                indices = features[fIdx].indices;
                values = features[fIdx].values;
                if(featureSquare[fIdx] != 0) {
                    //First Order L:
                    firstOrderL = 0;
                    oldValue = modelOfV.values[fIdx];
                    for (int i = 0; i < indices.length; i++) {
                        idx = indices[i];
                        xj = values[i];
                        LabeledData l = trainCorpus[idx];
                        //double tao = 1.0 / (1.0 + Math.exp(-l.label * predictValue[idx]));
                        //double tao = EXP(-l.label * predictValue[idx]);
                        double tao = EXP(-l.label * predictValue[idx]);
                        firstOrderL += l.label * xj * (tao - 1);
                    }
                    firstOrderL *= C;
                    double Uj = 0.25 * C * featureSquare[fIdx];
                    double updateValue = (1 - firstOrderL) / Uj;
                    if (updateValue > modelOfV.values[fIdx]) {
                        modelOfV.values[fIdx] = 0;
                    } else {
                        modelOfV.values[fIdx] -= updateValue;
                    }
                    if(Math.abs(modelOfV.values[fIdx] - oldValue) > 1e-5)
                    {
                        for (int i = 0; i < indices.length; i++) {
                            predictValue[indices[i]] -= values[i] * (modelOfV.values[fIdx] - oldValue);
                        }
                    }
                }
            }
        }
    }

    public class adjustPredictValueThread implements Runnable
    {
        int from, to;
        adjustPredictValueThread(int from, int to){
            this.from = from;
            this.to = to;
        }
        public void run() {
            for(int j = from; j < to; j++){
                LabeledData l = trainCorpus[j];
                predictValue[j] = model.dot(l.data);
            }
        }
    }

    private void adjustPredictValue(){
        ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
        for (int threadID = 0; threadID < threadNum; threadID++) {
            int from = trainCorpus.length * threadID / threadNum;
            int to = trainCorpus.length * (threadID + 1) / threadNum;
            threadPool.execute(new adjustPredictValueThread(from, to));
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
    }

    private void trainCore(LabeledData[] labeledData) {
        double startCompute = System.currentTimeMillis();
        shuffle(labeledData);
        SparseMap[] tmpFeatures = Utils.LoadLibSVMFromLabeledData(labeledData, featureDimension, trainRatio);
        features = Utils.generateSpareVector(tmpFeatures);
        int testBegin = (int)(labeledData.length * trainRatio);
        int testEnd = labeledData.length;
        trainCorpus = new LabeledData[testBegin];
        System.arraycopy(labeledData, 0, trainCorpus, 0, testBegin);
        LabeledData []testCorpus = new LabeledData[testEnd - testBegin];
        System.arraycopy(labeledData, testBegin, testCorpus, 0, testEnd - testBegin);

        predictValue = new double[trainCorpus.length];

        model = new DenseVector(featureDimension);
        DenseVector oldModel = new DenseVector(featureDimension);

        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;
        //Added according to the http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf Page 18.
        featureSquare = new double[featureDimension];
        Arrays.fill(featureSquare, 0);
        for(int idx = 0; idx < featureDimension; idx++){
            for (double xj: features[idx].values) {
                featureSquare[idx] += xj * xj;
            }
        }
        System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model, lambda);
            //if(threadNum != 1 || i == 0){
                adjustPredictValue();
            //}
            long startTrain = System.currentTimeMillis();
            //Update w+
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                threadPool.execute(new updateWPositive(from, to));
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
            //Update w-
            ExecutorService newThreadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                newThreadPool.execute(new updateWNegative(from, to));
            }
            newThreadPool.shutdown();
            while (!newThreadPool.isTerminated()) {
                try {
                    newThreadPool.awaitTermination(1, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            for(int fIdx = 0; fIdx < featureDimension; fIdx ++){
                model.values[fIdx] = modelOfU.values[fIdx] - modelOfV.values[fIdx];
            }
            System.out.println("[Information]trainTime " + trainTime + " ");
            totalIterationTime += trainTime;
            System.out.println("[Information]totalTrainTime " + totalIterationTime);
            System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin) );
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
            if(converge(oldModel, model, trainCorpus, lambda)){
                if (modelType == 2)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, featureDimension);
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
        }
    }


    public static void train(LabeledData[] labeledData) {
        LogisticRegressionArrayExpTable lrSCD = new LogisticRegressionArrayExpTable();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        modelOfU = new DenseVector(featureDimension);
        modelOfV = new DenseVector(featureDimension);
        Arrays.fill(modelOfU.values, 0);
        Arrays.fill(modelOfV.values, 0);
        long start = System.currentTimeMillis();
        lrSCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }
    @SuppressWarnings("unused")
    private void shuffle(LabeledData[] labeledData) {
        //Collections.shuffle(labeledData);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LogisticRegression threadNum FeatureDim train_path lamda trainRatio");
        for(int i = 0; i < 5; i++){
            double l = Math.pow(10, -2*i);
            for(int j = 0; j < 100; j++){
                expTable[i][j] = Math.exp(j * l);
            }
        }
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        if(lambda < 0){
            System.out.println("Please input a correct lambda (>0)");
            System.exit(2);
        }
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
        System.out.println("------------------------------------");
        long startLoad = System.currentTimeMillis();
        LabeledData[] labeledData = Utils.loadLibSVMArray(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}