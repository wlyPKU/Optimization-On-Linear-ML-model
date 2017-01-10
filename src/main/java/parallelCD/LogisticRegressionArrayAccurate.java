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
public class LogisticRegressionArrayAccurate extends model.LogisticRegression{
    private static double ExpAdjustment[] = {
                1.040389835,
                1.039159306,
                1.037945888,
                1.036749401,
                1.035569671,
                1.034406528,
                1.033259801,
                1.032129324,
                1.031014933,
                1.029916467,
                1.028833767,
                1.027766676,
                1.02671504,
                1.025678708,
                1.02465753,
                1.023651359,
                1.022660049,
                1.021683458,
                1.020721446,
                1.019773873,
                1.018840604,
                1.017921503,
                1.017016438,
                1.016125279,
                1.015247897,
                1.014384165,
                1.013533958,
                1.012697153,
                1.011873629,
                1.011063266,
                1.010265947,
                1.009481555,
                1.008709975,
                1.007951096,
                1.007204805,
                1.006470993,
                1.005749552,
                1.005040376,
                1.004343358,
                1.003658397,
                1.002985389,
                1.002324233,
                1.001674831,
                1.001037085,
                1.000410897,
                0.999796173,
                0.999192819,
                0.998600742,
                0.998019851,
                0.997450055,
                0.996891266,
                0.996343396,
                0.995806358,
                0.995280068,
                0.99476444,
                0.994259393,
                0.993764844,
                0.993280711,
                0.992806917,
                0.992343381,
                0.991890026,
                0.991446776,
                0.991013555,
                0.990590289,
                0.990176903,
                0.989773325,
                0.989379484,
                0.988995309,
                0.988620729,
                0.988255677,
                0.987900083,
                0.987553882,
                0.987217006,
                0.98688939,
                0.98657097,
                0.986261682,
                0.985961463,
                0.985670251,
                0.985387985,
                0.985114604,
                0.984850048,
                0.984594259,
                0.984347178,
                0.984108748,
                0.983878911,
                0.983657613,
                0.983444797,
                0.983240409,
                0.983044394,
                0.982856701,
                0.982677276,
                0.982506066,
                0.982343022,
                0.982188091,
                0.982041225,
                0.981902373,
                0.981771487,
                0.981648519,
                0.981533421,
                0.981426146,
                0.981326648,
                0.98123488,
                0.981150798,
                0.981074356,
                0.981005511,
                0.980944219,
                0.980890437,
                0.980844122,
                0.980805232,
                0.980773726,
                0.980749562,
                0.9807327,
                0.9807231,
                0.980720722,
                0.980725528,
                0.980737478,
                0.980756534,
                0.98078266,
                0.980815817,
                0.980855968,
                0.980903079,
                0.980955475,
                0.981017942,
                0.981085714,
                0.981160303,
                0.981241675,
                0.981329796,
                0.981424634,
                0.981526154,
                0.981634325,
                0.981749114,
                0.981870489,
                0.981998419,
                0.982132873,
                0.98227382,
                0.982421229,
                0.982575072,
                0.982735318,
                0.982901937,
                0.983074902,
                0.983254183,
                0.983439752,
                0.983631582,
                0.983829644,
                0.984033912,
                0.984244358,
                0.984460956,
                0.984683681,
                0.984912505,
                0.985147403,
                0.985388349,
                0.98563532,
                0.98588829,
                0.986147234,
                0.986412128,
                0.986682949,
                0.986959673,
                0.987242277,
                0.987530737,
                0.987825031,
                0.988125136,
                0.98843103,
                0.988742691,
                0.989060098,
                0.989383229,
                0.989712063,
                0.990046579,
                0.990386756,
                0.990732574,
                0.991084012,
                0.991441052,
                0.991803672,
                0.992171854,
                0.992545578,
                0.992924825,
                0.993309578,
                0.993699816,
                0.994095522,
                0.994496677,
                0.994903265,
                0.995315266,
                0.995732665,
                0.996155442,
                0.996583582,
                0.997017068,
                0.997455883,
                0.99790001,
                0.998349434,
                0.998804138,
                0.999264107,
                0.999729325,
                1.000199776,
                1.000675446,
                1.001156319,
                1.001642381,
                1.002133617,
                1.002630011,
                1.003131551,
                1.003638222,
                1.00415001,
                1.004666901,
                1.005188881,
                1.005715938,
                1.006248058,
                1.006785227,
                1.007327434,
                1.007874665,
                1.008426907,
                1.008984149,
                1.009546377,
                1.010113581,
                1.010685747,
                1.011262865,
                1.011844922,
                1.012431907,
                1.013023808,
                1.013620615,
                1.014222317,
                1.014828902,
                1.01544036,
                1.016056681,
                1.016677853,
                1.017303866,
                1.017934711,
                1.018570378,
                1.019210855,
                1.019856135,
                1.020506206,
                1.02116106,
                1.021820687,
                1.022485078,
                1.023154224,
                1.023828116,
                1.024506745,
                1.025190103,
                1.02587818,
                1.026570969,
                1.027268461,
                1.027970647,
                1.02867752,
                1.029389072,
                1.030114973,
                1.030826088,
                1.03155163,
                1.032281819,
                1.03301665,
                1.033756114,
                1.034500204,
                1.035248913,
                1.036002235,
                1.036760162,
                1.037522688,
                1.038289806,
                1.039061509,
                1.039837792,
                1.040618648
    };
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

    private class updateWPositive implements Runnable
    {
        int from, to;
        private updateWPositive(int from, int to){
            this.from = from;
            this.to = to;
        }
        public final double exp(double val) {
            final long tmp = (long) (1512775 * val + (1072693248 - 60801));
            int index = (int)(tmp >> 12) & 0xFF;
            return Double.longBitsToDouble(tmp << 32)* ExpAdjustment[index];
        }
        public double EXP(double x){
            if(x > 10){
                return 0;
            }else if(x < -10){
                return 1;
            }else{
                return 1.0 / (1 + Math.exp(x));
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
                        double tao = 1.0 / (1.0 + Math.exp(-l.label * predictValue[idx]));
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
        public final double exp(double val) {
            final long tmp = (long) (1512775 * val + (1072693248 - 60801));
            int index = (int)(tmp >> 12) & 0xFF;
            return Double.longBitsToDouble(tmp << 32)* ExpAdjustment[index];
        }
        public double EXP(double x){
            if(x > 10){
                return 0;
            }else if(x < -10){
                return 1;
            }else{
                return 1 / (1 + Math.exp(x));
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
                        double tao = 1.0 / (1.0 + Math.exp(-l.label * predictValue[idx]));
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
        LogisticRegressionArrayAccurate lrSCD = new LogisticRegressionArrayAccurate();
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