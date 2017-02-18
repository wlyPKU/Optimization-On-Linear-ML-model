package parallelCD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;
import math.SparseMap;
import math.SparseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
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
public class LogisticRegressionCDN extends model.LogisticRegression{

    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;

    private static SparseVector[] features;

    private static DenseVector model;
    private static double predictValue[];

    private static double featureSquare[];

    private static List<LabeledData> trainCorpus;

    private class update implements Runnable
    {
        int from, to;
        private update(int from, int to){
            this.from = from;
            this.to = to;

        }
        public double exp(double val) {
            final long tmp = (long) (1512775 * val + (1072693248 - 60801));
            return Double.longBitsToDouble(tmp << 32);
        }
        public double log(double x) {
            return 6 * (x - 1) / (x + 1 + 4 * (Math.sqrt(x)));
        }
        double Ldiff(int x_i, double diff){
            double result = 0;
            int []indices = features[x_i].indices;
            double []values = features[x_i].values;
            for(int i = 0; i < indices.length; i++){
                LabeledData l = trainCorpus.get(indices[i]);
                result += Math.log(1 + exp(- l.label * predictValue[i])) * (exp(-l.label * values[i] * diff) - 1);
            }
            return 1 / lambda * result;
        }

        double g_xi(double z, int x_i){
            double result = 0;
            result += Math.abs(model.values[x_i] + z) - Math.abs(model.values[x_i]);
            result += Ldiff(x_i, z);
            return result;
        }

        public void run() {
            double C = Double.MAX_VALUE;
            if(lambda != 0){
                C = 1.0 / lambda;
            }
            long costTime = 0;
            int[] indices;
            int idx;
            double firstOrderL, secondOrderL, xj;
            double[] values;
            for(int fIdx = from; fIdx < to; fIdx++){
                indices = features[fIdx].indices;
                values = features[fIdx].values;
                if(featureSquare[fIdx] != 0) {
                    double d = - model.values[fIdx];
                    //First Order L:
                    firstOrderL = 0;
                    secondOrderL = 0;
                    for (int i = 0; i < indices.length; i++) {
                        idx = indices[i];
                        xj = values[i];
                        LabeledData l = trainCorpus.get(idx);
                        double tao = 1 / (1 + exp(-l.label * predictValue[idx]));
                        firstOrderL += l.label * xj * (tao - 1);
                        secondOrderL += xj * xj * tao * (1 - tao);
                    }
                    firstOrderL *= C;
                    secondOrderL *= C;
                    if(firstOrderL + 1 <= secondOrderL * model.values[fIdx]){
                        d = - (firstOrderL + 1) / secondOrderL;
                    }else if(firstOrderL - 1 >= secondOrderL * model.values[fIdx]){
                        d = - (firstOrderL - 1) / secondOrderL;
                    }
                    double delta = (firstOrderL * d + Math.abs(model.values[fIdx] + d))
                            - Math.abs(model.values[fIdx]);
                    double gamma = 1;
                    double rhs_c = 0.01 * delta;

                    long startTime = System.currentTimeMillis();

                    for(int i = 0; i < 5; i++){
                        double change_in_obj = g_xi(d, fIdx);
                        if(change_in_obj <= gamma * rhs_c){
                            model.values[fIdx] += d;
                            for (int id = 0; id < indices.length; id++) {
                                predictValue[indices[id]] += values[id] * d;
                            }
                            break;
                        }
                        gamma *= 0.1;
                        d *= 0.1;
                    }
                    costTime += System.currentTimeMillis() - startTime;
                }
            }
            System.out.println(costTime);
        }
    }

    private void trainCore(List<LabeledData> labeledData) {
        double startCompute = System.currentTimeMillis();
        shuffle(labeledData);
        SparseMap[] tmpFeatures = Utils.LoadLibSVMFromLabeledData(labeledData, featureDimension, trainRatio);
        features = Utils.generateSpareVector(tmpFeatures);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        trainCorpus = labeledData.subList(0, testBegin + 1);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        predictValue = new double[trainCorpus.size()];

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
            //if(threadNum != 1 || i != 0)
            {
                for(int idx = 0; idx < trainCorpus.size(); idx++){
                    LabeledData l = trainCorpus.get(idx);
                    predictValue[idx] = model.dot(l.data);
                }
            }
            long startTrain = System.currentTimeMillis();
            //Update w+
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                threadPool.execute(new update(from, to));
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
            long trainTime = System.currentTimeMillis() - startTrain;
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


    public static void train(List<LabeledData> labeledData) {
        LogisticRegressionCDN lrSCD = new LogisticRegressionCDN();
        //http://www.csie.ntu.edu.tw/~cjlin/papers/l1.pdf 3197-3200+
        long start = System.currentTimeMillis();
        lrSCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }

    @SuppressWarnings("unused")
    private void shuffle(List<LabeledData> labeledData) {
        Collections.shuffle(labeledData);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.LogisticRegression threadNum FeatureDim train_path lamda trainRatio");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        featureDimension = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        if(lambda <= 0){
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
        List<LabeledData> labeledData = Utils.loadLibSVM(path, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}