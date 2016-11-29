package parallelCD;

import Utils.LabeledData;
import Utils.Utils;
import it.unimi.dsi.fastutil.ints.Int2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import math.DenseVector;
import math.SparseMap;
import math.SparseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by WLY on 2016/9/4.
 */

//  model       每个线程共享
//  residual    每个线程共享
//  可能会发生冲突
public class Lasso extends model.Lasso {

    private static double residual[];
    private static DenseVector model;
    private static double featureSquare[];
    private static SparseVector[] features;
    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;

    public class executeRunnable implements Runnable
    {
        int from, to;
        public executeRunnable(int from, int to){
            this.from = from;
            this.to = to;

        }
        public void run() {
            for(int j = from; j < to; j++){
                if(featureSquare[j] != 0) {
                    double oldValue = model.values[j];
                    double updateValue = 0;
                    for(int i = 0; i < features[j].indices.length; i++){
                        int idx = features[j].indices[i];
                        double xj = features[j].values[i];
                        updateValue += xj * residual[idx];
                    }
                    updateValue /= featureSquare[j];
                    model.values[j] += updateValue;
                    model.values[j] = Utils.soft_threshold(lambda / featureSquare[j], model.values[j]);
                    double deltaChange = model.values[j] - oldValue;
                    if (deltaChange != 0) {
                        for(int i = 0; i < features[j].indices.length; i++){
                            int idx = features[j].indices[i];
                            double value = features[j].values[i];
                            residual[idx] -= value * deltaChange;
                        }
                    }
                }
            }
        }
    }

    private void adjustResidual(DenseVector model, double[] residual){
        for (int i = 0; i < features[featureDimension].indices.length; i++) {
            int idx = features[featureDimension].indices[i];
            double value = features[featureDimension].values[i];
            residual[idx] = value;
        }

        for(int j = 0; j < featureDimension; j++) {
            for (int i = 0; i < features[j].indices.length; i++) {
                int idx = features[j].indices[i];
                double value = features[j].values[i];
                residual[idx] -= value * model.values[j];
            }
        }
    }

    private void shuffle(List<LabeledData> labeledData) {
        Collections.shuffle(labeledData);
        SparseMap[] tmpFeatures = Utils.LoadLibSVMFromLabeledData(labeledData, featureDimension, trainRatio);
        features = Utils.generateSpareVector(tmpFeatures);
    }

    private void trainCore(List<LabeledData> labeledData) {
        double startCompute = System.currentTimeMillis();
        shuffle(labeledData);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        List<LabeledData> trainCorpus = labeledData.subList(0, testBegin + 1);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);
        featureSquare = new double[featureDimension];
        residual = new double[trainCorpus.size()];
        for(int i = 0; i < featureDimension; i++){
            featureSquare[i] = 0;
            for(Double v: features[i].values){
                featureSquare[i] += v * v;
            }
        }

        for(int i = 0; i < features[featureDimension].indices.length; i++){
            residual[features[featureDimension].indices[i]] = features[featureDimension].values[i];
        }
        DenseVector oldModel = new DenseVector(featureDimension);

        System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");

        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;

        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model, lambda);
            long startTrain = System.currentTimeMillis();
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                int from = featureDimension * threadID / threadNum;
                int to = featureDimension * (threadID + 1) / threadNum;
                threadPool.execute(new executeRunnable(from, to));
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
            adjustResidual(model, residual);
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
            if(converge(oldModel, model)){
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
        Lasso lassoModelParallelCD = new Lasso();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new DenseVector(featureDimension);
        Arrays.fill(model.values, 0);
        long start = System.currentTimeMillis();
        lassoModelParallelCD.trainCore(labeledData);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelCD.Lasso threadNum FeatureDimension train_path lambda trainRatio");
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
