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
 * Created by WLY on 2016/9/4.
 */

//  model       每个线程共享
//  residual    每个线程共享
//  可能会发生冲突
public class LogisticRegressionHydra extends model.LogisticRegression {
    private static DenseVector model;
    private static double featureSquare[];

    private static SparseVector[] features;
    private static double lambda;
    private static double trainRatio = 0.5;
    private static int threadNum;
    private static int featureDimension;

    private static double beta = 4000;
    private static double allReducedGradient[];
    private static double nextIterationAllReducedGradient[];
    private static double partialDeltaSolution[];

    private List<LabeledData> trainCorpus;

    public class executeRunnable implements Runnable
    {
        int from, to;
        public executeRunnable(int from, int to){
            this.from = from;
            this.to = to;

        }
        public final double exp(double val) {
            final long tmp = (long) (1512775 * val + (1072693248 - 60801));
            return Double.longBitsToDouble(tmp << 32);
        }
        public void run() {
            for(int j = from; j < to; j++){
                double deltaF = 0;
                if(featureSquare[j] != 0) {
                    for(int i = 0; i < features[j].indices.length; i++){
                        deltaF += trainCorpus.get(features[j].indices[i]).label * features[j].values[i]
                                * Math.exp(allReducedGradient[features[j].indices[i]])
                                / (1.0 + Math.exp(allReducedGradient[features[j].indices[i]]));
                    }
                    double h = - (deltaF + lambda * model.values[j] * 2.0) / (featureSquare[j] / 4 * beta + lambda * 2.0);
                    partialDeltaSolution[j] = h;
                    model.values[j] += partialDeltaSolution[j];
                    //Things are different when distributed.
                    for(int i = 0; i < features[j].indices.length; i++){
                        nextIterationAllReducedGradient[features[j].indices[i]] -= features[j].values[i]
                                * partialDeltaSolution[j] * trainCorpus.get(features[j].indices[i]).label;
                    }
                }
            }
        }
    }

    @SuppressWarnings("unused")
    private void shuffle(List<LabeledData> labeledData) {
        Collections.shuffle(labeledData);
    }

    private void trainCore(List<LabeledData> labeledData) {
        double startCompute = System.currentTimeMillis();
        shuffle(labeledData);
        SparseMap[] tmpFeatures = Utils.LoadLibSVMFromLabeledData(labeledData, featureDimension, trainRatio);
        features = Utils.generateSpareVector(tmpFeatures);
        int testBegin = (int)(labeledData.size() * trainRatio);
        int testEnd = labeledData.size();
        trainCorpus = labeledData.subList(0, testBegin);
        List<LabeledData> testCorpus = labeledData.subList(testBegin, testEnd);

        allReducedGradient = new double[trainCorpus.size()];
        nextIterationAllReducedGradient = new double[trainCorpus.size()];
        Arrays.fill(allReducedGradient, 0);
        Arrays.fill(model.values, 0);
        featureSquare = new double[featureDimension];
        for(int i = 0; i < featureDimension; i++){
            featureSquare[i] = 0;
            for(Double v: features[i].values){
                featureSquare[i] += v * v;
            }
        }
        DenseVector oldModel = new DenseVector(featureDimension);
        System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");
        long totalBegin = System.currentTimeMillis();

        long totalIterationTime = 0;

        for(int i = 0; i < trainCorpus.size(); i++){
            allReducedGradient[i] = 0;
        }

        //beta = getBeta(trainCorpus);
        System.out.println("[Information]Beta " + beta );
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            System.arraycopy(allReducedGradient, 0, nextIterationAllReducedGradient, 0, allReducedGradient.length);
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
            if(converge(oldModel, model, trainCorpus, lambda)){
                if (modelType == 2)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values,0, featureDimension);
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
            System.arraycopy(nextIterationAllReducedGradient, 0, allReducedGradient, 0, allReducedGradient.length);
        }
    }

    private static double getBeta(List<LabeledData> labeledData){
        int nonZeroNumber = 0;
        for (LabeledData aLabeledData : labeledData) {
            if (aLabeledData.data.values.length > nonZeroNumber) {
                nonZeroNumber = aLabeledData.data.values.length;
            }
        }
        return 2 * (1 + (labeledData.size() / threadNum - 1) * (nonZeroNumber - 1) / Math.max(threadNum - 1, 1));
    }

    public static void train(List<LabeledData> labeledData) {
        LogisticRegressionHydra lrHydra = new LogisticRegressionHydra();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        model = new DenseVector(featureDimension);
        partialDeltaSolution = new double[featureDimension];
        long start = System.currentTimeMillis();
        lrHydra.trainCore(labeledData);
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
                if(trainRatio > 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }
            }
            if(argv[i].equals("Beta")){
                beta = Double.parseDouble(argv[i + 1]);
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
        labeledData = Utils.normalizeData(labeledData, featureDimension);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
        train(labeledData);
    }
}
