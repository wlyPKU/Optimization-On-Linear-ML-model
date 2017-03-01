package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.lang.management.ManagementFactory;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by WLY on 2016/9/4.
 */
public class SVMPSGD extends model.SVM{
    public static long start;

    private DenseVector globalModel;
    private static DenseVector[] localModel;
    public static double trainRatio = 0.5;
    public static int threadNum;
    public static double lambda = 0.1;

    private static double learningRate = 0.001;
    public int iteration = 0;

    private void setNewLearningRate(){
    }

    public class executeRunnable implements Runnable
    {
        List<LabeledData> localList;
        double lambda;
        int globalCorpusSize;
        int threadID;
        public executeRunnable(int threadID, List<LabeledData> list, double lambda, int globalCorpusSize){
            this.threadID = threadID;
            localList = list;
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;
        }
        public void run() {
            Collections.shuffle(localList);
            sgdOneEpoch(localList, learningRate, lambda);
        }
        void sgdOneEpoch(List<LabeledData> list, double lr, double lambda) {
            double modelPenalty = -2 * lr * lambda / globalCorpusSize;
            for (LabeledData labeledData : list) {
                //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf Pg 3.
                /* model pennalty */
                //model.value[i] -= model.value[i] * 2 * lr * lambda / N;
                localModel[threadID].multiplySparse(labeledData.data, modelPenalty);
                double dotProd = localModel[threadID].dot(labeledData.data);
                if (1 - dotProd * labeledData.label > 0) {
                    /* residual pennalty */
                    localModel[threadID].plusGradient(labeledData.data, lr * labeledData.label);
                }
            }
        }
    }

    public double train(List<LabeledData> corpus, DenseVector model, boolean verbose) {
        double startCompute = System.currentTimeMillis();
        List<List<LabeledData>> ThreadTrainCorpus = new ArrayList<List<LabeledData>>();
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        Collections.shuffle(trainCorpus);
        for(int threadID = 0; threadID < threadNum; threadID++){
            int from = end * threadID / threadNum;
            int to = end * (threadID + 1) / threadNum;
            List<LabeledData> threadCorpus = corpus.subList(from, to);
            ThreadTrainCorpus.add(threadCorpus);
        }

        DenseVector oldModel = new DenseVector(model.values.length);
        globalModel = new DenseVector(model.dim);
        localModel = new DenseVector[threadNum];

        for(int i = 0; i < threadNum; i++){
            localModel[i] = new DenseVector(model.dim);
        }

        long totalBegin = System.currentTimeMillis();
        if(verbose) {
            System.out.println("[Prepare]Pre-computation takes " + (System.currentTimeMillis() - startCompute) + " ms totally");
        }
        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            if(verbose) {
                System.out.println("[Information]Iteration " + i + " ---------------");
                System.out.println("[Information]Learning rate " + learningRate);
            }
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model, lambda, verbose);

            long startTrain = System.currentTimeMillis();
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(threadID, ThreadTrainCorpus.get(threadID), lambda, trainCorpus.size()));
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
            Arrays.fill(globalModel.values, 0);
            for(int id = 0; id < threadNum; id++){
                for(int j = 0; j < model.dim; j++){
                    globalModel.values[j] += localModel[id].values[j];
                }
            }
            globalModel.allDividedBy(threadNum);
            System.arraycopy(globalModel.values, 0, model.values, 0, model.dim);
            for(int id = 0; id < threadNum; id++){
                System.arraycopy(model.values, 0, localModel[id].values, 0, model.dim);
            }
            totalIterationTime += trainTime;
            if(verbose) {
                System.out.println("[Information]trainTime " + trainTime);
                System.out.println("[Information]totalTrainTime " + totalIterationTime);
                System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin));
                System.out.println("[Information]HeapUsed " + ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed()
                        / 1024 / 1024 + "M");
                System.out.println("[Information]MemoryUsed " + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                        / 1024 / 1024 + "M");
            }
            iteration++;
            setNewLearningRate();
            if(modelType == 1) {
                if (totalIterationTime > maxTimeLimit) {
                    break;
                }
            }else if(modelType == 0){
                if(i > maxIteration){
                    break;
                }
            }
            if(converge(oldModel, model, trainCorpus, lambda, verbose)){
                if (modelType == 2)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            if(diverge){
                if(verbose) {
                    System.out.println("[Warning]Diverge happens!");
                }
                break;
            }
        }
        return SVMLoss(trainCorpus, model, lambda);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.SVM threadNum dim train_path lambda [trainRatio]");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        corpus = Utils.normalizeData(corpus, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("[Prepare]Loading corpus completed, takes " + loadTime + " ms");
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
            if(argv[i].equals("DoNormalize")){
                doNormalize = Boolean.parseBoolean(argv[i + 1]);
            }
            if(argv[i].equals("TrainRatio")){
                trainRatio = Double.parseDouble(argv[i+1]);
                if(trainRatio > 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }
            }
        }
        if(doNormalize){
            corpus = Utils.normalizeData(corpus, dim);
        }
        System.out.println("[Parameter]ThreadNum " + threadNum);
        System.out.println("[Parameter]StopDelta " + stopDelta);
        System.out.println("[Parameter]FeatureDimension " + dim);
        System.out.println("[Parameter]File Path " + path);
        System.out.println("[Parameter]Lambda " + lambda);
        System.out.println("[Parameter]TrainRatio " + trainRatio);
        System.out.println("[Parameter]TimeLimit " + maxTimeLimit);
        System.out.println("[Parameter]ModelType " + modelType);
        System.out.println("[Parameter]Iteration Limit " + maxIteration);
        System.out.println("[Parameter]DoNormalize " + doNormalize);

        System.out.println("------------------------------------");

        SVMPSGD svm = new SVMPSGD();

        /* choose a good learning rate */
        List<LabeledData> miniCorpus = corpus.subList(0, Math.min(corpus.size(), Math.min(corpus.size() / 10, 10000)));
        Collections.shuffle(miniCorpus);
        double learningRates[] = {1, 0.33, 0.1, 0.033, 0.01, 0.0033, 0.001, 0.00033, 0.0001, 0.00033, 0.00001, 0.000033,
                0.00001, 0.0000033, 0.000001};
        double lowestObjectValue = 1e300;
        int minLearningRateIndex = 0;
        for(int i = 0; i < learningRates.length; i++){
            learningRate = learningRates[i];
            DenseVector model = new DenseVector(dim);
            double currentObjectValue = svm.train(miniCorpus, model, false);
            System.out.println("[Learning rate test]Learning rate " + learningRates[i] + " objective value "
                    + currentObjectValue + " on " + miniCorpus.size() + " samples.");
            if(lowestObjectValue >  currentObjectValue){
                minLearningRateIndex = i;
                lowestObjectValue = currentObjectValue;
            }
        }
        learningRate = learningRates[minLearningRateIndex];
        System.out.println("[Parameter]LearningRate " + learningRate);

        DenseVector model = new DenseVector(dim);
        start = System.currentTimeMillis();
        svm.train(corpus, model, true);

        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }
}
