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
public class SVMDataParallel extends model.SVM{
    public static long start;

    public DenseVector globalModel;
    public static double trainRatio = 0.5;
    public static int threadNum;
    public static double lambda = 0.1;

    public static double learningRate = 0.001;
    public int iteration = 0;
    public DenseVector localModel[];


    public void setNewLearningRate(){
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
            sgdOneEpoch(localList, learningRate, lambda, globalCorpusSize);
        }
        public void sgdOneEpoch(List<LabeledData> list, double lr, double lambda, double globalCorpusSize) {
            double modelPenalty = -2 * lr * lambda / globalCorpusSize;
            //double modelPenalty = - 2 * lr * lambda;
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
            globalModel.plusDense(localModel[threadID]);
        }
    }

    public void train(List<LabeledData> corpus, DenseVector model) {
        Collections.shuffle(corpus);
        List<List<LabeledData>> ThreadTrainCorpus = new ArrayList<List<LabeledData>>();
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
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

        long totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            testAndSummary(trainCorpus, testCorpus, model, lambda);
            Arrays.fill(globalModel.values, 0);
            long startTrain = System.currentTimeMillis();
            System.out.println("[Information]Learning rate " + learningRate);
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
            globalModel.allDividedBy(threadNum);

            System.arraycopy(globalModel.values, 0, model.values, 0, model.dim);

            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("[Information]trainTime " + trainTime);
            totalIterationTime += trainTime;
            System.out.println("[Information]totalTrainTime " + totalIterationTime);
            System.out.println("[Information]totalTime " + (System.currentTimeMillis() - totalBegin));
            System.out.println("[Information]HeapUsed " + ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed()
                    / 1024 / 1024 + "M");
            System.out.println("[Information]MemoryUsed " + (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                    / 1024 / 1024 + "M");
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
            if(converge(oldModel, model)){
                if (modelType == 2)
                    break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            for(int id = 0; id < threadNum; id++){
                System.arraycopy(globalModel.values, 0, localModel[id].values, 0, model.dim);
            }
        }
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.SVM threadNum dim train_path lambda learningRate [trainRatio]");
        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
        System.out.println(df.format(new Date()));// new Date()为获取当前系统时间
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        learningRate = Double.parseDouble(argv[4]);
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
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
        System.out.println("[Parameter]FeatureDimension " + dim);
        System.out.println("[Parameter]LearningRate " + learningRate);
        System.out.println("[Parameter]File Path " + path);
        System.out.println("[Parameter]Lambda " + lambda);
        System.out.println("[Parameter]TrainRatio " + trainRatio);
        System.out.println("[Parameter]TimeLimit " + maxTimeLimit);
        System.out.println("[Parameter]ModelType " + modelType);
        System.out.println("[Parameter]Iteration Limit " + maxIteration);
        System.out.println("------------------------------------");

        SVMDataParallel svm = new SVMDataParallel();
        DenseVector model = new DenseVector(dim);
        start = System.currentTimeMillis();
        svm.train(corpus, model);

        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }
}