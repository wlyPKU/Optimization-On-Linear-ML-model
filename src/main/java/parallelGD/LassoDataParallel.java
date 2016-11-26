
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
 * Created by 王羚宇 on 2016/7/20.
 */
public class LassoDataParallel extends model.Lasso{
    public static long start;

    public DenseVector globalModelOfU;
    public DenseVector globalModelOfV;
    public DenseVector localModelOfU[];
    public DenseVector localModelOfV[];
    public static int threadNum;
    public static double lambda = 0.1;
    public static double trainRatio = 0.5;

    public static double learningRate = 0.005;
    public int iteration = 0;

    public void setNewLearningRate(){
    }

    public class executeRunnable implements Runnable
    {
        public List<LabeledData> localList;
        public double lambda;
        public int globalCorpusSize;
        int threadID;
        public executeRunnable(int threadID, List<LabeledData> list, double lambda, int globalCorpusSize){
            this.threadID = threadID;
            localList = list;
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;

        }
        public void run() {
            sgdOneEpoch(localList, learningRate, lambda);
        }
        public void sgdOneEpoch(List<LabeledData> list, double lr, double lambda) {
            //Collections.shuffle(list);
            double modelPenalty = -lr * lambda / globalCorpusSize;
            //double modelPenalty = - lr * lambda;
            for (LabeledData labeledData: list) {
                double scala = labeledData.label - localModelOfU[threadID].dot(labeledData.data)
                        + localModelOfV[threadID].dot(labeledData.data);
                localModelOfU[threadID].plusSparse(labeledData.data, modelPenalty);
                localModelOfU[threadID].plusGradient(labeledData.data, scala * lr);
                localModelOfU[threadID].positiveOrZero(labeledData.data);
                localModelOfV[threadID].plusSparse(labeledData.data, modelPenalty);
                localModelOfV[threadID].plusGradient(labeledData.data, - scala * lr);
                localModelOfV[threadID].positiveOrZero(labeledData.data);

            }
            globalModelOfU.plusDense(localModelOfU[threadID]);
            globalModelOfV.plusDense(localModelOfV[threadID]);
        }
    }

    public void train(List<LabeledData> corpus, DenseVector modelOfU, DenseVector modelOfV) {
        Collections.shuffle(corpus);
        List<List<LabeledData>> ThreadTrainCorpus = new ArrayList<List<LabeledData>>();
        int size = corpus.size();
        int end = (int) (size * trainRatio);
        List<LabeledData> trainCorpus = corpus.subList(0, end);
        List<LabeledData> testCorpus = corpus.subList(end, size);
        localModelOfU = new DenseVector[threadNum];
        localModelOfV = new DenseVector[threadNum];
        for(int i = 0; i < threadNum; i++){
            localModelOfV[i] = new DenseVector(modelOfU.dim);
            localModelOfU[i] = new DenseVector(modelOfV.dim);
        }
        for(int threadID = 0; threadID < threadNum; threadID++){
            int from = end * threadID / threadNum;
            int to = end * (threadID + 1) / threadNum;
            List<LabeledData> threadCorpus = corpus.subList(from, to);
            ThreadTrainCorpus.add(threadCorpus);
        }
        DenseVector model = new DenseVector(modelOfU.dim);
        DenseVector oldModel = new DenseVector(modelOfU.dim);

        globalModelOfU = new DenseVector(modelOfU.dim);
        globalModelOfV = new DenseVector(modelOfU.dim);
        long totalBegin = System.currentTimeMillis();

        int totalIterationTime = 0;
        for (int i = 0; ; i ++) {
            System.out.println("[Information]Iteration " + i + " ---------------");
            boolean diverge = testAndSummary(trainCorpus, testCorpus, model, lambda);
            Arrays.fill(globalModelOfU.values, 0);
            Arrays.fill(globalModelOfV.values, 0);
            long startTrain = System.currentTimeMillis();
            System.out.println("[Information]Learning rate " + learningRate);
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(threadID, ThreadTrainCorpus.get(threadID), lambda, trainCorpus.size()));
            }
            globalModelOfV.allDividedBy(threadNum);
            globalModelOfU.allDividedBy(threadNum);
            threadPool.shutdown();
            while (!threadPool.isTerminated()) {
                try {
                    threadPool.awaitTermination(1, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    System.out.println("Waiting.");
                    e.printStackTrace();
                }
            }

            for(int j = 0; j < modelOfU.dim; j++){
                model.values[j] = globalModelOfU.values[j] - globalModelOfV.values[j];
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
            if(diverge){
                System.out.println("[Warning]Diverge happens!");
                break;
            }
            for(int id = 0; id < threadNum; id++){
                System.arraycopy(globalModelOfU.values, 0, localModelOfU[id].values, 0, globalModelOfU.dim);
                System.arraycopy(globalModelOfV.values, 0, localModelOfV[id].values, 0, globalModelOfV.dim);
            }
        }
    }



    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.Lasso threadNum dim train_path lambda learningRate [trainRatio]");
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

        LassoDataParallel lasso = new LassoDataParallel();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV);
        long cost = System.currentTimeMillis() - start;
        System.out.println("[Information]Training cost " + cost + " ms totally.");
    }
}
