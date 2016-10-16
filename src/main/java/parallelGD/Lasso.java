
package parallelGD;

import Utils.*;
import math.DenseVector;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by 王羚宇 on 2016/7/20.
 */
public class Lasso extends model.Lasso{
    private static long start;

    public DenseVector globalModelOfU;
    public DenseVector globalModelOfV;
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
        public DenseVector localModelOfU;
        public DenseVector localModelOfV;
        public double lambda;
        public int globalCorpusSize;
        public executeRunnable(List<LabeledData> list, DenseVector modelOfU, DenseVector modelOfV, double lambda, int globalCorpusSize){
            localList = list;
            localModelOfU = new DenseVector(modelOfU.dim);
            localModelOfV = new DenseVector(modelOfV.dim);
            System.arraycopy(modelOfU.values, 0, localModelOfU.values, 0, modelOfU.dim);
            System.arraycopy(modelOfV.values, 0, localModelOfV.values, 0, modelOfU.dim);
            this.lambda = lambda;
            this.globalCorpusSize = globalCorpusSize;

        }
        public void run() {
            sgdOneEpoch(localList, localModelOfU, localModelOfV, learningRate, lambda);
            globalModelOfU.plusDense(localModelOfU);
            globalModelOfV.plusDense(localModelOfV);
        }
        public void sgdOneEpoch(List<LabeledData> list, DenseVector modelOfU,
                                 DenseVector modelOfV, double lr, double lambda) {
            double modelPenalty = -lr * lambda / globalCorpusSize;
            for (LabeledData labeledData: list) {
                double scala = labeledData.label - modelOfU.dot(labeledData.data)
                        + modelOfV.dot(labeledData.data);
                modelOfU.plusSparse(labeledData.data, modelPenalty);
                modelOfU.plusGradient(labeledData.data, scala * lr);
                modelOfU.positiveOrZero(labeledData.data);
                scala = labeledData.label - modelOfU.dot(labeledData.data)
                        + modelOfV.dot(labeledData.data);
                modelOfV.plusGradient(labeledData.data, - scala * lr);
                modelOfV.plusSparse(labeledData.data, modelPenalty);
                modelOfV.positiveOrZero(labeledData.data);
            }
        }
    }

    public void train(List<LabeledData> corpus, DenseVector modelOfU,
                      DenseVector modelOfV) {
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
        DenseVector model = new DenseVector(modelOfU.dim);
        DenseVector oldModel = new DenseVector(model.dim);

        globalModelOfU = new DenseVector(modelOfU.dim);
        globalModelOfV = new DenseVector(modelOfV.dim);

        long totalBegin = System.currentTimeMillis();

        for (int i = 0; ; i ++) {
            long startTrain = System.currentTimeMillis();
            System.out.println("learning rate " + learningRate);
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(ThreadTrainCorpus.get(threadID),
                        modelOfU, modelOfV, lambda, corpus.size()));
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
            globalModelOfU.allDividedBy(threadNum);
            globalModelOfV.allDividedBy(threadNum);
            System.arraycopy(globalModelOfU.values, 0, modelOfU.values, 0, modelOfU.dim);
            System.arraycopy(globalModelOfV.values, 0, modelOfV.values, 0, modelOfV.dim);

            for(int j = 0; j < model.dim; j++){
                model.values[j] = modelOfU.values[j] - modelOfV.values[j];
            }
            long trainTime = System.currentTimeMillis() - startTrain;
            System.out.println("trainTime " + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model, lambda);

            if(converge(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            Arrays.fill(globalModelOfU.values, 0);
            Arrays.fill(globalModelOfV.values, 0);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );

            iteration++;
            setNewLearningRate();
            long nowCost = System.currentTimeMillis() - start;
            if(nowCost > 60000) {
                break;
                //break;
            }
        }
    }

    public static void train(List<LabeledData> corpus) {
        int dim = corpus.get(0).data.dim;
        Lasso lasso = new Lasso();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.Lasso threadNum dim train_path lambda learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        lambda = Double.parseDouble(argv[3]);
        learningRate = Double.parseDouble(argv[4]);
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        if(argv.length >= 6){
            trainRatio = Double.parseDouble(argv[5]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }

        Lasso lasso = new Lasso();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector modelOfU = new DenseVector(dim);
        DenseVector modelOfV = new DenseVector(dim);
        start = System.currentTimeMillis();
        lasso.train(corpus, modelOfU, modelOfV);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
