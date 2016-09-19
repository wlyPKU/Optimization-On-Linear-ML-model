package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class LinearRegressionRMSprop extends model.LinearRegression{
    public DenseVector globalModel;
    public static double trainRatio = 0.5;
    public static int threadNum;

    public int iteration = 0;
    double epsilon = 1e-8;

    double gamma = 0.9;
    public double[][]G2;
    public double learningRate = 0.001;

    public void setNewLearningRate(){
    }
    public class executeRunnable implements Runnable
    {
        List<LabeledData> localList;
        DenseVector localModel;
        int threadID;
        public executeRunnable(List<LabeledData> list, DenseVector model, int threadID){
            localList = list;
            localModel = new DenseVector(model.dim);
            System.arraycopy(model.values, 0, localModel.values, 0, model.dim);
            this.threadID = threadID;

        }
        public void run() {
            sgdOneEpoch(localList, localModel,learningRate);
            globalModel.plusDense(localModel);
        }
        public void sgdOneEpoch(List<LabeledData> list, DenseVector model, double lr) {
            for (LabeledData labeledData: list) {
                double scala = labeledData.label - model.dot(labeledData.data);
                for(int i = 0; i < labeledData.data.indices.length; i++){
                    double gradient;
                    if(labeledData.data.values != null) {
                        gradient = scala * labeledData.data.values[i];
                    }else{
                        gradient = scala;
                    }
                    G2[threadID][labeledData.data.indices[i]] *= gamma;
                    G2[threadID][labeledData.data.indices[i]] += (1 - gamma) * gradient * gradient;

                    model.values[labeledData.data.indices[i]] += lr * gradient
                        / Math.sqrt(G2[threadID][labeledData.data.indices[i]] + epsilon);
                }
            }
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
        DenseVector oldModel = new DenseVector(model.dim);

        globalModel = new DenseVector(model.dim);

        G2 = new double[threadNum][model.dim];
        for(int i = 0; i < threadNum; i++){
            for(int j = 0; j < model.dim; j++){
                G2[i][j] = 0;
            }
        }
        long totalBegin = System.currentTimeMillis();

        for (int i = 0; i < 200; i ++) {
            long startTrain = System.currentTimeMillis();
            //TODO StepSize tuning:  c/k(k=0,1,2...) or backtracking line search
            ExecutorService threadPool = Executors.newFixedThreadPool(threadNum);
            for (int threadID = 0; threadID < threadNum; threadID++) {
                threadPool.execute(new executeRunnable(ThreadTrainCorpus.get(threadID), model, threadID));
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
            System.out.println("trainTime " + trainTime + " ");
            testAndSummary(trainCorpus, testCorpus, model);
            if(converge(oldModel, model)){
                //break;
            }
            System.arraycopy(model.values, 0, oldModel.values, 0, oldModel.values.length);
            Arrays.fill(globalModel.values, 0);
            System.out.println("totaltime " + (System.currentTimeMillis() - totalBegin) );

            iteration++;
            setNewLearningRate();
        }
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.LinearRegressionRMSprop threadNum dim train_path [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        if(argv.length >= 4){
            trainRatio = Double.parseDouble(argv[3]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }

        LinearRegressionRMSprop linear = new LinearRegressionRMSprop();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dim);
        long start = System.currentTimeMillis();
        linear.train(corpus, model);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
