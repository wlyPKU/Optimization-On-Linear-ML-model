package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;

import java.util.List;

/**
 * Created by WLY on 2016/9/4.
 */
public class SVMStepDecay extends SVM{

    double decayRate = 0.7;
    int dacayIteration = 20;
    static double learningRate = 0.01;

    public void setNewLearningRate(){
        if(iteration % dacayIteration == 0){
            learningRate *= decayRate;
        }
    }


    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.SVMStepDecay threadNum dim train_path lambda learningRate [trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        lambda = Double.parseDouble(argv[3]);
        learningRate = Double.parseDouble(argv[4]);
        for(int i = 0; i < argv.length - 1; i++){
            if(argv[i].equals("EarlyStop")){
                earlyStop = Boolean.parseBoolean(argv[i + 1]);
            }
            if(argv[i].equals("TimeLimit")){
                maxTimeLimit = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("StopDelta")){
                stopDelta = Double.parseDouble(argv[i + 1]);
            }
            if(argv[i].equals("TrainRatio")){
                trainRatio = Double.parseDouble(argv[i+1]);
                if(trainRatio >= 1 || trainRatio <= 0){
                    System.out.println("Error Train Ratio!");
                    System.exit(1);
                }            }
        }
        System.out.println("ThreadNum " + threadNum);
        System.out.println("StopDelta " + stopDelta);
        System.out.println("FeatureDimension " + dim);
        System.out.println("LearningRate " + learningRate);
        System.out.println("File Path " + path);
        System.out.println("Lambda " + lambda);
        System.out.println("TrainRatio " + trainRatio);
        System.out.println("TimeLimit " + maxTimeLimit);
        System.out.println("EarlyStop " + earlyStop);
        SVMStepDecay svm = new SVMStepDecay();
        DenseVector model = new DenseVector(dim);
        start = System.currentTimeMillis();
        svm.train(corpus, model);

        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
