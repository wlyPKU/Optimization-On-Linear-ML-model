package parallelGD;

import Utils.LabeledData;
import Utils.Utils;
import javafx.scene.control.DatePicker;
import math.DenseVector;

import java.util.List;

public class LinearRegression1_tDecay extends LinearRegression{

    static double initalLearningRate = 0.01;
    double decayRate = 1;

    public void setNewLearningRate(){
        learningRate = initalLearningRate / (1.0 + iteration * decayRate);
    }

    public static void main(String[] argv) throws Exception {
        System.out.println("Usage: parallelGD.LinearRegression1_tDecay threadNum dim train_path learningRate[trainRatio]");
        threadNum = Integer.parseInt(argv[0]);
        int dim = Integer.parseInt(argv[1]);
        String path = argv[2];
        initalLearningRate = Double.parseDouble(argv[3]);
        learningRate = initalLearningRate;
        long startLoad = System.currentTimeMillis();
        List<LabeledData> corpus = Utils.loadLibSVM(path, dim);
        long loadTime = System.currentTimeMillis() - startLoad;
        System.out.println("Loading corpus completed, takes " + loadTime + " ms");
        if(argv.length >= 5){
            trainRatio = Double.parseDouble(argv[4]);
            if(trainRatio >= 1 || trainRatio <= 0){
                System.out.println("Error Train Ratio!");
                System.exit(1);
            }
        }
        for(int i = 0; i < argv.length - 1; i++){
            if(argv[i].equals("EarlyStop")){
                earlyStop = Boolean.parseBoolean(argv[i + 1]);
            }
            if(argv[i].equals("TimeLimit")){
                maxTimeLimit = Double.parseDouble(argv[i + 1]);
            }
        }
        LinearRegression1_tDecay linear = new LinearRegression1_tDecay();
        //https://www.microsoft.com/en-us/research/wp-content/uploads/2012/01/tricks-2012.pdf  Pg 3.
        DenseVector model = new DenseVector(dim);
        start = System.currentTimeMillis();
        linear.train(corpus, model);
        long cost = System.currentTimeMillis() - start;
        System.out.println(cost + " ms");
    }
}
