package model;

import Utils.LabeledData;
import Utils.Utils;
import math.DenseVector;
import java.util.List;

/**
 * Created by 王羚宇 on 2016/8/7.
 */
public class LinearRegression {
    public void testAndSummary(List<LabeledData>trainCorpus, List<LabeledData> testCorpus,
                                DenseVector model){
        long startTest = System.currentTimeMillis();
        double loss = test(trainCorpus, model);
        double accuracy = test(testCorpus, model);
        long testTime = System.currentTimeMillis() - startTest;
        System.out.println("trainLoss=" + loss + " testLoss=" + accuracy +
                " testTime=" + testTime);
        double []trainAccuracy = Utils.LinearAccuracy(trainCorpus, model);
        double []testAccuracy = Utils.LinearAccuracy(testCorpus, model);
        System.out.println("trainAccuracy:");
        Utils.printAccuracy(trainAccuracy);
        System.out.println("testAccuracy:");
        Utils.printAccuracy(testAccuracy);
    }


    public double test(List<LabeledData> list, DenseVector model) {
        double residual = 0;
        for (LabeledData labeledData : list) {
            double dot_prod = model.dot(labeledData.data);
            residual += 0.5 * Math.pow(labeledData.label - dot_prod, 2);
        }
        return residual;
    }
    public boolean converge(DenseVector oldModel, DenseVector newModel){
        double delta = 0;
        for(int i = 0; i < oldModel.values.length; i++){
            delta += Math.pow(oldModel.values[i] - newModel.values[i], 2);
        }
        System.out.println("This iteration average changes " + delta);
        if(delta < 0.00001){
            return true;
        }else{
            return false;
        }
    }
}
